"""Paired checkpoint screening without training or test-set model selection."""
import argparse,collections,hashlib,json,math,random,re,time
from pathlib import Path
from .core import read_jsonl,write_jsonl,exact_match,mine_pairs,encode_pair
from .rollout import generate_trajectory

def prepare(out):
 import pyarrow.parquet as pq
 root=Path(out);root.mkdir(parents=True,exist_ok=False)
 # Match the prior question-level holdout, then stratify without inspecting model outputs.
 buckets=collections.defaultdict(dict)
 for row in pq.read_table('/external/data/nq_hotpotqa_train/train.parquet',columns=['id','question','golden_answers','data_source','metadata']).to_pylist():
  q=' '.join(row['question'].casefold().split())
  h=hashlib.sha256(f'42:{q}'.encode()).digest()
  if int.from_bytes(h[:8],'big')/2**64>=.02: continue
  buckets[row['data_source']].setdefault(q,row)
 selected=[]
 for source in ('nq','hotpotqa'):
  items=list(buckets[source].values())
  # Hash ordering gives a fixed uniform-looking subset, unrelated to correctness.
  items.sort(key=lambda x:hashlib.sha256(('opsd_compare_20261019:'+x['question']).encode()).hexdigest())
  for row in items[:32]:
   meta=row['metadata'] or {}
   selected.append({'question':row['question'],'aliases':row['golden_answers'],'data_source':source,'id':row['id'],'type':meta.get('type'),'level':meta.get('level')})
 assert len(selected)==64 and len({x['question'] for x in selected})==64
 write_jsonl(root/'questions.jsonl',selected)
 (root/'protocol.json').write_text(json.dumps({'questions':64,'per_source':32,'k':5,'temperature':1.,'top_p':1.,'top_k':0,'retrieval_topk':3,'max_searches':4,'max_length':8192,'max_new_tokens':2048,'sampling_batch_size':1,'selection':'question-level 2% holdout of training corpus; source-stratified hash sample','seed':'SHA256(question, sibling), identical for both checkpoints','training':False},indent=2))
 print('Prepared 64 paired questions',flush=True)

def collect(model_path,questions,out):
 from transformers import set_seed
 from .train import load_model
 model,tokenizer=load_model(model_path);model.eval()
 path=Path(out);existing=read_jsonl(path) if path.exists() else []
 done={(x['question'],x['sibling']) for x in existing}
 with path.open('a') as f:
  for item in read_jsonl(questions):
   for sibling in range(5):
    if (item['question'],sibling) in done: continue
    seed=int.from_bytes(hashlib.sha256(f'opsd-compare:{item["question"]}:{sibling}'.encode()).digest()[:4],'big')
    set_seed(seed);start=time.time()
    row=generate_trajectory(model,tokenizer,item['question'],'http://127.0.0.1:8123/retrieve',max_searches=4,max_new_tokens=2048,max_length=8192,temperature=1.)
    row.update(item,sibling=sibling,seed=seed,policy_model=model_path,elapsed_seconds=time.time()-start)
    row['reward']=exact_match(row['trajectory'],item['aliases'])
    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    print(json.dumps({'completed':len(done)+1,'total':320,'question':item['question'],'sibling':sibling,'reward':row['reward'],'seconds':row['elapsed_seconds']}),flush=True)
    done.add((item['question'],sibling))

def analyze(out):
 from transformers import AutoTokenizer
 from verl.utils.reward_score.search_r1_like_qa_em import normalize_answer
 root=Path(out);questions=read_jsonl(root/'questions.jsonl');all_details={};summary={}
 for step in (200,250):
  rows=read_jsonl(root/f'pool_{step}.jsonl')
  assert len(rows)==320
  groups=collections.defaultdict(list)
  for row in rows: groups[row['question']].append(row)
  pairs,stats=mine_pairs(rows)
  refs={x['question']:x for x in pairs}
  tokenizer=AutoTokenizer.from_pretrained(rows[0]['policy_model'],local_files_only=True,use_fast=True)
  details=[]
  for item in questions:
   group=sorted(groups[item['question']],key=lambda x:x['sibling']);assert len(group)==5
   assert len({x['sibling'] for x in group})==5
   answers=[]
   for x in group:
    found=re.findall(r'<answer>(.*?)</answer>',x['trajectory'],re.S)
    answers.append(normalize_answer(found[-1]) if found else None)
   counts=collections.Counter(a for a in answers if a is not None)
   repeated=max(counts.values(),default=0)
   pair=refs.get(item['question']);encoded=encode_pair(tokenizer,pair,8192) if pair else None
   details.append({**item,'first_correct':group[0]['reward'],'correct_candidates':sum(x['reward'] for x in group),'pass5':int(any(x['reward'] for x in group)),'normalized_answers':answers,'all_wrong_same_answer':int(not any(x['reward'] for x in group) and repeated==5),'all_wrong_majority_same':int(not any(x['reward'] for x in group) and repeated>=4),'teacher_reference_em':exact_match(pair['reference'],item['aliases']) if pair else None,'trainable_pair':int(encoded is not None),'student_length':len(encoded['student_ids']) if encoded else None,'teacher_length':len(encoded['teacher_ids']) if encoded else None})
  def metrics(ds):
   n=len(ds);ref=[d for d in ds if d['teacher_reference_em'] is not None]
   return {'n_questions':n,'single_first_accuracy':sum(d['first_correct'] for d in ds)/n,'mean_single_accuracy_all5':sum(d['correct_candidates'] for d in ds)/(5*n),'pass_at_5':sum(d['pass5'] for d in ds)/n,'correct_candidates':sum(d['correct_candidates'] for d in ds),'teacher_reference_count':len(ref),'teacher_reference_em':sum(d['teacher_reference_em'] for d in ref)/len(ref) if ref else None,'trainable_pair_count':sum(d['trainable_pair'] for d in ds),'all_wrong_same_answer_count':sum(d['all_wrong_same_answer'] for d in ds),'all_wrong_majority_same_count':sum(d['all_wrong_majority_same'] for d in ds)}
  summary[str(step)]={'overall':metrics(details),'by_source':{source:metrics([d for d in details if d['data_source']==source]) for source in ('nq','hotpotqa')},'mining_stats':stats}
  write_jsonl(root/f'details_{step}.jsonl',details);write_jsonl(root/f'pairs_{step}.jsonl',pairs);all_details[step]=details
 paired=[]
 for a,b in zip(all_details[200],all_details[250]):
  assert a['question']==b['question']
  paired.append({'question':a['question'],'data_source':a['data_source'],'only200':int(a['pass5'] and not b['pass5']),'only250':int(b['pass5'] and not a['pass5']),'pass5_diff_250_minus_200':b['pass5']-a['pass5'],'usable_diff_250_minus_200':b['trainable_pair']-a['trainable_pair'],'stuck250_only':int(b['all_wrong_majority_same'] and not a['all_wrong_majority_same'])})
 rng=random.Random(42);diffs=[]
 for _ in range(20000):
  # Paired bootstrap within each source preserves the sample's source balance.
  draw=[]
  for source in ('nq','hotpotqa'):
   items=[x for x in paired if x['data_source']==source];draw.extend(rng.choices(items,k=len(items)))
  diffs.append(sum(x['pass5_diff_250_minus_200'] for x in draw)/64)
 diffs.sort();low,high=diffs[500],diffs[19499]
 summary['paired']={'only200':sum(x['only200'] for x in paired),'only250':sum(x['only250'] for x in paired),'pass5_difference_250_minus_200_95pct_paired_bootstrap':[low,high],'stuck250_only':sum(x['stuck250_only'] for x in paired)}
 a,b=summary['200']['overall'],summary['250']['overall']
 if high<0 and a['trainable_pair_count']>b['trainable_pair_count'] and b['all_wrong_majority_same_count']>a['all_wrong_majority_same_count']:
  decision='Evidence favors 200 under the requested candidate-coverage and wrong-answer persistence rule; inspect paired trajectories.'
 elif b['pass_at_5']>=a['pass_at_5'] and b['trainable_pair_count']>=a['trainable_pair_count']:
  decision='Retain 250: candidate coverage and usable reference yield are no worse; a small sample cannot establish universal superiority.'
 else: decision='Inconclusive: expand the paired sample before switching away from 250; no training started.'
 summary['decision']=decision
 summary['limitations']=['EM checks final answers, not factual validity of all reasoning or documents.','Teacher-reference EM is gold-filtered by the actual mining rule, not independent evidence of calibrated teacher distributions.','64 balanced questions give a screening comparison, not a full benchmark or corpus-weighted estimate.','Usable pairs also require both contexts <=8192; full-sequence reasoning correctness is not measured.']
 (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));write_jsonl(root/'paired_comparison.jsonl',paired)
 lines=['# OPSD checkpoint screening','', '| Metric | step 200 | step 250 |','|---|---:|---:|']
 for key in a: lines.append(f'| {key} | {a[key]} | {b[key]} |')
 lines.extend(['',decision,'','Paired comparison: '+json.dumps(summary['paired']),*['- '+x for x in summary['limitations']]])
 (root/'report.md').write_text('\n'.join(lines));print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','collect','analyze']);p.add_argument('--output',required=True);p.add_argument('--model');p.add_argument('--questions');a=p.parse_args()
 if a.mode=='prepare': prepare(a.output)
 elif a.mode=='collect': collect(a.model,a.questions,a.output)
 else: analyze(a.output)
