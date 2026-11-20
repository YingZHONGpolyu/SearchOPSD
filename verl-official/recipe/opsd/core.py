"""Paper section 3.2: prompts, exact-match rewards, pair mining and token masks.

All routines here are independent of a GPU so that the scientific choices can be
tested separately from distributed training. Paper text is data, not instructions.
"""

import difflib
import json
import re
from collections import defaultdict
from pathlib import Path


# The Base checkpoint has no conversation fine-tuning: use a completion prompt.
INSTRUCTION = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search> and it "
    "will return the top searched results between <information> and </information>. "
    "If you find no further external knowledge needed, provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
)


def student_prompt(question):
    """Inference context contains only the task and question (paper Eq. 3)."""
    return INSTRUCTION + "Question: " + question.strip() + "\n"


def teacher_prompt(question, reference):
    """Privileged context exposes a sibling, followed by the SAME student suffix."""
    return (
        student_prompt(question)
        + "\nExpert reference trajectory:\n"
        + reference
        + "\nRead the reference, then solve the same question in your own words.\n"
    )


def exact_match(trajectory, aliases):
    """Score the last complete answer using veRL's QA normalization, without bonuses."""
    from verl.utils.reward_score.search_r1_like_qa_em import em_check

    answers = re.findall(r"<answer>(.*?)</answer>", trajectory, flags=re.S)
    return int(bool(answers) and em_check(answers[-1].strip(), aliases))


def read_jsonl(path):
    """Read explicit JSONL inputs; never evaluate serialized Python objects."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    """Use exclusive creation to protect existing rollout pools and experiment files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def character_distance(left, right):
    """Deterministic character dissimilarity; the paper does not specify a metric.

    1 - SequenceMatcher ratio is our documented reproduction choice. Disable its
    popular-character heuristic so repeated XML tags do not distort the ranking.
    """
    return 1.0 - difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def mine_pairs(rows):
    """Pick shortest-search correct reference and most different wrong sibling.

    If all siblings are correct, use the most different remaining correct one.
    Questions without a correct rollout, or without another sibling, are dropped.
    Count actual retrieval calls when collection supplied that value.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["reward"] not in (0, 1):
            raise ValueError("OPSD mining requires binary EM rewards")
        groups[row["question"]].append(row)
    pairs = []
    stats = {"questions": len(groups), "no_correct": 0, "no_sibling": 0}
    for question, siblings in groups.items():
        correct = [i for i, row in enumerate(siblings) if row["reward"] == 1]
        if not correct:
            stats["no_correct"] += 1
            continue
        if len(siblings) < 2:
            stats["no_sibling"] += 1
            continue
        ref_index = min(correct, key=lambda i: (
            siblings[i].get("search_calls", siblings[i]["trajectory"].count("<search>")),
            len(siblings[i].get("response_ids", siblings[i]["trajectory"])), i,
        ))
        ref = siblings[ref_index]
        candidates = [i for i, row in enumerate(siblings) if row["reward"] == 0]
        if not candidates:
            candidates = [i for i in correct if i != ref_index]
        index = max(candidates, key=lambda i: character_distance(
            ref["trajectory"], siblings[i]["trajectory"]
        ))
        stu = siblings[index]
        if ref.get("policy_model") != stu.get("policy_model"):
            raise ValueError("Sibling trajectories must come from the same policy")
        pair = {
            "question": question, "reference": ref["trajectory"],
            "student": stu["trajectory"], "policy_model": stu.get("policy_model"),
            "aliases": stu.get("aliases", []),
            "reference_reward": 1, "student_reward": stu["reward"],
            "reference_search_calls": ref.get("search_calls", ref["trajectory"].count("<search>")),
        }
        # Preserve sampled IDs and environment masks instead of re-tokenizing a rollout.
        for field in ("response_ids", "policy_mask", "action_stats"):
            if field in stu:
                pair[field] = stu[field]
        pairs.append(pair)
    stats["pairs"] = len(pairs)
    return pairs, stats


def response_tokens(tokenizer, pair):
    """Reuse exact sampled IDs; imported text uses conservative character offsets.

    Environment delimiters and their contents are excluded. A token overlapping
    an environment boundary is excluded too, preventing partial-token leakage.
    An unclosed information span is masked through the end of the trajectory.
    """
    if "response_ids" in pair:
        ids, mask = pair["response_ids"], pair["policy_mask"]
        if len(ids) != len(mask) or any(x not in (0, 1) for x in mask):
            raise ValueError("Invalid sampled token mask")
        return ids, mask
    text = pair["student"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = [(m.start(), m.end()) for m in re.finditer(
        r"<information>.*?(?:</information>|\Z)", text, flags=re.S
    )]
    mask = [int(b > a and not any(a < end and b > start for start, end in spans))
            for a, b in encoded["offset_mapping"]]
    return encoded["input_ids"], mask


def encode_pair(tokenizer, pair, max_length):
    """Align causal predictions by response index despite different prefix lengths.

    Never truncate a reference or student silently: truncation changes the target
    method and may remove the answer. Let the caller skip overlength examples.
    """
    ids, mask = response_tokens(tokenizer, pair)
    sp = tokenizer.encode(student_prompt(pair["question"]), add_special_tokens=False)
    tp = tokenizer.encode(teacher_prompt(pair["question"], pair["reference"]), add_special_tokens=False)
    if max(len(sp), len(tp)) + len(ids) > max_length:
        return None
    positions = [i for i, active in enumerate(mask) if active]
    if not positions:
        return None
    return {
        "action_stats": pair.get("action_stats"),
        "student_ids": sp + ids, "teacher_ids": tp + ids,
        # Position p-1 predicts response token p; prefix lengths differ on each side.
        "student_positions": [len(sp) - 1 + i for i in positions],
        "teacher_positions": [len(tp) - 1 + i for i in positions],
    }
