"""Frozen-policy search rollout collection, and held-out EM evaluation."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, set_seed

from .core import exact_match, read_jsonl, student_prompt
from .collection import collection_output, persist_trajectory, trajectory_seed


def load_questions(path, limit=0, offset=0, *, stratified=False, seed=42):
    """Accept simple QA JSONL or existing Search-R1 parquet, without chat templates."""
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    else:
        rows = read_jsonl(path)
    output = []
    if stratified and offset:
        raise ValueError("Stratified validation does not support a prefix offset")
    selected = rows if stratified else (rows[offset:offset + limit] if limit else rows[offset:])
    for row in selected:
        question = row.get("question") or (row.get("extra_info") or {}).get("question")
        if not question:
            # Search-R1 parquet stores the original Base completion prompt in a user message.
            content = row["prompt"][-1]["content"]
            if "Question: " not in content:
                raise ValueError("Cannot extract question from parquet prompt")
            question = content.rsplit("Question: ", 1)[1].strip()
        aliases = row.get("aliases", row.get("golden_answers"))
        if aliases is None:
            aliases = row.get("reward_model", {}).get("ground_truth", {})
            if isinstance(aliases, dict):
                aliases = aliases.get("target", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        if not aliases:
            raise ValueError("QA rows require non-empty answer aliases")
        item = {"question": question, "aliases": list(aliases)}
        if row.get("data_source"):
            item["data_source"] = row["data_source"]
        output.append(item)
    if not output:
        raise ValueError("No questions selected")
    return stratified_questions(output, limit, seed) if stratified else output


def stratified_questions(questions, limit, seed=42):
    """Select a fixed, balanced validation subset without relying on input ordering."""
    if not all(item.get("data_source") for item in questions):
        raise ValueError("Stratified validation requires data_source; regenerate the QA split")
    groups = {}
    for item in questions:
        groups.setdefault(item["data_source"], []).append(item)
    if limit and limit < len(groups):
        raise ValueError("Validation limit is too small to include every data source")
    def key(item):
        return hashlib.sha256(f'{seed}:{item["data_source"]}:{item["question"]}'.encode()).hexdigest()
    groups = {source: sorted(items, key=key) for source, items in groups.items()}
    target = min(limit, len(questions)) if limit else len(questions)
    selected = []
    position = 0
    while len(selected) < target:
        for source in sorted(groups):
            if position < len(groups[source]):
                selected.append(groups[source][position])
                if len(selected) == target:
                    break
        position += 1
    return selected


class EndOfAction(StoppingCriteria):
    """Stop generation after an actual action, not after a fixed token fragment."""

    def __init__(self, tokenizer, prefix_length, allow_search=True):
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.allow_search = allow_search

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the new suffix: the prompt itself contains the XML examples.
        text = self.tokenizer.decode(input_ids[0, self.prefix_length:], skip_special_tokens=True)
        return "</answer>" in text or (self.allow_search and "</search>" in text)


def retrieve(url, query, topk=3):
    """Use the standard Search-R1 /retrieve API; failures abort rather than fake evidence."""
    payload = json.dumps({"queries": [query], "topk": topk, "return_scores": True}).encode()
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        docs = json.load(response)["result"][0]
    formatted = []
    for i, hit in enumerate(docs[:topk], 1):
        doc = hit.get("document", hit)
        contents = doc.get("contents", doc.get("text", ""))
        if not contents:
            raise ValueError("Retriever returned a document without contents")
        # Escape tag delimiters so document text cannot forge trajectory boundaries.
        contents = contents.replace("<", "&lt;").replace(">", "&gt;")
        formatted.append(f"Doc {i}: {contents}")
    return "\n".join(formatted)


FINAL_INSTRUCTION = (
    "Stop searching. Give the final answer now, using the required answer tags "
    "and the information already available."
)


def environment_ids(tokenizer, evidence):
    """Keep all environment scaffolding distinguishable from policy-generated tokens."""
    return tokenizer.encode("\n<information>\n" + evidence + "\n</information>\n",
                            add_special_tokens=False)


@torch.no_grad()
def generate_trajectory(model, tokenizer, question, retriever_url, max_searches=4,
                        max_new_tokens=2048, max_length=8192, temperature=1.0,
                        final_answer_reserve=128):
    """Generate within both budgets; use a no-search final phase after the search cap.

    A small part of the existing policy/context budgets is reserved for a final
    answer, never added on top of the configured limits. Observations are kept
    whole or rejected with an explicit failure reason, not silently truncated.
    """
    if min(max_new_tokens, max_length) <= 0 or max_searches < 0 or final_answer_reserve < 0:
        raise ValueError("Invalid rollout budgets")
    prefix = tokenizer.encode(student_prompt(question), add_special_tokens=False)
    response_ids, policy_mask = [], []
    generated, calls = 0, 0
    context_limit, final_only, final_notice_added = False, max_searches == 0, False
    turns = valid_actions = valid_searches = 0
    reserve = min(final_answer_reserve, max_new_tokens // 4)
    notice = environment_ids(tokenizer, FINAL_INSTRUCTION)
    reason = "policy_token_limit"
    while generated < max_new_tokens:
        room = max_length - len(prefix) - len(response_ids)
        if room <= 0:
            context_limit, reason = True, "context_limit"
            break
        if not final_only:
            budget = min(max_new_tokens - generated - reserve, room - reserve - len(notice))
            if budget <= 0:
                final_only = True
        if final_only:
            if not final_notice_added:
                # Leave at least one genuine policy token; never fabricate an answer.
                if len(notice) >= room:
                    context_limit, reason = True, "final_prompt_context_limit"
                    break
                response_ids.extend(notice)
                policy_mask.extend([0] * len(notice))
                final_notice_added = True
                room -= len(notice)
            budget = min(max_new_tokens - generated, room)
        if budget <= 0:
            reason = "policy_token_limit"
            break
        context = prefix + response_ids
        inputs = torch.tensor([context], device=next(model.parameters()).device)
        kwargs = {"do_sample": temperature > 0}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        if final_only:
            # Constrain the terminal phase as well as prompting it. Search actions
            # never trigger a retriever here, including alternative tokenizations.
            forbidden = [tokenizer.encode(text, add_special_tokens=False)
                         for text in ("<search>", " <search>", "\n<search>")]
            kwargs["bad_words_ids"] = [ids for ids in forbidden if ids]
        output = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=True,
            max_new_tokens=budget, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([
                EndOfAction(tokenizer, len(context), allow_search=not final_only)
            ]), **kwargs,
        )[0, len(context):].tolist()
        if len(output) > budget:
            raise RuntimeError("Generator exceeded its max_new_tokens budget")
        if not output:
            reason = "empty_generation"
            break
        turns += 1
        response_ids.extend(output)
        policy_mask.extend([1] * len(output))
        generated += len(output)
        text = tokenizer.decode(output, skip_special_tokens=True)
        if "</answer>" in text:
            full_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            reason = "answer" if re.search(r"<answer>.*?</answer>", full_text, re.S) else "malformed_answer"
            valid_actions += int(reason == "answer")
            break
        if output[-1] == tokenizer.eos_token_id:
            reason = "eos_without_answer"
            break
        if final_only:
            # A rejected search must not terminate the final-answer phase. The
            # remaining token budget bounds even a non-compliant generator.
            continue
        queries = re.findall(r"<search>(.*?)</search>", text, re.S)
        if not queries:
            if len(output) == budget:
                final_only = True
                continue
            reason = "invalid_action"
            break
        evidence = retrieve(retriever_url, queries[-1].strip())
        calls += 1
        valid_actions += 1
        valid_searches += 1
        final_only = calls >= max_searches
        if final_only:
            evidence += "\n" + FINAL_INSTRUCTION
        observation = environment_ids(tokenizer, evidence)
        room = max_length - len(prefix) - len(response_ids)
        # Also leave space for the final instruction when searches remain.
        required = len(observation) + max(1, reserve) + (0 if final_only else len(notice))
        if required > room:
            context_limit, reason = True, "observation_context_limit"
            break
        response_ids.extend(observation)
        policy_mask.extend([0] * len(observation))
        final_notice_added = final_only
    else:
        reason = "policy_token_limit"
    if reason != "answer" and len(prefix) + len(response_ids) >= max_length:
        context_limit, reason = True, "context_limit"
    return {
        "trajectory": tokenizer.decode(response_ids, skip_special_tokens=True),
        "response_ids": response_ids, "policy_mask": policy_mask,
        "search_calls": calls, "context_limit": context_limit,
        "termination_reason": reason, "forced_final": final_only,
        "policy_tokens": generated,
        "action_stats": dict(turns=turns, valid_actions=valid_actions, valid_searches=valid_searches),
    }


def collect(model, tokenizer, questions, output, policy_model, k=5, *, resume=False, seed=42, **kwargs):
    """Collect or safely resume a fixed pool, using an independent seed for each sibling."""
    if not questions or k <= 0:
        raise ValueError("Need nonempty questions and positive k")
    with collection_output(output, policy_model, questions, k, seed, kwargs, resume) as state:
        stream, completed, policy_model = state
        for index, item in enumerate(questions):
            for sibling in range(k):
                if (index, sibling) in completed:
                    continue
                sample_seed = trajectory_seed(seed, index, item["question"], sibling)
                set_seed(sample_seed)
                result = generate_trajectory(model, tokenizer, item["question"], **kwargs)
                result.update(item, sibling=sibling, question_index=index,
                              policy_model=policy_model, sampling_seed=sample_seed)
                result["reward"] = exact_match(result["trajectory"], item["aliases"])
                persist_trajectory(stream, result)
                completed.add((index, sibling))
                print(json.dumps({"question": item["question"], "sibling": sibling,
                                  "reward": result["reward"], "search_calls": result["search_calls"],
                                  "completed": len(completed), "total": len(questions) * k}), flush=True)


def evaluate(model, tokenizer, questions, **kwargs):
    """Evaluate inference-time student contexts only; no references or answer leakage."""
    training = model.training
    model.eval()
    records = []
    try:
        for item in questions:
            result = generate_trajectory(model, tokenizer, item["question"], temperature=0, **kwargs)
            result.update(item, reward=exact_match(result["trajectory"], item["aliases"]))
            records.append(result)
    finally:
        model.train(training)
    return sum(row["reward"] for row in records) / len(records), records
