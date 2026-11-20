"""Mine OPSD pairs with the unchanged reference rules and exact GPU character scores."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
from queue import Queue, Empty
import threading
import time

import orjson
import torch

from .gpu_sequence_matcher import GPUSequenceMatcher, build_kernel


def mine_file(pool, output, devices=None, batch_size=1024):
    """Parallelize distance calculations across all visible GPUs and publish progress."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if batch_size <= 0:
        raise ValueError('GPU batch size must be positive')
    devices = list(range(torch.cuda.device_count())) if devices is None else list(devices)
    if not devices:
        raise RuntimeError('No CUDA devices selected for GPU pair mining')
    progress_path = Path(str(output) + '.progress.json')
    started = time.monotonic()
    def progress(phase, **values):
        record = dict(phase=phase, elapsed_seconds=time.monotonic() - started, devices=devices, **values)
        temporary = Path(str(progress_path) + '.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(progress_path)
        print(json.dumps(record), flush=True)

    with Path(str(output) + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        groups = defaultdict(list)
        loaded = 0
        with Path(pool).open('rb') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = orjson.loads(line)
                if row['reward'] not in (0, 1):
                    raise ValueError('OPSD mining requires binary EM rewards')
                groups[row['question']].append(row)
                loaded += 1
                if loaded % 10000 == 0:
                    progress('loading', loaded_trajectories=loaded, questions=len(groups))
        stats = dict(questions=len(groups), no_correct=0, no_sibling=0)
        selected, jobs, scores = [], [], []
        for question, siblings in groups.items():
            correct = [index for index, row in enumerate(siblings) if row['reward'] == 1]
            if not correct:
                stats['no_correct'] += 1
                continue
            if len(siblings) < 2:
                stats['no_sibling'] += 1
                continue
            ref_index = min(correct, key=lambda index: (
                siblings[index].get('search_calls', siblings[index]['trajectory'].count('<search>')),
                len(siblings[index].get('response_ids', siblings[index]['trajectory'])), index))
            reference = siblings[ref_index]
            candidates = [index for index, row in enumerate(siblings) if row['reward'] == 0]
            if not candidates:
                candidates = [index for index in correct if index != ref_index]
            candidate_scores = []
            for index in candidates:
                a, b = reference['trajectory'], siblings[index]['trajectory']
                score_index = len(scores)
                candidate_scores.append(score_index)
                # Exact trivial cases need no dynamic matching; all other scores go to CUDA.
                if len(candidates) == 1 or a == b:
                    scores.append(0.0)
                elif a in b or b in a:
                    scores.append(1.0 - 2.0 * min(len(a), len(b)) / (len(a) + len(b)))
                else:
                    scores.append(None)
                    jobs.append((score_index, a, b))
            selected.append((question, siblings, ref_index, candidates, candidate_scores))
        binary = build_kernel()
        # Expensive comparisons are scheduled first and dynamically shared by seven GPU workers.
        jobs.sort(key=lambda job: len(job[1]) * len(job[2]), reverse=True)
        queue = Queue()
        for offset in range(0, len(jobs), batch_size):
            queue.put(jobs[offset:offset + batch_size])
        completed, gpu_counts = 0, {}
        mutex = threading.Lock()
        progress('gpu_matching', total_comparisons=len(jobs), completed_comparisons=0,
                 candidate_questions=len(selected), loaded_trajectories=loaded)
        def worker(device):
            nonlocal completed
            matcher = GPUSequenceMatcher(device, binary)
            count = 0
            while True:
                try:
                    batch = queue.get_nowait()
                except Empty:
                    break
                values = matcher.distances([(a, b) for _, a, b in batch])
                for (index, _, _), value in zip(batch, values):
                    scores[index] = value
                count += len(batch)
                with mutex:
                    completed += len(batch)
                    gpu_counts[str(device)] = count
                    progress('gpu_matching', completed_comparisons=completed, total_comparisons=len(jobs),
                             per_gpu_comparisons=dict(gpu_counts), candidate_questions=len(selected))
            return count
        with ThreadPoolExecutor(len(devices)) as executor:
            list(executor.map(worker, devices))
        pairs = []
        for question, siblings, ref_index, candidates, candidate_scores in selected:
            # Python max preserves the original first-candidate tie-breaking exactly.
            position = max(range(len(candidates)), key=lambda index: scores[candidate_scores[index]])
            reference, student = siblings[ref_index], siblings[candidates[position]]
            if reference.get('policy_model') != student.get('policy_model'):
                raise ValueError('Sibling trajectories must come from the same policy')
            pair = dict(question=question, reference=reference['trajectory'], student=student['trajectory'],
                        policy_model=student.get('policy_model'), aliases=student.get('aliases', []),
                        reference_reward=1, student_reward=student['reward'],
                        reference_search_calls=reference.get('search_calls', reference['trajectory'].count('<search>')))
            for field in ('response_ids', 'policy_mask', 'action_stats'):
                if field in student:
                    pair[field] = student[field]
            pairs.append(pair)
        if not pairs:
            raise ValueError('No usable siblings')
        progress('writing', pairs=len(pairs), total_comparisons=len(jobs))
        temporary = Path(str(output) + f'.tmp.{os.getpid()}')
        with temporary.open('xb') as stream:
            for pair in pairs:
                stream.write(orjson.dumps(pair) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Atomically publish a complete pool without overwriting another writer's result.
        os.link(temporary, output)
        temporary.unlink()
        stats['pairs'] = len(pairs)
        Path(str(output) + '.stats.json').write_text(json.dumps(stats, indent=2))
        progress('complete', stats=stats, total_comparisons=len(jobs), per_gpu_comparisons=gpu_counts)
        return stats
