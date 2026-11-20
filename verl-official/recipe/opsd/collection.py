"""Durable, provenance-checked checkpoints for ordinary OPSD collection."""

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def canonical(value):
    """Serialize protocol inputs identically across processes and restarts."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def trajectory_seed(seed, question_index, question, sibling):
    """Make each trajectory independent of prior samples and interrupted RNG state."""
    key = canonical([seed, question_index, question, sibling])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def collection_manifest(policy_model, questions, k, seed, kwargs):
    """Bind a pool to its checkpoint, ordered questions, sampler code and parameters."""
    model = Path(policy_model).resolve()
    files = []
    if model.is_dir():
        for path in sorted(model.iterdir()):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}:
                stat = path.stat()
                item = {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                if stat.st_size < 32 * 1024 * 1024:
                    item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                files.append(item)
    code = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("collection.py", "rollout.py", "core.py")}
    return {"schema_version": 1, "policy_model": str(model), "checkpoint_files": files,
            "questions_sha256": hashlib.sha256(canonical(questions).encode()).hexdigest(),
            "question_count": len(questions), "k": k, "seed": seed,
            "generation_parameters": kwargs, "sampler_code_sha256": code}


def recover_completed(path, questions, k, seed, policy_model):
    """Validate committed records; repair only a crash-damaged final JSON fragment."""
    completed = set()
    if not path.exists():
        return completed
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if not line.endswith(b"\n") and not stream.read(1):
                    stream.seek(start)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ValueError(f"Corrupt committed pool record at byte {start}") from None
            index, sibling = row.get("question_index"), row.get("sibling")
            if (type(index) is not int or not 0 <= index < len(questions)
                    or type(sibling) is not int or not 0 <= sibling < k):
                raise ValueError("Invalid question_index/sibling in resumed pool")
            expected_seed = trajectory_seed(seed, index, questions[index]["question"], sibling)
            if (row.get("question") != questions[index]["question"]
                    or row.get("aliases") != questions[index]["aliases"]
                    or row.get("policy_model") != policy_model
                    or row.get("sampling_seed") != expected_seed
                    or row.get("reward") not in (0, 1)):
                raise ValueError("Resumed record does not match its collection protocol")
            ids, mask = row.get("response_ids", []), row.get("policy_mask", [])
            if (not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask)
                    or any(type(i) is not int for i in ids) or any(m not in (0, 1) for m in mask)
                    or not isinstance(row.get("trajectory"), str)):
                raise ValueError("Invalid trajectory/token mask in resumed pool")
            key = (index, sibling)
            if key in completed:
                raise ValueError("Duplicate question_index/sibling in resumed pool")
            completed.add(key)
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                break
    return completed


@contextmanager
def collection_output(output, policy_model, questions, k, seed, kwargs, resume):
    """Hold a single-writer lock and yield a durable stream plus completed IDs."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.with_name(path.name + ".meta.json")
    expected = collection_manifest(policy_model, questions, k, seed, kwargs)
    with path.with_name(path.name + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another collector is writing this pool") from None
        if path.exists() or meta.exists():
            if not resume:
                raise FileExistsError(f"Pool exists; pass --resume explicitly: {path}")
            if not meta.exists():
                raise ValueError("Legacy pool has no provenance manifest; use a fresh output")
            if json.loads(meta.read_text()) != expected:
                raise ValueError("Resume protocol mismatch: checkpoint, questions, code or parameters changed")
        else:
            temporary = meta.with_name(meta.name + ".tmp")
            with temporary.open("w") as stream:
                json.dump(expected, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, meta)
        completed = recover_completed(path, questions, k, seed, expected["policy_model"])
        with path.open("a", encoding="utf-8") as stream:
            yield stream, completed, expected["policy_model"]


def persist_trajectory(stream, row):
    """Commit each expensive trajectory before starting the next one."""
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
