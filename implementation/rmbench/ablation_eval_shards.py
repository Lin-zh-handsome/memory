"""Merge checkpoint workers in the original reset order."""
import json
from pathlib import Path


RANGES = ((0, 26), (26, 50))


def shard_dir(root, span):
    first, stop = span
    return root / "shards" / f"{first:02d}_{stop:02d}"


def read(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def rows(path):
    episodes = path / "episodes.jsonl"
    if not episodes.exists():
        return []
    return [json.loads(line) for line in episodes.read_text().splitlines() if line]


def save(path, data):
    temp = path.with_suffix(".merge.tmp")
    temp.write_text(data)
    temp.replace(path)


def merge(root, spec, epoch):
    config = read(root / "evaluation_protocol.json")
    original = rows(root)
    previous = read(root / "summary.json", {})
    protocol = read(spec.root / "configs/protocol.json")
    reset_path = Path(protocol.get("shared_reset_seeds", spec.root / "evaluation/reset_seeds.json"))
    manifest = read(reset_path, {})
    if not manifest.get("complete"):
        return len(original)
    seeds = manifest["seeds"]
    pieces = {}
    summary = previous
    for span in RANGES:
        path = shard_dir(root, span)
        if not (part := rows(path)):
            continue
        first, stop = span
        case = read(path / "evaluation_protocol.json")
        if (case["batch_size"] != 2 or case["camera_shader"] != "rt" or case["seed_order"] != seeds
                or [row["episode_id"] for row in part] != list(range(first, first + len(part)))
                or [row["seed"] for row in part] != seeds[first:first + len(part)]
                or first + len(part) > stop or config is not None and case != config):
            raise ValueError(f"Shard does not match the fixed evaluation protocol: {path}")
        config = case
        summary = read(path / "summary.json", summary)
        for row in part:
            if row["video"] and (path / row["video"]).is_file():
                row["video"] = str(path.relative_to(root) / row["video"])
            pieces[row["episode_id"]] = row
    combined = original.copy()
    while len(combined) in pieces:
        combined.append(pieces[len(combined)])
    if not config or not summary or (len(combined) == len(original) == previous.get("episodes")):
        return len(combined)
    rate = sum(row["success"] for row in combined)
    complete = len(combined) == len(seeds)
    updated = {
        **summary, "model": spec.model, "epoch": epoch, "episodes": len(combined),
        "complete": complete, "successes": rate, "success_rate": rate / len(combined) if complete else None,
        "render_pcis": sorted({row["render_pci"] for row in combined}),
    }
    for key in ("steps", "replans", "observe_seconds", "predict_seconds"):
        updated[key] = sum(row[key] for row in combined)
    save(root / "evaluation_protocol.json", json.dumps(config, indent=2) + "\n")
    save(root / "episodes.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in combined))
    save(root / "summary.json", json.dumps(updated, indent=2) + "\n")
    return len(combined)
