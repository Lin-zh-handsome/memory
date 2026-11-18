"""Publish an already completed M5 report and compact, verifiable experiment records."""

import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    checkout = Path(__file__).resolve().parents[2]
    relative = Path("experiments/rmbench/m5_goal_keyframe_aux")
    branch = "codex/m5-goal-keyframe-aux"

    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()

    if git("branch", "--show-current") != branch or git("diff", "--cached", "--name-only"):
        raise RuntimeError("Expected the M5 branch and an empty Git index")
    if not (root / "REPORT.md").is_file():
        raise FileNotFoundError("Run the completed comparison first")
    gate = json.loads((root / "seed_gate.json").read_text())
    files = [
        root / name for name in (
            "REPORT.md", "comparison.json", "seed_gate.json", "videos.md", "task_plan.md",
            "findings.md", "progress.md",
        )
    ]
    plan_path = root / "task_plan.md"
    plan = plan_path.read_text()
    for phrase in ("完成种子 0", "按 40%", "完成对照报告"):
        plan = plan.replace(f"- [ ] {phrase}", f"- [x] {phrase}")
    plan_path.write_text(plan)
    progress = root / "progress.md"
    entry = "\n正式实验与报告已完成，发布范围含逐回合结果和规划文档。\n"
    if entry.strip() not in progress.read_text():
        with progress.open("a") as handle:
            handle.write(entry)
    for seed_root in (root, root / "seed1") if gate["run_seed_1"] else (root,):
        files += list((seed_root / "configs").glob("*.json"))
        files += [seed_root / "logs/M5_epochs.csv", seed_root / "evaluation/reset_seeds.json"]
        for directory in (seed_root / "evaluation/M5").glob("epoch_*"):
            files.extend(directory / name for name in ("episodes.jsonl", "summary.json", "evaluation_protocol.json"))
    for source in files:
        dest = checkout / relative / source.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    git("add", "--", *(str(relative / source.relative_to(root)) for source in files))
    if git("diff", "--cached", "--name-only"):
        print(git("commit", "-m", "Publish complete M5 Cover Blocks experiments"), flush=True)
    git("push", "origin", f"HEAD:refs/heads/{branch}")
    head = git("rev-parse", "HEAD")
    if git("ls-remote", "origin", f"refs/heads/{branch}").split()[0] != head:
        raise RuntimeError("Remote head differs from the published commit")
    print(f"https://github.com/Lin-zh-handsome/memory/commit/{head}", flush=True)


if __name__ == "__main__":
    main()
