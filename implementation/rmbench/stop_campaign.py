"""Stop only the selected RMBench campaign at its server-side deadline."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

CODE = Path(__file__).resolve().parent
ZONE = ZoneInfo("Asia/Shanghai")


def matched_processes(root):
    queue = root / "queue.json"
    runs = {str(Path(job["root"]).resolve()) for job in json.loads(queue.read_text())["jobs"]}
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = [os.fsdecode(v) for v in (proc / "cmdline").read_bytes().split(b"\0") if v]
            kind = None
            for script, flag, candidate_kind in (
                ("run_ablation.py", "--queue", "coordinator"),
                ("train.py", "--root", "train"),
                ("evaluate.py", "--experiment", "eval"),
            ):
                if str(CODE / script) not in args or flag not in args:
                    continue
                value = str(Path(args[args.index(flag) + 1]).resolve())
                if value == str(queue) if candidate_kind == "coordinator" else value in runs:
                    kind = candidate_kind
                    break
            if kind:
                pid = int(proc.name)
                found.append({"pid": pid, "group": os.getpgid(pid), "kind": kind})
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue
    return found


def write(path, value):
    temporary = path.with_suffix(".stop.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("inspect", "graceful", "force"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    schedule = json.loads((root / "logs/stop_schedule.json").read_text())
    now = datetime.now(ZONE)
    jobs = matched_processes(root)
    if args.stage == "inspect":
        print(json.dumps({"now": now.isoformat(), "schedule": schedule, "processes": jobs}, indent=2))
        return
    deadline = datetime.fromisoformat(schedule["force_at"])
    if now.timestamp() < datetime.fromisoformat(schedule[args.stage + "_at"]).timestamp() - 2:
        raise RuntimeError("The scheduled stop time has not arrived")
    if now.timestamp() > deadline.timestamp() + 86400:
        return
    # Stop dispatch before signaling workers, so no replacement work can start.
    for job in jobs:
        if job["kind"] == "coordinator":
            try:
                os.kill(job["pid"], signal.SIGTERM if args.stage == "graceful" else signal.SIGKILL)
            except ProcessLookupError:
                pass
    groups = {}
    for job in jobs:
        if job["kind"] != "coordinator":
            groups.setdefault(job["group"], set()).add(job["kind"])
    for group, kinds in groups.items():
        try:
            if args.stage == "force":
                os.killpg(group, signal.SIGKILL)
            elif "eval" in kinds:
                os.killpg(group, signal.SIGTERM)
            else:
                # Train's SIGTERM handler saves optimizer, RNG and Mamba carry state.
                os.kill(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    event = {"stage": args.stage, "time": now.isoformat(), "target_time": schedule["release_before"],
             "processes": jobs}
    write(root / ("logs/stop_" + args.stage + ".json"), event)
    if args.stage == "force":
        time.sleep(1)
        remaining = matched_processes(root)
        checkpoints = []
        for job in json.loads((root / "queue.json").read_text())["jobs"]:
            run = Path(job["root"])
            log = run / ("logs/" + job["model"] + "_train.log")
            if not log.exists():
                continue
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 100000))
                tail = handle.read().decode(errors="replace").splitlines()
            saved = []
            for line in tail:
                try:
                    row = json.loads(line)
                    if row.get("resume_saved"):
                        saved.append(row)
                except (ValueError, TypeError):
                    continue
            checkpoints.append({"root": str(run), "checkpoint": str(run / "checkpoints" / job["model"] / "latest.pt"),
                                "last_resume_record": saved[-1] if saved else None})
        outcome = {"status": "paused" if not remaining else "stop_incomplete",
                   "reason": "scheduled_0730_beijing", "time": now.isoformat(),
                   "target_time": schedule["release_before"], "forced": bool(jobs),
                   "remaining": remaining, "checkpoints": checkpoints}
        write(root / "logs/campaign_status.json", outcome)
        write(root / "logs/stop_result.json", outcome)
        marker = schedule["cron_marker"]
        previous = subprocess.run(["crontab", "-l"], text=True, capture_output=True)
        if previous.returncode == 0:
            kept = [line for line in previous.stdout.splitlines() if marker not in line]
            subprocess.run(["crontab", "-"], input="\n".join(kept) + "\n", text=True, check=True)
        with (root / "progress.md").open("a") as handle:
            handle.write("\n## 定时停止 " + now.isoformat() + "\n\n" + json.dumps(outcome, ensure_ascii=False) + "\n")
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()
