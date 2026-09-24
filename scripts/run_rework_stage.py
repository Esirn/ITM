#!/usr/bin/env python3
"""Serial, bounded CPU protocol runner. Default is a preflight; no GPU launch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from audit_itm_rework import artifact, dump


CPU_SCRIPTS = {
    "audit_itm_rework.py", "audit_rework_sources.py", "audit_rework_cache_full.py",
    "check_rework_cache_synthesis.py", "recompute_rework_statistics.py",
    "evaluate_rework_records.py", "plan_rework_validation.py",
}


def validate_job(job):
    if job.get("device") != "cpu" or job.get("script") not in CPU_SCRIPTS:
        raise ValueError("only explicitly supported CPU protocol scripts can run")
    limit = job.get("timeout_seconds")
    if not isinstance(limit, int) or not 1 <= limit <= 1800:
        raise ValueError("finite CPU timeout of 1..1800 seconds required")
    if job.get("output_kind") not in {"file", "directory"}:
        raise ValueError("output_kind must be file or directory")
    arguments = job.get("arguments")
    if not isinstance(arguments, list) or any(not isinstance(a, str) for a in arguments):
        raise ValueError("arguments must be a string list")
    if any(a == "--output" or a.startswith("--output=") for a in arguments):
        raise ValueError("runner assigns output inside a new run directory")
    if not job.get("job_id") or Path(job["job_id"]).name != job["job_id"] or job["job_id"] in {".", ".."}:
        raise ValueError("job_id must be a plain filename")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    jobs = queue["jobs"]
    if not jobs or len({j["job_id"] for j in jobs}) != len(jobs):
        parser.error("nonempty queue with distinct job IDs required")
    for job in jobs: validate_job(job)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"queue": artifact(args.queue), "mode": "execute" if args.execute else "dry_run",
              "gpu_jobs": 0, "concurrency": 1, "jobs": []}
    failure = False
    for job in jobs:
        missing = [p for p in job.get("required_files", []) if not Path(p).is_file()]
        blockers = list(job.get("blockers", []))
        blockers.extend(f"missing input: {p}" for p in missing)
        output = args.output / job["job_id"]
        if job["output_kind"] == "file": output = output.with_suffix(".json")
        command = [sys.executable, str(Path("scripts") / job["script"]), *job["arguments"], "--output", str(output)]
        row = {"job_id": job["job_id"], "command": command, "timeout_seconds": job["timeout_seconds"], "blockers": blockers}
        if blockers:
            row["status"] = "blocked"
            failure = True
        elif not args.execute:
            row["status"] = "ready_not_executed"
        elif failure:
            row["status"] = "not_run_after_failure"
        else:
            start = time.monotonic()
            with (args.output / f"{job['job_id']}.log").open("w") as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=job["timeout_seconds"], check=False)
                    row.update(exit_code=result.returncode, status="passed" if result.returncode == 0 else "failed")
                except subprocess.TimeoutExpired:
                    row.update(status="timeout", exit_code=None)
            row["elapsed_seconds"] = time.monotonic()-start
            failure = row["status"] != "passed"
        report["jobs"].append(row)
        dump(args.output / "run.json", report)
    print(json.dumps({"output": str(args.output), "jobs": [{"job_id":j["job_id"], "status":j["status"]} for j in report["jobs"]]}))
    return 2 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
