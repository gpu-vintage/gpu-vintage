#!/usr/bin/env python3
"""Run the rows of a campaign list, on this machine or through Slurm, skipping
runs that have already finished.

    python3 scripts/run_campaign.py campaigns/e1.tsv --gpu a100              # here, one run after another
    python3 scripts/run_campaign.py campaigns/e1.tsv --gpu a100 \\
        --sbatch "-p gpu --gres=gpu:1 -c 8 --mem=60G"                       # one Slurm job per run

Each run writes <out>/<run>/{config,env,metrics}.json; --out defaults to
out/runs, never to the shipped results/runs, where every run is already done.
The models must be in the local Hugging Face cache first (see the README).
--only keeps the runs whose name contains a substring; --dry-run prints the
commands without running them. Build the tables from the new runs with

    GPUVINTAGE_RUNS=out/runs python3 scripts/build_results.py --tex out/paper --results out/tables
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def finished(run_dir: Path) -> bool:
    m = run_dir / "metrics.json"
    if not m.is_file():
        return False
    try:
        return bool(json.loads(m.read_text()).get("finished"))
    except json.JSONDecodeError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", help="a campaign list, campaigns/e<n>.tsv")
    ap.add_argument("--gpu", default="", help="only the rows of this GPU tag (column 'gpu')")
    ap.add_argument("--only", default="", help="only the runs whose name contains this")
    ap.add_argument("--out", default="out/runs", help="directory of the run outputs")
    ap.add_argument("--sbatch", default=None, metavar="FLAGS",
                    help="submit each run with sbatch and these resource flags, instead of running it here")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lines = Path(a.campaign).read_text().splitlines()
    if lines[0].split("\t") != ["run", "gpu", "time", "args"]:
        print(f"{a.campaign}: not a campaign list", file=sys.stderr)
        return 2
    rows = [l.split("\t") for l in lines[1:] if l.strip()]
    rows = [r for r in rows if (not a.gpu or r[1] == a.gpu) and a.only in r[0]]
    done = started = 0
    for name, _, time, args in rows:
        run_dir = Path(a.out) / name
        if finished(run_dir):
            done += 1
            continue
        argv = shlex.split(args)
        script = argv.pop(0) if argv[0].startswith("scripts/") else "scripts/bench.py"
        cmd = [sys.executable, str(ROOT / script), *argv, f"--run-dir={run_dir}"]
        if a.sbatch is not None:
            cmd = (["sbatch", f"--job-name={name}", f"--time={time}", *shlex.split(a.sbatch),
                    "--wrap", " ".join(shlex.quote(c) for c in cmd)])
        print(" ".join(shlex.quote(c) for c in cmd), flush=True)
        if not a.dry_run:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"{name}: exit {res.returncode}", file=sys.stderr)
                return 1
        started += 1
    verb = "submit" if a.sbatch is not None else "run"
    verb = f"to {verb}" if a.dry_run else ("submitted" if a.sbatch is not None else "run")
    print(f"{started} {verb}, {done} already finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
