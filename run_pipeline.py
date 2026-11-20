"""Runs every stage in order, each in its own process (so each model's GPU memory is freed),
skipping stages whose output already exists, and records how long each one took.

    python run_pipeline.py --demo              # whole pipeline, simulated, ~30 s
    python run_pipeline.py                     # real run on a GPU (Colab T4: ~1.5-2 h)
    python run_pipeline.py --only small large  # just some stages
    python run_pipeline.py --force grade       # redo a stage even if its output exists
"""
import argparse
import json
import os
import subprocess
import sys
import time

import config

STAGES = [
    ("dataset", ["build_dataset.py"], config.DATASET_PATH),
    ("small", ["generate.py", "--role", "small"], config.GEN_PATH.format(role="small")),
    ("large", ["generate.py", "--role", "large"], config.GEN_PATH.format(role="large")),
    ("grade", ["grade.py"], config.JUDGE_PATH),
    ("verify", ["verify.py"], config.VERIFY_PATH),
    ("evaluate", ["evaluate.py"], os.path.join(config.RESULTS_DIR, "summary.md")),
]

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--force", nargs="*", default=[])
ap.add_argument("--splits", type=int, default=None)
args = ap.parse_args()

timing_path = os.path.join(config.OUT_DIR, "timings.json")
timings = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
total = time.time()
for name, cmd, out in STAGES:
    if args.only and name not in args.only:
        continue
    if os.path.exists(out) and name not in args.force and name not in ("small", "large"):
        print(f"[{name}] skipped: {out} exists (use --force {name} to redo)")
        continue
    full = [sys.executable] + cmd + (["--demo"] if config.DEMO else [])
    if name == "evaluate" and args.splits:
        full += ["--splits", str(args.splits)]
    print(f"\n[{name}] {' '.join(full[1:])}", flush=True)
    t0 = time.time()
    subprocess.run(full, check=True)            # generate.py resumes by itself
    timings[name] = round(time.time() - t0, 1)
    json.dump(timings, open(timing_path, "w"), indent=2)
    print(f"[{name}] {timings[name] / 60:.1f} min")

print("\nStage times (min): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in timings.items()))
print(f"This run: {(time.time() - total) / 60:.1f} min. Results: {config.RESULTS_DIR}/summary.md")
