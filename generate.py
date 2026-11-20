"""Stage 1: one model answers every query, in batches (vLLM). Resumable by chunk.

    python generate.py --role small
    python generate.py --role large
    python generate.py --role small --demo
"""
import argparse
import json
import os
import time

import pandas as pd

import config
from riskroute import parsing, simulator

ap = argparse.ArgumentParser()
ap.add_argument("--role", choices=["small", "large"], required=True)
ap.add_argument("--limit", type=int, default=None, help="first N queries only (timing test)")
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()

t0 = time.time()
model_name = config.SMALL_MODEL if args.role == "small" else config.LARGE_MODEL
out_path = config.GEN_PATH.format(role=args.role)
df = pd.read_csv(config.DATASET_PATH, keep_default_na=False)
if args.limit:
    df = df.head(args.limit)

done = pd.read_csv(out_path, keep_default_na=False) if os.path.exists(out_path) else pd.DataFrame()
todo = df[~df["id"].isin(done["id"])] if len(done) else df
print(f"{args.role}: {model_name} | {len(todo)} to answer, {len(done)} already done")

backend = None
if len(todo) and not config.DEMO:
    from riskroute.backend import VLLM
    backend = VLLM(model_name)

r = args.role
for start in range(0, len(todo), config.CHUNK):
    chunk = todo.iloc[start:start + config.CHUNK]
    rows = []
    for kind, g in chunk.groupby("answer_type", sort=False):
        if config.DEMO:
            res = [simulator.reply(row, r) for row in g.to_dict("records")]
        else:
            prompts = [parsing.build_prompt(q, kind, c or None) for q, c in zip(g["query"], g["context"])]
            res = backend.chat(prompts, parsing.SYSTEM, config.MAX_TOKENS[kind])
        for rid, (text, tin, tout) in zip(g["id"], res):
            answer, conf = parsing.parse_confidence(text)
            rows.append({"id": rid, f"{r}_raw": text, f"{r}_answer": answer, f"{r}_conf": conf,
                         f"{r}_tin": tin, f"{r}_tout": tout})
    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    done.to_csv(out_path, index=False)                 # checkpoint after every chunk
    print(f"  saved {len(done)}/{len(df)}  ({time.time() - t0:.0f}s)")

# Per-user latency needs single-stream speed, not batch throughput.
speed_path = os.path.join(config.OUT_DIR, f"speed_{r}.json")
if config.DEMO:
    tps = {"small": 45.0, "large": 14.0}[r]
elif backend is not None:
    sample = [parsing.build_prompt(q, "open") for q in df["query"].head(5)]
    tps = backend.single_stream_speed(sample, parsing.SYSTEM)
else:
    tps = json.load(open(speed_path))["tokens_per_s"] if os.path.exists(speed_path) else None
if tps:
    json.dump({"model": model_name, "tokens_per_s": round(tps, 2)}, open(speed_path, "w"))
    print(f"single-stream speed: {tps:.1f} tokens/s")
print(f"{args.role} stage done in {time.time() - t0:.0f}s -> {out_path}")
