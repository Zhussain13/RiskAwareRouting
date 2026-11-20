"""Stage 0: assemble StakesQA into one table (no GPU).

    python build_dataset.py --demo     # offline placeholders for the downloaded slices
    python build_dataset.py            # real run: downloads GSM8K, TruthfulQA, MedMCQA and
                                       # compiles every generated C program with gcc
"""
import argparse
import time

import config
from riskroute import dataset

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
ap.add_argument("--small", action="store_true", help="tiny sizes for a quick smoke test")
args = ap.parse_args()

t0 = time.time()
sizes = {"gsm8k": 20, "truthfulqa": 15, "medmcqa": 15, "code_output": 25} if args.small else None
df = dataset.build(sizes)
dataset.check(df)
df.to_csv(config.DATASET_PATH, index=False)

print(dataset.summary(df).to_string())
print(f"\n{len(df)} queries -> {config.DATASET_PATH}  ({time.time() - t0:.0f}s)")
print(f"FAQ context attached to {int((df.context.str.len() > 0).sum())} queries")
if config.DEMO:
    print("DEMO: GSM8K / TruthfulQA / MedMCQA rows are placeholders and C programs were not compiled.")
