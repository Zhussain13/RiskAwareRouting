"""Human severity labelling: export sheets, measure agreement, settle disagreements.

    python labels.py export                     # annotator_A.csv, annotator_B.csv
    python labels.py merge annotator_A.csv annotator_B.csv
    python labels.py finalise                   # -> data/severity_human.csv

See data/rubric/severity_rubric.md. Annotators never see the prior, the LLM label or
any model answer.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import config
from riskroute import severity

LABEL_DIR = os.path.join(config.OUT_DIR, "labelling")
HUMAN_PATH = os.path.join(config.DATA_DIR, "severity_human.csv")
os.makedirs(LABEL_DIR, exist_ok=True)

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("export")
m = sub.add_parser("merge")
m.add_argument("a")
m.add_argument("b")
sub.add_parser("finalise")
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()

if args.cmd == "export":
    df = pd.read_csv(config.DATASET_PATH)[["id", "query"]]
    for name, seed in (("A", 1), ("B", 2)):
        sheet = df.sample(frac=1, random_state=seed).assign(severity="", note="")
        path = os.path.join(LABEL_DIR, f"annotator_{name}.csv")
        sheet.to_csv(path, index=False)
        print(f"wrote {path} ({len(sheet)} queries)")
    print("Give one sheet to each annotator with data/rubric/severity_rubric.md.")

elif args.cmd == "merge":
    a = pd.read_csv(args.a)[["id", "query", "severity"]].rename(columns={"severity": "a"})
    b = pd.read_csv(args.b)[["id", "severity"]].rename(columns={"severity": "b"})
    df = a.merge(b, on="id")
    done = df.dropna(subset=["a", "b"])
    if done.empty:
        sys.exit("No query has labels from both annotators yet.")
    k = severity.weighted_kappa(done.a, done.b)
    exact = float((done.a == done.b).mean())
    print(f"{len(done)} queries labelled by both | exact agreement {exact:.1%} | "
          f"weighted kappa {k:.3f} ({'OK' if k >= 0.6 else 'below the 0.6 target'})")
    print(pd.crosstab(done.a.astype(int), done.b.astype(int), rownames=["A"], colnames=["B"]))
    dis = done[done.a != done.b].assign(final="")
    agreed = done[done.a == done.b].assign(final=lambda d: d.a)
    dis.to_csv(os.path.join(LABEL_DIR, "disagreements.csv"), index=False)
    agreed.to_csv(os.path.join(LABEL_DIR, "agreed.csv"), index=False)
    pd.DataFrame([{"n": len(done), "exact": exact, "weighted_kappa": k}]).to_csv(
        os.path.join(LABEL_DIR, "agreement.csv"), index=False)
    print(f"{len(dis)} disagreements -> {LABEL_DIR}/disagreements.csv (fill the 'final' column)")

elif args.cmd == "finalise":
    agreed = pd.read_csv(os.path.join(LABEL_DIR, "agreed.csv"))
    dis = pd.read_csv(os.path.join(LABEL_DIR, "disagreements.csv"))
    if dis["final"].isna().any():
        sys.exit(f"{int(dis['final'].isna().sum())} disagreements still have no 'final' label.")
    out = pd.concat([agreed, dis])[["id", "final"]].rename(columns={"final": "severity_human"})
    out["severity_human"] = out["severity_human"].astype(int)
    assert np.all(out.severity_human.between(0, 3))
    out.to_csv(HUMAN_PATH, index=False)
    print(f"{len(out)} final labels -> {HUMAN_PATH}")
