"""Stage 4 (CPU only): train, calibrate and compare every method over many random splits.

Everything replays from the cached answers, so this never calls a model.

    python evaluate.py               # 100 splits (config.N_SPLITS)
    python evaluate.py --splits 10   # quicker
    python evaluate.py --demo
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config
from riskroute import experiment, features, table

ap = argparse.ArgumentParser()
ap.add_argument("--splits", type=int, default=config.N_SPLITS)
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()
t0 = time.time()
R = config.RESULTS_DIR

df = table.build()
df.to_csv(config.TABLE_PATH, index=False)
src = df.attrs["severity_source"]
print(f"{len(df)} queries | severity labels: {src} | alpha = {config.ALPHA} | weights = {config.SEVERITY_WEIGHTS}")

X = features.build(df)
ok, cost, lat, avail = table.matrices(df)
T = {"ok": ok, "cost": cost, "lat": lat, "avail": avail, "verify_large": df.verify_large.to_numpy(),
     "faq_sim": df.faq_sim.to_numpy(), "query": df["query"].to_numpy(), "small_conf": df.small_conf.to_numpy(),
     "severity_llm": df.severity_llm.to_numpy(int), "severity_prior": df.severity_prior.to_numpy(int)}
sev = df.severity.to_numpy(int)
kinds = df.answer_type.to_numpy()

shift = df.slice.to_numpy() == "kanglish"          # held out entirely: shift test only
main_idx = np.where(~shift)[0]
kang_idx = np.where(shift)[0]

rows, extras = [], []
for s in range(args.splits):
    parts = experiment.stratified_split(sev[main_idx], seed=config.SEED + s)
    idx = [main_idx[p] for p in parts]
    split = experiment.Split(T, X, idx, sev, kinds)
    res, extra = experiment.run(split, kang_idx)
    for method, m in res.items():
        rows.append({"split": s, "method": method, **m})
    extras.append({"split": s, **extra})
    if s == 0:
        curve = pd.DataFrame(experiment.curves(split, np.geomspace(0.005, 0.2, 12)))
        curve.to_csv(os.path.join(R, "curves_split0.csv"), index=False)
    if (s + 1) % 10 == 0 or s == args.splits - 1:
        print(f"  {s + 1}/{args.splits} splits ({time.time() - t0:.0f}s)")

allres = pd.DataFrame(rows)
allres.to_csv(os.path.join(R, "all_splits.csv"), index=False)
ex = pd.DataFrame(extras)

order = list(dict.fromkeys(allres.method))
cols = ["cost_per_1k_usd", "accuracy", "swer", "hser", "large_calls", "flag_rate", "verify_share",
        "faq_share", "lat_p50_s", "lat_p95_s"]
mean = allres.groupby("method")[cols].mean().loc[order]
std = allres.groupby("method")[cols].std().loc[order]
mean.round(4).to_csv(os.path.join(R, "summary_mean.csv"))
std.round(4).to_csv(os.path.join(R, "summary_std.csv"))

# ---- the guarantee check
g = {}
for m in ("ours_crc", "ours_ltt"):
    v = allres[allres.method == m]["swer"]
    g[m] = {"mean_test_swer": float(v.mean()), "share_splits_swer_le_alpha": float((v <= config.ALPHA).mean())}
g["alpha"], g["delta"] = config.ALPHA, config.DELTA
g["note"] = ("crc bounds the EXPECTED risk (check mean_test_swer <= alpha); ltt bounds risk with "
             "probability >= 1 - delta. Test sets are finite, so the share of splits under alpha "
             "is a noisy, conservative check of the population guarantee.")
g["severity_source"] = src
g["no_human_feasible_share"] = float(ex.no_human_feasible.mean())
large_swer = float(allres[allres.method == "always_large"]["swer"].mean())
if config.ALPHA < large_swer:
    print(f"NOTE: alpha {config.ALPHA} is below the large model's own SWER ({large_swer:.4f}); "
          "only flagging to a human can meet it. See ours_no_human and the curves.")
g["stakes_estimator_accuracy"] = float(ex.stakes_acc.mean())
g["lambda_crc_median"] = float(ex.lam_crc.replace(np.inf, np.nan).median())
json.dump(g, open(os.path.join(R, "guarantee.json"), "w"), indent=2)

# ---- plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
names = {"ours": "Risk-aware (ours)", "stakes_blind": "Same, stakes-blind", "smr_learned": "SmartModelRouter learned"}
for ax, metric, label in ((axes[0], "swer", "Severity-weighted error (SWER)"),
                          (axes[1], "hser", "Error rate on severity-3 queries (HSER)")):
    for m, gdf in curve.groupby("method"):
        gdf = gdf.sort_values("cost_per_1k_usd")
        style = {"ours": ("#1f5fbf", "-", "o"), "stakes_blind": ("#555555", "--", "s"),
                 "smr_learned": ("#a8a8a8", ":", "^")}[m]
        ax.plot(gdf.cost_per_1k_usd, gdf[metric], marker=style[2], ms=3.5, label=names.get(m, m),
                color=style[0], ls=style[1], lw=1.6)
    ax.set_xlabel("Cost per 1,000 queries (USD, paid list prices)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[0].axhline(config.ALPHA, color="#c0392b", lw=1, ls=":", label=f"alpha = {config.ALPHA}")
axes[0].legend(fontsize=8)
title = "Cost vs risk on one test split" + ("  [DEMO: SIMULATED]" if config.DEMO else "")
fig.suptitle(title)
fig.tight_layout()
fig.savefig(os.path.join(R, "cost_vs_risk.png"), dpi=150)

# ---- a readable summary
show = ["cost_per_1k_usd", "accuracy", "swer", "hser", "large_calls", "flag_rate", "verify_share"]
lines = [f"# Results{'  (DEMO: SIMULATED, NOT RESULTS)' if config.DEMO else ''}", "",
         f"- queries: {len(df)} ({len(kang_idx)} Kanglish held out as a shift test)",
         f"- severity labels: **{src}**", f"- alpha = {config.ALPHA}, delta = {config.DELTA}, "
         f"weights = {config.SEVERITY_WEIGHTS}, splits = {args.splits}",
         f"- ours (CRC): mean test SWER {g['ours_crc']['mean_test_swer']:.4f} (bound: <= {config.ALPHA} in expectation)",
         f"- ours (LTT): SWER <= alpha in {g['ours_ltt']['share_splits_swer_le_alpha']:.0%} of splits "
         f"(bound: >= {1 - config.DELTA:.0%} for the population risk)",
         f"- without humans, alpha was reachable in {g['no_human_feasible_share']:.0%} of splits",
         f"- stakes estimator exact-level accuracy: {g['stakes_estimator_accuracy']:.1%}", "",
         "Mean over splits:", "", mean[show].round(4).to_markdown(), ""]
open(os.path.join(R, "summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))
print(f"\ndone in {time.time() - t0:.0f}s -> {R}/")
