"""The whole pipeline in demo mode, in a temporary output folder."""
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_demo_pipeline_end_to_end(tmp_path):
    env = {**os.environ, "RAR_DEMO": "1", "RAR_OUT": str(tmp_path)}
    subprocess.run([sys.executable, "run_pipeline.py", "--demo", "--splits", "3"],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    res = tmp_path / "results"
    for f in ("summary.md", "summary_mean.csv", "all_splits.csv", "guarantee.json",
              "cost_vs_risk.png", "curves_split0.csv"):
        assert (res / f).exists(), f
    table = pd.read_csv(tmp_path / "table.csv")
    assert len(table) > 1000
    for a in ("faq", "small", "verify", "large", "flag"):
        assert {f"{a}_ok", f"{a}_cost", f"{a}_avail"} <= set(table.columns)
    mean = pd.read_csv(res / "summary_mean.csv", index_col=0)
    assert {"ours_crc", "ours_ltt", "smr_learned", "uniform_risk_matched", "abl_no_verify"} <= set(mean.index)
    g = json.load(open(res / "guarantee.json"))
    assert "SIMULATED" in g["severity_source"]
    # flagging everything always meets alpha, so the router must do better than that
    assert mean.loc["ours_crc", "flag_rate"] < 0.9
    assert "DEMO" in (res / "summary.md").read_text()
