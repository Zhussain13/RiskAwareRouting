"""Joins every stage into one table: per query, what each action would have cost, how long
it would have taken, and whether its answer would have been correct.

Because all five actions are evaluated for every query, any routing policy can be replayed
offline on CPU, with no further model calls.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import config
from . import simulator

ACTIONS = config.ACTIONS
VERIFIER_SECONDS = {"code": 0.5, "numeric": 0.01, "faq": 0.15, "none": 0.0}
FAQ_SECONDS = 0.05


def _price(tin, tout, large: bool):
    if large:
        return (tin * config.LARGE_PRICE_IN + tout * config.LARGE_PRICE_OUT) / 1e6
    return (tin * config.SMALL_PRICE_IN + tout * config.SMALL_PRICE_OUT) / 1e6


def _speed(role):
    p = os.path.join(config.OUT_DIR, f"speed_{role}.json")
    if os.path.exists(p):
        return json.load(open(p))["tokens_per_s"]
    return {"small": 45.0, "large": 14.0}[role]


def severity_column(df: pd.DataFrame, source: str = None):
    """Picks the severity labels and says which ones were used.

    human -> data/severity_human.csv (the only publishable labels)
    llm   -> judge-model pre-labels;  prior -> keyword rules
    """
    source = source or config.SEVERITY_SOURCE
    if config.DEMO:
        sev = [simulator.severity_true(q, p) for q, p in zip(df["query"], df["severity_prior"])]
        return np.asarray(sev), "human (SIMULATED in demo)"
    human = os.path.join(config.DATA_DIR, "severity_human.csv")
    if source == "human" and os.path.exists(human):
        h = df[["id"]].merge(pd.read_csv(human), on="id", how="left")["severity_human"]
        if h.notna().all():
            return h.astype(int).to_numpy(), "human"
        warnings.warn(f"{int(h.isna().sum())} queries lack human labels; falling back to LLM labels")
    if source in ("human", "llm") and "severity_llm" in df:
        if source == "human":
            warnings.warn("NO HUMAN SEVERITY LABELS: using LLM pre-labels. Do not report these "
                          "results as final; run labels.py first.")
        return df["severity_llm"].astype(int).to_numpy(), "llm"
    return df["severity_prior"].astype(int).to_numpy(), "prior"


def build() -> pd.DataFrame:
    df = pd.read_csv(config.DATASET_PATH, keep_default_na=False)
    for role in ("small", "large"):
        g = pd.read_csv(config.GEN_PATH.format(role=role), keep_default_na=False)
        df = df.merge(g[["id", f"{role}_conf", f"{role}_tin", f"{role}_tout"]], on="id")
    df = df.merge(pd.read_csv(config.JUDGE_PATH), on="id").merge(pd.read_csv(config.VERIFY_PATH), on="id")

    df["severity"], src = severity_column(df)
    df.attrs["severity_source"] = src

    s_tps, l_tps = _speed("small"), _speed("large")
    df["small_cost"] = _price(df.small_tin, df.small_tout, False)
    df["large_cost"] = _price(df.large_tin, df.large_tout, True)
    df["small_lat"] = 0.2 + df.small_tout / s_tps
    df["large_lat"] = 0.4 + df.large_tout / l_tps

    df["faq_avail"] = (df.faq_sim >= config.FAQ_MIN_SIM).astype(int)
    df["faq_ok"] = ((df.faq_id >= 0) & (df.faq_match == df.faq_id)).astype(int)
    df["faq_cost"], df["faq_lat"] = 0.0, FAQ_SECONDS

    df["small_avail"] = df["large_avail"] = df["flag_avail"] = 1
    df["verify_avail"] = (df.verifier != "none").astype(int)
    p = df.verify_pass
    df["verify_ok"] = np.where(p == 1, df.small_ok, df.large_ok)
    df["verify_cost"] = df.small_cost + config.VERIFY_COST + (1 - p) * df.large_cost
    df["verify_lat"] = df.small_lat + df.verifier.map(VERIFIER_SECONDS) + (1 - p) * df.large_lat
    df["verify_large"] = (1 - p) * df.verify_avail      # verify fell back to the large model

    df["flag_ok"] = 1
    df["flag_cost"] = config.HUMAN_COST_X_LARGE * df.large_cost.mean()
    df["flag_lat"] = np.nan                              # human time is reported separately
    return df


def matrices(df: pd.DataFrame):
    """(ok, cost, lat, avail) as arrays of shape (n, 5) in ACTIONS order."""
    ok = df[[f"{a}_ok" for a in ACTIONS]].to_numpy(float)
    cost = df[[f"{a}_cost" for a in ACTIONS]].to_numpy(float)
    lat = df[[f"{a}_lat" for a in ACTIONS]].to_numpy(float)
    avail = df[[f"{a}_avail" for a in ACTIONS]].to_numpy(bool)
    return ok, cost, lat, avail
