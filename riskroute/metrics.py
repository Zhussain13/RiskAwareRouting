"""Outcome of a routing policy on a set of queries."""
import numpy as np

import config

ACTIONS = config.ACTIONS


def outcome_from_actions(a, ok, cost, lat, verify_large):
    """Per-query (ok, cost, latency, large_called, route) for chosen action indices."""
    r = np.arange(len(a))
    large_called = (a == ACTIONS.index("large")) | ((a == ACTIONS.index("verify")) & (verify_large == 1))
    route = np.asarray(ACTIONS)[a]
    return ok[r, a], cost[r, a], lat[r, a], large_called.astype(int), route


def summarise(o_ok, o_cost, o_lat, o_large, route, severity, weights=config.SEVERITY_WEIGHTS) -> dict:
    severity = np.asarray(severity, int)
    w = np.asarray(weights, float)[severity]
    err = 1 - np.asarray(o_ok, float)
    hi = severity == 3
    human = route == "flag"
    lat = np.asarray(o_lat, float)[~human]
    return {
        "cost_per_1k_usd": 1000 * float(np.mean(o_cost)),
        "accuracy": float(1 - err.mean()),
        "swer": float(np.mean(w * err)),
        "hser": float(err[hi].mean()) if hi.any() else np.nan,
        "large_calls": float(np.mean(o_large)),
        "flag_rate": float(human.mean()),
        "verify_share": float(np.mean(route == "verify")),
        "faq_share": float(np.mean(route == "faq")),
        "small_share": float(np.mean(route == "small")),
        "lat_p50_s": float(np.percentile(lat, 50)) if len(lat) else np.nan,
        "lat_p95_s": float(np.percentile(lat, 95)) if len(lat) else np.nan,
    }
