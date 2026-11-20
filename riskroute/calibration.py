"""Choosing lambda so that severity-weighted risk stays within the budget alpha.

loss_i(lam) = w(severity_i) * [answer of the action chosen at lam is wrong], in [0, 1].

crc  Conformal risk control (Angelopoulos et al., 2024). Needs losses that do not increase
     with lam. Ours can (a safer-looking action can still be wrong), so each loss curve is
     first monotonised: L~_i(lam) = max over lam' >= lam of L_i(lam'). L~ >= L, so bounding
     E[L~] also bounds E[L]. Guarantee: E[risk on new queries] <= alpha.
ltt  Learn-then-Test (Angelopoulos et al., 2021) with Hoeffding-Bentkus p-values and fixed-
     sequence testing from the safest lam down. Needs no monotonicity. Guarantee:
     P(risk on new queries <= alpha) >= 1 - delta.
"""
import numpy as np
from scipy.stats import binom


def lambda_grid(scale: float, n: int = 200) -> np.ndarray:
    """0, then log-spaced values around the cost scale, then +inf (= always safest action)."""
    return np.concatenate([[0.0], scale * np.geomspace(1e-3, 1e8, n), [np.inf]])


def loss_matrix(choose, lams, true_w, ok) -> np.ndarray:
    """L[i, j] = loss of query i at lams[j]. `choose(lam)` returns action indices."""
    L = np.empty((len(true_w), len(lams)))
    rows = np.arange(len(true_w))
    for j, lam in enumerate(lams):
        a = choose(lam)
        L[:, j] = true_w * (1 - ok[rows, a])
    return L


def crc(L, lams, alpha: float, B: float = 1.0):
    """Smallest lam whose monotonised CRC bound is <= alpha. Returns (lam, index).
    If even the safest setting misses alpha, returns lam = inf: callers check feasible()."""
    n = L.shape[0]
    Lm = np.maximum.accumulate(L[:, ::-1], axis=1)[:, ::-1]   # max over lam' >= lam
    bound = n / (n + 1) * Lm.mean(axis=0) + B / (n + 1)
    ok = np.where(bound <= alpha)[0]
    if len(ok) == 0:
        return np.inf, len(lams) - 1
    j = ok.min()
    return lams[j], j


def _hb_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: risk > alpha, losses in [0, 1]."""
    r = min(risk_hat, alpha)
    if r <= 0:
        h = np.log(1 / (1 - alpha))
    else:
        h = r * np.log(r / alpha) + (1 - r) * np.log((1 - r) / (1 - alpha))
    hoeffding = np.exp(-n * h)
    bentkus = np.e * binom.cdf(np.ceil(n * risk_hat), n, alpha)
    return float(min(hoeffding, bentkus, 1.0))


def ltt(L, lams, alpha: float, delta: float):
    """Fixed-sequence test from the largest lam downward; stop at the first non-rejection."""
    n = L.shape[0]
    best = len(lams) - 1
    for j in range(len(lams) - 1, -1, -1):
        if _hb_pvalue(L[:, j].mean(), n, alpha) <= delta:
            best = j
        else:
            break
    return lams[best], best


def feasible(L, alpha: float, B: float = 1.0) -> bool:
    """Can the safest setting (lam = inf, the last column) meet alpha at all?"""
    n = L.shape[0]
    return bool(n / (n + 1) * L[:, -1].mean() + B / (n + 1) <= alpha)
