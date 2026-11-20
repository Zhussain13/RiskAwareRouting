"""The risk-aware router: stakes estimator, one failure head per action, and the selector.

For a query x and action a (faq, small, verify, large, flag):

    score_lam(x, a) = cost_hat(x, a) + lam * w_hat(x) * p_fail_hat(x, a)

w_hat(x)  = expected severity weight from the stakes estimator
p_fail_hat = probability that action a answers x wrongly (flag = a human: 0)
The selector picks the available action with the lowest score; lam comes from calibration.
As lam grows, the chosen action's predicted risk can only go down (lower envelope of lines),
which is what makes the calibration step well behaved.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

from .severity import LEVELS

ACTIONS = config.ACTIONS
HEADS = ("faq", "small", "verify", "large")


def _lr(multi: bool = False):
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, C=0.5, class_weight="balanced"))


class StakesEstimator:
    """Predicts P(severity = k) for k = 0..3; returns the expected weight."""

    def fit(self, X, levels):
        self.classes = np.unique(levels)
        self.const = None
        if len(self.classes) == 1:
            self.const = int(self.classes[0])
            return self
        self.model = _lr(multi=True).fit(X, levels)
        return self

    def proba(self, X) -> np.ndarray:
        P = np.zeros((len(X), len(LEVELS)))
        if self.const is not None:
            P[:, self.const] = 1
            return P
        P[:, self.model.classes_.astype(int)] = self.model.predict_proba(X)
        return P

    def expected_weight(self, X, weights=config.SEVERITY_WEIGHTS) -> np.ndarray:
        return self.proba(X) @ np.asarray(weights, float)

    def predict(self, X) -> np.ndarray:
        return self.proba(X).argmax(1)


class FailureHeads:
    """One classifier per action: P(answer wrong | x). Trained only where the action exists."""

    def fit(self, X, ok, avail):
        self.models, self.base = {}, {}
        for a in HEADS:
            j = ACTIONS.index(a)
            m = avail[:, j]
            y = 1 - ok[m, j]
            self.base[a] = float(y.mean()) if m.any() else 1.0
            if m.sum() >= 10 and 0 < y.sum() < len(y):
                self.models[a] = _lr().fit(X[m], y)
        return self

    def predict(self, X) -> np.ndarray:
        P = np.zeros((len(X), len(ACTIONS)))
        for a in HEADS:
            j = ACTIONS.index(a)
            P[:, j] = self.models[a].predict_proba(X)[:, 1] if a in self.models else self.base[a]
        return P   # flag column stays 0: a human answer is taken as correct


def expected_costs(cost, avail, kinds, kinds_train, cost_train, avail_train):
    """Cost known at routing time: the training mean of each action's cost for queries of
    the same answer type (the real cost depends on the answer length, unknown in advance)."""
    out = np.zeros_like(cost)
    overall = np.array([cost_train[avail_train[:, j], j].mean() if avail_train[:, j].any() else 0
                        for j in range(cost.shape[1])])
    for k in np.unique(kinds):
        m, mt = kinds == k, kinds_train == k
        for j in range(cost.shape[1]):
            sel = mt & avail_train[:, j]
            out[m, j] = cost_train[sel, j].mean() if sel.any() else overall[j]
    return out


def choose(lam, w_hat, p_fail, cost_hat, avail) -> np.ndarray:
    """Index of the chosen action per query. lam = inf -> lowest predicted risk."""
    if np.isinf(lam):
        score = w_hat[:, None] * p_fail + 1e-12 * cost_hat
    else:
        score = cost_hat + lam * w_hat[:, None] * p_fail
    score = np.where(avail, score, np.inf)
    return score.argmin(1)
