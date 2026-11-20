"""One train / calibration / test split: fit the router, calibrate it, and replay every
method (ours, baselines, ablations) on the test queries."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

from . import calibration, metrics
from .router import ACTIONS, FailureHeads, StakesEstimator, choose, expected_costs
from .severity import keyword_high_stakes

A = {a: i for i, a in enumerate(ACTIONS)}
SMR_FAQ_SIM = 0.47 if config.DEMO else 0.75     # SmartModelRouter's own FAQ threshold
WEIGHT_VARIANTS = {"weights_flat": (0, 0.25, 0.5, 1), "weights_steep": (0, 0.05, 0.2, 1)}


def stratified_split(severity, seed, fractions=config.SPLIT):
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for lvl in np.unique(severity):
        idx = rng.permutation(np.where(severity == lvl)[0])
        c1 = int(round(fractions[0] * len(idx)))
        c2 = c1 + int(round(fractions[1] * len(idx)))
        for p, chunk in zip(parts, (idx[:c1], idx[c1:c2], idx[c2:])):
            p.extend(chunk.tolist())
    return [np.sort(np.array(p, int)) for p in parts]


def cheapest_threshold(scores, ok_low, ok_high, cost_low, cost_high, grid, target):
    """SmartModelRouter's rule: cheapest t (high route if score >= t) whose accuracy is
    >= target x always-high accuracy. Falls back to the most accurate t."""
    goal = target * ok_high.mean()
    best, most = None, None
    for t in grid:
        hi = scores >= t
        acc = np.where(hi, ok_high, ok_low).mean()
        cost = np.where(hi, cost_high, cost_low).mean()
        if most is None or acc > most[1] or (acc == most[1] and cost < most[2]):
            most = (t, acc, cost)
        if acc >= goal and (best is None or cost < best[2]):
            best = (t, acc, cost)
    return (best or most)[0]


class Split:
    def __init__(self, T, X, idx, sev_true, kinds):
        self.T, self.X = T, X
        self.tr, self.cal, self.te = idx
        self.sev = sev_true
        self.kinds = kinds
        self.ok, self.cost, self.lat, self.avail = T["ok"], T["cost"], T["lat"], T["avail"]
        self.vlarge = T["verify_large"]

    # ------------------------------------------------------------------ helpers
    def sub(self, M, idx):
        return M[idx]

    def outcome(self, a, idx):
        return metrics.outcome_from_actions(a, self.ok[idx], self.cost[idx], self.lat[idx], self.vlarge[idx])

    def summarise(self, out, idx, weights=config.SEVERITY_WEIGHTS):
        return metrics.summarise(*out, self.sev[idx], weights)

    # ------------------------------------------------------------------ our router
    def fit(self, train_sev, avail=None):
        avail = self.avail if avail is None else avail
        fit_idx = self.tr
        stakes = StakesEstimator().fit(self.X[fit_idx], train_sev[fit_idx])
        heads = FailureHeads().fit(self.X[fit_idx], self.ok[fit_idx], avail[fit_idx])
        return stakes, heads

    def policy(self, stakes, heads, idx, avail, weights, uniform=False, w_override=None):
        X = self.X[idx]
        w_hat = np.ones(len(idx)) if uniform else stakes.expected_weight(X, weights)
        if w_override is not None:
            w_hat = w_override
        p = heads.predict(X)
        c = expected_costs(self.cost[idx], avail[idx], self.kinds[idx], self.kinds[self.tr],
                           self.cost[self.tr], avail[self.tr])
        return lambda lam: choose(lam, w_hat, p, c, avail[idx])

    def calibrate(self, chooser, alpha, weights, method="crc", true_sev=None, cal_weights=None):
        sev = self.sev if true_sev is None else true_sev
        w = np.asarray(cal_weights if cal_weights is not None else weights, float)[sev[self.cal]]
        lams = calibration.lambda_grid(scale=float(np.mean(self.cost[self.tr, A["large"]])))
        L = calibration.loss_matrix(chooser, lams, w, self.ok[self.cal])
        self.last_feasible = calibration.feasible(L, alpha)
        if method == "ltt":
            return calibration.ltt(L, lams, alpha, config.DELTA)[0]
        return calibration.crc(L, lams, alpha)[0]

    def run_ours(self, alpha=config.ALPHA, method="crc", weights=config.SEVERITY_WEIGHTS,
                 train_sev=None, avail=None, uniform=False, stakes_noise=0.0, lam=None):
        train_sev = self.sev if train_sev is None else train_sev
        avail = self.avail if avail is None else avail
        stakes, heads = self.fit(train_sev, avail)
        cal_choose = self.policy(stakes, heads, self.cal, avail, weights, uniform)
        if lam is None:
            lam = self.calibrate(cal_choose, alpha, weights, method, true_sev=train_sev)
        w_override = None
        if stakes_noise:
            w_override = stakes.expected_weight(self.X[self.te], weights)
            rng = np.random.default_rng(0)
            hit = (self.sev[self.te] == 3) & (rng.random(len(self.te)) < stakes_noise)
            w_override[hit] = weights[1]        # a severity-3 query mistaken for a level-1 one
        te_choose = self.policy(stakes, heads, self.te, avail, weights, uniform, w_override)
        return te_choose(lam), lam, (stakes, heads, cal_choose)

    # ------------------------------------------------------------------ SmartModelRouter
    def smr_route(self, scores_te, t, idx):
        """FAQ if similarity >= SMR's threshold, else large if score >= t, else small."""
        faq = self.T["faq_sim"][idx] >= SMR_FAQ_SIM
        return np.where(faq, A["faq"], np.where(scores_te >= t, A["large"], A["small"]))

    def smr_threshold(self, scores_fit, fit_idx, target=config.ACCURACY_TARGET):
        grid = np.unique(np.concatenate([[0.0, 1.01], np.quantile(scores_fit, np.linspace(0, 1, 101))]))
        return cheapest_threshold(scores_fit, self.ok[fit_idx, A["small"]], self.ok[fit_idx, A["large"]],
                                  self.cost[fit_idx, A["small"]], self.cost[fit_idx, A["large"]],
                                  grid, target)

    def smr_learned_scores(self):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5,
                                                              class_weight="balanced"))
        y = 1 - self.ok[self.tr, A["small"]]
        m.fit(self.X[self.tr], y)
        return lambda idx: m.predict_proba(self.X[idx])[:, 1]

    def cascade(self, t, idx):
        conf = self.T["small_conf"][idx]
        esc = conf < t
        ok = np.where(esc, self.ok[idx, A["large"]], self.ok[idx, A["small"]])
        cost = self.cost[idx, A["small"]] + esc * self.cost[idx, A["large"]]
        lat = self.lat[idx, A["small"]] + esc * self.lat[idx, A["large"]]
        route = np.where(esc, "large", "small")
        return ok, cost, lat, esc.astype(int), route


def cost_on(split, a, idx):
    return float(np.mean(split.cost[idx][np.arange(len(idx)), a]))


def match_cost(split, chooser, idx, target_cost, lams):
    """The lam whose cost on idx is closest to target_cost without exceeding it."""
    best, best_c = lams[-1], None
    for lam in lams:
        c = cost_on(split, chooser(lam), idx)
        if c <= target_cost * 1.0001 and (best_c is None or c > best_c):
            best, best_c = lam, c
    return best


def run(split: Split, kanglish_idx=None) -> dict:
    S = split
    te, fit_idx = S.te, np.concatenate([S.tr, S.cal])
    res, extra = {}, {}

    # ---- ours
    a, lam, (stakes, heads, cal_choose) = S.run_ours(method="crc")
    res["ours_crc"] = S.summarise(S.outcome(a, te), te)
    extra["lam_crc"] = lam
    extra["stakes_acc"] = float((stakes.predict(S.X[te]) == S.sev[te]).mean())
    a_ltt, lam_ltt, _ = S.run_ours(method="ltt")
    res["ours_ltt"] = S.summarise(S.outcome(a_ltt, te), te)
    our_cal_cost = cost_on(S, cal_choose(lam), S.cal)

    # ---- simple references
    n = len(te)
    res["always_small"] = S.summarise(S.outcome(np.full(n, A["small"]), te), te)
    res["always_large"] = S.summarise(S.outcome(np.full(n, A["large"]), te), te)
    ok_te, cost_te, av_te = S.ok[te].copy(), S.cost[te].copy(), S.avail[te].copy()
    av_te[:, A["flag"]] = False
    c = np.where(av_te & (ok_te == 1), cost_te, np.inf)
    oracle = np.where(np.isfinite(c.min(1)), c.argmin(1), A["large"])
    res["oracle"] = S.summarise(S.outcome(oracle, te), te)

    # ---- SmartModelRouter baselines (their own 95%-accuracy thresholds)
    from .features import rule_score
    rules = np.array([rule_score(q) for q in S.T["query"]])
    t = S.smr_threshold(rules[fit_idx], fit_idx)
    res["smr_rules"] = S.summarise(S.outcome(S.smr_route(rules[te], t, te), te), te)
    learned = S.smr_learned_scores()
    t_l = S.smr_threshold(learned(fit_idx), fit_idx)
    a_smr = S.smr_route(learned(te), t_l, te)
    res["smr_learned"] = S.summarise(S.outcome(a_smr, te), te)
    conf_fit = S.T["small_conf"][fit_idx]
    best_t, best_cost = 11, None   # cascade: cheapest confidence cut keeping 95% accuracy
    goal = config.ACCURACY_TARGET * S.ok[fit_idx, A["large"]].mean()
    for tc in range(1, 12):
        o = S.cascade(tc, fit_idx)
        if o[0].mean() >= goal and (best_cost is None or o[1].mean() < best_cost):
            best_t, best_cost = tc, o[1].mean()
    res["smr_cascade"] = S.summarise(S.cascade(best_t, te), te)
    kw = np.array([keyword_high_stakes(q) for q in S.T["query"][te]])
    res["keyword_floor"] = S.summarise(S.outcome(np.where(kw, A["large"], a_smr), te), te)

    # ---- same budget as ours: does the money go to the right queries?
    lams = calibration.lambda_grid(scale=float(np.mean(S.cost[S.tr, A["large"]])))
    stakes_u, heads_u = S.fit(S.sev)
    uni_cal = S.policy(stakes_u, heads_u, S.cal, S.avail, config.SEVERITY_WEIGHTS, uniform=True)
    lam_u = match_cost(S, uni_cal, S.cal, our_cal_cost, lams)
    uni_te = S.policy(stakes_u, heads_u, te, S.avail, config.SEVERITY_WEIGHTS, uniform=True)
    res["uniform_risk_matched"] = S.summarise(S.outcome(uni_te(lam_u), te), te)
    lc = learned(S.cal)
    ts = np.unique(np.quantile(lc, np.linspace(0, 1, 201)))
    # lowest threshold (= most large-model calls) that still fits our budget
    t_m = min([tt for tt in ts if cost_on(S, S.smr_route(lc, tt, S.cal), S.cal) <= our_cal_cost * 1.0001]
              or [1.01])
    res["smr_learned_matched"] = S.summarise(S.outcome(S.smr_route(learned(te), t_m, te), te), te)

    # ---- no human in the loop: can the models alone meet alpha?
    av = S.avail.copy()
    av[:, A["flag"]] = False
    a_nf = S.run_ours(avail=av)[0]
    res["ours_no_human"] = S.summarise(S.outcome(a_nf, te), te)
    extra["no_human_feasible"] = S.last_feasible

    # ---- ablations of ours
    av = S.avail.copy()
    av[:, A["verify"]] = False
    res["abl_no_verify"] = S.summarise(S.outcome(S.run_ours(avail=av)[0], te), te)
    av = S.avail.copy()
    av[:, A["faq"]] = False
    res["abl_no_faq"] = S.summarise(S.outcome(S.run_ours(avail=av)[0], te), te)
    res["abl_stakes_blind"] = S.summarise(S.outcome(S.run_ours(uniform=True)[0], te), te)
    # no calibration: cheapest lam keeping 95% of large accuracy on train + calibration
    ch_fit = S.policy(stakes, heads, fit_idx, S.avail, config.SEVERITY_WEIGHTS)
    goal = config.ACCURACY_TARGET * S.ok[fit_idx, A["large"]].mean()
    lam95 = lams[-1]
    for lm in lams:
        aa = ch_fit(lm)
        if S.ok[fit_idx][np.arange(len(fit_idx)), aa].mean() >= goal:
            lam95 = lm
            break
    res["abl_no_calibration"] = S.summarise(S.outcome(S.run_ours(lam=lam95)[0], te), te)
    for name in ("llm", "prior"):
        col = S.T[f"severity_{name}"]
        res[f"abl_{name}_labels"] = S.summarise(S.outcome(S.run_ours(train_sev=col)[0], te), te)
    res["abl_stakes_noise_30"] = S.summarise(S.outcome(S.run_ours(stakes_noise=0.3)[0], te), te)
    for name, wv in WEIGHT_VARIANTS.items():
        aw = S.run_ours(weights=wv)[0]
        res[f"abl_{name}"] = S.summarise(S.outcome(aw, te), te, weights=wv)

    # ---- distribution shift: Kanglish queries were never in train or calibration
    if kanglish_idx is not None and len(kanglish_idx):
        ch_k = S.policy(stakes, heads, kanglish_idx, S.avail, config.SEVERITY_WEIGHTS)
        res["shift_kanglish"] = S.summarise(S.outcome(ch_k(lam), kanglish_idx), kanglish_idx)
    return res, extra


def curves(split: Split, alphas):
    """Cost vs risk on the test split, sweeping each method's knob."""
    S, te = split, split.te
    rows = []
    for alpha in alphas:
        a, _, _ = S.run_ours(alpha=alpha)
        rows.append({"method": "ours", "knob": alpha, **S.summarise(S.outcome(a, te), te)})
        a, _, _ = S.run_ours(alpha=alpha, uniform=True)
        rows.append({"method": "stakes_blind", "knob": alpha, **S.summarise(S.outcome(a, te), te)})
    learned = S.smr_learned_scores()
    lt = learned(te)
    for t in np.quantile(lt, np.linspace(0, 1, 25)):
        rows.append({"method": "smr_learned", "knob": float(t),
                     **S.summarise(S.outcome(S.smr_route(lt, t, te), te), te)})
    return rows
