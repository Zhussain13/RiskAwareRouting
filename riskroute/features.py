"""Router input features: everything is computed from the query alone (plus the FAQ lookup),
so routing never needs a model call. SmartModelRouter's rule score is kept as a feature."""
import re

import numpy as np

from .embed import embed
from .severity import keyword_high_stakes

HARD_WORDS = ["why", "debug", "error", "segfault", "crash", "prove", "derive", "compare",
              "design", "optimi", "complexity", "step by step", "analy", "difference between",
              "trace", "output of"]
CODE_PATTERN = r"[{};]|#include|int main|def |printf\(|\bclass\b|=>"
TYPES = ("open", "numeric", "mc", "code_output")


def rule_score(q: str) -> float:
    """SmartModelRouter's hand-written difficulty score in [0, 1]."""
    s, ql = 0.0, q.lower()
    if len(q.split()) > 40:
        s += 0.4
    if re.search(CODE_PATTERN, q):
        s += 0.4
    if any(w in ql for w in HARD_WORDS):
        s += 0.3
    if re.search(r"\d+\s*[-+*/^]\s*\d+", q):
        s += 0.2
    return min(s, 1.0)


def handcrafted(df) -> np.ndarray:
    rows = []
    for q, kind, sim, margin in zip(df["query"], df["answer_type"], df["faq_sim"], df["faq_margin"]):
        rows.append([rule_score(q), min(len(q.split()), 200) / 200, float(keyword_high_stakes(q)),
                     float(sim), float(margin)] + [float(kind == t) for t in TYPES])
    return np.asarray(rows)


def build(df) -> np.ndarray:
    """384 embedding numbers + 9 hand-made features per query."""
    return np.hstack([embed(df["query"]), handcrafted(df)])
