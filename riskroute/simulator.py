"""Demo-mode stand-in for every model (small, large, judge, severity labeller, NLI).

Deterministic (same query -> same result) and loosely realistic, so the full pipeline,
the parsers and the guarantee check all run offline in seconds. It writes replies in the
same formats real models are asked for, so the real parsers and graders are exercised.

DEMO NUMBERS ARE SIMULATED, NOT RESULTS. Never report them.
"""
import hashlib
import re

# Base accuracy per slice for (small, large), before the difficulty adjustment.
BASE = {"cs_concepts": (0.86, 0.97), "kanglish": (0.66, 0.90), "logistics": (0.55, 0.80),
        "code_output": (0.58, 0.82), "gsm8k": (0.72, 0.92), "truthfulqa": (0.45, 0.66),
        "medmcqa": (0.44, 0.66)}
_HARD = ["why", "debug", "prove", "derive", "design", "analy", "compare", "trace", "explain how",
         "step by step", "complexity", "segfault", "recursion", "fork", "static", "switch"]


def u(text: str, salt: str) -> float:
    h = hashlib.md5(f"{salt}|{text}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0x100000000


def difficulty(q: str) -> float:
    ql = q.lower()
    d = 0.2 + 0.3 * any(w in ql for w in _HARD) + 0.15 * min(len(q.split()), 60) / 60
    d += 0.4 * (u(q, "noise") - 0.5)
    return min(max(d, 0.0), 1.0)


def correct(q: str, slice_name: str, role: str, faq_id: int = -1, has_context: bool = False) -> bool:
    small, large = BASE.get(slice_name, (0.7, 0.9))
    p = small if role == "small" else large
    p -= (0.35 if role == "small" else 0.15) * (difficulty(q) - 0.4)
    if slice_name == "logistics":
        # Without the right FAQ entry in context, nobody can know college-specific facts.
        if faq_id >= 0 and not has_context:
            p -= 0.35
        if faq_id < 0:          # near-miss: the right answer is "not in the FAQ"
            p -= 0.15 if role == "small" else 0.05
    return u(q, f"ok-{role}") < min(max(p, 0.02), 0.99)


def confidence(q: str, ok: bool) -> int:
    c = 8.8 - 4 * difficulty(q) + 3 * (u(q, "conf") - 0.5) - (1.3 if not ok else 0)
    return int(min(max(round(c), 1), 10))


def reply(row: dict, role: str):
    """Returns (text, tokens_in, tokens_out) in the format asked for that answer type."""
    q, kind, gold = row["query"], row["answer_type"], str(row.get("gold", ""))
    ok = correct(q, row["slice"], role, int(row.get("faq_id", -1)), bool(row.get("context")))
    conf = confidence(q, ok) if role == "small" else 9
    d = difficulty(q)
    tin = int(len(q.split()) * 1.4) + 60
    if kind == "code_output":
        out = gold if ok else gold + " 0"
        text, tout = f"OUTPUT_START\n{out}\nOUTPUT_END\nCONFIDENCE: {conf}", 20
    elif kind == "numeric":
        g = float(gold)
        val = g if ok else g + 1 + int(10 * u(q, "off"))
        consistent = ok or u(q, f"cons-{role}") < 0.35   # wrong answers are often self-consistent
        expr = f"{val:g}" if consistent else f"{val + 7:g}"
        text = f"Working...\nANSWER: {val:g}\nPYTHON: {expr}\nCONFIDENCE: {conf}"
        tout = int(80 + 60 * d)
    elif kind == "mc":
        letters = sorted(set(re.findall(r"^([A-H])\)", q, flags=re.M))) or ["A", "B", "C"]
        wrong = [x for x in letters if x != gold] or letters
        ans = gold if ok else wrong[int(u(q, "letter") * len(wrong))]
        text, tout = f"ANSWER: {ans}\nCONFIDENCE: {conf}", 8
    else:
        tag = "SIMOK" if ok else "SIMBAD"
        text = f"[{role}:{tag}] Simulated answer to: {q[:80]}\nCONFIDENCE: {conf}"
        tout = int(70 + 130 * d) if role == "small" else int(110 + 90 * d)
    return text, tin, tout


def judge_open(answer: str) -> bool:
    return "SIMOK" in answer


def severity_true(q: str, prior: int) -> int:
    """Stand-in for human labels: the prior, with some disagreement."""
    r = u(q, "sev")
    if r < 0.12:
        return max(prior - 1, 0)
    if r > 0.93:
        return min(prior + 1, 3)
    return prior


def severity_llm(q: str, prior: int) -> int:
    r = u(q, "sevllm")
    return max(prior - 1, 0) if r < 0.2 else prior


def nli_pass(small_ok: bool, q: str) -> bool:
    return small_ok if u(q, "nli") < 0.88 else not small_ok
