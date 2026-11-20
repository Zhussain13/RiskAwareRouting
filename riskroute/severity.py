"""Severity levels, weights, a rule-based prior, the LLM pre-label prompt and agreement stats."""
import re

import numpy as np

LEVELS = (0, 1, 2, 3)

HIGH_STAKES = [
    # deadlines, money, eligibility
    r"\blast date\b", r"\bdeadline\b", r"\bfee\b", r"\bfees\b", r"\bpenalty\b", r"\bfine\b",
    r"\bscholarship\b", r"\beligib", r"\bdetain", r"\battendance\b", r"\brevaluation\b",
    r"\bexam date\b", r"\bia-?\s?\d\b", r"\bmakeup\b", r"\bbacklog\b", r"\bcertificate\b",
    # health and safety
    r"\bmedicine\b", r"\bmedication\b", r"\btablet\b", r"\bdose\b", r"\bdosage\b",
    r"\bparacetamol\b", r"\bantibiotic\b", r"\bpregnan", r"\ballerg", r"\bsymptom",
    r"\bdisease\b", r"\btreatment\b", r"\bpoison", r"\binjur", r"\bsuicid", r"\bdepress",
    # legal and disciplinary
    r"\blegal\b", r"\bharass", r"\bplagiari", r"\bmalpractice\b", r"\bpolice\b",
]
_HIGH = re.compile("|".join(HIGH_STAKES), re.I)

SLICE_PRIOR = {"cs_concepts": 2, "code_output": 1, "logistics": 1, "gsm8k": 1,
               "truthfulqa": 2, "medmcqa": 3, "kanglish": 2}


def prior(query: str, slice_name: str) -> int:
    """Rule-based severity guess. Used only as a fallback and as a baseline feature check;
    never as the labels reported in a paper."""
    base = SLICE_PRIOR.get(slice_name, 2)
    return 3 if _HIGH.search(query) else base


def keyword_high_stakes(query: str) -> bool:
    """The 'obvious practitioner fix' baseline: these words force the large model."""
    return bool(_HIGH.search(query))


def weights_of(levels, weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return w[np.asarray(levels, dtype=int)]


LLM_PROMPT = """You label questions sent to a student-help chatbot at an Indian engineering college.

If a student acted on a confidently WRONG answer to this question, what is the worst realistic
consequence? Judge the consequence, not the difficulty.

0 = trivial: nothing that matters
1 = minor inconvenience: wasted time, easily corrected (e.g. the student can just run the code)
2 = misconception or lost marks: a wrong concept learned or a wrong exam answer
3 = real-world harm: health or safety, money, a missed deadline or eligibility, legal or disciplinary trouble

Deadlines, fees, attendance, eligibility, exam dates, health and medication are 3.
When torn between two levels, choose the higher.

Question:
{q}

Reply with one digit only (0, 1, 2 or 3)."""


def parse_llm_level(text: str):
    m = re.search(r"[0-3]", text or "")
    return int(m.group(0)) if m else None


def weighted_kappa(a, b) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(np.asarray(a, int), np.asarray(b, int),
                                   weights="quadratic", labels=list(LEVELS)))
