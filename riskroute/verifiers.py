"""Cheap, external checks of the small model's answer. No verifier asks a model to grade
itself: each one is grounded in something outside the model.

    code_output : compile and run the C program in the question with gcc, compare outputs
    numeric     : evaluate the model's own PYTHON expression safely; it must agree with
                  the model's stated ANSWER (a consistency check, not the gold answer)
    faq         : the answer must be entailed by the retrieved FAQ entry (NLI model)

verifiability(row) says which verifier applies to a query, from the query's form only.
"""
import ast
import operator
import os
import shutil
import subprocess
import tempfile

from . import parsing
from .cprograms import extract_code

# ---------------------------------------------------------------- which verifier applies


def verifiability(query: str, answer_type: str, faq_sim: float, context_min_sim: float) -> str:
    """'code', 'numeric', 'faq' or 'none'. Uses only what is visible at routing time."""
    if answer_type == "code_output" and extract_code(query):
        return "code"
    if answer_type == "numeric":
        return "numeric"
    if faq_sim >= context_min_sim and answer_type == "open":
        return "faq"
    return "none"


# ---------------------------------------------------------------- C programs


def gcc_available() -> bool:
    return shutil.which("gcc") is not None


def run_c(code: str, timeout: float = 5.0):
    """Compiles and runs a C program. Returns (ok, normalised stdout or error message).

    Programs here come from our own generator or from student questions; they run in a
    temporary folder with a time limit and no input.
    """
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "p.c"), os.path.join(d, "p")
        with open(src, "w") as f:
            f.write(code)
        c = subprocess.run(["gcc", "-std=c99", "-O0", "-w", src, "-o", exe],
                           capture_output=True, text=True, timeout=60)
        if c.returncode != 0:
            return False, "compile error: " + c.stderr[-300:]
        try:
            r = subprocess.run([exe], capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, parsing.normalise_output(r.stdout)


def verify_code(query: str, small_answer: str) -> bool:
    code = extract_code(query)
    if not code:
        return False
    ok, out = run_c(code)
    return ok and parsing.parse_output_block(small_answer) == out


# ---------------------------------------------------------------- arithmetic

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str):
    """Evaluates + - * / // % ** and brackets on numbers only. Anything else -> None."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("not plain arithmetic")
    try:
        return float(ev(ast.parse(expr.replace(",", "").replace("^", "**"), mode="eval")))
    except Exception:
        return None


def verify_numeric(small_answer: str) -> bool:
    stated = parsing.parse_number(small_answer)
    expr = parsing.parse_python_expr(small_answer)
    if stated is None or not expr:
        return False
    return parsing.numbers_equal(safe_eval(expr), stated)


# ---------------------------------------------------------------- FAQ entailment

_nli = None


def _load_nli(model_name: str):
    global _nli
    if _nli is None:
        import torch
        from transformers import pipeline
        _nli = pipeline("text-classification", model=model_name,
                        device=0 if torch.cuda.is_available() else -1)
    return _nli


def verify_faq_batch(premises, answers, model_name: str, threshold: float = 0.5):
    """True where the FAQ entry entails the answer (entailment prob >= threshold)."""
    nli = _load_nli(model_name)
    pairs = [{"text": p, "text_pair": a[:1000]} for p, a in zip(premises, answers)]
    out = nli(pairs, top_k=None, batch_size=32, truncation=True)
    res = []
    for scores in out:
        ent = next((s["score"] for s in scores if s["label"].lower().startswith("entail")), 0.0)
        res.append(ent >= threshold)
    return res
