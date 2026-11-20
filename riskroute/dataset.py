"""Assembles StakesQA: one table, one row per query.

Columns
    id, slice, query, answer_type (open | numeric | mc | code_output),
    reference_answer (open questions; may be empty), gold (numeric value, option letter or
    program output), faq_id (logistics: the FAQ entry that answers it, -1 = not in the FAQ),
    severity_prior, faq_sim, faq_margin, faq_match, context (FAQ entry given to the models)
"""
import os
import random

import numpy as np
import pandas as pd

import config
from . import cprograms, severity, verifiers
from .embed import FAQ

NOT_IN_FAQ = ("This is not covered in the college FAQ. The student should check the official "
              "notice or ask the department office instead of relying on a guess.")

TARGET_SIZES = {"gsm8k": 200, "truthfulqa": 150, "medmcqa": 150, "code_output": 250}


def _rows(df, slice_name, answer_type, **const):
    out = df.copy()
    out["slice"], out["answer_type"] = slice_name, answer_type
    for k, v in const.items():
        out[k] = v
    return out


def cs_concepts():
    df = pd.read_csv(f"{config.DATA_DIR}/slices/cs_concepts.csv")
    return _rows(df, "cs_concepts", "open", gold="", faq_id=-1)


def _human(name):
    """Rows added through the human workbook (import_human_workbook.py), if any."""
    p = f"{config.DATA_DIR}/slices/{name}_human.csv"
    return pd.read_csv(p, keep_default_na=False) if os.path.exists(p) else None


def kanglish():
    df = pd.read_csv(f"{config.DATA_DIR}/slices/kanglish.csv")
    extra = _human("kanglish")
    if extra is not None:
        df = pd.concat([df, extra[["query", "reference_answer"]]], ignore_index=True)
    return _rows(df, "kanglish", "open", gold="", faq_id=-1)


def logistics(faq: FAQ):
    df = pd.read_csv(f"{config.DATA_DIR}/slices/logistics.csv")
    extra = _human("logistics")
    if extra is not None:   # paraphrases point at their FAQ entry by question text; "" = near-miss
        pos = {q: i for i, q in enumerate(faq.df["question"])}
        extra["faq_id"] = [pos.get(q, -1) if q else -1 for q in extra["faq_question"]]
        df = pd.concat([df, extra[["query", "faq_id"]]], ignore_index=True)
    df["reference_answer"] = [faq.answer(i) if i >= 0 else NOT_IN_FAQ for i in df.faq_id]
    return _rows(df, "logistics", "open", gold="")


def code_output(n: int, seed: int):
    rows = []
    for fam, code in cprograms.generate(n, seed=seed):
        q = cprograms.QUESTION.format(code=code)
        if config.DEMO or not verifiers.gcc_available():
            gold = "(demo: not compiled)"
        else:
            ok, out = verifiers.run_c(code)
            if not ok:
                continue
            gold = out
        rows.append(dict(query=q, reference_answer="", gold=gold, family=fam))
    return _rows(pd.DataFrame(rows), "code_output", "code_output", faq_id=-1)


def _letters(choices):
    return "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))


def gsm8k(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(n))
    rows = [dict(query=r["question"], reference_answer="",
                 gold=r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds]
    return _rows(pd.DataFrame(rows), "gsm8k", "numeric", faq_id=-1)


def truthfulqa(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=seed).select(range(n))
    rng, rows = random.Random(seed), []
    for r in ds:
        ch, lab = list(r["mc1_targets"]["choices"]), list(r["mc1_targets"]["labels"])
        order = list(range(len(ch)))
        rng.shuffle(order)       # mc1 lists the right answer first: shuffle it away
        ch = [ch[i] for i in order]
        gold = chr(65 + [lab[i] for i in order].index(1))
        rows.append(dict(query=f"{r['question']}\n\n{_letters(ch)}", reference_answer="", gold=gold))
    return _rows(pd.DataFrame(rows), "truthfulqa", "mc", faq_id=-1)


def medmcqa(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    ds = ds.filter(lambda r: r["choice_type"] == "single").shuffle(seed=seed).select(range(n))
    rows = [dict(query=f"{r['question']}\n\n{_letters([r['opa'], r['opb'], r['opc'], r['opd']])}",
                 reference_answer="", gold="ABCD"[int(r["cop"])]) for r in ds]
    return _rows(pd.DataFrame(rows), "medmcqa", "mc", faq_id=-1)


def _demo_slice(name, answer_type, n, seed):
    """Placeholder rows so demo mode runs offline. Never used for real results."""
    rng = random.Random(f"{seed}{name}")
    topics = {"gsm8k": "A shop sells {a} pens a day for {b} days. How many pens in total?",
              "truthfulqa": "Demo misconception question #{a}: what happens if you {b}?\n\nA) x\nB) y\nC) z",
              "medmcqa": "Demo medical question #{a}: which drug is used for condition {b}?\n\nA) p\nB) q\nC) r\nD) s"}
    rows = []
    for i in range(n):
        a, b = rng.randint(2, 90), rng.randint(2, 30)
        gold = str(a * b) if answer_type == "numeric" else rng.choice("ABC")
        rows.append(dict(query=topics[name].format(a=a + 100 * i, b=b), reference_answer="", gold=gold))
    return _rows(pd.DataFrame(rows), name, answer_type, faq_id=-1)


def build(sizes=None, seed: int = config.SEED) -> pd.DataFrame:
    sizes = {**TARGET_SIZES, **(sizes or {})}
    faq = FAQ()
    parts = [cs_concepts(), logistics(faq), code_output(sizes["code_output"], seed), kanglish()]
    for name, fn, kind in (("gsm8k", gsm8k, "numeric"), ("truthfulqa", truthfulqa, "mc"),
                           ("medmcqa", medmcqa, "mc")):
        parts.append(_demo_slice(name, kind, sizes[name], seed) if config.DEMO
                     else fn(sizes[name], seed))
    df = pd.concat(parts, ignore_index=True)
    df["reference_answer"] = df["reference_answer"].fillna("").astype(str)
    df["gold"] = df["gold"].fillna("").astype(str)
    df["faq_id"] = df["faq_id"].fillna(-1).astype(int)
    df = df.drop_duplicates("query").reset_index(drop=True)

    sim, margin, match = faq.lookup_many(df["query"])
    df["faq_sim"], df["faq_margin"], df["faq_match"] = sim.round(4), margin.round(4), match
    df["context"] = [faq.entry(m) if s >= config.CONTEXT_MIN_SIM else ""
                     for s, m in zip(sim, match)]
    df["severity_prior"] = [severity.prior(q, s) for q, s in zip(df["query"], df["slice"])]
    df.insert(0, "id", [f"q{i:05d}" for i in range(len(df))])
    return df[["id", "slice", "query", "answer_type", "reference_answer", "gold", "faq_id",
               "severity_prior", "faq_sim", "faq_margin", "faq_match", "context"]
              + (["family"] if "family" in df else [])]


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("slice").agg(n=("id", "size"), answer_type=("answer_type", "first"),
                                    mean_prior=("severity_prior", "mean"))
            .round(2).sort_values("n", ascending=False))


def check(df: pd.DataFrame):
    """Sanity checks that fail loudly before any GPU time is spent."""
    assert df["id"].is_unique
    assert set(df["answer_type"]) <= {"open", "numeric", "mc", "code_output"}
    if not config.DEMO:
        bad = df[(df.answer_type != "open") & (df.gold.str.len() == 0)]
        assert bad.empty, f"{len(bad)} closed-form rows have no gold answer"
    assert np.all(df["severity_prior"].between(0, 3))
