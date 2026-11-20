"""Stage 2: is each answer correct? Plus LLM severity pre-labels. One judge model load.

Closed-form answers are graded by script (no model): numeric match, option letter,
program output. Only open answers go to the LLM judge, which is a different model family
from the small and large models. The same model then pre-labels severity for every query.

    python grade.py
    python grade.py --demo
"""
import argparse
import time

import pandas as pd

import config
from riskroute import parsing, severity, simulator

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()
t0 = time.time()

df = pd.read_csv(config.DATASET_PATH, keep_default_na=False)
for role in ("small", "large"):
    g = pd.read_csv(config.GEN_PATH.format(role=role), keep_default_na=False)
    df = df.merge(g[["id", f"{role}_answer", f"{role}_raw"]], on="id", how="inner")
print(f"grading {len(df)} queries")


def script_grade(row, answer):
    kind, gold = row["answer_type"], row["gold"]
    if kind == "numeric":
        try:
            return parsing.numbers_equal(parsing.parse_number(answer), float(gold))
        except ValueError:
            return False
    if kind == "mc":
        return parsing.parse_letter(answer) == gold.strip().upper()
    if kind == "code_output":
        return parsing.parse_output_block(answer) == parsing.normalise_output(gold)
    return None


JUDGE = """You are grading a student-help chatbot.

Question:
{q}

Reference answer:
{ref}

Candidate answer:
{cand}

Is the candidate answer correct and complete enough to genuinely help a student? If the
reference says the information is not available, the candidate is correct only if it also
avoids inventing an answer. Reply with only one word: YES or NO."""

for role in ("small", "large"):
    df[f"{role}_ok"] = [script_grade(r, r[f"{role}_raw"]) for r in df.to_dict("records")]
df["graded_by"] = ["script" if k != "open" else "judge" for k in df["answer_type"]]

open_rows = df[df.answer_type == "open"]
has_ref = open_rows["reference_answer"].str.len() > 0
print(f"script-graded: {int((df.graded_by == 'script').sum())} | judge: {len(open_rows)} open "
      f"queries ({int(has_ref.sum())} with a reference, the rest graded against the large answer)")

if config.DEMO:
    for role in ("small", "large"):
        df.loc[open_rows.index, f"{role}_ok"] = [simulator.judge_open(a) for a in open_rows[f"{role}_answer"]]
    df.loc[open_rows.index[~has_ref.values], "large_ok"] = True
    df["severity_llm"] = [simulator.severity_llm(q, p) for q, p in zip(df["query"], df["severity_prior"])]
else:
    from riskroute.backend import VLLM
    judge = VLLM(config.JUDGE_MODEL)
    sys_msg = "You are a strict but fair grader."
    # small answers: against the reference, or the large answer when there is none
    refs = [r if r else la for r, la in zip(open_rows["reference_answer"], open_rows["large_answer"])]
    prompts = [JUDGE.format(q=q, ref=ref, cand=a)
               for q, ref, a in zip(open_rows["query"], refs, open_rows["small_answer"])]
    ref_rows = open_rows[has_ref.values]
    prompts += [JUDGE.format(q=q, ref=ref, cand=a) for q, ref, a in
                zip(ref_rows["query"], ref_rows["reference_answer"], ref_rows["large_answer"])]
    verdicts = [t.strip().upper().lstrip("*").startswith("YES")
                for t, _, _ in judge.chat(prompts, sys_msg, max_tokens=3)]
    n = len(open_rows)
    df.loc[open_rows.index, "small_ok"] = verdicts[:n]
    df.loc[open_rows.index, "large_ok"] = True          # no reference: large is the reference
    df.loc[ref_rows.index, "large_ok"] = verdicts[n:]
    sev = judge.chat([severity.LLM_PROMPT.format(q=q) for q in df["query"]],
                     "You are a careful annotator.", max_tokens=2)
    df["severity_llm"] = [severity.parse_llm_level(t) for t, _, _ in sev]
    df["severity_llm"] = df["severity_llm"].fillna(df["severity_prior"]).astype(int)

for role in ("small", "large"):
    df[f"{role}_ok"] = df[f"{role}_ok"].astype(bool).astype(int)

# Hand-audit sample: 10% of judge-graded rows, for reporting the judge's accuracy.
audit = df[df.graded_by == "judge"].sample(frac=0.10, random_state=config.SEED)
audit[["id", "query", "reference_answer", "small_answer", "small_ok", "large_answer", "large_ok"]] \
    .assign(human_small_ok="", human_large_ok="").to_csv(
        f"{config.OUT_DIR}/judge_audit.csv", index=False)

keep = ["id", "small_ok", "large_ok", "graded_by", "severity_llm"]
df[keep].to_csv(config.JUDGE_PATH, index=False)
print(df.groupby("slice")[["small_ok", "large_ok"]].mean().round(3).to_string())
print(f"done in {time.time() - t0:.0f}s -> {config.JUDGE_PATH} (+ judge_audit.csv for hand-checking)")
