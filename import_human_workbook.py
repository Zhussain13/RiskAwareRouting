"""Reads the filled StakesQA_human_work.xlsx back into the project.

    python import_human_workbook.py StakesQA_human_work.xlsx
    python import_human_workbook.py StakesQA_human_work.xlsx --b StakesQA_human_work_B.xlsx
        (when annotator B filled a separate copy of the workbook)

Writes
    data/severity_human.csv            final human severity per query (keyed by query text)
    data/faq.csv                       + the new official FAQ entries
    data/slices/logistics_human.csv    student paraphrases (answered by the FAQ) + near-misses
    data/slices/kanglish_human.csv     new Kanglish questions
    <RAR_OUT>/results/human_import.json   agreement (kappa), judge accuracy, counts

Disagreements between A and B are settled by the rubric's own rule (take the higher level)
unless --resolve points to a CSV with columns id,final.
Then rebuild and re-run everything: python run_pipeline.py --force dataset grade verify evaluate
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import config
from riskroute import severity

ap = argparse.ArgumentParser()
ap.add_argument("workbook")
ap.add_argument("--b", help="separate workbook filled by annotator B (optional)")
ap.add_argument("--resolve", help="CSV id,final settling A/B disagreements (optional)")
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()

D = config.DATA_DIR
report = {}


def sheet(path, name, **kw):
    return pd.read_excel(path, sheet_name=name, **kw)


def num(s):
    return pd.to_numeric(s, errors="coerce")


# ------------------------------------------------------------------ 1+2: severity
A = sheet(args.workbook, "1_Severity_A")
B = sheet(args.b or args.workbook, "2_Severity_B")
A = A.rename(columns={"ID": "id", "Question": "query", "Severity (A)": "a"})[["id", "query", "a"]]
B = B.rename(columns={"ID": "id", "Severity (B)": "b"})[["id", "b"]]
S = A.merge(B, on="id", how="outer")
S["a"], S["b"] = num(S.a), num(S.b)
both = S.dropna(subset=["a", "b"])
if len(both):
    report["severity"] = {"labelled_by_both": len(both),
                          "exact_agreement": round(float((both.a == both.b).mean()), 4),
                          "weighted_kappa": round(severity.weighted_kappa(both.a, both.b), 4)}
S["final"] = np.where(S.a.notna() & S.b.notna(), np.fmax(S.a, S.b), S.a.fillna(S.b))
S["rule"] = np.where(S.a.notna() & S.b.notna(),
                     np.where(S.a == S.b, "agreed", "higher of A/B"), "single annotator")
if args.resolve:
    res = pd.read_csv(args.resolve)[["id", "final"]].rename(columns={"final": "resolved"})
    S = S.merge(res, on="id", how="left")
    S["rule"] = np.where(S.resolved.notna(), "third person", S.rule)
    S["final"] = S.resolved.fillna(S.final)
S.loc[S.a.notna() & S.b.notna() & (S.a != S.b)].to_csv(
    os.path.join(config.OUT_DIR, "severity_disagreements.csv"), index=False)
report["severity_rules"] = S.rule.value_counts().to_dict()
report["severity_missing"] = int(S.final.isna().sum())
final = S.dropna(subset=["final"])[["id", "query", "final", "a", "b", "rule"]].rename(
    columns={"final": "severity_human"})
final["severity_human"] = final.severity_human.astype(int)
by_id = dict(zip(final.id, final.severity_human))

# ------------------------------------------------------------------ 3: judge audit
J = sheet(args.workbook, "3_Judge_Audit").rename(columns={"ID": "id"})
judged_path = config.JUDGE_PATH
if len(J) and os.path.exists(judged_path):
    jd = pd.read_csv(judged_path)[["id", "small_ok", "large_ok"]]
    J = J.merge(jd, on="id", how="left")
    rows = []
    for col, human in (("small_ok", "Small correct? (Y/N)"), ("large_ok", "Large correct? (Y/N)")):
        h = J[human].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
        ok = h.notna() & J[col].notna()
        rows += list(zip(h[ok], J.loc[ok, col]))
    if rows:
        h, j = np.array(rows).T
        report["judge_audit"] = {"n_grades_checked": len(rows),
                                 "judge_agrees_with_human": round(float((h == j).mean()), 4),
                                 "judge_says_correct_but_human_says_wrong": int(((j == 1) & (h == 0)).sum()),
                                 "judge_says_wrong_but_human_says_correct": int(((j == 0) & (h == 1)).sum())}

# ------------------------------------------------------------------ 4: new FAQ entries
Fq = sheet(args.workbook, "4_New_FAQ")
Fq = Fq[num(Fq["#"]) >= 1]                         # drop the example row
c = Fq.columns
faq = pd.read_csv(f"{D}/faq.csv")
new_faq, log_rows, new_sev = [], [], []
for _, r in Fq.iterrows():
    q, a = str(r[c[1]]).strip(), str(r[c[2]]).strip()
    if q in ("", "nan") or a in ("", "nan"):
        continue
    if q not in set(faq.question):
        new_faq.append({"question": q, "answer": a, "source": str(r[c[3]]).strip()})
    sev = np.nanmax([num(pd.Series([r[c[8]]]))[0], num(pd.Series([r[c[9]]]))[0], np.nan])
    for p in (r[c[4]], r[c[5]], r[c[6]]):
        p = str(p).strip()
        if p and p != "nan":
            log_rows.append({"query": p, "faq_question": q})
            new_sev.append((p, sev))
    nm = str(r[c[7]]).strip()
    if nm and nm != "nan":
        log_rows.append({"query": nm, "faq_question": ""})
        new_sev.append((nm, np.nanmax([num(pd.Series([r[c[10]]]))[0], num(pd.Series([r[c[11]]]))[0], np.nan])))
if new_faq:
    if "source" not in faq:
        faq["source"] = ""
    faq = pd.concat([faq, pd.DataFrame(new_faq)], ignore_index=True)
    faq.to_csv(f"{D}/faq.csv", index=False)
if log_rows:
    pd.DataFrame(log_rows).drop_duplicates("query").to_csv(f"{D}/slices/logistics_human.csv", index=False)
report["new_faq_entries"], report["new_logistics_queries"] = len(new_faq), len(log_rows)

# ------------------------------------------------------------------ 5: Kanglish
K = sheet(args.workbook, "5_Kanglish")
K = K[num(K["#"]) >= 1]
kc = K.columns
kang = []
for _, r in K.iterrows():
    t = str(r[kc[3]]).strip()
    if t and t != "nan":
        kang.append({"query": t, "reference_answer": "", "source_id": r[kc[1]]})
        new_sev.append((t, by_id.get(r[kc[1]], np.nan)))   # same meaning, same severity
if kang:
    pd.DataFrame(kang).drop_duplicates("query").to_csv(f"{D}/slices/kanglish_human.csv", index=False)
report["new_kanglish_queries"] = len(kang)

# ------------------------------------------------------------------ write the labels
extra = pd.DataFrame(new_sev, columns=["query", "severity_human"]).dropna()
extra["severity_human"] = extra.severity_human.astype(int)
extra["rule"] = "new row (entry / source label)"
out = pd.concat([final, extra], ignore_index=True).drop_duplicates("query", keep="first")
out.to_csv(f"{D}/severity_human.csv", index=False)
report["severity_labels_written"] = len(out)
report["severity_distribution"] = out.severity_human.value_counts().sort_index().to_dict()

os.makedirs(config.RESULTS_DIR, exist_ok=True)
json.dump(report, open(os.path.join(config.RESULTS_DIR, "human_import.json"), "w"), indent=2, default=int)
print(json.dumps(report, indent=2, default=int))
if report.get("severity_missing"):
    print(f"WARNING: {report['severity_missing']} questions have no severity yet; "
          "the pipeline will fall back to LLM labels until every question is labelled.")
print("\nNext: python run_pipeline.py --force dataset grade verify evaluate")
