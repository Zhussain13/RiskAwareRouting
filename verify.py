"""Stage 3: run the external verifier on every small-model answer it applies to.

    code    -> gcc compiles and runs the program in the question (CPU)
    numeric -> safe evaluation of the model's own PYTHON expression (CPU)
    faq     -> NLI entailment of the answer by the retrieved FAQ entry (GPU or CPU)

    python verify.py
    python verify.py --demo
"""
import argparse
import time

import pandas as pd

import config
from riskroute import simulator, verifiers
from riskroute.embed import FAQ

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()
t0 = time.time()

df = pd.read_csv(config.DATASET_PATH, keep_default_na=False)
df = df.merge(pd.read_csv(config.GEN_PATH.format(role="small"), keep_default_na=False)
              [["id", "small_raw"]], on="id")
df = df.merge(pd.read_csv(config.JUDGE_PATH)[["id", "small_ok"]], on="id")

df["verifier"] = [verifiers.verifiability(q, k, s, config.CONTEXT_MIN_SIM)
                  for q, k, s in zip(df["query"], df["answer_type"], df["faq_sim"])]
df["verify_pass"] = 0

if config.DEMO:
    code = df.verifier == "code"
    df.loc[code, "verify_pass"] = df.loc[code, "small_ok"]          # running the code is exact
    num = df.verifier == "numeric"
    df.loc[num, "verify_pass"] = [int(verifiers.verify_numeric(a)) for a in df.loc[num, "small_raw"]]
    faq = df.verifier == "faq"
    df.loc[faq, "verify_pass"] = [int(simulator.nli_pass(bool(ok), q)) for ok, q in
                                  zip(df.loc[faq, "small_ok"], df.loc[faq, "query"])]
else:
    for i in df.index[df.verifier == "code"]:
        df.at[i, "verify_pass"] = int(verifiers.verify_code(df.at[i, "query"], df.at[i, "small_raw"]))
    for i in df.index[df.verifier == "numeric"]:
        df.at[i, "verify_pass"] = int(verifiers.verify_numeric(df.at[i, "small_raw"]))
    faq_rows = df[df.verifier == "faq"]
    if len(faq_rows):
        kb = FAQ()
        premises = [kb.answer(m) for m in faq_rows["faq_match"]]
        answers = [a.split("CONFIDENCE")[0] for a in faq_rows["small_raw"]]
        df.loc[faq_rows.index, "verify_pass"] = [
            int(p) for p in verifiers.verify_faq_batch(premises, answers, config.NLI_MODEL)]

df[["id", "verifier", "verify_pass"]].to_csv(config.VERIFY_PATH, index=False)
s = df[df.verifier != "none"]
tab = s.groupby("verifier").apply(lambda g: pd.Series({
    "n": len(g), "pass_rate": g.verify_pass.mean(),
    "precision": g[g.verify_pass == 1].small_ok.mean() if g.verify_pass.any() else float("nan"),
    "catch_rate": 1 - g[g.small_ok == 0].verify_pass.mean() if (g.small_ok == 0).any() else float("nan"),
}), include_groups=False).round(3)
print(tab.to_string())
print("precision = P(small correct | verifier passed); catch_rate = P(verifier fails | small wrong)")
print(f"done in {time.time() - t0:.0f}s -> {config.VERIFY_PATH}")
