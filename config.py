"""All settings in one place. Every value can be overridden with an environment variable.

Run any script with --demo (or RAR_DEMO=1) to use the built-in simulator: no GPU, no
downloads, results in seconds. Demo numbers are SIMULATED and never go in a paper.
"""
import os
import sys

DEMO = os.getenv("RAR_DEMO", "0") == "1" or "--demo" in sys.argv

# ---------------------------------------------------------------- models (all free)
# Small and large keep SmartModelRouter's Qwen2.5 pair so results stay comparable.
SMALL_MODEL = os.getenv("SMALL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LARGE_MODEL = os.getenv("LARGE_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
# The judge is a different model family, so no model grades its own answers.
# Llama needs you to accept Meta's licence on Hugging Face (set HF_TOKEN in Colab).
# Ungated fallback: JUDGE_MODEL=microsoft/Phi-3.5-mini-instruct
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# vLLM settings for a 16 GB T4 (no bf16 on T4, so fp16).
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "half")
VLLM_GPU_UTIL = float(os.getenv("VLLM_GPU_UTIL", "0.90"))
VLLM_MAX_LEN = int(os.getenv("VLLM_MAX_LEN", "2048"))
VLLM_MAX_SEQS = int(os.getenv("VLLM_MAX_SEQS", "64"))
CHUNK = int(os.getenv("CHUNK", "300"))            # rows per checkpoint inside a stage

# Answer length caps per answer type (short answers = fast runs + easier grading).
MAX_TOKENS = {"open": 200, "numeric": 160, "mc": 40, "code_output": 60}

# ---------------------------------------------------------------- cost accounting
# Everything runs free. For comparison, calls are priced at paid list prices
# (USD per 1M tokens). Large = Gemini 2.5 Flash list price, as in SmartModelRouter.
# Small = a typical hosted 3B price (assumption; set to 0 to match SmartModelRouter).
LARGE_PRICE_IN = float(os.getenv("LARGE_PRICE_IN", "0.30"))
LARGE_PRICE_OUT = float(os.getenv("LARGE_PRICE_OUT", "2.50"))
SMALL_PRICE_IN = float(os.getenv("SMALL_PRICE_IN", "0.02"))
SMALL_PRICE_OUT = float(os.getenv("SMALL_PRICE_OUT", "0.04"))
# A human answering a flagged question. Expressed as a multiple of the mean
# large-model cost per question, so it scales with whatever prices you set.
HUMAN_COST_X_LARGE = float(os.getenv("HUMAN_COST_X_LARGE", "20"))
VERIFY_COST = float(os.getenv("VERIFY_COST", "0.0"))  # gcc / Python / NLI run locally

# ---------------------------------------------------------------- severity
# Levels: 0 trivial, 1 minor inconvenience, 2 misconception or lost marks,
# 3 health, safety, money, legal or deadline harm. See data/rubric/severity_rubric.md
SEVERITY_WEIGHTS = tuple(float(x) for x in os.getenv("SEVERITY_WEIGHTS", "0,0.1,0.4,1").split(","))
# Where severity labels come from: "human" (final), "llm" (pre-labels) or "prior" (rules).
# Only "human" is publishable. The pipeline falls back down this list and warns.
SEVERITY_SOURCE = os.getenv("SEVERITY_SOURCE", "human")

# ---------------------------------------------------------------- routing
FAQ_MIN_SIM = float(os.getenv("FAQ_MIN_SIM", "0.30"))      # below this the FAQ action is unavailable
CONTEXT_MIN_SIM = float(os.getenv("CONTEXT_MIN_SIM", "0.50"))  # FAQ entries given to models as context
ACTIONS = ("faq", "small", "verify", "large", "flag")

# Risk budget and guarantee settings.
# Bound on expected severity-weighted error. It must sit above what the safest available
# actions can reach; evaluate.py warns when alpha < the large model's own SWER (then only
# human flagging can meet it). Sweep it (curves) rather than trusting one value.
ALPHA = float(os.getenv("ALPHA", "0.05"))
DELTA = float(os.getenv("DELTA", "0.10"))        # LTT: guarantee holds with prob >= 1 - delta
CALIBRATION = os.getenv("CALIBRATION", "crc")    # "crc" (monotonised CRC) or "ltt"
N_SPLITS = int(os.getenv("N_SPLITS", "100"))
SPLIT = (0.50, 0.25, 0.25)                       # train / calibration / test

# SmartModelRouter baselines tune the cheapest threshold keeping this share of large accuracy.
ACCURACY_TARGET = float(os.getenv("ACCURACY_TARGET", "0.95"))
SEED = int(os.getenv("SEED", "42"))

# ---------------------------------------------------------------- paths
DATA_DIR = "data"
FAQ_PATH = os.path.join(DATA_DIR, "faq.csv")
OUT_DIR = os.getenv("RAR_OUT", os.path.join("outputs", "demo" if DEMO else "real"))
DATASET_PATH = os.path.join(OUT_DIR, "stakesqa.csv")
GEN_PATH = os.path.join(OUT_DIR, "gen_{role}.csv")
JUDGE_PATH = os.path.join(OUT_DIR, "judged.csv")
VERIFY_PATH = os.path.join(OUT_DIR, "verified.csv")
TABLE_PATH = os.path.join(OUT_DIR, "table.csv")    # one row per query, everything joined
RESULTS_DIR = os.path.join(OUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
