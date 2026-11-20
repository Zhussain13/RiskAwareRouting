import numpy as np
import pytest

from riskroute import calibration, cprograms, parsing, severity, verifiers
from riskroute.router import choose


# ---------------------------------------------------------------- parsing
def test_confidence_line_removed():
    clean, conf = parsing.parse_confidence("A pointer stores an address.\n**CONFIDENCE:** 8")
    assert clean == "A pointer stores an address." and conf == 8
    assert parsing.parse_confidence("no line")[1] == 5
    assert parsing.parse_confidence("CONFIDENCE: 42")[1] == 10


def test_number_letter_and_output():
    assert parsing.parse_number("so 3 boxes\nANSWER: 1,250\nPYTHON: 125*10") == 1250
    assert parsing.parse_python_expr("ANSWER: 4\nPYTHON: `2+2`") == "2+2"
    assert parsing.parse_letter("ANSWER: (c)\nCONFIDENCE: 3") == "C"
    out = parsing.parse_output_block("OUTPUT_START\n6 0  \n\nOUTPUT_END\nCONFIDENCE: 9")
    assert out == "6 0"
    assert parsing.numbers_equal(0.1 + 0.2, 0.3)


# ---------------------------------------------------------------- verifiers
def test_safe_eval_only_arithmetic():
    assert verifiers.safe_eval("(18 - 3 - 4) * 2") == 22
    assert verifiers.safe_eval("__import__('os').system('ls')") is None
    assert verifiers.safe_eval("2 ** 1000") is None


def test_numeric_verifier_checks_consistency():
    assert verifiers.verify_numeric("ANSWER: 18\nPYTHON: (16-3-4)*2")
    assert not verifiers.verify_numeric("ANSWER: 20\nPYTHON: (16-3-4)*2")
    assert not verifiers.verify_numeric("ANSWER: 20")


def test_verifiability_uses_query_form_only():
    q = cprograms.QUESTION.format(code="int main(void){return 0;}\n")
    assert verifiers.verifiability(q, "code_output", 0.1, 0.5) == "code"
    assert verifiers.verifiability("x", "numeric", 0.1, 0.5) == "numeric"
    assert verifiers.verifiability("When is IA-2?", "open", 0.9, 0.5) == "faq"
    assert verifiers.verifiability("What is a pointer?", "open", 0.1, 0.5) == "none"


@pytest.mark.skipif(not verifiers.gcc_available(), reason="gcc not installed")
def test_generated_programs_compile_and_verify():
    progs = cprograms.generate(32, seed=3)
    assert len({c for _, c in progs}) == 32
    fam, code = progs[0]
    ok, out = verifiers.run_c(code)
    assert ok and out
    q = cprograms.QUESTION.format(code=code)
    assert cprograms.extract_code(q) == code
    assert verifiers.verify_code(q, f"OUTPUT_START\n{out}\nOUTPUT_END")
    assert not verifiers.verify_code(q, "OUTPUT_START\nwrong\nOUTPUT_END")


# ---------------------------------------------------------------- severity
def test_severity_prior_and_kappa():
    assert severity.prior("What is the last date to pay the exam fee?", "logistics") == 3
    assert severity.prior("Where is the CSE office?", "logistics") == 1
    assert severity.parse_llm_level("Level: 2") == 2
    assert severity.weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)


# ---------------------------------------------------------------- selector + calibration
def test_selector_risk_never_rises_with_lambda():
    rng = np.random.default_rng(0)
    n = 200
    p = rng.random((n, 5))
    p[:, 4] = 0
    cost = np.sort(rng.random((n, 5)), axis=1)
    avail = rng.random((n, 5)) > 0.2
    avail[:, [1, 4]] = True
    w = rng.random(n)
    prev = None
    for lam in calibration.lambda_grid(1.0):
        a = choose(lam, w, p, cost, avail)
        risk = p[np.arange(n), a]
        if prev is not None:
            assert np.all(risk <= prev + 1e-12)
        prev = risk
    assert np.all(choose(np.inf, w, p, cost, avail) == 4)


def test_crc_and_ltt_respect_alpha():
    rng = np.random.default_rng(1)
    n, lams = 400, np.array([0.0, 1.0, 2.0, np.inf])
    base = np.array([0.30, 0.10, 0.04, 0.0])
    L = (rng.random((n, 4)) < base).astype(float)
    lam, j = calibration.crc(L, lams, alpha=0.08)
    assert lam == 2.0
    lam_ltt, _ = calibration.ltt(L, lams, alpha=0.08, delta=0.1)
    assert lam_ltt >= lam
    assert calibration.feasible(L, 0.08)
    L[:, -1] = 0.5
    assert not calibration.feasible(L, 0.08)
