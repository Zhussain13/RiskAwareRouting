"""Prompts for each answer type and parsers for the models' replies."""
import re

SYSTEM = ("You are a helpful teaching assistant for engineering students in India. "
          "Answer clearly and concisely. If the question is about college rules, dates or "
          "fees and the answer is not in the context you are given, say you do not know "
          "and tell the student to check the official notice.")

CONF_LINE = ("After your answer, write one final line exactly like 'CONFIDENCE: N' where N "
             "is 1-10 (how sure you are that your answer is fully correct).")

FORMAT = {
    "open": CONF_LINE,
    "numeric": ("Solve it briefly. Then write exactly three final lines:\n"
                "ANSWER: <the final number only>\n"
                "PYTHON: <one Python arithmetic expression that computes the answer>\n"
                "CONFIDENCE: <1-10>"),
    "mc": ("Reply with exactly two lines:\nANSWER: <the letter of the correct option>\n"
           "CONFIDENCE: <1-10>"),
    "code_output": ("Reply with the program's exact output between the lines OUTPUT_START "
                    "and OUTPUT_END, nothing else, then a final line 'CONFIDENCE: N' (1-10)."),
}


def build_prompt(query: str, answer_type: str, context: str | None = None) -> str:
    ctx = f"College FAQ context:\n{context}\n\n" if context else ""
    return f"{ctx}Question:\n{query}\n\n{FORMAT[answer_type]}"


def parse_confidence(text: str):
    """Returns (text without the confidence line, confidence 1-10; 5 if missing)."""
    m = re.findall(r"CONFIDENCE:\**\s*(\d+)", text)
    conf = int(m[-1]) if m else 5
    clean = re.sub(r"\s*\**CONFIDENCE:\**\s*\d+.*$", "", text, flags=re.S).strip()
    return clean, max(1, min(conf, 10))


_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def parse_number(text: str):
    """The number after 'ANSWER:', else the last number in the text."""
    m = re.findall(rf"ANSWER:\**\s*\$?\s*({_NUM})", text)
    if not m:
        m = re.findall(_NUM, text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def parse_python_expr(text: str):
    m = re.findall(r"PYTHON:\**\s*`*([^\n`]+)", text)
    return m[-1].strip() if m else None


def parse_letter(text: str):
    m = re.findall(r"ANSWER:\**\s*\(?([A-Ha-h])\b", text)
    if not m:
        m = re.findall(r"^\s*\(?([A-H])\)?[.)\s]*$", text, flags=re.M)
    return m[-1].upper() if m else None


def parse_output_block(text: str):
    m = re.search(r"OUTPUT_START\s*\n?(.*?)\n?\s*OUTPUT_END", text, flags=re.S)
    if m:
        out = m.group(1)
    else:  # fall back to a fenced block
        m = re.search(r"```(?:\w+)?\n(.*?)```", text, flags=re.S)
        out = m.group(1) if m else text
    return normalise_output(out)


def normalise_output(s: str) -> str:
    """Compare program outputs ignoring trailing spaces and blank lines at the ends."""
    lines = [ln.rstrip() for ln in str(s).strip("\n").splitlines()]
    return "\n".join(lines).strip()


def numbers_equal(a, b, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
