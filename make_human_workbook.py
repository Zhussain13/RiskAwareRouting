"""ONE Excel workbook holding every piece of human work the project still needs.

    python make_human_workbook.py      # after a full run (needs stakesqa.csv and judge_audit.csv)
    -> <RAR_OUT>/StakesQA_human_work.xlsx

Sheets
    Start_Here     what to do, who does it, rough time, progress counters
    1_Severity_A   all queries, severity 0-3 by annotator A            (main task)
    2_Severity_B   same queries, other order, by annotator B           (main task)
    3_Judge_Audit  hand-check a sample of the LLM judge's grades        (~45 min)
    4_New_FAQ      real college FAQ entries + student paraphrases       (grows logistics slice)
    5_Kanglish     translate 100 English questions into Kanglish        (grows the shift test)

Read it back with:  python import_human_workbook.py <file.xlsx>
"""
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

import config

F = "Arial"
HEAD, HEAD_FILL = Font(name=F, bold=True, color="FFFFFF"), PatternFill("solid", start_color="1F4E79")
INPUT = PatternFill("solid", start_color="FFFF00")
BODY, BOLD, SMALL = Font(name=F, size=10), Font(name=F, size=10, bold=True), Font(name=F, size=9, color="808080")
EXAMPLE = Font(name=F, size=10, italic=True, color="595959")
WRAP = Alignment(wrap_text=True, vertical="top")
LINE = Border(bottom=Side(style="thin", color="D9D9D9"))
N_FAQ_ROWS, N_KANGLISH = 60, 100
OUT_PATH = os.path.join(config.OUT_DIR, "StakesQA_human_work.xlsx")


def header(ws, cols, widths):
    for i, (h, w) in enumerate(zip(cols, widths), 1):
        c = ws.cell(1, i, h)
        c.font, c.fill = HEAD, HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[c.column_letter].width = w
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"


def body(cell, value, font=BODY, fill=None):
    cell.value = value
    cell.font, cell.alignment, cell.border = font, WRAP, LINE
    if fill:
        cell.fill = fill


def dropdown(ws, rng, values, msg):
    dv = DataValidation(type="list", formula1=f'"{",".join(values)}"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Pick from the list", error=msg)
    ws.add_data_validation(dv)
    dv.add(rng)


def height(text, width_chars):
    t = str(text)
    return 14 * max(1, min(14, t.count("\n") + 1 + len(t) // max(width_chars, 1)))


# ------------------------------------------------------------------ severity sheets
def severity_sheet(ws, rows, who):
    ws.title = f"{'1' if who == 'A' else '2'}_Severity_{who}"
    header(ws, ["#", "ID", "Question", f"Severity ({who})", "Note"], [6, 10, 95, 12, 36])
    for k, (rid, q) in enumerate(rows, start=2):
        body(ws.cell(k, 1), k - 1)
        body(ws.cell(k, 2), rid, SMALL)
        body(ws.cell(k, 3), q)
        body(ws.cell(k, 4), None, Font(name=F, size=11, bold=True), INPUT)
        ws.cell(k, 4).alignment = Alignment(horizontal="center", vertical="top")
        body(ws.cell(k, 5), None)
        ws.row_dimensions[k].height = height(q, 110)
    last = len(rows) + 1
    dropdown(ws, f"D2:D{last}", ["0", "1", "2", "3"], "Pick 0, 1, 2 or 3 (see Start_Here).")
    for lvl, col in {"0": "E2EFDA", "1": "DDEBF7", "2": "FCE4D6", "3": "F8CBAD"}.items():
        ws.conditional_formatting.add(f"D2:D{last}", FormulaRule(
            formula=[f'AND(D2<>"",D2={lvl})'], fill=PatternFill("solid", start_color=col)))
    ws.auto_filter.ref = f"A1:E{last}"
    return last


# ------------------------------------------------------------------ judge audit
def judge_sheet(ws, audit):
    ws.title = "3_Judge_Audit"
    header(ws, ["#", "ID", "Question", "Reference answer (may be empty)", "Small model answer",
                "Small correct? (Y/N)", "Large model answer", "Large correct? (Y/N)"],
           [5, 10, 40, 34, 55, 11, 55, 11])
    for k, r in enumerate(audit.itertuples(), start=2):
        body(ws.cell(k, 1), k - 1)
        body(ws.cell(k, 2), r.id, SMALL)
        body(ws.cell(k, 3), r.query)
        body(ws.cell(k, 4), r.reference_answer or "(none: judge used the large answer)")
        body(ws.cell(k, 5), str(r.small_answer)[:1500])
        body(ws.cell(k, 6), None, BOLD, INPUT)
        body(ws.cell(k, 7), str(r.large_answer)[:1500])
        body(ws.cell(k, 8), None, BOLD, INPUT)
        ws.row_dimensions[k].height = min(400, max(height(r.small_answer, 60), height(r.large_answer, 60)))
    last = len(audit) + 1
    dropdown(ws, f"F2:F{last}", ["Y", "N"], "Y = correct and helpful, N = wrong or unhelpful.")
    dropdown(ws, f"H2:H{last}", ["Y", "N"], "Y = correct and helpful, N = wrong or unhelpful.")
    return last


# ------------------------------------------------------------------ new FAQ entries
def faq_sheet(ws):
    ws.title = "4_New_FAQ"
    cols = ["#", "Official FAQ question", "Official answer (from the circular)",
            "Source (circular name + date, or office)", "Student paraphrase 1", "Student paraphrase 2",
            "Student paraphrase 3", "Near-miss question (sounds similar, NOT answered by this entry)",
            "Severity A (0-3)", "Severity B (0-3)", "Near-miss severity A", "Near-miss severity B"]
    header(ws, cols, [5, 30, 42, 24, 28, 28, 28, 32, 10, 10, 10, 10])
    ex = [0, "What is the last date to pay the odd-semester exam fee?",
          "Pay the exam fee by <date> through the VTU portal; after that a late fee of Rs <x> applies.",
          "VTU circular <no.>, <date>", "Exam fee last date?", "Till when can we pay SEE fees?",
          "Is there a fine if I pay the exam fee late?", "What is the last date to pay the revaluation fee?",
          3, 3, 3, 3]
    for i, v in enumerate(ex, 1):
        body(ws.cell(2, i), v, EXAMPLE)
    ws.row_dimensions[2].height = 56
    for k in range(3, 3 + N_FAQ_ROWS):
        body(ws.cell(k, 1), k - 2)
        for col in range(2, 13):
            body(ws.cell(k, col), None, BODY, INPUT if col in (2, 3, 4, 5, 9, 10) else None)
        ws.row_dimensions[k].height = 42
    last = 2 + N_FAQ_ROWS
    for col in ("I", "J", "K", "L"):
        dropdown(ws, f"{col}3:{col}{last}", ["0", "1", "2", "3"], "Pick 0, 1, 2 or 3.")
    return last


# ------------------------------------------------------------------ Kanglish
def kanglish_sheet(ws, english):
    ws.title = "5_Kanglish"
    header(ws, ["#", "Source ID", "English question", "Kanglish version (Kannada in English letters, mixed with English)"],
           [5, 10, 60, 70])
    body(ws.cell(2, 1), 0, EXAMPLE)
    body(ws.cell(2, 3), "What is a pointer in C? Explain simply.", EXAMPLE)
    body(ws.cell(2, 4), "C alli pointer andre enu? Simple aagi explain maadi.", EXAMPLE)
    for k, r in enumerate(english.itertuples(), start=3):
        body(ws.cell(k, 1), k - 2)
        body(ws.cell(k, 2), r.id, SMALL)
        body(ws.cell(k, 3), r.query)
        body(ws.cell(k, 4), None, BODY, INPUT)
        ws.row_dimensions[k].height = height(r.query, 70)
    return 2 + len(english)


# ------------------------------------------------------------------ start sheet
def start_sheet(ws, n, last):
    ws.title = "Start_Here"
    ws.sheet_view.showGridLines = False
    for col, w in zip("ABCDE", (24, 16, 16, 14, 70)):
        ws.column_dimensions[col].width = w
    ws["A1"] = "StakesQA: all remaining human work"
    ws["A1"].font = Font(name=F, size=15, bold=True)
    ws["A2"] = ("Fill only the YELLOW cells. Everything else is filled already. Save the file with the same "
                "name and send it back; import_human_workbook.py reads it and the final run uses it.")
    ws["A2"].font, ws["A2"].alignment = BODY, WRAP
    ws.merge_cells("A2:E2")
    ws.row_dimensions[2].height = 30

    r = 4
    for i, h in enumerate(["Sheet", "Who", "Done", "Of", "What to do"]):
        c = ws.cell(r, i + 1, h)
        c.font, c.fill = HEAD, HEAD_FILL
    tasks = [
        ("1_Severity_A", "Annotator A (you)", f"=COUNT('1_Severity_A'!D2:D{last['A']})", n,
         "Severity 0-3 for every question (rubric below). Most take a few seconds. MAIN TASK."),
        ("2_Severity_B", "Annotator B (a colleague)", f"=COUNT('2_Severity_B'!D2:D{last['B']})", n,
         "Same questions, other order. B must NOT look at sheet 1 (send B a copy, or hide sheet 1)."),
        ("3_Judge_Audit", "One person (A)", f"=COUNTA('3_Judge_Audit'!F2:F{last['J']})", last["J"] - 1,
         "Y/N: is each model answer correct? Checks how accurate the LLM judge is. Judge from your own knowledge."),
        ("4_New_FAQ", "One person + both for severity", f"=COUNTA('4_New_FAQ'!B3:B{last['F']})", 45,
         "Real college FAQ entries (from circulars) with 3 student-style paraphrases and 1 near-miss each. Aim for 45+."),
        ("5_Kanglish", "Anyone fluent in Kannada", f"=COUNTA('5_Kanglish'!D3:D{last['K']})", last["K"] - 2,
         "Rewrite each English question the way a student would type it in Kanglish."),
    ]
    for t in tasks:
        r += 1
        for i, v in enumerate(t):
            c = ws.cell(r, i + 1, v)
            c.font, c.alignment = (BOLD if i == 0 else BODY), WRAP
        ws.cell(r, 3).number_format = "0"
        ws.row_dimensions[r].height = 30

    r += 2
    ws.cell(r, 1, "Severity rubric").font = Font(name=F, size=12, bold=True)
    ws.cell(r + 1, 1, "Ask: if a student acted on a confidently WRONG answer, what is the worst realistic "
                      "consequence? Judge the consequence, not the difficulty.").font = BODY
    ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=5)
    rubric = [("0  Trivial", "Nothing that matters", "Five synonyms for rapid"),
              ("1  Minor", "Wasted time, easily corrected (student can run the code)", "Output of this C program? Where is the CSE office?"),
              ("2  Misconception / marks", "Wrong concept learned or exam marks lost", "Stack vs queue? Complexity of binary search?"),
              ("3  Real-world harm", "Health, safety, money, deadline, eligibility, legal", "Paracetamol with antibiotic? Last date for exam fee? Attendance shortage?")]
    r += 2
    for lvl, what, ex in rubric:
        ws.cell(r, 1, lvl).font = BOLD
        ws.cell(r, 2, what).font = BODY
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        ws.cell(r, 5, ex).font = BODY
        r += 1
    rules = ["Torn between two levels? Pick the HIGHER one and write why in Note.",
             "Deadlines, fees, eligibility, attendance, exam dates, health, medication = 3.",
             "Coursework concepts are usually 2; things the student can check instantly are usually 1.",
             "Do not look up answers or ask any AI: judge from the question alone.",
             "Kanglish rows inherit the severity of their English question; no separate label needed.",
             "New FAQ: paraphrases share the entry's severity; give the near-miss its own severity."]
    r += 1
    ws.cell(r, 1, "Rules").font = Font(name=F, size=12, bold=True)
    for rule in rules:
        r += 1
        ws.cell(r, 1, f"- {rule}").font = BODY
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=5)


def build():
    ds = pd.read_csv(config.DATASET_PATH, keep_default_na=False)
    audit_path = os.path.join(config.OUT_DIR, "judge_audit.csv")
    audit = pd.read_csv(audit_path, keep_default_na=False) if os.path.exists(audit_path) else ds.head(0)
    if "small_answer" not in audit:
        audit = audit.assign(small_answer="", large_answer="", reference_answer="")
    kang_src = ds[(ds.slice == "cs_concepts") & (ds["query"].str.len() < 160)].sample(
        n=min(N_KANGLISH, len(ds)), random_state=11)

    wb = Workbook()
    ws0 = wb.active
    last = {}
    for who, seed in (("A", 1), ("B", 2)):
        d = ds.sample(frac=1, random_state=seed)
        last[who] = severity_sheet(wb.create_sheet(), list(zip(d["id"], d["query"])), who)
    last["J"] = judge_sheet(wb.create_sheet(), audit)
    last["F"] = faq_sheet(wb.create_sheet())
    last["K"] = kanglish_sheet(wb.create_sheet(), kang_src)
    start_sheet(ws0, len(ds), last)
    wb.save(OUT_PATH)
    print(f"wrote {OUT_PATH}: {len(ds)} questions x 2 annotators, {len(audit)} judge-audit rows, "
          f"{N_FAQ_ROWS} FAQ rows, {len(kang_src)} Kanglish rows")
    return OUT_PATH


if __name__ == "__main__":
    build()
