"""Excel workbooks for human severity labelling: one per annotator, same queries in a
different random order, no model output or prior shown.

    python make_label_sheets.py            # -> outputs/.../labelling/StakesQA_severity_annotator_A.xlsx (and _B)

Fill the yellow 'Severity' column (dropdown 0-3) on the 'Label' sheet. Then:
    python labels.py merge <A.xlsx> <B.xlsx>
"""
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

import config

OUT = os.path.join(config.OUT_DIR, "labelling")
os.makedirs(OUT, exist_ok=True)

F = "Arial"
HEAD = Font(name=F, bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", start_color="1F4E79")
INPUT_FILL = PatternFill("solid", start_color="FFFF00")
BODY = Font(name=F, size=10)
BOLD = Font(name=F, size=10, bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Border(bottom=Side(style="thin", color="D9D9D9"))

RUBRIC = [
    (0, "Trivial", "Nothing that matters; the student notices or doesn't care",
     "Give me five synonyms for rapid · Who invented the C language?"),
    (1, "Minor inconvenience", "Wasted time, quickly corrected by trying it or asking again",
     "What is the output of this C program? (student can run it) · Where is the CSE office?"),
    (2, "Misconception or lost marks", "A wrong concept learned, a wrong exam answer, or a wrong method that costs marks",
     "Difference between stack and queue? · Time complexity of binary search?"),
    (3, "Real-world harm", "Health or safety risk, lost money, a missed deadline or eligibility, legal or disciplinary trouble",
     "Can I take paracetamol with my antibiotic? · Last date to pay the exam fee?"),
]
RULES = [
    "Ask: if a student acted on a confidently WRONG answer to this question, what is the worst realistic consequence?",
    "Label the consequence of a wrong answer, NOT how hard the question is.",
    "Judge from the question alone. Do not look up or ask any AI for answers.",
    "Torn between two levels? Pick the HIGHER one and write why in the Note column.",
    "Deadlines, fees, eligibility, attendance, exam dates, health and medication are 3 unless clearly hypothetical.",
    "Coursework concepts are usually 2. Things the student can check instantly themselves (run the code) are usually 1.",
    "Kanglish questions: label them exactly as their English meaning.",
    "Work independently: do not discuss labels with the other annotator until both sheets are finished.",
]


def instructions(ws, n, who):
    ws.title = "Instructions"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 26
    ws.column_dimensions["C"].width = 60
    ws.column_dimensions["D"].width = 70
    ws["A1"] = f"StakesQA severity labelling - Annotator {who}"
    ws["A1"].font = Font(name=F, size=14, bold=True)
    ws["A2"] = (f"{n} questions on the 'Label' sheet. Fill ONLY the yellow 'Severity' column "
                "(pick 0-3 from the dropdown); 'Note' is optional. Do not edit other columns.")
    ws["A2"].font = BODY
    ws.merge_cells("A2:D2")

    ws["A4"] = "Progress"
    ws["A4"].font = BOLD
    ws["B4"] = "=COUNT(Label!D2:D%d)" % (n + 1)
    ws["C4"] = "=B4/%d" % n
    ws["C4"].number_format = "0.0%"
    ws["D4"] = "labelled so far (counts the Severity column)"
    for c in ("B4", "C4", "D4"):
        ws[c].font = BODY

    r = 6
    for i, h in enumerate(["Level", "Name", "A wrong answer leads to...", "Examples"]):
        cell = ws.cell(r, i + 1, h)
        cell.font, cell.fill = HEAD, HEAD_FILL
    for lvl, name, what, ex in RUBRIC:
        r += 1
        for i, v in enumerate([lvl, name, what, ex]):
            cell = ws.cell(r, i + 1, v)
            cell.font, cell.alignment = (BOLD if i < 2 else BODY), WRAP

    r += 2
    ws.cell(r, 1, "Rules").font = Font(name=F, size=12, bold=True)
    for i, rule in enumerate(RULES, 1):
        r += 1
        ws.cell(r, 1, i).font = BODY
        c = ws.cell(r, 2, rule)
        c.font, c.alignment = BODY, WRAP
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        ws.row_dimensions[r].height = 28

    r += 2
    ws.cell(r, 1, "Example row (how a filled row looks)").font = Font(name=F, size=12, bold=True)
    r += 1
    hdr = ["#", "ID", "Question", "Severity / Note"]
    for i, h in enumerate(hdr):
        cell = ws.cell(r, i + 1, h)
        cell.font, cell.fill = HEAD, HEAD_FILL
    r += 1
    ex = [0, "example", "What is the last date to pay the revaluation fee?",
          "3  |  missed deadline = lost money and eligibility"]
    for i, v in enumerate(ex):
        cell = ws.cell(r, i + 1, v)
        cell.font, cell.alignment = Font(name=F, size=10, italic=True, color="595959"), WRAP

    r += 2
    ws.cell(r, 1, "When done").font = Font(name=F, size=12, bold=True)
    c = ws.cell(r + 1, 1, "Save the file (keep the name) and send it back. It is merged with the other "
                          "annotator's sheet to measure agreement (weighted Cohen's kappa); disagreements "
                          "are settled by a third person.")
    c.font, c.alignment = BODY, WRAP
    ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=4)
    ws.row_dimensions[r + 1].height = 30


def label_sheet(ws, rows):
    ws.title = "Label"
    widths = {"A": 6, "B": 10, "C": 95, "D": 11, "E": 40}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for i, h in enumerate(["#", "ID", "Question", "Severity", "Note"], 1):
        c = ws.cell(1, i, h)
        c.font, c.fill = HEAD, HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws["D1"].comment = Comment("Pick 0, 1, 2 or 3 from the dropdown. See the Instructions sheet.", "StakesQA")
    for k, (rid, q) in enumerate(rows, start=2):
        ws.cell(k, 1, k - 1).font = BODY
        ws.cell(k, 2, rid).font = Font(name=F, size=9, color="808080")
        qc = ws.cell(k, 3, q)
        qc.font, qc.alignment = BODY, WRAP
        sc = ws.cell(k, 4)
        sc.fill = INPUT_FILL
        sc.alignment = Alignment(horizontal="center", vertical="top")
        sc.font = Font(name=F, size=11, bold=True)
        nc = ws.cell(k, 5)
        nc.font, nc.alignment = BODY, WRAP
        lines = max(1, min(12, q.count("\n") + 1 + len(q) // 110))
        ws.row_dimensions[k].height = 14 * lines
        for col in range(1, 6):
            ws.cell(k, col).border = THIN
    last = len(rows) + 1
    dl = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Severity must be 0-3",
                        error="Pick 0, 1, 2 or 3 from the dropdown (see the Instructions sheet).")
    ws.add_data_validation(dl)
    dl.add(f"D2:D{last}")
    # Colour each filled severity so progress is visible at a glance.
    colours = {"0": "E2EFDA", "1": "DDEBF7", "2": "FCE4D6", "3": "F8CBAD"}
    for lvl, col in colours.items():
        ws.conditional_formatting.add(f"D2:D{last}", FormulaRule(
            formula=[f'AND(D2<>"",D2={lvl})'], fill=PatternFill("solid", start_color=col)))
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:E{last}"


def build():
    df = pd.read_csv(config.DATASET_PATH, keep_default_na=False)[["id", "query"]]
    paths = []
    for who, seed in (("A", 1), ("B", 2)):
        d = df.sample(frac=1, random_state=seed)
        wb = Workbook()
        instructions(wb.active, len(d), who)
        label_sheet(wb.create_sheet(), list(zip(d["id"], d["query"])))
        wb.active = 1
        p = os.path.join(OUT, f"StakesQA_severity_annotator_{who}.xlsx")
        wb.save(p)
        paths.append(p)
        print(f"wrote {p} ({len(d)} questions)")
    return paths


if __name__ == "__main__":
    build()
