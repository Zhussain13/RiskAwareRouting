# Severity labels: provenance

- File: `StakesQA_labels_2026-11-20.xlsx`, supplied by the author on 2026-11-20 (16:02 IST).
- Sheet `1_Severity_A`: 1,125 questions labelled 0-3. Sheet `2_Severity_B`: a 300-question
  stratified subset labelled 0-3. Sheet `3_Judge_Audit`: 38 judge-graded answers checked Y/N.
- The author states these are human ratings. The workbook's own title reads "draft dual-rater
  annotation workbook", and the raw per-rater responses (e.g. a form export) were not supplied.
- Agreement on the 300 overlapping questions is computed by `import_human_workbook.py`
  and reported in `results/human_import.json`.
- Final label per question: agreed label, otherwise the higher of A and B (rubric rule),
  otherwise A's label.

Before describing these as independent human annotations in a publication, keep the raw
per-rater records (who rated what, when) with the paper's data.
