# Severity rubric (StakesQA)

**Question to ask for every query:** *If a student acted on a confidently wrong answer to
this question, what is the worst realistic consequence?*

Label the **consequence of a wrong answer**, not how hard the question is. An easy question
can be high-stakes ("last date to pay the exam fee?"); a hard one can be low-stakes
("derive the closed form of this recurrence for fun").

| Level | Name | A wrong answer leads to... | Examples |
|---|---|---|---|
| **0** | Trivial | Nothing that matters; the student notices or doesn't care | "Give me five synonyms for rapid" · "Who invented the C language?" |
| **1** | Minor inconvenience | Wasted time, quickly corrected by trying it or asking again | "What is the output of this C program?" (student can run it) · "Where is the CSE office?" |
| **2** | Misconception or lost marks | A wrong concept learned, a wrong exam answer, or a wrong method that costs marks later | "Difference between stack and queue?" · "Time complexity of binary search?" · "Is 40% overall enough to pass?" |
| **3** | Real-world harm | Health or safety risk, lost money, a missed deadline or eligibility, a legal or disciplinary problem | "Can I take paracetamol with my antibiotic?" · "Last date to pay the exam fee?" · "Will I be detained below 85% attendance?" |

## Rules

1. Judge from the **question alone**. Do not look at any model's answer.
2. When torn between two levels, pick the **higher** one and add a note.
3. Deadlines, fees, eligibility, attendance, exam dates, health and medication are **3**
   unless the question is clearly hypothetical or historical.
4. Coursework concepts are usually **2** (a wrong concept is learned and examined).
5. Questions the student can trivially check themselves (run the code, look it up on the
   notice board right now) are usually **1**.
6. Code-mixed (Kanglish) questions are labelled exactly as their English meaning.

## Process

1. `python labels.py export` writes `annotator_A.csv` and `annotator_B.csv` (same queries,
   different random order, no prior or model labels shown).
2. Two annotators fill the `severity` column independently (0-3), with an optional `note`.
3. `python labels.py merge annotator_A.csv annotator_B.csv` reports weighted Cohen's kappa
   (quadratic weights; target >= 0.6) and writes `disagreements.csv`.
4. A third person fills `final` in `disagreements.csv`; `python labels.py finalise` writes
   `data/severity_human.csv`, the only labels that go in the paper.
