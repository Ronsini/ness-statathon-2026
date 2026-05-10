# NESS Statathon 2026 — Policy Retention Model

## What this project is

Kaggle multiclass classification competition. Predict whether a Travelers
property insurance policy will be:

- 0 = won't cancel (~71% of training data)
- 1 = may cancel but can be convinced (~7%, the hardest class)
- 2 = will cancel (~22%)

Metric: accuracy. Public/private split: 30/70. Submission deadline: May 15, 2026.

## Pipeline location

- `src/train_model.py` — main training pipeline (5-fold stratified CV with LightGBM)
- `src/tune_params.py` — Optuna hyperparameter search (separate, not auto-run)
- `src/eda.py` — exploratory analysis
- `data/train.csv`, `data/test.csv` — gitignored, downloaded from Kaggle
- `output/submission.csv` — current Kaggle submission
- `output/results_log.txt` — raw, append-only log of every attempt
- `docs/ATTEMPTS.md` — curated, human-readable attempt history (newest first)
- `docs/PROJECT_STATE.md` — current snapshot (overwritten on each PASS)
- `docs/ATTEMPT_TEMPLATE.md` — blank hypothesis form for planning the next attempt

## Workflow: when the user pastes terminal output from a run

1. Parse the terminal output for: attempt label, fold scores, mean CV, std,
   confusion matrix, raw OOF accuracy, tuned OOF accuracy if present,
   multipliers if present, runtime
2. Compare against the current best in `docs/PROJECT_STATE.md`
3. Determine verdict (auto-classify; user can override):
   - **PASS**: new CV beats prior best by > 0.001
   - **MIXED**: within ±0.001 of prior best
   - **FAIL**: more than 0.001 below prior best
4. Prepend a new entry to `docs/ATTEMPTS.md` using the format below
5. If verdict is PASS, overwrite `docs/PROJECT_STATE.md` with the new state.
   Otherwise leave PROJECT_STATE.md unchanged.
6. Write a 2-3 sentence summary explaining what changed, why it moved (or
   didn't), and what the next attempt should consider trying. Reference the
   confusion matrix shifts to ground the analysis.
7. Do NOT modify `src/train_model.py` unless the user explicitly asks.

## ATTEMPTS.md entry format (newest at top)

```
attempt-N: [one-line description] — [PASS/FAIL/MIXED]
Branch: improve/attempt-N
Date: YYYY-MM-DD
Runtime: Xs
CV accuracy: 0.XXXXX (raw) → 0.XXXXX (tuned)
Std across folds: 0.XXXXX
Naive baseline: 0.XXXXX
Hypothesis: [what the user expected and why — pull from ATTEMPT_NOTES in code]
Changes vs prior attempt:
  - [bulleted from ATTEMPT_NOTES]

Result:
  [1-3 sentences of what actually happened]
  [per-class recall comparison vs prior attempt if relevant]
  [any surprises]

Confusion matrix:
true \ pred    0         1         2
0              .         .         .
1              .         .         .
2              .         .         .
Per-class recall: class-0: XX.X%,  class-1: XX.X%,  class-2: XX.X%
Verdict: [why this is/isn't the new best]
Next attempt should try: [concrete suggestion]
```

---

## PROJECT_STATE.md format (overwritten on each PASS)

```
# Project State (last updated YYYY-MM-DD)

## Current best
Branch: improve/attempt-N
CV accuracy: 0.XXXXX
Public leaderboard: 0.XXXXX (or "not submitted")
Submission file: output/submission.csv

## Pipeline summary
Data: 1.045M training rows, 2412 test rows
Features: [count] base + [count] engineered
Engineered features: [list]
Model: LightGBM, 5-fold stratified CV
Key hyperparameters: learning_rate=X, num_leaves=Y, min_data_in_leaf=Z, N_ROUNDS=W
Decision rule: argmax / threshold-tuned (multipliers t1=X, t2=Y)

## What's been tried (newest first, one line each)
attempt-N:   [verdict] [delta vs previous] — [one-line description]
attempt-N-1: ...

## Lessons learned
- [bullets capturing durable insights]

## Open ideas (untried)
- [bullets]
```

---

## Style and behavior preferences

- The user is a UConn statistics student, comfortable with stats but new to
  Kaggle competition strategy. Explain ML jargon when first introducing it;
  don't repeat explanations across sessions.
- Default tone: direct, plain prose, minimal bullets. Use bullets only for
  parallel structure (like attempt change-lists).
- Don't be sycophantic. If an attempt is bad, say so and explain why.
- When suggesting next attempts, lead with the highest expected gain, not
  the safest or easiest.
- The user prefers seeing the actual confusion matrix and per-class recall,
  not just headline accuracy.
- Keep responses short by default. The user will ask for depth if they want it.
- The user runs the pipeline on their own laptop (~60 min). Don't try to run
  `train_model.py` from this environment — just analyze the output the user pastes.

## Things to never do

- Never modify `src/train_model.py` unless explicitly asked
- Never delete or rewrite past entries in `docs/ATTEMPTS.md`
- Never overwrite `docs/PROJECT_STATE.md` unless the new attempt is a PASS
- Never auto-commit changes — the user reviews before committing
- Never invent results — if the user pastes incomplete output, ask for what's missing

## Current attempt context

Latest attempt as of this writing: improve/attempt-4
Latest CV: 0.72516
Class weights {0:1, 1:2, 2:1.5} are active
Open hypothesis: post-hoc threshold tuning will recover ~0.5-2.0pt by correcting
the argmax decision rule for accuracy maximization.
