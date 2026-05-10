# Attempt History (newest first)

---

## attempt-4: class weights + zip_cancel1_rate OOF, dropped threshold tuning — FAIL

Branch: improve/attempt-4
Date: 2026-05-09
Runtime: 3542s
CV accuracy: 0.72516 (raw, no tuning)
Std across folds: 0.00033
Naive baseline: 0.70900
Hypothesis: Class-2 recall was stuck at 22-26% because the model underestimates
P(cancel=2). Upweighting class-2 (and class-1) rows during training via sample_weight
would shift probabilities upward and break the recall ceiling. A new zip_cancel1_rate
OOF feature would add signal for the difficult class-1 group.
Changes vs prior attempt:
  - Class weights {0:1.0, 1:2.0, 2:1.5} via sample_weight in lgb.Dataset
  - New OOF feature: zip_cancel1_rate (smoothed class-1 rate per zip per fold)
  - Dropped threshold tuning (3 prior attempts showed it converged to conservative values)

Result:
  Class weights successfully broke the class-2 recall ceiling (26% → 35.8%) and
  dramatically boosted class-1 recall (near-0% → 36.9%). However, class-0 recall
  dropped from ~94% to 87.5%. With class-0 at 71% of rows, the 6.5pp drop in
  class-0 recall costs ~4.6pp of overall accuracy while only ~4.4pp is recovered
  from classes 1 and 2 combined. Net: -0.0013 vs attempt-3.
  4 of 5 folds hit the N_ROUNDS=5000 cap without converging — model wanted more rounds.

Confusion matrix:
true \ pred    0          1          2
0              648136     22821      70032
1              42407      27834      5126
2              134331     12522      81914
Per-class recall: class-0: 87.5%,  class-1: 36.9%,  class-2: 35.8%
Verdict: FAIL — 0.72516 vs prior best 0.7265 (attempt-3), delta = -0.00134
Next attempt should try: Keep class weights for calibrated probabilities, but add
post-hoc multiplier grid search on OOF predictions to correct the argmax decision
rule for accuracy. Also raise N_ROUNDS to 10000 to allow full convergence.

---

## attempt-3: 4 new interaction features + class-2 threshold tuning — MIXED

Branch: improve/attempt-3
Date: [unknown]
Runtime: [unknown]
CV accuracy: 0.72650 (raw; tuning selected t2=0.49, net +0.0007 after tuning)
Std across folds: [unknown]
Naive baseline: [unknown]
Hypothesis: Class-2 recall was stuck near 26%. New features targeting financial
pressure (premium_credit_stress) and mobility (res_tenure_ratio) should add signal
for class-2 cancel. Threshold tuning on class-2 probability would recover
misclassified rows without retraining.
Changes vs prior attempt:
  - 4 new features: tenure_x_credit, res_tenure_ratio, premium_credit_stress,
    premium_vs_dwelling (42 features total, up from 38)
  - Class-2 threshold tuning via OOF grid search (t2 in 0.10–0.60)

Result:
  Gained +0.0005 over attempt-2 (0.7265 vs 0.7260). Class-2 recall remained near
  26%. Threshold tuning selected t2=0.49 — more conservative on class-2, not less
  — and only recovered +0.0007. This reveals the model probability estimates are
  well-calibrated; lowering the scalar threshold alone isn't the right lever.
  LightGBM was already finding the interaction splits internally; numeric interaction
  features (tenure×credit, premium/credit) added marginal gain.

Confusion matrix: [unknown — not recorded]
Per-class recall: class-0: ~94%,  class-1: ~1–2%,  class-2: ~26%
Verdict: MIXED — +0.0005 vs prior best, within ±0.001
Next attempt should try: Address class-2 recall at the training level via class
weights rather than post-hoc threshold adjustment.

---

## attempt-2: slower/deeper training + OOF zip encoding + credit ordinal — MIXED

Branch: improve/slower-deeper-oof-encoding
Date: [unknown]
Runtime: [unknown]
CV accuracy: 0.72600
Std across folds: [unknown]
Naive baseline: [unknown]
Hypothesis: The baseline used num_leaves=31, lr=0.1, ~300 rounds and only 300k rows.
Switching to num_leaves=127, lr=0.02 with early stopping on 1M rows plus OOF target
encoding on zip.code should extract more signal from the dominant feature.
Changes vs prior attempt:
  - Full 1M rows (was 300k subsample)
  - num_leaves=31 → 127, learning_rate=0.1 → 0.02
  - OOF zip_cancel2_rate encoding (smoothed, k=20, no leakage)
  - credit encoded as ordinal numeric (low=0, medium=1, high=2)
  - 38 features total

Result:
  +0.0003 over baseline. Early stopping triggered at ~1780–2102 rounds (much later
  than the 300-round baseline), confirming the deeper model was finding more signal.
  zip.code emerged as the dominant feature by gain. Class-2 recall remained near
  26%, establishing a pattern that would persist through attempts 3 and 4.

Confusion matrix: [unknown — not recorded]
Per-class recall: class-0: ~94%,  class-1: ~1–2%,  class-2: ~26%
Verdict: MIXED — +0.0003 vs prior best, within ±0.001
Next attempt should try: Feature engineering targeting the strong cancel signals
identified in zip and sales.channel diagnostics.

---

## attempt-1: LightGBM baseline on 300k subsample — PASS

Branch: main
Date: [unknown]
Runtime: [unknown]
CV accuracy: 0.72570
Std across folds: [unknown]
Naive baseline: ~0.70990
Hypothesis: Establish a working LightGBM pipeline with default parameters as a
reproducible starting point above the naive all-zero baseline.
Changes vs prior attempt:
  - First attempt; LightGBM replacing naive baseline
  - num_leaves=31, learning_rate=0.1, ~300 rounds
  - 300k row subsample for speed

Result:
  Cleared the naive baseline (0.7257 vs ~0.7099). Class-1 and class-2 recall were
  near zero — model predicted almost everything as class-0. Established the pipeline
  and showed LightGBM can find signal in this dataset.

Confusion matrix: [unknown — not recorded]
Per-class recall: class-0: ~99%,  class-1: ~0%,  class-2: ~0%
Verdict: PASS — establishes baseline, beats naive all-zero by ~1.6pp
Next attempt should try: Full dataset, deeper model (more leaves, slower lr, early stopping).
