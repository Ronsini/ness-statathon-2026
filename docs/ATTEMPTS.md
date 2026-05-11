# Attempt History (newest first)

---

## attempt-9: Optuna hyperparameter search — FAIL (tuning run, not submitted)

Branch: improve/attempt-9
Date: 2026-05-11
Runtime: 13307s (~3.7 hours)
Type: Optuna tuning on 350k-row subsample (3-fold CV, 40 trials, N_ROUNDS=5000)
Best trial: #11
Best tuned OOF accuracy (subsample): 0.72621
Best raw OOF accuracy (subsample): 0.72119
Best multipliers (subsample): c1×0.300, c2×0.840
Naive baseline: 0.70900
Hypothesis: 40 Optuna trials using the full attempt-8 feature pipeline (56 features,
  nested OOF encoding, CLASS_WEIGHTS={0:1.0, 1:1.3, 2:1.1}) would find hyperparameters
  that outperform the hand-tuned attempt-8 params on the full 1M dataset.
Search space: learning_rate [0.01-0.05], num_leaves [63-511],
  min_data_in_leaf [20-300], feature_fraction [0.65-0.95],
  bagging_fraction [0.65-0.95], bagging_freq [1-10], lambda_l1 [1e-4-2.0],
  lambda_l2 [1e-3-20.0], max_depth {-1,6,8,10,12}, min_gain_to_split [0.0-1.0]

Best params found:
  learning_rate=0.025, num_leaves=336, min_data_in_leaf=287,
  feature_fraction=0.666, bagging_fraction=0.843, bagging_freq=6,
  lambda_l1=0.004, lambda_l2=0.308, max_depth=6, min_gain_to_split=0.004

Result:
  Tuning run considered failed — params not carried forward into train_model.py.
  The best trial's c1×0.300 multiplier hit the fine-search floor (the coarse grid
  starts at 0.40, fine grid bottom is 0.30), meaning the optimal c1 multiplier is
  at or below the search boundary. This signals that the found hyperparameters
  produce over-confident class-1 predictions that require extreme suppression to
  maximize accuracy — a sign the params are not well-calibrated for this objective.
  The subsample (350k) and reduced folds (3) may also have introduced enough
  noise that the identified optimum doesn't transfer to the full 1M/5-fold setting.
  All top-5 trials show the same c1×0.300 boundary pattern, confirming this is
  systematic rather than a single-trial fluke.

Confusion matrix: N/A (tuning run, no full-dataset predictions)
Per-class recall: N/A
Verdict: FAIL — tuning run results not used; attempt-8 params remain in effect
Next attempt should try: Run a second Optuna pass with a different objective
  (minimize class-1 calibration error, or use a coarse search with a lower c1
  multiplier floor of 0.05) or try Optuna with no class weights so the multiplier
  search is not fighting against extreme probability miscalibration.

---

## attempt-8: lighter weights, nested OOF, 4 new OOF features — MIXED

Branch: improve/attempt-8
Date: 2026-05-11
Runtime: 5326s
CV accuracy: 0.72783 (raw) → 0.72963 (tuned)
Std across folds: 0.00047
Naive baseline: 0.70900
Hypothesis: Heavy class weights from attempts 4–7 sacrificed class-0 recall, which
  dominates accuracy (71% of rows). Lighter weights {0:1.0, 1:1.3, 2:1.1} plus
  post-hoc multiplier tuning should recover class-0 recall while the tuner corrects
  the decision rule. New OOF features (zip×sales_channel, coverage×dwelling) and
  nested OOF encoding for training rows (no in-fold leakage) should add marginal
  signal and improve calibration.
Changes vs attempt-7:
  - Class weights lightened: {0:1.0, 1:1.3, 2:1.1} (was {0:1.0, 1:2.0, 2:1.5})
  - Fixed fold-specific test OOF encoding bug: each fold builds X_test_fold with
    that fold's maps (was running average of prior folds' maps)
  - 4 new OOF features: zip_sales_cancel{1,2}_rate, cov_dwell_cancel{1,2}_rate
    (total OOF features: 8; zip + age_credit retained from attempt-7)
  - Two-stage multiplier search: coarse (step 0.05) then fine (step 0.01, +-0.10)
  - Nested OOF encoding for training rows: each outer training fold is split into
    inner folds so training rows are encoded using other inner-fold rows only
  - Explicit reset_index on X, X_test, y, and all key Series before fold loop

Result:
  Tuned accuracy 0.72963 — new high watermark but MIXED (+0.00036 over attempt-5).
  The lighter weights shifted the optimal multipliers to c1×1.26, c2×0.90 — the
  opposite direction from attempt-5 (c1×0.95, c2×0.65). With lighter upweighting,
  the model under-predicts class-1 at argmax; the tuner compensates by scaling up.
  Class-2 recall fell to 24.8% (vs 35.8% in attempt-4) — confirming that class-2
  recall tracks directly with class weight strength. Class-0 recall recovered to
  91.6% (vs 87.5% in attempt-4), and class-1 recall held at 35.9%.
  All 5 folds converged below 5000 rounds (best iters: 3477, 4079, 3439, 4428, 4865).

Confusion matrix:
true \ pred    0        1        2
0              678658   23037    39294
1              47709    27068    590
2              159096   12846    56825
Per-class recall: class-0: 91.6%,  class-1: 35.9%,  class-2: 24.8%
Verdict: MIXED — 0.72963 tuned (+0.00036 vs prior best 0.72927, attempt-5).
  New high watermark among all attempts; below +0.001 threshold for PASS.
Next attempt should try: Optuna hyperparameter tuning — num_leaves, learning_rate,
  regularization, and min_data_in_leaf have not been tuned since the baseline.
  The pipeline has 56 features and nested OOF encoding; the hyperparameters should
  be searched against this full pipeline, not a simplified version.

---

## attempt-7: OOF age×credit encoding + 6 missingness flags — MIXED

Branch: improve/attempt-7
Date: 2026-05-11
Runtime: [unknown]
CV accuracy: [unknown raw] → 0.72945 (tuned)
Std across folds: [unknown]
Naive baseline: 0.70900
Hypothesis: The ni.age×credit interaction shows a 42pp c2-rate spread (10% for
  elderly/high-credit to 52% for young/low-credit), which is the strongest untapped
  signal found in feature exploration. OOF target encoding of this pair would give
  the model the aggregate signal as a numeric feature. Missingness flags for 6
  columns (~1,000 missing each) have 2–5pp class-rate shifts.
Changes vs attempt-6:
  - Dropped XGBoost ensemble (hurt accuracy in attempt-6)
  - OOF age_credit_cancel{1,2}_rate: smoothed cancel rates per (age_bucket × credit)
    cell, same implementation as zip cancel rates (k=20 smoothing)
  - 6 missingness flags: credit_missing, ni_age_missing, n_adults_missing,
    coverage_type_missing, ni_marital_status_missing, n_children_missing
  - 52 total features (44 baseline + 8 new)

Result:
  +0.00018 over attempt-5 (0.72945 vs 0.72927). The age×credit OOF encoding added
  marginal signal; the missingness flags likely contributed small gains. Result
  confirms that OOF encoding is most valuable for high-cardinality groups (zip has
  400+ values; age×credit has only 15 cells and LightGBM already finds those splits).

Confusion matrix: [not recorded]
Per-class recall: [not recorded]
Verdict: MIXED — 0.72945 tuned (+0.00018 vs prior best 0.72927, attempt-5)
Next attempt should try: More OOF features for higher-cardinality interactions,
  lighter class weights so the model doesn't need such extreme multiplier correction,
  and a proper nested OOF encoding to eliminate training-fold leakage.

---

## attempt-6: XGBoost ensemble blend — FAIL

Branch: improve/attempt-6
Date: 2026-05-11
Runtime: [unknown]
CV accuracy: [unknown raw] → 0.72711 (tuned)
Std across folds: [unknown]
Naive baseline: 0.70900
Hypothesis: XGBoost uses a different inductive bias and approximation method.
  Averaging LightGBM and XGBoost OOF probabilities before argmax should reduce
  variance and move toward an ensemble ceiling above either model alone.
Changes vs attempt-5:
  - Added XGBoost (xgb.XGBClassifier) trained per fold
  - Final predictions: average of LightGBM and XGBoost per-fold probabilities

Result:
  -0.00216 vs attempt-5. XGBoost handles high-cardinality categoricals (especially
  zip.code with 400+ values) worse than LightGBM's native categorical splits,
  resulting in ~0.016 higher logloss. Blending with an inferior model pulled
  the ensemble down rather than up.

Confusion matrix: [not recorded]
Per-class recall: [not recorded]
Verdict: FAIL — 0.72711 tuned (-0.00216 vs prior best 0.72927, attempt-5)
Next attempt should try: Drop XGBoost. Use LightGBM only with new OOF features
  targeting the high-signal interactions found in feature exploration.

---

## attempt-5: post-hoc threshold tuning + raised N_ROUNDS — PASS

Branch: improve/attempt-5
Date: 2026-05-09
Runtime: [unknown]
CV accuracy: [unknown raw] → 0.72927 (tuned)
Std across folds: [unknown]
Naive baseline: 0.70900
Hypothesis: Attempt-4 showed that class weights produce well-calibrated
  probabilities, but the argmax decision rule is wrong for accuracy because
  class-0 (71% of rows) was sacrificed. Post-hoc multiplier grid search on
  OOF predictions should correct the decision rule without retraining. Raising
  N_ROUNDS from 5000 to 10000 allows full convergence (4/5 folds hit the cap
  in attempt-4).
Changes vs attempt-4:
  - Post-hoc 2D multiplier grid search on OOF predictions (c1 and c2 in [0.40, 2.00],
    step 0.05); apply tuned multipliers to test predictions before argmax
  - N_ROUNDS raised from 5000 to 10000
  - Class weights {0:1.0, 1:2.0, 2:1.5} unchanged

Result:
  +0.00277 vs attempt-4. Tuning found c1×0.95, c2×0.65 — the model was
  over-calling classes 1 and 2 relative to class-0. Scaling them down recovered
  substantial class-0 accuracy. This confirms the class weights produce good
  probability calibration; the issue was purely the argmax decision rule.

Confusion matrix: [not recorded]
Per-class recall: [not recorded]
Verdict: PASS — 0.72927 tuned (+0.00277 vs prior attempt-4, new best overall)
Next attempt should try: New OOF features from feature exploration (age×credit
  interaction, missingness indicators); XGBoost ensemble blend.

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
