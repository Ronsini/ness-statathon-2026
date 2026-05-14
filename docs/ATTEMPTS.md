# Attempt History (newest first)

---

## attempt-20: LogisticRegression stacking on XGB+LGB OOF probs — PASS

Branch: improve/attempt-20-stacking-logreg
Date: 2026-05-14
Runtime: <5 min (no retraining)
CV accuracy: 0.73680 (raw) → 0.73788 (tuned)
Public leaderboard: 0.76038
Std across folds: n/a (meta-model, no per-fold tracking)
Naive baseline: 0.70900
Hypothesis: A LogisticRegression meta-model trained on 6 meta-features
  (xgb_prob_0-2, lgb_prob_0-2) can learn a smarter combination than a fixed
  blend weight — trusting XGB when it's confident, LGB when XGB is uncertain,
  and adjusting class-1 calibration separately.
Changes vs attempt-19:
  - Replaced fixed blend weight with LogisticRegression meta-model (C=0.03, lbfgs)
  - 5-fold CV on meta features to produce OOF meta-probs
  - Searched C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0] × class_weight [None, balanced]
  - Best: C=0.03, class_weight=None

Result:
  Tuned OOF 0.73788 — PASS (+0.00127 vs attempt-14's 0.73661, +0.00106 vs attempt-19
  blend 0.73682). Public 0.76038 is a new best (+0.00415 over attempt-14's 0.75623).
  The C value was essentially irrelevant — all None configs clustered at 0.73787–0.73788,
  confirming logistic regression saturates early on these 6 well-separated meta-features.
  balanced class_weight hurt (0.733 vs 0.738), consistent with the pattern across all
  prior attempts that lighter/no weighting is better for accuracy.
  Class-0 recall improved vs attempt-14 (+0.6pp: 90.9% → 91.5%), while class-1 slipped
  slightly (-1.1pp: 54.3% → 53.2%) and class-2 fell slightly (-0.9pp: 24.2% → 23.3%).
  Net accuracy is higher because class-0 dominates. The c1 multiplier rose to 1.510
  (vs 1.260 for XGB alone) — the stacker underestimates class-1, needing more boosting.

Confusion matrix:
true \ pred    0         1         2
0              677794    28250     34945
1              35127     40089     151
2              160194    15280     53293
Per-class recall: class-0: 91.5%,  class-1: 53.2%,  class-2: 23.3%
Verdict: PASS — 0.73788 tuned (+0.00127 vs attempt-14). New best public: 0.76038 (+0.00415).
Next attempt should try: Add CatBoost OOF probs as a third base model to the stacking
  meta-features. CatBoost handles categoricals differently from both XGB and LGB; if its
  errors are orthogonal it may push the stack higher. Alternatively, try a gradient-boosted
  meta-model (XGBoost or LGB on the 6 meta-features) which can learn non-linear combinations.

---

## attempt-19: XGB + LGB soft blend weight search — MIXED

Branch: improve/attempt-19-xgb-lgb-blend
Date: 2026-05-14
Runtime: <1 min (no retraining)
CV accuracy: n/a (raw) → 0.73682 (tuned, best weight xgb=0.82)
Public leaderboard: not submitted
Std across folds: n/a
Naive baseline: 0.70900
Hypothesis: A weighted average of XGB and LGB probabilities with tuned multipliers
  may outperform either model alone if their errors are partially orthogonal.
Changes vs attempt-18:
  - Searched xgb_weight from 0.70 to 1.00 (step 0.01)
  - lgb_weight = 1 - xgb_weight
  - Tuned class multipliers on blended OOF probs

Result:
  Best blend (xgb=0.82, lgb=0.18) tuned OOF 0.73682 — MIXED (+0.00021 vs attempt-14's
  0.73661). Marginal gain over XGB-only, confirming LGB adds small orthogonal signal.
  Not submitted — gap too small to justify a submission slot ahead of stacking (attempt-20).

Verdict: MIXED — +0.00021 over attempt-14 OOF. Not submitted.
Next attempt should try: Stacking meta-model (attempt-20).

---

## attempt-18: save XGB + LGB probability arrays for blending — INFRASTRUCTURE

Branch: improve/attempt-18-save-probs
Date: 2026-05-14
Runtime: 3638s (XGB) + 3332s (LGB)
CV accuracy: 0.73661 (XGB, identical to attempt-14) / 0.72963 (LGB, identical to attempt-8)
Public leaderboard: not submitted
Hypothesis: Blending/stacking requires raw class probabilities, not just submission labels.
  Re-run attempt-14 (XGB) and attempt-8 (LGB) pipelines unchanged to generate and save
  probability arrays for use in attempt-19 soft blend and attempt-20 stacking.
Changes vs attempt-14:
  - Added np.save calls to train_model.py to write xgb_oof_probs.npy, xgb_test_probs.npy,
    y_train.npy, test_ids.npy (no model or preprocessing changes)
  - Created save_lgb_probs.py replicating the exact attempt-8 LGB pipeline, saving
    lgb_oof_probs.npy and lgb_test_probs.npy

Result:
  Both runs reproduced their originals exactly. XGB: 0.73661 tuned, c1×1.260, c2×0.890,
  fold convergence at 7336, 7261, 6788, 7737, 6643. LGB: 0.72963 tuned, c1×1.260,
  c2×0.900, fold convergence at 3477, 4079, 3439, 4428, 4865. Probability arrays saved
  to output/ and committed to the branch. Ready for attempt-19 blending.

Outputs saved:
  output/xgb_oof_probs.npy   (1,045,123 × 3)
  output/xgb_test_probs.npy  (2,412 × 3)
  output/lgb_oof_probs.npy   (1,045,123 × 3)
  output/lgb_test_probs.npy  (2,412 × 3)
  output/y_train.npy         (1,045,123,)
  output/test_ids.npy        (2,412,)
Verdict: Infrastructure run. No submission. Attempt-14 remains the best public score (0.75623).
Next attempt should try: Soft blend attempt-19 — weighted average of XGB and LGB probabilities
  (XGB 85–95%, LGB 5–15%) with multiplier tuning on OOF blend.

---

## attempt-14: XGBoost n_estimators 5000 → 10000 — PASS

Branch: improve/attempt-14-xgb-10000-rounds
Date: 2026-05-13
Runtime: 3486s
CV accuracy: 0.73426 (raw) → 0.73661 (tuned)
Public leaderboard: 0.75623
Std across folds: 0.00035
Naive baseline: 0.70900
Hypothesis: All 5 attempt-13 folds hit the 5000 estimator cap with logloss still
  declining at round 4999. The model was undertrained. Raising n_estimators to 10000
  with early_stopping_rounds=100 should let each fold converge naturally and recover
  the remaining accuracy.
Changes vs attempt-13:
  - n_estimators: 5000 → 10000 (only change)

Result:
  Tuned accuracy 0.73661 — PASS (+0.00103 vs attempt-13's 0.73558, +0.00734 vs
  all-time prior best 0.72927). Public 0.75623 is a new best (+0.00416 over attempt-13).
  All folds converged naturally via early stopping (best iters: 7336, 7261, 6788, 7737,
  6643) — none hit the 10000 cap. This confirms n_estimators=10000 is the right ceiling.
  Class-1 recall improved further: 52.7% → 54.3% (+1.6pp). Class-0 and class-2 held
  flat at 90.9% and 24.2%. Optimal multipliers identical to attempt-13 (c1×1.260, c2×0.890),
  confirming the model calibration is stable across round counts.

Confusion matrix:
true \ pred    0         1         2
0              673544    29368     38077
1              34035     40922     410
2              157347    16043     55377
Per-class recall: class-0: 90.9%,  class-1: 54.3%,  class-2: 24.2%
Verdict: PASS — 0.73661 tuned (+0.00103 vs attempt-13; +0.00734 vs prior all-time best).
  New best CV and new best public (0.75623).
Next attempt should try: Optuna hyperparameter search on XGBoost (learning_rate,
  max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, reg_alpha)
  using the current 91-feature pipeline. Parameters haven't been tuned since attempt-9
  which used LightGBM. Alternatively, an LGB+XGB blend using attempt-8 and attempt-14
  probabilities could add orthogonal signal.

---

## attempt-13: pure XGBoost with clean numeric preprocessing — PASS

Branch: improve/attempt-13-xgb-clean-preprocess
Date: 2026-05-13
Runtime: 2277s
CV accuracy: 0.73312 (raw) → 0.73558 (tuned)
Public leaderboard: 0.75207
Std across folds: 0.00034
Naive baseline: 0.70900
Hypothesis: Pure XGBoost with proper numeric preprocessing has never been tried.
  Attempt-6 was a blend (not pure XGBoost) and had no preprocessing for high-cardinality
  categoricals. With count/frequency encoding for zip.code, house.color, email_domain and
  one-hot encoding for low-cardinality categoricals, XGBoost gets clean numeric input and
  may handle the feature space differently than LightGBM — potentially surfacing signal
  that LightGBM's native categorical splits miss, especially for class-1 (7% of data).
Changes vs attempt-8:
  - Model: LightGBM → pure XGBoost (XGBClassifier, multi:softprob)
  - XGB_PARAMS: learning_rate=0.03, max_depth=7, min_child_weight=10,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, reg_alpha=0.2,
    tree_method=hist, max_bin=256, n_estimators=5000, early_stopping_rounds=100
  - Count/frequency encoding for high-cardinality categoricals:
    zip.code, house.color, email_domain → {col}_count, {col}_freq
  - One-hot encoding (pd.get_dummies, dummy_na=True) for low-cardinality categoricals:
    credit, coverage.type, dwelling.type, ni.gender, original_quote_weekday,
    season_of_renewal, sales.channel
  - Remaining string/category columns dropped; X/X_test cast to float
  - Added safe row-level features from attempt-12: is_first_year_with_claim,
    len_at_res_missing, sales_channel_missing, tenure_missing,
    premium_missing, square_footage_missing
  - All 8 OOF target-encoded features and nested OOF structure retained from attempt-8
  - 91 total features (was 56 in attempt-8)

Result:
  Tuned accuracy 0.73558 — PASS (+0.00631 vs prior best 0.72927, attempt-5).
  Public score 0.75207 is the new best by a large margin (was 0.74099, attempt-8).
  XGBoost dramatically improved class-1 recall: 35.9% → 52.7% (+16.8pp). This is the
  key driver — the one-hot encoding gives XGBoost clean categorical boundaries for the
  features that distinguish class-1, which LightGBM's native splits were underusing.
  Class-0 recall dropped slightly (91.6% → 90.9%, -0.7pp) and class-2 held near flat
  (24.8% → 24.2%, -0.6pp). The net gain is large because class-1 improvements are
  nearly pure additions at modest class-0 cost.
  All 5 folds hit the n_estimators=5000 cap (best iters: 4999, 4999, 4999, 4995, 4994)
  — the model was still improving at round 5000. Raising n_estimators to 10000 for
  attempt-14 should yield further gains.

Confusion matrix:
true \ pred    0         1         2
0              673717    29878     37394
1              35356     39755     256
2              156902    16562     55303
Per-class recall: class-0: 90.9%,  class-1: 52.7%,  class-2: 24.2%
Verdict: PASS — 0.73558 tuned (+0.00631 vs prior best 0.72927, attempt-5).
  New best CV and new best public (0.75207). Previous lesson that XGBoost is inferior
  was wrong — it was based on attempt-6 which was a blend without proper preprocessing.
Next attempt should try: Raise n_estimators to 10000 — all folds hit the 5000 cap,
  logloss still declining at the last round. Pure XGBoost with more rounds is the
  highest-expected-gain next step.

---

## attempt-8: lighter weights, nested OOF, 4 new OOF features — MIXED

Branch: improve/attempt-8
Date: 2026-05-11
Runtime: 5326s
CV accuracy: 0.72783 (raw) → 0.72963 (tuned)
Public leaderboard: 0.74099
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
Public leaderboard: 0.73961
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
