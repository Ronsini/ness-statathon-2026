# Project State (last updated 2026-05-14)

## Current best

Branch: improve/attempt-20-stacking-logreg
CV accuracy: 0.73788 (tuned; C=0.03, class_weight=None, c0×1.00, c1×1.510, c2×0.980)
Public leaderboard: 0.76038
Submission file: output/submission_stack.csv

## Pipeline summary (attempt-20)

Data: 1,045,123 training rows, 2,412 test rows

### Base model 1: XGBoost (attempt-14)
Features: 91 total (see attempt-14 pipeline summary)
Key params: learning_rate=0.03, max_depth=7, min_child_weight=10,
  subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, reg_alpha=0.2,
  tree_method=hist, max_bin=256, n_estimators=10000, early_stopping_rounds=100
Class weights: {0: 1.0, 1: 1.3, 2: 1.1}
OOF tuned accuracy: 0.73661  Multipliers: c0×1.00, c1×1.260, c2×0.890
Saved: output/xgb_oof_probs.npy, output/xgb_test_probs.npy

### Base model 2: LightGBM (attempt-8)
Features: 56 total (native categoricals, 6 missingness flags, 8 OOF encodings)
Key params: learning_rate=0.02, num_leaves=127, min_data_in_leaf=30,
  feature_fraction=0.8, bagging_fraction=0.8, lambda_l1=0.1, lambda_l2=1.0
Class weights: {0: 1.0, 1: 1.3, 2: 1.1}
OOF tuned accuracy: 0.72963  Multipliers: c0×1.00, c1×1.260, c2×0.900
Saved: output/lgb_oof_probs.npy, output/lgb_test_probs.npy

### Meta-model: LogisticRegression (attempt-20)
Meta features: xgb_prob_0, xgb_prob_1, xgb_prob_2, lgb_prob_0, lgb_prob_1, lgb_prob_2
Training: 5-fold StratifiedKFold on OOF probs (StandardScaler per fold)
Best config: C=0.03, solver=lbfgs, class_weight=None
Decision rule: two-stage multiplier search (coarse step 0.05, fine step 0.01 ±0.10)
  attempt-20 best multipliers: c0×1.00, c1×1.510, c2×0.980

## What's been tried (newest first)

attempt-20: PASS  +0.00127 — LogisticRegression stacking on XGB+LGB probs; public 0.76038
attempt-19: MIXED +0.00021 — XGB/LGB soft blend (best xgb=0.82); not submitted
attempt-18: INFRA         — save XGB/LGB probability arrays for blending
attempt-17: FAIL  -0.00871 — Optuna best params (depth=5, gamma=2.7) collapsed class-1 recall
attempt-16: INFRA         — Optuna XGBoost tuning on 350k subsample (best trial: 0.72830)
attempt-15: MIXED -0.00017 — lr 0.03→0.02, n_estimators 10000→15000
attempt-14: PASS  +0.00103 — XGBoost n_estimators 10000, natural convergence; public 0.75623
attempt-13: PASS  +0.00631 — pure XGBoost clean preprocessing, 91 features; public 0.75207
attempt-8:  MIXED +0.00036 — LGB lighter weights, nested OOF, 4 new OOF; public 0.74099
attempt-7:  MIXED +0.00018 — OOF age×credit encoding + 6 missingness flags (52 features)
attempt-6:  FAIL  -0.00216 — XGBoost blend (not pure XGB; no proper preprocessing)
attempt-5:  PASS  +0.00277 — post-hoc 2D multiplier tuning + N_ROUNDS raised to 10000; public 0.73961
attempt-4:  FAIL  -0.00134 — class weights {0:1, 1:2, 2:1.5} + zip_cancel1_rate OOF
attempt-3:  MIXED +0.00050 — 4 new interaction features + class-2 threshold tuning
attempt-2:  MIXED +0.00030 — slower/deeper training + OOF zip encoding + credit ordinal
attempt-1:  PASS  baseline — LightGBM on 300k subsample (0.72570)

## Lessons learned

- Stacking (logistic regression on base model OOF probs) outperforms both single models
  and fixed blending. Attempt-20 gained +0.00415 public over attempt-14 XGB alone.
  The meta-model learns adaptive combination vs fixed blend weight.
- XGBoost with clean numeric preprocessing outperforms LightGBM on this dataset.
  The key driver was class-1 recall: 35.9% (LGB) → 52.7% (XGB attempt-13) → 54.3% (attempt-14).
- Attempt-6's conclusion that "XGBoost is inferior" was based on a blend without proper
  preprocessing. Pure XGB with freq+OHE encoding is the stronger base model.
- n_estimators=10000 with early_stopping_rounds=100 is the right XGB ceiling.
  Attempt-14 folds converged naturally at 6643–7737 rounds.
- Optuna on 350k subsample failed to generalize: params with c1×0.300 floor on subsample
  collapsed class-1 recall to 0% on full data (attempt-17). The c1 multiplier behavior
  on the subsample is a reliable signal — avoid params that floor c1.
- Post-hoc 2D multiplier search is essential. Raw OOF → tuned gains ~0.002–0.006.
- Optimal XGB multipliers are stable: c1×1.260, c2×0.890 across attempts-13 and 14.
  Stack multipliers shifted to c1×1.510, c2×0.980 — the meta-model underestimates class-1.
- Class-0 dominance (71% of rows) is the binding constraint. Lighter class weights
  with multiplier correction outperform aggressive upweighting.
- LogisticRegression C value is irrelevant on these meta-features (all configs 0.01–10.0
  score 0.73787–0.73788) — the 6-feature space is well-separated.
- balanced class_weight consistently hurts accuracy across all meta-model configs.

## Open ideas (untried)

- Add CatBoost as a third base model: CatBoost handles categoricals natively and
  differently from both XGB and LGB. If its errors are orthogonal, a 9-feature meta-model
  (XGB+LGB+CAT probs) may push the stack higher.
- Non-linear meta-model: XGBoost or LightGBM on the 6 meta-features. LogReg is saturating
  (C insensitive) so a tree-based meta-model might capture non-linear combinations.
- More base models: train XGB with different feature subsets or hyperparameters to
  generate diverse OOF probs for stacking.
- Class-1 feature hunt: find features that specifically separate class-1 from class-0/2.
  Current class-1 recall at 53.2% — if there are behavioral/engagement columns, they
  could add signal that no amount of ensemble will recover.
- Ordinal regression: two binary classifiers P(cancel≥1) and P(cancel≥2) exploiting
  the ordered class structure.
