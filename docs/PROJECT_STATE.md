# Project State (last updated 2026-05-15)

## Current best

Branch: improve/attempt-21-cat-stack
CV accuracy: 0.73856 (tuned)
Public leaderboard: 0.76454
Submission file: output/submission_stack_cat.csv

## Pipeline summary (attempt-21)

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

### Base model 3: CatBoost (attempt-21)
Features: same attempt-8 feature set (native categoricals via Pool API)
Key params: iterations=8000, learning_rate=0.03, depth=7, l2_leaf_reg=5,
  early_stopping_rounds=100, loss_function=MultiClass
Class weights: {0: 1.0, 1: 1.3, 2: 1.1}
Saved: output/cat_oof_probs.npy, output/cat_test_probs.npy

### Meta-model: LogisticRegression (attempt-21)
Meta features: xgb_prob_0-2, lgb_prob_0-2, cat_prob_0-2 (9 total)
Training: 5-fold StratifiedKFold on OOF probs (StandardScaler per fold)
Best config: not captured (all C values near-equivalent as in attempt-20)
Decision rule: two-stage multiplier search (coarse step 0.05, fine step 0.01 ±0.10)

## What's been tried (newest first)

attempt-21: PASS  +0.00068 OOF / +0.00416 public — XGB+LGB+CAT 9-feature stack; public 0.76454
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
- Adding CatBoost as a third base model added +0.00416 public (attempt-21). Its different
  categorical encoding produces partially orthogonal errors vs XGB and LGB.
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
  score 0.73787–0.73788) — the feature space is well-separated.
- balanced class_weight consistently hurts accuracy across all meta-model configs.

## Open ideas (untried)

- Richer meta features: confidence (max prob), top-2 margin, cross-model agreement,
  class-level mean/std/range across models. A 34-feature meta-space may help the stacker
  know when a row is uncertain or when one model should be trusted more.
- Finer multiplier search: current step 0.05 coarse / 0.01 fine. Step 0.02 coarse /
  0.005 fine focused on [1.20, 2.20] × [0.70, 1.20] (the known good region).
- Non-linear meta-model: XGBoost or LightGBM on meta features.
- Class-1 feature hunt: current recall ~53%. Behavioral/engagement columns might add
  signal no ensemble can recover.
- Ordinal regression: two binary classifiers P(cancel≥1) and P(cancel≥2).
