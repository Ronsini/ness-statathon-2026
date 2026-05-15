# Project State (last updated 2026-05-15)

## Current best

Branch: improve/attempt-23-blend-stack-versions
CV accuracy: 0.73983 (tuned)
Public leaderboard: 0.76869
Submission file: output/submission_stack_blend_21_22.csv

## Pipeline summary (attempt-23)

Data: 1,045,123 training rows, 2,412 test rows

### Base model 1: XGBoost (attempt-14)
Features: 91 total (see attempt-14 pipeline summary)
Key params: learning_rate=0.03, max_depth=7, min_child_weight=10,
  subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, reg_alpha=0.2,
  tree_method=hist, max_bin=256, n_estimators=10000, early_stopping_rounds=100
Class weights: {0: 1.0, 1: 1.3, 2: 1.1}
Saved: output/xgb_oof_probs.npy, output/xgb_test_probs.npy

### Base model 2: LightGBM (attempt-8)
Features: 56 total (native categoricals, 6 missingness flags, 8 OOF encodings)
Key params: learning_rate=0.02, num_leaves=127, min_data_in_leaf=30,
  feature_fraction=0.8, bagging_fraction=0.8, lambda_l1=0.1, lambda_l2=1.0
Class weights: {0: 1.0, 1: 1.3, 2: 1.1}
Saved: output/lgb_oof_probs.npy, output/lgb_test_probs.npy

### Base model 3: CatBoost (attempt-21)
Features: same attempt-8 feature set (native categoricals via Pool API)
Key params: iterations=8000, learning_rate=0.03, depth=7, l2_leaf_reg=5,
  early_stopping_rounds=100, loss_function=MultiClass
Class weights: {0: 1.0, 1: 1.3, 2: 1.1}
Saved: output/cat_oof_probs.npy, output/cat_test_probs.npy

### Layer 1 meta-models:
- attempt-21 stack (LogReg, 9 raw probs): output/stack_cat_oof_probs.npy, stack_cat_test_probs.npy
- attempt-22 stack (LogReg, 34 meta features): output/stack_cat_meta_oof_probs.npy, stack_cat_meta_test_probs.npy

### Final blend (attempt-23):
final = w * stack_cat + (1-w) * stack_cat_meta
Best w: not captured (search w in [0.70, 1.00])
Multiplier search: c1 in [1.45, 1.85], c2 in [0.85, 1.05], step 0.01

## What's been tried (newest first)

attempt-23: PASS  +0.00127 OOF / +0.00415 public — blend att21+att22 stack probs; public 0.76869
attempt-22: MIXED -0.00139 public — LogReg stack with 34 meta features; public 0.76315
attempt-21: PASS  +0.00068 OOF / +0.00416 public — XGB+LGB+CAT 9-feature stack; public 0.76454
attempt-20: PASS  +0.00127 OOF — LogisticRegression stacking on XGB+LGB probs; public 0.76038
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
- Adding CatBoost as a third base model added +0.00416 public (attempt-21).
- Extra meta features (34 vs 9) hurt LogReg stacker generalization (attempt-22 public
  regressed -0.00139). LogReg is not the right model for high-dimensional meta-feature spaces.
- Blending attempt-21 (9 raw, better generalization) with attempt-22 (34 features, overfit)
  recovered the signal while anchoring generalization (attempt-23: +0.00415 public).
- XGBoost with clean numeric preprocessing outperforms LightGBM on this dataset.
- n_estimators=10000 with early_stopping_rounds=100 is the right XGB ceiling.
- Optuna on 350k subsample failed to generalize (attempt-17). c1 floor on subsample = warning.
- Post-hoc 2D multiplier search is essential. Raw OOF → tuned gains ~0.002–0.006.
- Class-0 dominance (71% of rows) is the binding constraint. Lighter class weights
  with multiplier correction outperform aggressive upweighting.
- LogisticRegression C value is irrelevant on well-separated meta-features.
- balanced class_weight consistently hurts accuracy across all meta-model configs.

## Open ideas (untried)

- LightGBM meta-model on 34 meta features: a tree-based stacker with heavy regularization
  (reg_lambda=20, min_child_samples=200) may capture non-linear combinations that LogReg
  cannot. Small num_leaves (7–31) to prevent overfitting.
- Class-1 feature hunt: current recall ~53%. Behavioral/engagement columns might add
  signal no ensemble can recover.
- Ordinal regression: two binary classifiers P(cancel≥1) and P(cancel≥2).
