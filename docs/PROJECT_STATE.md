# Project State (last updated 2026-05-30)

## FINAL RESULT — Competition complete

**3rd place finish**
Branch: improve/attempt-24-lgb-meta-stacker
CV accuracy: 0.75623 (tuned OOF)
Public leaderboard (30% test): 0.77839
**Private leaderboard (70% test — final): 0.76941**
Submission file: output/submission_stack_meta_lgb.csv

## Pipeline summary (attempt-24)

Data: 1,045,123 training rows, 2,412 test rows

### Base model 1: XGBoost (attempt-14)
Features: 91 total
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

### Meta-model: LightGBM (attempt-24)
Meta features: 34 total (9 raw probs + 25 derived confidence/agreement features)
Training: 5-fold StratifiedKFold, LGBMClassifier, early_stopping_rounds=100
Grid: num_leaves in [7, 15, 31], max_depth in [3, 4, 5]
Common params: lr=0.02, n_estimators=3000, subsample=0.8, colsample_bytree=0.8,
  reg_lambda=20, reg_alpha=5, min_child_samples=200
Multiplier search: c1 in [1.20, 1.90], c2 in [0.80, 1.15], step 0.01
Saved: output/stack_meta_lgb_oof_probs.npy, output/stack_meta_lgb_test_probs.npy

## What's been tried (newest first)

attempt-24: PASS  +0.01640 OOF / +0.00970 public — LGB meta-stacker on 34 features; public 0.77839
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

- LightGBM meta-stacker on 34 meta features dramatically outperforms LogisticRegression
  (+0.01640 OOF, +0.00970 public in one step). Non-linear interactions between base model
  confidence, agreement, and class probabilities carry substantial signal a linear model cannot.
- Stacking (logistic regression on base model OOF probs) outperforms fixed blending.
  Attempt-20 gained +0.00415 public over attempt-14 XGB alone.
- Adding CatBoost as a third base model added +0.00416 public (attempt-21).
- Extra meta features (34 vs 9) hurt LogReg stacker but massively help LGB stacker.
- Blending attempt-21 (9 raw) with attempt-22 (34 features) at w=0.70-1.00 was a useful
  intermediate step — confirmed extra features carry signal when combined carefully.
- XGBoost with clean numeric preprocessing outperforms LightGBM on this dataset.
- n_estimators=10000 with early_stopping_rounds=100 is the right XGB ceiling.
- Optuna on 350k subsample failed to generalize (attempt-17).
- Post-hoc 2D multiplier search is essential.
- LogisticRegression C value is irrelevant on well-separated meta-features.
- balanced class_weight consistently hurts accuracy.

## Open ideas (untried)

- Grid search more LGB meta configs: num_leaves up to 63, reg_lambda as low as 5.
  Attempt-24 used conservative settings; slightly less regularization may improve further.
- Wider multiplier search now that LGB meta-model calibrates class-1 differently.
  The optimal c1 range may have shifted from ~1.5-1.8 toward ~0.8-1.3.
- Class-1 feature hunt: behavioral/engagement columns that no ensemble can recover.
- Ordinal regression: two binary classifiers P(cancel≥1) and P(cancel≥2).
