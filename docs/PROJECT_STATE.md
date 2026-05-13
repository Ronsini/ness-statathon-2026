# Project State (last updated 2026-05-13)

## Current best

Branch: improve/attempt-14-xgb-10000-rounds
CV accuracy: 0.73661 (tuned; c0×1.00, c1×1.260, c2×0.890)
Public leaderboard: 0.75623
Submission file: output/submission.csv

## Pipeline summary (attempt-14)

Data: 1,045,123 training rows, 2,412 test rows
Features: 91 total

Base numeric features (from train.csv, passed through directly):
  year, ni.age, len.at.res, premium, n.adults, n.children, tenure,
  claim.ind, square_footage, num_windows_front, ni.marital.status

Frequency/count features (computed from training X, no target):
  zip.code_count, zip.code_freq,
  house.color_count, house.color_freq,
  email_domain_count, email_domain_freq

One-hot encoded features (pd.get_dummies on combined train+test, dummy_na=True):
  credit_*, coverage.type_*, dwelling.type_*, ni.gender_*,
  original_quote_weekday_*, season_of_renewal_*, sales.channel_*

Row-level engineered features (add_engineered_features):
  household_size, has_children, premium_per_sqft, claim_x_tenure,
  young_with_claim, new_customer, log_premium, log_tenure, log_square_footage,
  premium_x_claim, age_x_tenure, is_married_adult, windows_per_sqft,
  long_resident, credit_ordinal, tenure_x_credit, res_tenure_ratio,
  premium_credit_stress,
  credit_missing, ni_age_missing, n_adults_missing, coverage_type_missing,
  ni_marital_status_missing, n_children_missing,
  is_first_year_with_claim, len_at_res_missing, sales_channel_missing,
  tenure_missing, premium_missing, square_footage_missing

Group features (add_group_features, computed on training ref):
  premium_vs_zip, premium_vs_credit, premium_vs_dwelling

OOF features (nested fold encoding, no leakage):
  zip_cancel2_rate, zip_cancel1_rate,
  age_credit_cancel2_rate, age_credit_cancel1_rate,
  zip_sales_cancel2_rate, zip_sales_cancel1_rate,
  cov_dwell_cancel2_rate, cov_dwell_cancel1_rate

Model: XGBoost (XGBClassifier, multi:softprob, num_class=3), 5-fold stratified CV
Key hyperparameters: learning_rate=0.03, max_depth=7, min_child_weight=10,
  subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, reg_alpha=0.2,
  tree_method=hist, max_bin=256, n_estimators=10000, early_stopping_rounds=100
Class weights: {0: 1.0, 1: 1.3, 2: 1.1}
Decision rule: two-stage multiplier search (coarse step 0.05, fine step 0.01 ±0.10)
  attempt-14 best multipliers: c0×1.00, c1×1.260, c2×0.890
Convergence: folds converged at 7336, 7261, 6788, 7737, 6643 — none hit the 10000 cap.

## What's been tried (newest first)

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

- XGBoost with clean numeric preprocessing outperforms LightGBM on this dataset.
  Attempt-13 gained +0.00631 CV and +0.01108 public over the best LightGBM attempt.
  The key driver was class-1 recall: 35.9% (LGB) → 52.7% (XGB attempt-13) → 54.3%
  (XGB attempt-14). One-hot encoding gives XGBoost clean categorical boundaries for
  class-1 separation that LightGBM's native splits were missing.
- Attempt-6's conclusion that "XGBoost is inferior" was based on a blend without proper
  preprocessing. Pure XGB with freq+OHE encoding is the stronger model on this dataset.
- n_estimators=10000 with early_stopping_rounds=100 is the right setup. Attempt-13
  showed 5000 was insufficient (all folds hit cap). Attempt-14 folds converged naturally
  at 6643–7737 rounds. The 10000 ceiling is correct.
- Optimal multipliers are stable: c1×1.260, c2×0.890 in both attempts-13 and 14,
  suggesting the calibration is consistent and the search is reliable.
- Post-hoc 2D multiplier search is essential. Raw OOF 0.73426 → tuned 0.73661 (+0.00235).
- Class-0 dominance is the binding constraint (71% of rows). XGBoost found a better
  class-1/class-0 trade-off than LightGBM: +18.4pp class-1 recall at only -0.7pp class-0.
- Profile OOF and zip×year OOF (attempt-12) hurt public score — avoid high-cardinality
  OOF keys beyond the 8 in the current pipeline.
- OOF target encoding (8 features, nested OOF) carries over cleanly to XGBoost.
- CV and public scores can diverge; always submit and verify.

## Open ideas (untried)

- Optuna hyperparameter search on XGBoost using a 300k subsample + 3 folds for speed,
  then run best params on full 5-fold CV. Key params: max_depth, min_child_weight,
  subsample, colsample_bytree, reg_lambda, reg_alpha. Keep lr=0.03 fixed (confirmed
  best by attempt-15). Highest-expected-gain lever remaining.
- LightGBM + XGBoost blend: attempt-8 LGB (public 0.74099) + attempt-14 XGB (public
  0.75623). If errors are orthogonal, a small LGB contribution might diversify.
- Ordinal regression approach: two binary classifiers P(cancel≥1) and P(cancel≥2)
  exploiting the ordered class structure (cancel 0 < 1 < 2).
