# Project State (last updated 2026-05-13)

## Current best

Branch: improve/attempt-13-xgb-clean-preprocess
CV accuracy: 0.73558 (tuned; c0×1.00, c1×1.260, c2×0.890)
Public leaderboard: 0.75207
Submission file: output/submission.csv

## Pipeline summary (attempt-13)

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
  tree_method=hist, max_bin=256, n_estimators=5000, early_stopping_rounds=100
Class weights: {0: 1.0, 1: 1.3, 2: 1.1}
Decision rule: two-stage multiplier search (coarse step 0.05, fine step 0.01 ±0.10)
  attempt-13 best multipliers: c0×1.00, c1×1.260, c2×0.890

Note: All 5 folds hit the n_estimators=5000 cap — model still improving at round 4999.
Raising n_estimators to 10000 is the top priority for attempt-14.

## What's been tried (newest first)

attempt-13: PASS  +0.00631 — pure XGBoost, clean preprocessing, 91 features; public 0.75207
attempt-8:  MIXED +0.00036 — lighter weights {0:1, 1:1.3, 2:1.1}, nested OOF, 4 new OOF; public 0.74099
attempt-7:  MIXED +0.00018 — OOF age×credit encoding + 6 missingness flags (52 features)
attempt-6:  FAIL  -0.00216 — XGBoost blend (not pure XGB; no proper preprocessing)
attempt-5:  PASS  +0.00277 — post-hoc 2D multiplier tuning + N_ROUNDS raised to 10000; public 0.73961
attempt-4:  FAIL  -0.00134 — class weights {0:1, 1:2, 2:1.5} + zip_cancel1_rate OOF
attempt-3:  MIXED +0.00050 — 4 new interaction features + class-2 threshold tuning
attempt-2:  MIXED +0.00030 — slower/deeper training + OOF zip encoding + credit ordinal
attempt-1:  PASS  baseline — LightGBM on 300k subsample (0.72570)

## Lessons learned

- XGBoost with clean numeric preprocessing outperforms LightGBM on this dataset.
  Attempt-13 gained +0.00631 CV and +0.01108 public over the LightGBM best. The key
  driver was class-1 recall: 35.9% (LGB attempt-8) → 52.7% (XGB attempt-13). One-hot
  encoding gives XGBoost clean categorical boundaries that LightGBM's native splits
  were underusing for class-1 separation.
- Attempt-6's conclusion that "XGBoost is inferior" was wrong. That was a blend without
  proper preprocessing, not a fair comparison. Pure XGB with freq+OHE encoding is superior.
- All attempt-13 folds hit n_estimators=5000 cap — logloss still declining at round 4999.
  n_estimators=5000 is insufficient; raise to 10000 for attempt-14.
- Class-0 dominance is the binding constraint: class-0 is 71% of rows, so every
  1pp drop in class-0 recall costs 0.71pp accuracy. But XGBoost found a better
  class-1/class-0 trade-off than LightGBM: +16.8pp class-1 at only -0.7pp class-0.
- Post-hoc 2D multiplier search is essential. Attempt-13 found c1×1.26, c2×0.89 —
  same direction as attempt-8 (upscale class-1). The pattern holds across models.
- OOF target encoding (8 features, nested OOF) carries over cleanly to XGBoost.
  The zip.code OOF rates are especially important since the raw zip.code is dropped
  and replaced by freq encoding.
- Profile OOF and zip×year OOF (from attempt-12) hurt public score — avoid high-cardinality
  OOF keys. The 8 existing OOF keys (attempt-8 level) are the safe set.
- CV and public score can diverge. Attempt-12 beat attempt-11 by +0.00020 CV but lost
  by -0.00138 public. Always submit and check — don't trust CV alone.

## Open ideas (untried)

- Raise n_estimators to 10000 for XGBoost — all attempt-13 folds hit the 5000 cap.
  Highest expected gain, lowest risk. Direct next attempt.
- Optuna hyperparameter search on XGBoost (learning_rate, max_depth, min_child_weight,
  subsample, colsample_bytree, reg_lambda, reg_alpha). Now that XGBoost is the best
  model, tuning its params is the right target.
- LightGBM + XGBoost blend: attempt-8 LGB (public 0.74099) + attempt-13 XGB (public 0.75207).
  If errors are orthogonal, a 10–20% LGB contribution might add diversity.
- Ordinal regression approach: two binary classifiers P(cancel≥1) and P(cancel≥2)
  exploiting the ordered class structure (cancel 0 < 1 < 2)
