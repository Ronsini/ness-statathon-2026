# Project State (last updated 2026-05-11)

## Current best

Branch: improve/attempt-5
CV accuracy: 0.72927 (tuned; c0×1.00, c1×0.95, c2×0.65)
Public leaderboard: not submitted
Submission file: output/submission.csv

> Note: attempt-8 (0.72963 tuned) is the new high watermark but MIXED (+0.00036).
> PROJECT_STATE.md updates on PASS only. Current best stays at attempt-5 until
> an attempt beats 0.72927 + 0.001 = 0.73027.

## Pipeline summary (attempt-8)

Data: 1,045,123 training rows, 2,412 test rows
Features: 21 base + 24 engineered (row-level + missingness) + 3 group + 8 OOF = 56 total

Base features (from train.csv):
  year, zip.code, house.color, ni.age, len.at.res, credit, coverage.type,
  dwelling.type, premium, sales.channel, ni.gender, ni.marital.status,
  n.adults, n.children, tenure, claim.ind, original_quote_weekday,
  season_of_renewal, square_footage, email_domain, num_windows_front

Row-level engineered features (add_engineered_features):
  household_size, has_children, premium_per_sqft, claim_x_tenure,
  young_with_claim, new_customer, log_premium, log_tenure, log_square_footage,
  premium_x_claim, age_x_tenure, is_married_adult, windows_per_sqft,
  long_resident, credit_ordinal, tenure_x_credit, res_tenure_ratio,
  premium_credit_stress,
  credit_missing, ni_age_missing, n_adults_missing, coverage_type_missing,
  ni_marital_status_missing, n_children_missing

Group features (add_group_features, computed on training ref):
  premium_vs_zip, premium_vs_credit, premium_vs_dwelling

OOF features (nested fold encoding, no leakage):
  zip_cancel2_rate, zip_cancel1_rate,
  age_credit_cancel2_rate, age_credit_cancel1_rate,
  zip_sales_cancel2_rate, zip_sales_cancel1_rate,
  cov_dwell_cancel2_rate, cov_dwell_cancel1_rate

Model: LightGBM multiclass (num_class=3), 5-fold stratified CV
Key hyperparameters: learning_rate=0.02, num_leaves=127, min_data_in_leaf=30,
  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
  lambda_l1=0.1, lambda_l2=1.0, N_ROUNDS=10000, EARLY_STOPPING=100
Class weights: {0: 1.0, 1: 1.3, 2: 1.1} (attempt-8; was {0:1.0, 1:2.0, 2:1.5} in attempt-5)
Decision rule: two-stage multiplier search (coarse step 0.05, fine step 0.01 ±0.10)
  attempt-8 best multipliers: c0×1.00, c1×1.260, c2×0.900

## What's been tried (newest first)

attempt-9: FAIL (tuning) — Optuna 40-trial search on 350k subsample; params not used (c1 multiplier hit search floor)
attempt-8: MIXED +0.00036 — lighter weights {0:1, 1:1.3, 2:1.1}, nested OOF encoding, 4 new OOF features
attempt-7: MIXED +0.00018 — OOF age×credit encoding + 6 missingness flags (52 features)
attempt-6: FAIL  -0.00216 — XGBoost ensemble blend (zip.code handling inferior)
attempt-5: PASS  +0.00277 — post-hoc 2D multiplier tuning + N_ROUNDS raised to 10000
attempt-4: FAIL  -0.00134 — class weights {0:1, 1:2, 2:1.5} + zip_cancel1_rate OOF
attempt-3: MIXED +0.00050 — 4 new interaction features + class-2 threshold tuning
attempt-2: MIXED +0.00030 — slower/deeper training + OOF zip encoding + credit ordinal
attempt-1: PASS  baseline — LightGBM on 300k subsample (0.72570)

## Lessons learned

- Class-0 dominance is the binding constraint: class-0 is 71% of rows, so every
  1pp drop in class-0 recall costs 0.71pp accuracy. Heavy class weights buy
  class-1/2 recall at too high a price. Lighter weights + multiplier tuning is
  the right balance.
- Post-hoc 2D multiplier search is essential when class weights are active. Attempt-5
  found c1×0.95, c2×0.65 (downscale both); attempt-8 with lighter weights found
  c1×1.26, c2×0.90 (upscale class-1). The direction depends on how much the weights
  push each class.
- OOF target encoding is only novel over LightGBM's native splits when the group
  has high cardinality (>50–100 unique cells). zip.code (400+ values) benefits
  substantially; age×credit (15 cells) adds minimal signal since LightGBM already
  finds those splits.
- N_ROUNDS=5000 is insufficient — raise to 10000. Folds in attempt-8 converged at
  3477–4865 rounds, confirming 10000 is the right cap.
- zip.code is the dominant feature by gain. OOF encoding of zip cancel rates adds
  meaningful signal on top of the native categorical.
- XGBoost is inferior to LightGBM on this dataset because zip.code (400+ values,
  #1 feature) is handled better by LightGBM's native categorical splits.
- email_domain, house.color, original_quote_weekday, season_of_renewal: near-zero
  class-2 signal spread. Confirmed noise; not worth investigating.

## Open ideas (untried)

- Optuna re-run with no class weights (attempt-9 Optuna ran with weights active;
  optimal c1 multiplier hit the search floor at 0.30, suggesting extreme probability
  miscalibration — weights-free tuning would avoid this)
- is_first_year_with_claim: (tenure < 1.1) & (claim.ind == 1), 31% c2 rate vs 21.9%
  overall (+9.1pp). Small segment but cleanly separable.
- zip_frequency: count of training rows per zip code. Rare zips have 40–50% c2 rate
  vs 18–24% for common zips — clean monotone signal complementing OOF rates.
- Ordinal regression approach: two binary classifiers P(cancel≥1) and P(cancel≥2)
  exploiting the ordered class structure (cancel 0 < 1 < 2)
- Feature pruning: drop near-zero-gain features (email_domain, house.color,
  season_of_renewal, original_quote_weekday) to reduce noise
