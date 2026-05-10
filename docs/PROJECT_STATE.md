# Project State (last updated 2026-05-09)

## Current best

Branch: improve/attempt-3
CV accuracy: 0.72650
Public leaderboard: not submitted
Submission file: output/submission.csv

> Note: attempt-4 (0.72516) is a FAIL vs attempt-3. Current pipeline code reflects
> attempt-4/5 (class weights active, zip_cancel1_rate added), but the best CV
> on record is 0.72650 from attempt-3. PROJECT_STATE.md will update on next PASS.

## Pipeline summary

Data: 1,045,123 training rows, 2,412 test rows
Features: 21 base + 21 engineered (row-level + group) + 2 OOF = 44 total at train time

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
  premium_credit_stress

Group features (add_group_features, computed on training ref):
  premium_vs_zip, premium_vs_credit, premium_vs_dwelling

OOF features (computed per fold, no leakage):
  zip_cancel2_rate, zip_cancel1_rate

Model: LightGBM multiclass (num_class=3), 5-fold stratified CV
Key hyperparameters: learning_rate=0.02, num_leaves=127, min_data_in_leaf=30,
  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
  lambda_l1=0.1, lambda_l2=1.0, N_ROUNDS=5000 (attempt-4) / 10000 (attempt-5)
Class weights: {0: 1.0, 1: 2.0, 2: 1.5} active since attempt-4
Decision rule: argmax (attempt-4); threshold-tuned multiplier search planned for attempt-5

## What's been tried (newest first)

attempt-4: FAIL  -0.00134 — class weights + zip_cancel1_rate OOF; dropped threshold tuning
attempt-3: MIXED +0.00050 — 4 new interaction features + class-2 threshold tuning
attempt-2: MIXED +0.00030 — slower/deeper training + OOF zip encoding + credit ordinal
attempt-1: PASS  baseline — LightGBM on 300k subsample (0.72570)

## Lessons learned

- Class weights {0:1, 1:2, 2:1.5} produce calibrated probabilities but hurt argmax
  accuracy ~0.7pt — pair with threshold tuning to correct the decision rule.
- Class-2 recall ceiling at ~26% (without weights) appears to be a training signal
  problem, not a feature capacity problem — the model needs to be pushed toward
  class-2 during training, not just post-hoc.
- zip.code is the dominant feature by gain (1.76M vs 872k for sales.channel #2).
  OOF encoding of zip cancel rates adds meaningful signal on top of the categorical.
- Numeric interaction features (tenure×credit, premium/credit) add marginal gain
  because LightGBM already finds those splits via trees; the encoding only helps
  for smooth linear combinations across features.
- Threshold tuning on a single class-2 scalar (attempt-3 style) selected a more
  conservative value (t2=0.49), confirming the model probabilities are well-
  calibrated. A 2D multiplier search (class-1 and class-2) is the right form.
- N_ROUNDS=5000 is insufficient — 4/5 folds in attempt-4 hit the cap. Use 10000.
- email_domain, house.color, original_quote_weekday, season_of_renewal have near-zero
  class-2 signal spread (~0.001pp difference across values). Likely pure noise.

## Open ideas (untried)

- Post-hoc threshold tuning on OOF predictions — 2D multiplier search for class-1
  and class-2 (planned for attempt-5, code already written)
- XGBoost ensemble averaging — different model bias, competition leaders likely ensembling
- Real Optuna tuning that matches train_model.py's full feature pipeline
  (tune_params.py exists but uses different features than the main pipeline)
- Feature pruning based on LightGBM gain importance — drop near-zero-gain features
  to reduce overfitting risk
- New features: claim_per_year, email × credit interaction, is_first_year
- Ordinal regression approach: two binary classifiers P(cancel≥1) and P(cancel≥2)
  exploiting the ordered class structure
