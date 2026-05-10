# Project State (last updated 2026-05-10)

## Current best

Branch: improve/attempt-5
CV accuracy: 0.72927 (tuned; raw argmax 0.72528)
Multipliers: c0×1.00, c1×0.95, c2×0.65
Public leaderboard: not submitted
Submission file: output/submission.csv

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
  lambda_l1=0.1, lambda_l2=1.0, N_ROUNDS=10000, EARLY_STOPPING=100
Class weights: {0: 1.0, 1: 2.0, 2: 1.5}
Decision rule: 2D multiplier tuned (c0×1.00, c1×0.95, c2×0.65)
Early stopping triggered at: 6929, 4269, 7075, 7715, 7370 rounds (mean ~6672)

## What's been tried (newest first)

attempt-5: PASS  +0.00277 — N_ROUNDS→10000 + 2D post-hoc multiplier tuning
attempt-4: FAIL  -0.00134 — class weights + zip_cancel1_rate OOF; dropped threshold tuning
attempt-3: MIXED +0.00050 — 4 new interaction features + class-2 threshold tuning
attempt-2: MIXED +0.00030 — slower/deeper training + OOF zip encoding + credit ordinal
attempt-1: PASS  baseline — LightGBM on 300k subsample (0.72570)

## Lessons learned

- **Class weights + threshold tuning must be paired.** Class weights {0:1, 1:2, 2:1.5}
  shift raw probabilities toward minority classes. Without threshold tuning, this
  hurts argmax accuracy (~-0.0013, attempt-4). With 2D tuning, it's a net +0.0028
  over no weights (attempt-5 vs attempt-3).
- **c2×0.65 optimal multiplier means class-2 weight of 1.5 is ~35% too high.**
  The model is over-predicting class-2, and tuning compensates by scaling it down.
  For attempt-6: reduce class-2 weight toward 1.1–1.2 so raw probabilities are
  better calibrated and the tuned multiplier lands closer to 1.0.
- **N_ROUNDS=10000 is the right ceiling.** Early stopping now fires naturally at
  4269–7715 rounds (mean ~6672). The model converges fully.
- **zip.code is the dominant feature** by gain (1.76M vs 872k for sales.channel).
  OOF encoding of zip cancel rates adds meaningful signal on top of the categorical.
- **Numeric interaction features** (tenure×credit, premium/credit) add marginal gain
  because LightGBM already finds those splits via trees.
- **Threshold tuning on a single class-2 scalar** (attempt-3) found a conservative
  value (t2=0.49). A 2D multiplier search is strictly more powerful.
- **email_domain, house.color, original_quote_weekday, season_of_renewal** have
  near-zero class spread — likely pure noise features.

## Open ideas (untried)

- Reduce class-2 weight from 1.5 → 1.1 so raw probabilities are better calibrated
  before 2D threshold tuning (highest expected gain for attempt-6)
- XGBoost ensemble averaging — different model bias, competition leaders likely ensembling
- Real Optuna tuning that matches train_model.py's full feature pipeline
  (tune_params.py exists but uses different features than the main pipeline)
- Feature pruning: drop email_domain, house.color, original_quote_weekday,
  season_of_renewal (near-zero signal)
- New features: claim_per_year, email × credit interaction, is_first_year
- Ordinal regression: two binary classifiers P(cancel≥1) and P(cancel≥2)
