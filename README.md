# NESS Statathon 2026 — Policy Retention Model

**Competition:** [2026 NESS Statathon — Travelers Policy Retention](https://www.kaggle.com/competitions/2026-ness-statathon)

**Final result: 3rd place &nbsp;·&nbsp; Private leaderboard accuracy: 0.76941**

## Problem

Predict whether a Travelers property insurance policy will be:
- **0** — Will not cancel (~71% of training data)
- **1** — May cancel but can be convinced to stay (~7%)
- **2** — Will cancel (~22%)

Training data: 1,045,123 policies from 2013–2017 with cancellation outcome labeled.
Test data: 2,412 policies with hidden outcomes.
Evaluation metric: **accuracy** (public/private split: 30%/70%).

## Final Model

A three-layer stacked ensemble:

1. **Base models** (trained with 5-fold out-of-fold CV):
   - XGBoost — 91 features, numeric preprocessing, OOF target encodings
   - LightGBM — 56 features, native categoricals, 6 missingness flags, OOF encodings
   - CatBoost — same feature set as LightGBM, native categorical handling via Pool API

2. **Meta-model** — LightGBM trained on 34 meta-features:
   - 9 raw class probabilities (3 per base model)
   - 25 derived features: max prob, top-2 margin, argmax per model, cross-model agreement indicators, vote counts, mean/std/range per class

All base models use 5-fold StratifiedKFold. The meta-model is trained on OOF predictions only — it never sees a row it helped predict.

## Score Progression

| Stage | Public (30%) | Private (70%) |
|-------|-------------|---------------|
| LGB baseline | 0.74099 | 0.74570 |
| XGBoost (best single model) | 0.75623 | 0.74629 |
| XGB + LGB logistic stack | 0.76038 | 0.74807 |
| XGB + LGB + CatBoost stack | 0.76454 | 0.74214 |
| Stack blend | 0.76869 | 0.74096 |
| **LGB meta-stacker (final)** | **0.77839** | **0.76941** |

The blending steps overfit the public leaderboard — private scores dipped even as public improved. The LGB meta-stacker was the only step that lifted both.

## Repository Structure

```
.
├── data/                       # train.csv, test.csv (gitignored — download from Kaggle)
├── src/                        # Training and stacking scripts
│   ├── train_model.py              # LightGBM base model
│   ├── save_xgb_probs.py           # XGBoost base model (attempt-14)
│   ├── save_cat_probs.py           # CatBoost base model (attempt-21)
│   ├── stack_meta_lgb.py           # LGB meta-stacker — final submission (attempt-24)
│   ├── feature_leak_audit.py       # EDA audit script (attempt-25)
│   └── postprocess_attempt24.py    # Rule-based postprocessing (attempt-27)
├── output/                     # Submission CSVs, saved OOF/test probability arrays
├── report/                     # Quarto presentation (competition-report.qmd)
├── docs/                       # Attempt history, project state, notes
└── requirements.txt
```

## Reproducing the Final Submission

```bash
pip install -r requirements.txt

# Place train.csv and test.csv in data/

# Step 1: Train base models and save OOF + test probability arrays
python src/train_model.py          # LightGBM → lgb_oof_probs.npy, lgb_test_probs.npy
python src/save_xgb_probs.py       # XGBoost  → xgb_oof_probs.npy, xgb_test_probs.npy
python src/save_cat_probs.py       # CatBoost → cat_oof_probs.npy, cat_test_probs.npy

# Step 2: Train LGB meta-stacker and generate final submission
python src/stack_meta_lgb.py       # → output/submission_stack_meta_lgb.csv
```

## Key Findings

- **Credit level** is the strongest cancellation signal — low-credit policies cancel at 2.5× the rate of high-credit policies
- **Sales channel** matters: Phone and Online policies cancel at ~40% vs ~22% for Broker
- **Class 1** (may cancel) is the most valuable group for retention outreach — customers who have not decided to leave yet
- **Stacking > blending**: non-linear interactions between base model probabilities carry signal that linear meta-models cannot capture

## Author

Ronnie Orsini — University of Connecticut, NESS Statathon 2026
