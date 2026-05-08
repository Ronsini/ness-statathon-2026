# NESS Statathon 2026 — Policy Retention Model

**Competition:** [2026 NESS Statathon — Travelers Policy Retention](https://www.kaggle.com/competitions/2026-ness-statathon)

## Problem

Predict whether a Travelers property insurance policy will be:
- **0** — Not cancelled (renews)
- **1** — May cancel but can be convinced to stay
- **2** — Will cancel

Training data is ~1M policies from 2013–2017 with the cancellation outcome labeled. Test data is 2,412 policies where the outcome is hidden. Evaluation metric is **accuracy**, with a public/private split (30%/70%).

## Goals

1. Identify policies likely to be cancelled before end of term
2. Understand key drivers of cancellation
3. Provide actionable recommendations for Travelers

## Approach

1. **EDA** — class balance, missingness, feature distributions, cancellation rates by feature
2. **Preprocessing** — handle ~0.1% missingness, encode 9 categorical features, drop ~0.3% of rows where `cancel = -1`
3. **Feature engineering** — household size, premium-per-sqft, claim×tenure interactions, target encoding on high-cardinality features (zip, email)
4. **Model** — LightGBM with 5-fold stratified cross-validation
5. **Validation** — trust local CV over public leaderboard (which is only 30% of test set)

## Repository Structure

```
.
├── data/             # train.csv, test.csv (gitignored — download from Kaggle)
├── src/              # Python scripts
│   ├── eda.py            # exploratory data analysis
│   ├── train_model.py    # main training pipeline
│   └── make_submission.py # generate Kaggle submission CSV
├── notebooks/        # Jupyter / Quarto notebooks
├── output/           # submission CSVs, saved models
└── docs/             # presentation slides, write-ups
```

## Reproducing Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place train.csv and test.csv in data/

# 3. Run pipeline
python src/eda.py              # generate EDA summary
python src/train_model.py      # train model + 5-fold CV
python src/make_submission.py  # generate submission.csv in output/
```

## Current Results

| Model                          | 5-fold CV Accuracy |
|--------------------------------|--------------------|
| Naive baseline (predict all 0) | 0.7099             |
| LightGBM (subsampled 300k)     | 0.7257             |
| LightGBM (full 1M, tuned)      | _to be added_      |

## Team

UConn Huskies — NESS Statathon 2026
