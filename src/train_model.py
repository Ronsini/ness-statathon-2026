"""
NESS Statathon 2026 — Policy Retention Model
Main training pipeline: 5-fold stratified cross-validation with LightGBM.

Outputs:
  - output/submission.csv        Kaggle submission file
  - output/cv_results.txt        CV scores and confusion matrix

Usage:
  python src/train_model.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Training subsample size. Set to None to use all rows (slower but higher accuracy).
SAMPLE_N = 300_000
N_FOLDS = 5
RANDOM_STATE = 42

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log(msg: str, t0: float) -> None:
    """Print a timestamped message."""
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs from data/, drop the rare cancel=-1 rows from train."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    # ~0.3% of rows have cancel = -1 (likely data quality issue) — drop them
    train = train[train["cancel"].isin([0, 1, 2])].reset_index(drop=True)
    return train, test


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven features. Modifies df in place and returns it."""
    df["household_size"] = df["n.adults"].fillna(0) + df["n.children"].fillna(0)
    df["has_children"] = (df["n.children"].fillna(0) > 0).astype(int)
    df["premium_per_sqft"] = df["premium"] / df["square_footage"].replace(0, np.nan)
    df["claim_x_tenure"] = df["claim.ind"].fillna(0) * df["tenure"].fillna(0)
    df["young_with_claim"] = (
        (df["ni.age"].fillna(99) < 30) & (df["claim.ind"].fillna(0) == 1)
    ).astype(int)
    df["new_customer"] = (df["tenure"].fillna(99) < 3).astype(int)
    return df


def align_categoricals(
    X: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert categorical columns to pandas Categorical with shared categories
    across train and test. LightGBM uses these natively."""
    for c in cat_cols:
        combined = pd.concat(
            [X[c].astype("string"), X_test[c].astype("string")], ignore_index=True
        )
        cats = combined.astype("category").cat.categories
        X[c] = pd.Categorical(X[c].astype("string"), categories=cats)
        X_test[c] = pd.Categorical(X_test[c].astype("string"), categories=cats)
    return X, X_test


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    log("Loading data...", t0)
    train, test = load_data()
    log(f"Train rows: {len(train):,}  Test rows: {len(test):,}", t0)

    # Optional subsample for faster iteration
    if SAMPLE_N is not None and SAMPLE_N < len(train):
        train = train.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
        log(f"Subsampled to {len(train):,} rows", t0)

    y = train["cancel"].astype(int).values
    test_ids = test["id"].values

    X = train.drop(columns=["id", "cancel"])
    X_test = test.drop(columns=["id"])

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    log(f"Categorical columns: {cat_cols}", t0)

    # Align categories across train/test so test doesn't fail on unseen levels
    X, X_test = align_categoricals(X, X_test, cat_cols)

    # Feature engineering
    X = add_engineered_features(X)
    X_test = add_engineered_features(X_test)
    log(f"Total features: {X.shape[1]}", t0)

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        log(f"Fold {fold + 1}/{N_FOLDS}...", t0)
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)

        model = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=300,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        va_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[va_idx] = va_pred
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS

        fold_acc = accuracy_score(y_va, va_pred.argmax(axis=1))
        fold_scores.append(fold_acc)
        log(f"  Fold {fold + 1} accuracy: {fold_acc:.5f}", t0)

    # Aggregate results
    oof_class = oof_preds.argmax(axis=1)
    overall_acc = accuracy_score(y, oof_class)
    cm = confusion_matrix(y, oof_class)

    summary = []
    summary.append("=" * 60)
    summary.append(f"5-fold CV accuracy (mean): {np.mean(fold_scores):.5f}")
    summary.append(f"5-fold CV accuracy (std):  {np.std(fold_scores):.5f}")
    summary.append(f"OOF accuracy overall:      {overall_acc:.5f}")
    summary.append(f"Naive baseline (all 0):    {(y == 0).mean():.5f}")
    summary.append("=" * 60)
    summary.append("\nConfusion matrix (rows=true, cols=pred):")
    summary.append(
        str(
            pd.DataFrame(
                cm,
                index=["true_0", "true_1", "true_2"],
                columns=["pred_0", "pred_1", "pred_2"],
            )
        )
    )
    print("\n" + "\n".join(summary))

    # Save CV results
    (OUTPUT_DIR / "cv_results.txt").write_text("\n".join(summary))

    # Save submission
    sub = pd.DataFrame({"Id": test_ids, "Predicted": test_preds.argmax(axis=1)})
    sub.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    log(f"Saved submission to {OUTPUT_DIR / 'submission.csv'}", t0)
    log(f"Total runtime: {time.time() - t0:.0f}s", t0)


if __name__ == "__main__":
    main()
