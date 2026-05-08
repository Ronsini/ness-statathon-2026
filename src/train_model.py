"""
NESS Statathon 2026 — Policy Retention Model
Training pipeline: 5-fold stratified CV with LightGBM on the full 1M-row dataset.

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

SAMPLE_N = None  # None = full 1M rows; set an int to subsample for quick iteration
N_FOLDS = 5
RANDOM_STATE = 42
N_ROUNDS = 5000
EARLY_STOPPING = 100

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.02,
    "num_leaves": 127,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train = train[train["cancel"].isin([0, 1, 2])].reset_index(drop=True)
    return train, test


CREDIT_ORDINAL = {"low": 0, "medium": 1, "high": 2}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features — no cross-row statistics needed."""
    df["household_size"] = df["n.adults"].fillna(0) + df["n.children"].fillna(0)
    df["has_children"] = (df["n.children"].fillna(0) > 0).astype(int)
    df["premium_per_sqft"] = df["premium"] / df["square_footage"].replace(0, np.nan)
    df["claim_x_tenure"] = df["claim.ind"].fillna(0) * df["tenure"].fillna(0)
    df["young_with_claim"] = (
        (df["ni.age"].fillna(99) < 30) & (df["claim.ind"].fillna(0) == 1)
    ).astype(int)
    df["new_customer"] = (df["tenure"].fillna(99) < 3).astype(int)
    df["log_premium"] = np.log1p(df["premium"].fillna(0))
    df["log_tenure"] = np.log1p(df["tenure"].fillna(0))
    df["log_square_footage"] = np.log1p(df["square_footage"].fillna(0))
    df["premium_x_claim"] = df["premium"].fillna(0) * df["claim.ind"].fillna(0)
    df["age_x_tenure"] = df["ni.age"].fillna(0) * df["tenure"].fillna(0)
    df["is_married_adult"] = (
        (df["ni.marital.status"].fillna(0) == 1) & (df["n.adults"].fillna(0) >= 2)
    ).astype(int)
    df["windows_per_sqft"] = df["num_windows_front"] / df["square_footage"].replace(0, np.nan)
    df["long_resident"] = (df["len.at.res"].fillna(0) > 10).astype(int)
    # credit is ordinal (low < medium < high) — encode numerically so the
    # model can learn monotone relationships, not just arbitrary splits.
    df["credit_ordinal"] = df["credit"].map(CREDIT_ORDINAL)
    # Convert zip.code to string so LightGBM treats it as categorical (nominal),
    # not numeric. Must happen after add_group_features (which merges on float zip).
    df["zip.code"] = df["zip.code"].astype("string")
    return df


def add_group_features(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Group-level statistics computed on `ref` (training data), merged into `df`."""
    # Premium relative to zip-code median
    zip_med = (
        ref.groupby("zip.code")["premium"].median()
        .reset_index()
        .rename(columns={"premium": "_zip_prem_med"})
    )
    df = df.merge(zip_med, on="zip.code", how="left")
    df["premium_vs_zip"] = df["premium"].fillna(0) / df["_zip_prem_med"].replace(0, np.nan)
    df = df.drop(columns=["_zip_prem_med"])

    # Premium relative to credit-tier median
    credit_med = (
        ref.groupby("credit")["premium"].median()
        .reset_index()
        .rename(columns={"premium": "_cred_prem_med"})
    )
    df = df.merge(credit_med, on="credit", how="left")
    df["premium_vs_credit"] = df["premium"].fillna(0) / df["_cred_prem_med"].replace(0, np.nan)
    df = df.drop(columns=["_cred_prem_med"])

    return df


def align_categoricals(
    X: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    if SAMPLE_N is not None and SAMPLE_N < len(train):
        train = train.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
        log(f"Subsampled to {len(train):,} rows", t0)

    y = train["cancel"].astype(int).values
    test_ids = test["id"].values

    X = train.drop(columns=["id", "cancel"])
    X_test = test.drop(columns=["id"])

    # Group features — computed on full training set to give stable statistics
    log("Adding group features...", t0)
    X = add_group_features(X, X)
    X_test = add_group_features(X_test, X)

    # Row-level feature engineering
    X = add_engineered_features(X)
    X_test = add_engineered_features(X_test)

    # Align categoricals
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    log(f"Categorical columns: {cat_cols}", t0)
    X, X_test = align_categoricals(X, X_test, cat_cols)

    log(f"Total features: {X.shape[1]}", t0)

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []

    # Placeholder columns for OOF zip target encoding (filled per fold)
    X["zip_cancel2_rate"] = np.nan
    X_test["zip_cancel2_rate"] = np.nan
    zip_test_accum = np.zeros(len(X_test))  # average across folds for test

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        log(f"Fold {fold + 1}/{N_FOLDS}...", t0)
        y_tr, y_va = y[tr_idx], y[va_idx]

        # OOF target encoding: smoothed fraction of cancel==2 per zip,
        # computed from training fold only to avoid leakage on val/test.
        # At ~2600 rows per zip in the training fold, the self-inclusion
        # leakage for training rows is ~0.04% — negligible.
        global_rate = (y_tr == 2).mean()
        zip_rates = (
            pd.DataFrame({"zip": X.iloc[tr_idx]["zip.code"].values, "is2": (y_tr == 2).astype(float)})
            .groupby("zip")["is2"]
            .agg(["sum", "count"])
        )
        smooth_k = 20
        zip_rates["rate"] = (zip_rates["sum"] + smooth_k * global_rate) / (zip_rates["count"] + smooth_k)
        zip_map = zip_rates["rate"].to_dict()

        X.loc[tr_idx, "zip_cancel2_rate"] = X.iloc[tr_idx]["zip.code"].map(zip_map).fillna(global_rate).values
        X.loc[va_idx, "zip_cancel2_rate"] = X.iloc[va_idx]["zip.code"].map(zip_map).fillna(global_rate).values
        zip_test_accum += X_test["zip.code"].map(zip_map).fillna(global_rate).values
        X_test["zip_cancel2_rate"] = zip_test_accum / (fold + 1)

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]

        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)

        model = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=N_ROUNDS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(100)],
        )

        va_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[va_idx] = va_pred
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS

        fold_acc = accuracy_score(y_va, va_pred.argmax(axis=1))
        fold_scores.append(fold_acc)
        log(f"  Fold {fold + 1} accuracy: {fold_acc:.5f}  (best iter: {model.best_iteration})", t0)

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

    (OUTPUT_DIR / "cv_results.txt").write_text("\n".join(summary))

    sub = pd.DataFrame({"Id": test_ids, "Predicted": test_preds.argmax(axis=1)})
    sub.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    log(f"Saved submission to {OUTPUT_DIR / 'submission.csv'}", t0)
    log(f"Total runtime: {time.time() - t0:.0f}s", t0)


if __name__ == "__main__":
    main()
