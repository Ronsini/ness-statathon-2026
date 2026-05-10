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
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
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
N_ROUNDS = 10000
EARLY_STOPPING = 100

# Update this every branch so results_log.txt stays self-documenting
ATTEMPT_LABEL = "improve/attempt-6"
ATTEMPT_NOTES = """
Changes vs attempt-5:
  - Added XGBoost as second base model trained on same 5 folds
  - Same sample weights, same categorical features (XGBoost native cat support
    via enable_categorical=True)
  - Ensemble: 0.5 * LightGBM probs + 0.5 * XGBoost probs
  - Threshold tuning applied to ensemble OOF (not per-model)
  - LightGBM hyperparameters unchanged from attempt-5

Why it should help:
  - Raw CV has been essentially flat across attempts 4 and 5 (0.72516 vs
    0.72528) despite very different LightGBM configurations. Single-model
    LightGBM appears near its ceiling on this data.
  - XGBoost uses a different splitting algorithm and slightly different feature
    subsampling logic. When two strong models disagree on a row, the average
    often beats either individually.
  - Expected gain: +0.2 to +0.5pt over attempt-5's 0.72927 tuned accuracy.
"""

# Per-class sample weights — upweight minority classes so the model
# pays proportionally more attention to class-1 (7%) and class-2 (22%).
CLASS_WEIGHTS = {0: 1.0, 1: 2.0, 2: 1.5}

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

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "tree_method": "hist",
    "enable_categorical": True,
    "n_jobs": -1,
    "verbosity": 1,
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
    # Long tenure × high credit = loyal; short tenure × low credit = cancel risk
    df["tenure_x_credit"] = df["log_tenure"] * df["credit_ordinal"].fillna(1)
    # Frequent mover relative to tenure signals instability
    df["res_tenure_ratio"] = df["len.at.res"].fillna(0) / (df["tenure"].fillna(0) + 1)
    # Premium burden relative to credit quality — price-stressed customers cancel
    df["premium_credit_stress"] = df["premium"].fillna(0) / (df["credit_ordinal"].fillna(0) + 1)
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

    # Premium relative to dwelling-type median
    dwell_med = (
        ref.groupby("dwelling.type")["premium"].median()
        .reset_index()
        .rename(columns={"premium": "_dwell_prem_med"})
    )
    df = df.merge(dwell_med, on="dwelling.type", how="left")
    df["premium_vs_dwelling"] = df["premium"].fillna(0) / df["_dwell_prem_med"].replace(0, np.nan)
    df = df.drop(columns=["_dwell_prem_med"])

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
    lgb_oof_preds = np.zeros((len(X), 3))
    xgb_oof_preds = np.zeros((len(X), 3))
    lgb_test_preds = np.zeros((len(X_test), 3))
    xgb_test_preds = np.zeros((len(X_test), 3))
    fold_lgb_scores = []
    fold_xgb_scores = []
    fold_ens_scores = []

    # Placeholder columns for OOF zip target encoding (filled per fold)
    X["zip_cancel2_rate"] = np.nan
    X["zip_cancel1_rate"] = np.nan
    X_test["zip_cancel2_rate"] = np.nan
    X_test["zip_cancel1_rate"] = np.nan
    zip2_test_accum = np.zeros(len(X_test))
    zip1_test_accum = np.zeros(len(X_test))

    smooth_k = 20

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        log(f"Fold {fold + 1}/{N_FOLDS}...", t0)
        y_tr, y_va = y[tr_idx], y[va_idx]

        # OOF target encoding: smoothed cancel-2 and cancel-1 rates per zip.
        # Computed from training fold only to avoid leakage on val/test.
        global_rate2 = (y_tr == 2).mean()
        global_rate1 = (y_tr == 1).mean()

        # Cast to str to avoid Categorical dtype interfering with map/fillna
        tr_zips = X.iloc[tr_idx]["zip.code"].astype(str)
        zip_df = pd.DataFrame({
            "zip": tr_zips.values,
            "is2": (y_tr == 2).astype(float),
            "is1": (y_tr == 1).astype(float),
        })
        zip_agg = zip_df.groupby("zip")[["is2", "is1"]].agg(["sum", "count"])
        zip_agg.columns = ["sum2", "count2", "sum1", "count1"]
        zip_agg["rate2"] = (zip_agg["sum2"] + smooth_k * global_rate2) / (zip_agg["count2"] + smooth_k)
        zip_agg["rate1"] = (zip_agg["sum1"] + smooth_k * global_rate1) / (zip_agg["count1"] + smooth_k)
        zip_map2 = zip_agg["rate2"].to_dict()
        zip_map1 = zip_agg["rate1"].to_dict()

        for col, zmap, grate, accum, test_col in [
            ("zip_cancel2_rate", zip_map2, global_rate2, zip2_test_accum, "zip_cancel2_rate"),
            ("zip_cancel1_rate", zip_map1, global_rate1, zip1_test_accum, "zip_cancel1_rate"),
        ]:
            X.loc[tr_idx, col] = X.iloc[tr_idx]["zip.code"].astype(str).map(zmap).fillna(grate).values
            X.loc[va_idx, col] = X.iloc[va_idx]["zip.code"].astype(str).map(zmap).fillna(grate).values
            accum += X_test["zip.code"].astype(str).map(zmap).fillna(grate).values
            X_test[test_col] = accum / (fold + 1)

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]

        # Sample weights — upweight minority classes during training
        sample_weights_tr = np.array([CLASS_WEIGHTS[yi] for yi in y_tr])

        # --- LightGBM ---
        dtr = lgb.Dataset(X_tr, y_tr, weight=sample_weights_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)

        lgb_model = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=N_ROUNDS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(100)],
        )

        lgb_va_pred = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
        lgb_oof_preds[va_idx] = lgb_va_pred
        lgb_test_preds += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / N_FOLDS

        # --- XGBoost ---
        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weights_tr, enable_categorical=True)
        dval_xgb = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dtest_xgb = xgb.DMatrix(X_test, enable_categorical=True)

        xgb_model = xgb.train(
            XGB_PARAMS,
            dtrain_xgb,
            num_boost_round=3000,
            evals=[(dval_xgb, "valid")],
            early_stopping_rounds=100,
            verbose_eval=100,
        )

        xgb_va_pred = xgb_model.predict(dval_xgb).reshape(-1, 3)
        xgb_oof_preds[va_idx] = xgb_va_pred
        xgb_test_preds += xgb_model.predict(dtest_xgb).reshape(-1, 3) / N_FOLDS

        lgb_fold_acc = accuracy_score(y_va, lgb_va_pred.argmax(axis=1))
        xgb_fold_acc = accuracy_score(y_va, xgb_va_pred.argmax(axis=1))
        ens_fold_acc = accuracy_score(y_va, (0.5 * lgb_va_pred + 0.5 * xgb_va_pred).argmax(axis=1))
        fold_lgb_scores.append(lgb_fold_acc)
        fold_xgb_scores.append(xgb_fold_acc)
        fold_ens_scores.append(ens_fold_acc)
        log(
            f"  Fold {fold + 1}  LGB: {lgb_fold_acc:.5f}  "
            f"XGB: {xgb_fold_acc:.5f}  ENS: {ens_fold_acc:.5f}  "
            f"(lgb_iter: {lgb_model.best_iteration}, xgb_iter: {xgb_model.best_iteration})",
            t0,
        )

    # Ensemble OOF and test predictions
    ensemble_oof = 0.5 * lgb_oof_preds + 0.5 * xgb_oof_preds
    ensemble_test = 0.5 * lgb_test_preds + 0.5 * xgb_test_preds

    lgb_raw_acc = accuracy_score(y, lgb_oof_preds.argmax(axis=1))
    xgb_raw_acc = accuracy_score(y, xgb_oof_preds.argmax(axis=1))
    ens_raw_acc = accuracy_score(y, ensemble_oof.argmax(axis=1))
    log(f"Raw OOF — LGB: {lgb_raw_acc:.5f}  XGB: {xgb_raw_acc:.5f}  ENS: {ens_raw_acc:.5f}", t0)

    # Post-hoc threshold tuning on the ensemble OOF
    print("\nTuning class-probability multipliers on ensemble OOF predictions...")
    best_acc = ens_raw_acc
    best_mult = (1.0, 1.0, 1.0)

    for t1, t2 in product(
        np.arange(0.40, 2.01, 0.05),
        np.arange(0.40, 2.01, 0.05),
    ):
        multipliers = np.array([1.0, t1, t2])
        pred = (ensemble_oof * multipliers).argmax(axis=1)
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)

    print(f"Best multipliers: class0×{best_mult[0]:.2f}, "
          f"class1×{best_mult[1]:.2f}, class2×{best_mult[2]:.2f}")
    print(f"Tuned ensemble OOF accuracy: {best_acc:.5f} (was {ens_raw_acc:.5f})")

    multipliers_arr = np.array(best_mult)
    oof_class = (ensemble_oof * multipliers_arr).argmax(axis=1)
    test_class = (ensemble_test * multipliers_arr).argmax(axis=1)

    overall_acc = accuracy_score(y, oof_class)
    cm = confusion_matrix(y, oof_class)

    summary = []
    summary.append("=" * 60)
    summary.append(f"5-fold LGB  (mean / std): {np.mean(fold_lgb_scores):.5f} / {np.std(fold_lgb_scores):.5f}")
    summary.append(f"5-fold XGB  (mean / std): {np.mean(fold_xgb_scores):.5f} / {np.std(fold_xgb_scores):.5f}")
    summary.append(f"5-fold ENS  (mean / std): {np.mean(fold_ens_scores):.5f} / {np.std(fold_ens_scores):.5f}")
    summary.append(f"OOF LightGBM  (raw):      {lgb_raw_acc:.5f}")
    summary.append(f"OOF XGBoost   (raw):      {xgb_raw_acc:.5f}")
    summary.append(f"OOF Ensemble  (raw):      {ens_raw_acc:.5f}")
    summary.append(f"OOF Ensemble  (tuned):    {overall_acc:.5f}")
    summary.append(f"Multipliers: c0×{best_mult[0]:.2f}  c1×{best_mult[1]:.2f}  c2×{best_mult[2]:.2f}")
    summary.append(f"Naive baseline (all 0):   {(y == 0).mean():.5f}")
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

    # Append to cumulative results log so every attempt is preserved
    import datetime
    log_path = OUTPUT_DIR / "results_log.txt"
    entry = "\n".join([
        "=" * 72,
        f"Attempt : {ATTEMPT_LABEL}",
        f"Date    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Runtime : {time.time() - t0:.0f}s",
        ATTEMPT_NOTES.strip(),
        "-" * 72,
        *summary,
        "",
    ])
    with open(log_path, "a") as f:
        f.write(entry + "\n")

    sub = pd.DataFrame({"Id": test_ids, "Predicted": test_class})
    sub.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    log(f"Saved submission to {OUTPUT_DIR / 'submission.csv'}", t0)
    log(f"Total runtime: {time.time() - t0:.0f}s", t0)


if __name__ == "__main__":
    main()
