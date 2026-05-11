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

ATTEMPT_LABEL = "improve/attempt-8"
ATTEMPT_NOTES = """
Changes vs attempt-7:
  - Lighter class weights: {0:1.0, 1:1.3, 2:1.1} (was {0:1.0, 1:2.0, 2:1.5})
    Heavy weights sacrificed 6.5pp class-0 recall for only 4.6pp recovery in
    classes 1+2. Lighter weights let threshold tuning do more of the work.
  - Fixed fold-specific test OOF encoding bug: previously X_test OOF cols used
    a running average of maps from folds 0..k when fold k predicted on test.
    Now each fold builds X_test_fold = X_test.copy() with that fold's maps,
    so fold k's model predicts on test encoded with fold k's map only.
  - 4 new OOF target-encoded features (in addition to zip + age_credit from attempt-7):
      zip_sales_cancel2_rate  / zip_sales_cancel1_rate   (zip x sales channel)
      cov_dwell_cancel2_rate  / cov_dwell_cancel1_rate   (coverage x dwelling)
  - Two-stage multiplier search: coarse (step 0.05, full range) then fine
    (step 0.01, +-0.10 around coarse best) for higher-precision tuning.
  - Retained from attempt-7: 6 missingness flags + age_credit OOF features.
  - Training-fold target encodings are now nested OOF encodings: each outer
    training fold is split into inner folds so training rows are encoded with
    rates computed from other inner-fold rows, not their own labels.
"""

# Lighter weights: class-0 recall matters more than class-1/2 recall
# because class-0 is 71% of rows. Threshold tuning corrects the decision rule.
CLASS_WEIGHTS = {0: 1.0, 1: 1.3, 2: 1.1}

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

AGE_BINS = [-np.inf, 25, 35, 50, 65, np.inf]
AGE_LABELS = ["lt25", "25-35", "35-50", "50-65", "65plus"]


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
    df["credit_ordinal"] = df["credit"].map(CREDIT_ORDINAL)
    df["tenure_x_credit"] = df["log_tenure"] * df["credit_ordinal"].fillna(1)
    df["res_tenure_ratio"] = df["len.at.res"].fillna(0) / (df["tenure"].fillna(0) + 1)
    df["premium_credit_stress"] = df["premium"].fillna(0) / (df["credit_ordinal"].fillna(0) + 1)
    # Missingness indicators — ~1,000 rows missing per column; class-rate shifts confirmed
    df["credit_missing"] = df["credit"].isna().astype(int)
    df["ni_age_missing"] = df["ni.age"].isna().astype(int)
    df["n_adults_missing"] = df["n.adults"].isna().astype(int)
    df["coverage_type_missing"] = df["coverage.type"].isna().astype(int)
    df["ni_marital_status_missing"] = df["ni.marital.status"].isna().astype(int)
    df["n_children_missing"] = df["n.children"].isna().astype(int)
    # Convert zip.code to string so LightGBM treats it as categorical (nominal),
    # not numeric. Must happen after add_group_features (which merges on float zip).
    df["zip.code"] = df["zip.code"].astype("string")
    return df


def add_group_features(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Group-level statistics computed on `ref` (training data), merged into `df`."""
    zip_med = (
        ref.groupby("zip.code")["premium"].median()
        .reset_index()
        .rename(columns={"premium": "_zip_prem_med"})
    )
    df = df.merge(zip_med, on="zip.code", how="left")
    df["premium_vs_zip"] = df["premium"].fillna(0) / df["_zip_prem_med"].replace(0, np.nan)
    df = df.drop(columns=["_zip_prem_med"])

    credit_med = (
        ref.groupby("credit")["premium"].median()
        .reset_index()
        .rename(columns={"premium": "_cred_prem_med"})
    )
    df = df.merge(credit_med, on="credit", how="left")
    df["premium_vs_credit"] = df["premium"].fillna(0) / df["_cred_prem_med"].replace(0, np.nan)
    df = df.drop(columns=["_cred_prem_med"])

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


def make_age_credit_key(df_sub: pd.DataFrame) -> pd.Series:
    """Combine age bucket and credit tier into a group key for OOF encoding."""
    age = df_sub["ni.age"]
    age_bucket = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS).astype(object)
    age_bucket = pd.Series(age_bucket, index=df_sub.index).where(~age.isna(), other="unknown")
    credit_str = df_sub["credit"].astype(object).fillna("missing").astype(str)
    return (age_bucket.astype(str) + "_" + credit_str).rename(None)


def make_zip_sales_key(df_sub: pd.DataFrame) -> pd.Series:
    """Combine zip code and sales channel into a group key for OOF encoding."""
    zip_str = df_sub["zip.code"].astype(object).fillna("missing").astype(str)
    chan_str = df_sub["sales.channel"].astype(object).fillna("missing").astype(str)
    return (zip_str + "_" + chan_str).rename(None)


def make_cov_dwell_key(df_sub: pd.DataFrame) -> pd.Series:
    """Combine coverage type and dwelling type into a group key for OOF encoding."""
    cov_str = df_sub["coverage.type"].astype(object).fillna("missing").astype(str)
    dwell_str = df_sub["dwelling.type"].astype(object).fillna("missing").astype(str)
    return (cov_str + "_" + dwell_str).rename(None)


def compute_oof_rates(
    keys: pd.Series, labels: np.ndarray, smooth_k: int = 20
) -> tuple[dict, dict, float, float]:
    """Smoothed cancel-2 and cancel-1 rates per group key (Bayesian smoothing)."""
    global_rate2 = (labels == 2).mean()
    global_rate1 = (labels == 1).mean()
    df = pd.DataFrame({
        "key": keys.values,
        "is2": (labels == 2).astype(float),
        "is1": (labels == 1).astype(float),
    })
    agg = df.groupby("key")[["is2", "is1"]].agg(["sum", "count"])
    agg.columns = ["sum2", "count2", "sum1", "count1"]
    agg["rate2"] = (agg["sum2"] + smooth_k * global_rate2) / (agg["count2"] + smooth_k)
    agg["rate1"] = (agg["sum1"] + smooth_k * global_rate1) / (agg["count1"] + smooth_k)
    return agg["rate2"].to_dict(), agg["rate1"].to_dict(), global_rate2, global_rate1


def build_encoding_maps(
    X_source: pd.DataFrame,
    row_idx: np.ndarray,
    y_source: np.ndarray,
    age_credit_keys: pd.Series,
    zip_sales_keys: pd.Series,
    cov_dwell_keys: pd.Series,
    smooth_k: int,
) -> dict:
    """Build all target-encoding maps using only the selected source rows."""
    y_part = y_source[row_idx]
    zip_maps = compute_oof_rates(
        X_source.iloc[row_idx]["zip.code"].astype(str).reset_index(drop=True),
        y_part, smooth_k,
    )
    ac_maps = compute_oof_rates(
        age_credit_keys.iloc[row_idx].reset_index(drop=True),
        y_part, smooth_k,
    )
    zs_maps = compute_oof_rates(
        zip_sales_keys.iloc[row_idx].reset_index(drop=True),
        y_part, smooth_k,
    )
    cd_maps = compute_oof_rates(
        cov_dwell_keys.iloc[row_idx].reset_index(drop=True),
        y_part, smooth_k,
    )
    return {"zip": zip_maps, "age_credit": ac_maps, "zip_sales": zs_maps, "cov_dwell": cd_maps}


def fill_rate_columns(
    X_target: pd.DataFrame,
    row_idx: np.ndarray,
    zip_keys: pd.Series,
    age_credit_keys: pd.Series,
    zip_sales_keys: pd.Series,
    cov_dwell_keys: pd.Series,
    maps: dict,
) -> None:
    """Fill target-encoding columns for selected rows using already-built maps."""
    zip_map2, zip_map1, gz2, gz1 = maps["zip"]
    ac_map2, ac_map1, ac_gz2, ac_gz1 = maps["age_credit"]
    zs_map2, zs_map1, zs_gz2, zs_gz1 = maps["zip_sales"]
    cd_map2, cd_map1, cd_gz2, cd_gz1 = maps["cov_dwell"]

    X_target.loc[row_idx, "zip_cancel2_rate"] = zip_keys.iloc[row_idx].map(zip_map2).fillna(gz2).values
    X_target.loc[row_idx, "zip_cancel1_rate"] = zip_keys.iloc[row_idx].map(zip_map1).fillna(gz1).values
    X_target.loc[row_idx, "age_credit_cancel2_rate"] = age_credit_keys.iloc[row_idx].map(ac_map2).fillna(ac_gz2).values
    X_target.loc[row_idx, "age_credit_cancel1_rate"] = age_credit_keys.iloc[row_idx].map(ac_map1).fillna(ac_gz1).values
    X_target.loc[row_idx, "zip_sales_cancel2_rate"] = zip_sales_keys.iloc[row_idx].map(zs_map2).fillna(zs_gz2).values
    X_target.loc[row_idx, "zip_sales_cancel1_rate"] = zip_sales_keys.iloc[row_idx].map(zs_map1).fillna(zs_gz1).values
    X_target.loc[row_idx, "cov_dwell_cancel2_rate"] = cov_dwell_keys.iloc[row_idx].map(cd_map2).fillna(cd_gz2).values
    X_target.loc[row_idx, "cov_dwell_cancel1_rate"] = cov_dwell_keys.iloc[row_idx].map(cd_map1).fillna(cd_gz1).values


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

    # Row-level feature engineering (including missingness flags)
    X = add_engineered_features(X)
    X_test = add_engineered_features(X_test)

    # Pre-compute group keys for OOF encoding (done once on full data, no target used)
    age_credit_keys_train = make_age_credit_key(X)
    age_credit_keys_test = make_age_credit_key(X_test)
    zip_sales_keys_train = make_zip_sales_key(X)
    zip_sales_keys_test = make_zip_sales_key(X_test)
    cov_dwell_keys_train = make_cov_dwell_key(X)
    cov_dwell_keys_test = make_cov_dwell_key(X_test)
    zip_keys_train = X["zip.code"].astype(str).reset_index(drop=True)
    zip_keys_test = X_test["zip.code"].astype(str).reset_index(drop=True)

    # Align categoricals
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    log(f"Categorical columns: {cat_cols}", t0)
    X, X_test = align_categoricals(X, X_test, cat_cols)

    # OOF placeholder columns (filled per fold inside the loop — no leakage)
    OOF_COLS = [
        "zip_cancel2_rate", "zip_cancel1_rate",
        "age_credit_cancel2_rate", "age_credit_cancel1_rate",
        "zip_sales_cancel2_rate", "zip_sales_cancel1_rate",
        "cov_dwell_cancel2_rate", "cov_dwell_cancel1_rate",
    ]
    for col in OOF_COLS:
        X[col] = np.nan
        X_test[col] = np.nan  # placeholder; overwritten per fold in X_test_fold

    log(f"Total features: {X.shape[1]}", t0)

    # Guarantee positional RangeIndex so .loc[tr_idx/va_idx] is unambiguous
    X = X.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y = np.asarray(y)
    age_credit_keys_train = age_credit_keys_train.reset_index(drop=True)
    age_credit_keys_test = age_credit_keys_test.reset_index(drop=True)
    zip_sales_keys_train = zip_sales_keys_train.reset_index(drop=True)
    zip_sales_keys_test = zip_sales_keys_test.reset_index(drop=True)
    cov_dwell_keys_train = cov_dwell_keys_train.reset_index(drop=True)
    cov_dwell_keys_test = cov_dwell_keys_test.reset_index(drop=True)
    zip_keys_train = zip_keys_train.reset_index(drop=True)
    zip_keys_test = zip_keys_test.reset_index(drop=True)

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []

    smooth_k = 20

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        log(f"Fold {fold + 1}/{N_FOLDS}...", t0)
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Step 1: Fill training rows via nested OOF — each training row is encoded
        # using rates from the other inner-fold rows, not its own label.
        inner_skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold + 100,
        )
        for inner_tr_pos, inner_va_pos in inner_skf.split(X.iloc[tr_idx], y_tr):
            inner_tr_idx = tr_idx[inner_tr_pos]
            inner_va_idx = tr_idx[inner_va_pos]
            inner_maps = build_encoding_maps(
                X_source=X,
                row_idx=inner_tr_idx,
                y_source=y,
                age_credit_keys=age_credit_keys_train,
                zip_sales_keys=zip_sales_keys_train,
                cov_dwell_keys=cov_dwell_keys_train,
                smooth_k=smooth_k,
            )
            fill_rate_columns(
                X_target=X,
                row_idx=inner_va_idx,
                zip_keys=zip_keys_train,
                age_credit_keys=age_credit_keys_train,
                zip_sales_keys=zip_sales_keys_train,
                cov_dwell_keys=cov_dwell_keys_train,
                maps=inner_maps,
            )

        # Step 2: Build outer maps from the full outer training fold
        outer_maps = build_encoding_maps(
            X_source=X,
            row_idx=tr_idx,
            y_source=y,
            age_credit_keys=age_credit_keys_train,
            zip_sales_keys=zip_sales_keys_train,
            cov_dwell_keys=cov_dwell_keys_train,
            smooth_k=smooth_k,
        )

        # Step 3: Fill validation rows using outer maps only (no leakage)
        fill_rate_columns(
            X_target=X,
            row_idx=va_idx,
            zip_keys=zip_keys_train,
            age_credit_keys=age_credit_keys_train,
            zip_sales_keys=zip_sales_keys_train,
            cov_dwell_keys=cov_dwell_keys_train,
            maps=outer_maps,
        )

        # Step 4: Build fold-specific test copy using outer maps
        X_test_fold = X_test.copy()
        zip_map2, zip_map1, gz2, gz1 = outer_maps["zip"]
        ac_map2, ac_map1, ac_gz2, ac_gz1 = outer_maps["age_credit"]
        zs_map2, zs_map1, zs_gz2, zs_gz1 = outer_maps["zip_sales"]
        cd_map2, cd_map1, cd_gz2, cd_gz1 = outer_maps["cov_dwell"]

        X_test_fold["zip_cancel2_rate"] = zip_keys_test.map(zip_map2).fillna(gz2).values
        X_test_fold["zip_cancel1_rate"] = zip_keys_test.map(zip_map1).fillna(gz1).values
        X_test_fold["age_credit_cancel2_rate"] = age_credit_keys_test.map(ac_map2).fillna(ac_gz2).values
        X_test_fold["age_credit_cancel1_rate"] = age_credit_keys_test.map(ac_map1).fillna(ac_gz1).values
        X_test_fold["zip_sales_cancel2_rate"] = zip_sales_keys_test.map(zs_map2).fillna(zs_gz2).values
        X_test_fold["zip_sales_cancel1_rate"] = zip_sales_keys_test.map(zs_map1).fillna(zs_gz1).values
        X_test_fold["cov_dwell_cancel2_rate"] = cov_dwell_keys_test.map(cd_map2).fillna(cd_gz2).values
        X_test_fold["cov_dwell_cancel1_rate"] = cov_dwell_keys_test.map(cd_map1).fillna(cd_gz1).values

        # Step 5: Train LightGBM
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]

        sample_weights_tr = np.array([CLASS_WEIGHTS[yi] for yi in y_tr])

        dtr = lgb.Dataset(X_tr, y_tr, weight=sample_weights_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)

        model = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=N_ROUNDS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(100)],
        )

        # Steps 6 & 7: Predict on validation and test
        va_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[va_idx] = va_pred
        test_preds += model.predict(X_test_fold, num_iteration=model.best_iteration) / N_FOLDS

        fold_acc = accuracy_score(y_va, va_pred.argmax(axis=1))
        fold_scores.append(fold_acc)
        log(f"  Fold {fold + 1} accuracy: {fold_acc:.5f}  (best iter: {model.best_iteration})", t0)

    # Post-hoc threshold tuning — two-stage grid search on OOF predictions
    raw_acc = accuracy_score(y, oof_preds.argmax(axis=1))
    log(f"Raw OOF accuracy (argmax): {raw_acc:.5f}", t0)

    print("\nTuning multipliers — stage 1: coarse grid (step 0.05, range 0.40-2.00)...")
    best_acc = raw_acc
    best_mult = (1.0, 1.0, 1.0)

    for t1, t2 in product(
        np.arange(0.40, 2.01, 0.05),
        np.arange(0.40, 2.01, 0.05),
    ):
        adjusted = oof_preds * np.array([1.0, t1, t2])
        acc = accuracy_score(y, adjusted.argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)

    coarse_t1, coarse_t2 = best_mult[1], best_mult[2]
    print(f"  Coarse best: c1x{coarse_t1:.2f}, c2x{coarse_t2:.2f}  acc={best_acc:.5f}")

    print("Tuning multipliers — stage 2: fine grid (step 0.01, +-0.10 around coarse best)...")
    t1_lo = max(0.01, coarse_t1 - 0.10)
    t1_hi = min(3.00, coarse_t1 + 0.11)
    t2_lo = max(0.01, coarse_t2 - 0.10)
    t2_hi = min(3.00, coarse_t2 + 0.11)

    for t1, t2 in product(
        np.arange(t1_lo, t1_hi, 0.01),
        np.arange(t2_lo, t2_hi, 0.01),
    ):
        adjusted = oof_preds * np.array([1.0, t1, t2])
        acc = accuracy_score(y, adjusted.argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)

    print(f"  Fine best: c1x{best_mult[1]:.3f}, c2x{best_mult[2]:.3f}  acc={best_acc:.5f}")
    print(f"Best multipliers: class0x{best_mult[0]:.2f}, "
          f"class1x{best_mult[1]:.3f}, class2x{best_mult[2]:.3f}")
    print(f"Tuned OOF accuracy: {best_acc:.5f} (was {raw_acc:.5f})")

    multipliers_arr = np.array(best_mult)
    oof_class = (oof_preds * multipliers_arr).argmax(axis=1)
    test_class = (test_preds * multipliers_arr).argmax(axis=1)

    overall_acc = accuracy_score(y, oof_class)
    cm = confusion_matrix(y, oof_class)

    summary = []
    summary.append("=" * 60)
    summary.append(f"5-fold CV accuracy (mean): {np.mean(fold_scores):.5f}")
    summary.append(f"5-fold CV accuracy (std):  {np.std(fold_scores):.5f}")
    summary.append(f"OOF accuracy (raw argmax): {raw_acc:.5f}")
    summary.append(f"OOF accuracy (tuned):      {overall_acc:.5f}")
    summary.append(f"Multipliers: c0x{best_mult[0]:.2f}  c1x{best_mult[1]:.3f}  c2x{best_mult[2]:.3f}")
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
