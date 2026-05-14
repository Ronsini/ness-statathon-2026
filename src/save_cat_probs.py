"""
NESS Statathon 2026 — Save CatBoost probability arrays for stacking (attempt-21).

Mirrors the attempt-8 feature pipeline (same engineered features, same OOF
target encodings, same class weights, same 5-fold CV) but trains CatBoost
with native categorical support instead of LightGBM.

Outputs:
  output/cat_oof_probs.npy   shape (n_train, 3)
  output/cat_test_probs.npy  shape (n_test, 3)
  output/y_train.npy         shape (n_train,)   — shared with XGB/LGB runs
  output/test_ids.npy        shape (n_test,)    — shared with XGB/LGB runs

Usage:
  python src/save_cat_probs.py
"""

import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42

CLASS_WEIGHTS = {0: 1.0, 1: 1.3, 2: 1.1}

CAT_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 8000,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 5,
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "allow_writing_files": False,
}

AGE_BINS = [-np.inf, 25, 35, 50, 65, np.inf]
AGE_LABELS = ["lt25", "25-35", "35-50", "50-65", "65plus"]
CREDIT_ORDINAL = {"low": 0, "medium": 1, "high": 2}

OOF_COLS = [
    "zip_cancel2_rate", "zip_cancel1_rate",
    "age_credit_cancel2_rate", "age_credit_cancel1_rate",
    "zip_sales_cancel2_rate", "zip_sales_cancel1_rate",
    "cov_dwell_cancel2_rate", "cov_dwell_cancel1_rate",
]


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train = train[train["cancel"].isin([0, 1, 2])].reset_index(drop=True)
    return train, test


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["credit_missing"] = df["credit"].isna().astype(int)
    df["ni_age_missing"] = df["ni.age"].isna().astype(int)
    df["n_adults_missing"] = df["n.adults"].isna().astype(int)
    df["coverage_type_missing"] = df["coverage.type"].isna().astype(int)
    df["ni_marital_status_missing"] = df["ni.marital.status"].isna().astype(int)
    df["n_children_missing"] = df["n.children"].isna().astype(int)
    df["zip.code"] = df["zip.code"].astype("string")
    return df


def add_group_features(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    for col, new_col in [
        ("zip.code", "premium_vs_zip"),
        ("credit", "premium_vs_credit"),
        ("dwelling.type", "premium_vs_dwelling"),
    ]:
        med = (
            ref.groupby(col)["premium"].median()
            .reset_index()
            .rename(columns={"premium": "_med"})
        )
        df = df.merge(med, on=col, how="left")
        df[new_col] = df["premium"].fillna(0) / df["_med"].replace(0, np.nan)
        df = df.drop(columns=["_med"])
    return df


def make_age_credit_key(df_sub: pd.DataFrame) -> pd.Series:
    age = df_sub["ni.age"]
    age_bucket = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS).astype(object)
    age_bucket = pd.Series(age_bucket, index=df_sub.index).where(~age.isna(), other="unknown")
    credit_str = df_sub["credit"].astype(object).fillna("missing").astype(str)
    return (age_bucket.astype(str) + "_" + credit_str).rename(None)


def make_zip_sales_key(df_sub: pd.DataFrame) -> pd.Series:
    zip_str = df_sub["zip.code"].astype(object).fillna("missing").astype(str)
    chan_str = df_sub["sales.channel"].astype(object).fillna("missing").astype(str)
    return (zip_str + "_" + chan_str).rename(None)


def make_cov_dwell_key(df_sub: pd.DataFrame) -> pd.Series:
    cov_str = df_sub["coverage.type"].astype(object).fillna("missing").astype(str)
    dwell_str = df_sub["dwelling.type"].astype(object).fillna("missing").astype(str)
    return (cov_str + "_" + dwell_str).rename(None)


def compute_oof_rates(
    keys: pd.Series, labels: np.ndarray, smooth_k: int = 20
) -> tuple[dict, dict, float, float]:
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
    y_part = y_source[row_idx]
    return {
        "zip": compute_oof_rates(
            X_source.iloc[row_idx]["zip.code"].astype(str).reset_index(drop=True),
            y_part, smooth_k,
        ),
        "age_credit": compute_oof_rates(
            age_credit_keys.iloc[row_idx].reset_index(drop=True), y_part, smooth_k,
        ),
        "zip_sales": compute_oof_rates(
            zip_sales_keys.iloc[row_idx].reset_index(drop=True), y_part, smooth_k,
        ),
        "cov_dwell": compute_oof_rates(
            cov_dwell_keys.iloc[row_idx].reset_index(drop=True), y_part, smooth_k,
        ),
    }


def fill_rate_columns(
    X_target: pd.DataFrame,
    row_idx: np.ndarray,
    zip_keys: pd.Series,
    age_credit_keys: pd.Series,
    zip_sales_keys: pd.Series,
    cov_dwell_keys: pd.Series,
    maps: dict,
) -> None:
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


def main():
    t0 = time.time()
    log("Loading data...", t0)
    train, test = load_data()
    log(f"Train rows: {len(train):,}  Test rows: {len(test):,}", t0)

    y = train["cancel"].astype(int).values
    test_ids = test["id"].values

    X = train.drop(columns=["id", "cancel"])
    X_test = test.drop(columns=["id"])

    log("Adding group features...", t0)
    X = add_group_features(X, X)
    X_test = add_group_features(X_test, X)

    X = add_engineered_features(X)
    X_test = add_engineered_features(X_test)

    # Pre-compute OOF keys before any categorical transforms
    age_credit_keys_train = make_age_credit_key(X).reset_index(drop=True)
    age_credit_keys_test = make_age_credit_key(X_test).reset_index(drop=True)
    zip_sales_keys_train = make_zip_sales_key(X).reset_index(drop=True)
    zip_sales_keys_test = make_zip_sales_key(X_test).reset_index(drop=True)
    cov_dwell_keys_train = make_cov_dwell_key(X).reset_index(drop=True)
    cov_dwell_keys_test = make_cov_dwell_key(X_test).reset_index(drop=True)
    zip_keys_train = X["zip.code"].astype(str).reset_index(drop=True)
    zip_keys_test = X_test["zip.code"].astype(str).reset_index(drop=True)

    # Fill NaN in string/object columns with "missing" for CatBoost
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    log(f"Categorical columns ({len(cat_cols)}): {cat_cols}", t0)
    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str)
        X_test[c] = X_test[c].fillna("missing").astype(str)

    # OOF placeholder columns
    for col in OOF_COLS:
        X[col] = np.nan
        X_test[col] = np.nan

    log(f"Total features: {X.shape[1]}", t0)

    X = X.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []
    smooth_k = 20

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        log(f"Fold {fold + 1}/{N_FOLDS}...", t0)
        y_tr = y[tr_idx]

        # Nested OOF encoding for training rows
        inner_skf = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True,
            random_state=RANDOM_STATE + fold + 100,
        )
        for inner_tr_pos, inner_va_pos in inner_skf.split(X.iloc[tr_idx], y_tr):
            inner_tr_idx = tr_idx[inner_tr_pos]
            inner_va_idx = tr_idx[inner_va_pos]
            inner_maps = build_encoding_maps(
                X_source=X, row_idx=inner_tr_idx, y_source=y,
                age_credit_keys=age_credit_keys_train,
                zip_sales_keys=zip_sales_keys_train,
                cov_dwell_keys=cov_dwell_keys_train,
                smooth_k=smooth_k,
            )
            fill_rate_columns(
                X, inner_va_idx, zip_keys_train,
                age_credit_keys_train, zip_sales_keys_train, cov_dwell_keys_train,
                inner_maps,
            )

        outer_maps = build_encoding_maps(
            X_source=X, row_idx=tr_idx, y_source=y,
            age_credit_keys=age_credit_keys_train,
            zip_sales_keys=zip_sales_keys_train,
            cov_dwell_keys=cov_dwell_keys_train,
            smooth_k=smooth_k,
        )
        fill_rate_columns(
            X, va_idx, zip_keys_train,
            age_credit_keys_train, zip_sales_keys_train, cov_dwell_keys_train,
            outer_maps,
        )

        # Fold-specific test encoding using outer maps
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

        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr_fold, y_va = y[tr_idx], y[va_idx]
        sample_weights_tr = np.array([CLASS_WEIGHTS[yi] for yi in y_tr_fold])

        # CatBoost Pools — pass categorical column names directly
        train_pool = Pool(X_tr, label=y_tr_fold, cat_features=cat_cols, weight=sample_weights_tr)
        val_pool   = Pool(X_va, label=y_va,      cat_features=cat_cols)
        test_pool  = Pool(X_test_fold,            cat_features=cat_cols)

        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

        va_pred   = model.predict_proba(val_pool)
        test_pred = model.predict_proba(test_pool)

        oof_preds[va_idx] = va_pred
        test_preds += test_pred / N_FOLDS

        fold_acc = accuracy_score(y_va, va_pred.argmax(axis=1))
        fold_scores.append(fold_acc)
        log(f"  Fold {fold + 1} accuracy: {fold_acc:.5f}  (best iter: {model.best_iteration_})", t0)

    raw_acc = accuracy_score(y, oof_preds.argmax(axis=1))

    # Two-stage multiplier tuning
    best_acc = raw_acc
    best_mult = (1.0, 1.0, 1.0)
    for t1, t2 in product(np.arange(0.40, 2.01, 0.05), np.arange(0.40, 2.01, 0.05)):
        acc = accuracy_score(y, (oof_preds * np.array([1.0, t1, t2])).argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)
    c1_lo = max(0.01, best_mult[1] - 0.10)
    c1_hi = min(3.00, best_mult[1] + 0.11)
    c2_lo = max(0.01, best_mult[2] - 0.10)
    c2_hi = min(3.00, best_mult[2] + 0.11)
    for t1, t2 in product(np.arange(c1_lo, c1_hi, 0.01), np.arange(c2_lo, c2_hi, 0.01)):
        acc = accuracy_score(y, (oof_preds * np.array([1.0, t1, t2])).argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)

    print(f"\nCatBoost 5-fold CV (mean): {np.mean(fold_scores):.5f}  std: {np.std(fold_scores):.5f}")
    print(f"Raw OOF: {raw_acc:.5f}  Tuned OOF: {best_acc:.5f}")
    print(f"Multipliers: c0x{best_mult[0]:.2f}  c1x{best_mult[1]:.3f}  c2x{best_mult[2]:.3f}")

    np.save(OUTPUT_DIR / "cat_oof_probs.npy",  oof_preds)
    np.save(OUTPUT_DIR / "cat_test_probs.npy", test_preds)
    np.save(OUTPUT_DIR / "y_train.npy",        y)
    np.save(OUTPUT_DIR / "test_ids.npy",       test_ids)
    log("Saved CatBoost probability arrays to output/", t0)
    log(f"Total runtime: {time.time() - t0:.0f}s", t0)


if __name__ == "__main__":
    main()
