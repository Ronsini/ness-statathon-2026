"""
NESS Statathon 2026 — Optuna hyperparameter search (attempt-9)

Uses the full attempt-8 feature pipeline (56 features, nested OOF encoding).
Objective: tuned OOF accuracy (2-stage multiplier search on c1 and c2).

Usage:
  pip install optuna
  python src/tune_params.py

Results saved to: output/optuna_attempt9_results.txt
"""

import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_N = 350_000
N_FOLDS = 3
N_TRIALS = 40
N_ROUNDS = 5000
EARLY_STOPPING = 100
RANDOM_STATE = 42
SMOOTH_K = 20

CLASS_WEIGHTS = {0: 1.0, 1: 1.3, 2: 1.1}

AGE_BINS = [-np.inf, 25, 35, 50, 65, np.inf]
AGE_LABELS = ["lt25", "25-35", "35-50", "50-65", "65plus"]
CREDIT_ORDINAL = {"low": 0, "medium": 1, "high": 2}

OOF_COLS = [
    "zip_cancel2_rate", "zip_cancel1_rate",
    "age_credit_cancel2_rate", "age_credit_cancel1_rate",
    "zip_sales_cancel2_rate", "zip_sales_cancel1_rate",
    "cov_dwell_cancel2_rate", "cov_dwell_cancel1_rate",
]


# -----------------------------------------------------------------------------
# Feature engineering helpers — identical to train_model.py
# -----------------------------------------------------------------------------

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
    zip_med = (
        ref.groupby("zip.code")["premium"].median()
        .reset_index().rename(columns={"premium": "_zip_prem_med"})
    )
    df = df.merge(zip_med, on="zip.code", how="left")
    df["premium_vs_zip"] = df["premium"].fillna(0) / df["_zip_prem_med"].replace(0, np.nan)
    df = df.drop(columns=["_zip_prem_med"])

    credit_med = (
        ref.groupby("credit")["premium"].median()
        .reset_index().rename(columns={"premium": "_cred_prem_med"})
    )
    df = df.merge(credit_med, on="credit", how="left")
    df["premium_vs_credit"] = df["premium"].fillna(0) / df["_cred_prem_med"].replace(0, np.nan)
    df = df.drop(columns=["_cred_prem_med"])

    dwell_med = (
        ref.groupby("dwelling.type")["premium"].median()
        .reset_index().rename(columns={"premium": "_dwell_prem_med"})
    )
    df = df.merge(dwell_med, on="dwelling.type", how="left")
    df["premium_vs_dwelling"] = df["premium"].fillna(0) / df["_dwell_prem_med"].replace(0, np.nan)
    df = df.drop(columns=["_dwell_prem_med"])

    return df


def align_categoricals(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    for c in cat_cols:
        X[c] = pd.Categorical(X[c].astype("string"))
    return X


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


# -----------------------------------------------------------------------------
# Data loading + full pipeline prep
# -----------------------------------------------------------------------------

def load_and_prep():
    train = pd.read_csv(DATA_DIR / "train.csv")
    train = train[train["cancel"].isin([0, 1, 2])].reset_index(drop=True)
    train = train.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)

    y = train["cancel"].astype(int).values
    X = train.drop(columns=["id", "cancel"])

    X = add_group_features(X, X)
    X = add_engineered_features(X)

    age_credit_keys = make_age_credit_key(X)
    zip_sales_keys = make_zip_sales_key(X)
    cov_dwell_keys = make_cov_dwell_key(X)
    zip_keys = X["zip.code"].astype(str).reset_index(drop=True)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X = align_categoricals(X, cat_cols)

    for col in OOF_COLS:
        X[col] = np.nan

    X = X.reset_index(drop=True)
    y = np.asarray(y)
    age_credit_keys = age_credit_keys.reset_index(drop=True)
    zip_sales_keys = zip_sales_keys.reset_index(drop=True)
    cov_dwell_keys = cov_dwell_keys.reset_index(drop=True)
    zip_keys = zip_keys.reset_index(drop=True)

    return X, y, cat_cols, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys


# -----------------------------------------------------------------------------
# Optuna objective
# -----------------------------------------------------------------------------

def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    zip_keys: pd.Series,
    age_credit_keys: pd.Series,
    zip_sales_keys: pd.Series,
    cov_dwell_keys: pd.Series,
) -> float:
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "feature_pre_filter": False,
        "zero_as_missing": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 511),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 2.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 20.0, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12]),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "verbose": -1,
        "n_jobs": -1,
    }

    # Fresh copy per trial so stale OOF values never carry over between trials
    X = X.copy()
    for col in OOF_COLS:
        X[col] = np.nan

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros((len(X), 3))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        y_tr = y[tr_idx]

        # Nested OOF encoding for training rows
        inner_skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold + 100,
        )
        for inner_tr_pos, inner_va_pos in inner_skf.split(X.iloc[tr_idx], y_tr):
            inner_tr_idx = tr_idx[inner_tr_pos]
            inner_va_idx = tr_idx[inner_va_pos]
            inner_maps = build_encoding_maps(
                X, inner_tr_idx, y, age_credit_keys, zip_sales_keys, cov_dwell_keys, SMOOTH_K,
            )
            fill_rate_columns(
                X, inner_va_idx, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys, inner_maps,
            )

        # Outer maps from training fold; fill validation rows
        outer_maps = build_encoding_maps(
            X, tr_idx, y, age_credit_keys, zip_sales_keys, cov_dwell_keys, SMOOTH_K,
        )
        fill_rate_columns(
            X, va_idx, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys, outer_maps,
        )

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        sample_weights_tr = np.array([CLASS_WEIGHTS[yi] for yi in y_tr])

        dtr = lgb.Dataset(X_tr, y_tr, weight=sample_weights_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, y[va_idx], categorical_feature=cat_cols, reference=dtr)

        model = lgb.train(
            params,
            dtr,
            num_boost_round=N_ROUNDS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)],
        )

        oof_preds[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)

    # Two-stage multiplier search on OOF predictions
    raw_acc = accuracy_score(y, oof_preds.argmax(axis=1))
    best_acc = raw_acc
    best_t1, best_t2 = 1.0, 1.0

    for t1, t2 in product(np.arange(0.40, 2.01, 0.05), np.arange(0.40, 2.01, 0.05)):
        acc = accuracy_score(y, (oof_preds * np.array([1.0, t1, t2])).argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_t1, best_t2 = t1, t2

    for t1, t2 in product(
        np.arange(max(0.01, best_t1 - 0.10), min(3.00, best_t1 + 0.11), 0.01),
        np.arange(max(0.01, best_t2 - 0.10), min(3.00, best_t2 + 0.11), 0.01),
    ):
        acc = accuracy_score(y, (oof_preds * np.array([1.0, t1, t2])).argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_t1, best_t2 = t1, t2

    trial.set_user_attr("raw_acc", float(raw_acc))
    trial.set_user_attr("c1_mult", float(best_t1))
    trial.set_user_attr("c2_mult", float(best_t2))

    return best_acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"Loading {SAMPLE_N:,}-row subsample...")
    X, y, cat_cols, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys = load_and_prep()
    print(f"Features: {X.shape[1]}  Rows: {len(X):,}")
    print(f"Running {N_TRIALS} trials  ({N_FOLDS}-fold CV each, N_ROUNDS={N_ROUNDS})...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(
            trial, X, y, cat_cols, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    raw_acc = best.user_attrs["raw_acc"]
    c1_mult = best.user_attrs["c1_mult"]
    c2_mult = best.user_attrs["c2_mult"]

    lines = []
    lines.append("=" * 60)
    lines.append(f"Optuna attempt-9 tuning results")
    lines.append(f"Runtime:               {time.time() - t0:.0f}s")
    lines.append(f"Sample size:           {SAMPLE_N:,} rows ({N_FOLDS}-fold, {N_TRIALS} trials)")
    lines.append(f"Best trial:            #{best.number}")
    lines.append(f"Best raw OOF accuracy: {raw_acc:.5f}")
    lines.append(f"Best tuned OOF accuracy: {best.value:.5f}")
    lines.append(f"Best multipliers:      c1x{c1_mult:.3f}  c2x{c2_mult:.3f}")
    lines.append("=" * 60)
    lines.append("")

    # Top 5 trials by tuned accuracy
    lines.append("Top 5 trials by tuned OOF accuracy:")
    top5 = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    for t in top5:
        lines.append(
            f"  trial #{t.number:3d}: tuned={t.value:.5f}  "
            f"raw={t.user_attrs.get('raw_acc', 0):.5f}  "
            f"c1x{t.user_attrs.get('c1_mult', 1):.3f}  "
            f"c2x{t.user_attrs.get('c2_mult', 1):.3f}"
        )
    lines.append("")

    lines.append("--- Paste into src/train_model.py LGB_PARAMS ---")
    lines.append("LGB_PARAMS = {")
    lines.append('    "objective": "multiclass",')
    lines.append('    "num_class": 3,')
    lines.append('    "metric": "multi_logloss",')
    for k, v in best.params.items():
        if isinstance(v, float):
            lines.append(f'    "{k}": {v:.6f},')
        else:
            lines.append(f'    "{k}": {v},')
    lines.append('    "verbose": -1,')
    lines.append('    "n_jobs": -1,')
    lines.append("}")

    output = "\n".join(lines)
    print("\n" + output)

    out_path = OUTPUT_DIR / "optuna_attempt9_results.txt"
    out_path.write_text(output)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
