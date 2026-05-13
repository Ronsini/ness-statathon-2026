"""
NESS Statathon 2026 — XGBoost Optuna hyperparameter search.
Runs N_TRIALS Optuna trials on a SAMPLE_N-row subsample with 3-fold CV.
Replicates the exact attempt-14 preprocessing pipeline (no test set needed).

Outputs:
  - output/optuna_xgb_results.txt    Per-trial results (append)
  - output/best_xgb_params.json      Best params ready to paste into train_model.py

Usage:
  python src/tune_xgb_params.py
"""

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
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
RANDOM_STATE = 42

CLASS_WEIGHTS = {0: 1.0, 1: 1.3, 2: 1.1}

# Fixed params — not tuned
FIXED_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_estimators": 10000,
    "early_stopping_rounds": 100,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

HIGH_CARD_COLS = ["zip.code", "house.color", "email_domain"]
LOW_CARD_COLS = [
    "credit", "coverage.type", "dwelling.type", "ni.gender",
    "original_quote_weekday", "season_of_renewal", "sales.channel",
]
OOF_COLS = [
    "zip_cancel2_rate", "zip_cancel1_rate",
    "age_credit_cancel2_rate", "age_credit_cancel1_rate",
    "zip_sales_cancel2_rate", "zip_sales_cancel1_rate",
    "cov_dwell_cancel2_rate", "cov_dwell_cancel1_rate",
]

AGE_BINS = [-np.inf, 25, 35, 50, 65, np.inf]
AGE_LABELS = ["lt25", "25-35", "35-50", "50-65", "65plus"]
CREDIT_ORDINAL = {"low": 0, "medium": 1, "high": 2}


# -----------------------------------------------------------------------------
# Preprocessing (mirrors train_model.py exactly)
# -----------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / "train.csv")
    train = train[train["cancel"].isin([0, 1, 2])].reset_index(drop=True)
    return train


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
    df["is_first_year_with_claim"] = (
        (df["tenure"].fillna(99) < 1.1) & (df["claim.ind"].fillna(0) == 1)
    ).astype(int)
    df["len_at_res_missing"] = df["len.at.res"].isna().astype(int)
    df["sales_channel_missing"] = df["sales.channel"].isna().astype(int)
    df["tenure_missing"] = df["tenure"].isna().astype(int)
    df["premium_missing"] = df["premium"].isna().astype(int)
    df["square_footage_missing"] = df["square_footage"].isna().astype(int)
    df["zip.code"] = df["zip.code"].astype("string")
    return df


def add_group_features(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    for col, grp in [("zip.code", "premium_vs_zip"), ("credit", "premium_vs_credit"),
                     ("dwelling.type", "premium_vs_dwelling")]:
        med = (
            ref.groupby(col)["premium"].median()
            .reset_index()
            .rename(columns={"premium": "_med"})
        )
        df = df.merge(med, on=col, how="left")
        df[grp] = df["premium"].fillna(0) / df["_med"].replace(0, np.nan)
        df = df.drop(columns=["_med"])
    return df


def add_frequency_features(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        x_vals = X[c].astype("string").fillna("missing")
        counts = x_vals.value_counts()
        freqs = counts / len(X)
        X[f"{c}_count"] = x_vals.map(counts).fillna(0).astype(float)
        X[f"{c}_freq"] = x_vals.map(freqs).fillna(0).astype(float)
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
    row_idx: np.ndarray,
    y_source: np.ndarray,
    zip_keys: pd.Series,
    age_credit_keys: pd.Series,
    zip_sales_keys: pd.Series,
    cov_dwell_keys: pd.Series,
    smooth_k: int = 20,
) -> dict:
    y_part = y_source[row_idx]
    return {
        "zip": compute_oof_rates(zip_keys.iloc[row_idx].reset_index(drop=True), y_part, smooth_k),
        "age_credit": compute_oof_rates(age_credit_keys.iloc[row_idx].reset_index(drop=True), y_part, smooth_k),
        "zip_sales": compute_oof_rates(zip_sales_keys.iloc[row_idx].reset_index(drop=True), y_part, smooth_k),
        "cov_dwell": compute_oof_rates(cov_dwell_keys.iloc[row_idx].reset_index(drop=True), y_part, smooth_k),
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
# Multiplier tuning (same two-stage search as train_model.py)
# -----------------------------------------------------------------------------

def tune_multipliers(oof_preds: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    raw_acc = accuracy_score(y, oof_preds.argmax(axis=1))
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

    return best_acc, best_mult


# -----------------------------------------------------------------------------
# Optuna objective
# -----------------------------------------------------------------------------

def make_objective(X, y, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys, results_path):
    def objective(trial: optuna.Trial) -> float:
        params = {
            **FIXED_PARAMS,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        }

        for col in OOF_COLS:
            X[col] = np.nan

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof_preds = np.zeros((len(X), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            y_tr = y[tr_idx]

            # Nested OOF for training rows
            inner_skf = StratifiedKFold(
                n_splits=N_FOLDS, shuffle=True,
                random_state=RANDOM_STATE + fold + 100,
            )
            for inner_tr_pos, inner_va_pos in inner_skf.split(X.iloc[tr_idx], y_tr):
                inner_tr_idx = tr_idx[inner_tr_pos]
                inner_va_idx = tr_idx[inner_va_pos]
                inner_maps = build_encoding_maps(
                    inner_tr_idx, y, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys,
                )
                fill_rate_columns(X, inner_va_idx, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys, inner_maps)

            # Outer maps for validation rows
            outer_maps = build_encoding_maps(
                tr_idx, y, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys,
            )
            fill_rate_columns(X, va_idx, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys, outer_maps)

            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            sample_weights_tr = np.array([CLASS_WEIGHTS[yi] for yi in y_tr])

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                sample_weight=sample_weights_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            oof_preds[va_idx] = model.predict_proba(X_va)

        raw_acc = accuracy_score(y, oof_preds.argmax(axis=1))
        tuned_acc, best_mult = tune_multipliers(oof_preds, y)

        line = (
            f"Trial {trial.number:>3d} | "
            f"raw={raw_acc:.5f} tuned={tuned_acc:.5f} | "
            f"c1x{best_mult[1]:.3f} c2x{best_mult[2]:.3f} | "
            f"lr={params['learning_rate']:.4f} depth={params['max_depth']} "
            f"mcw={params['min_child_weight']} sub={params['subsample']:.2f} "
            f"csb={params['colsample_bytree']:.2f} lam={params['reg_lambda']:.3f} "
            f"alp={params['reg_alpha']:.3f} gam={params['gamma']:.3f} "
            f"bin={params['max_bin']}"
        )
        print(line, flush=True)
        with open(results_path, "a") as f:
            f.write(line + "\n")

        return tuned_acc

    return objective


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"Loading and preparing {SAMPLE_N:,}-row subsample...", flush=True)

    train = load_data()
    train = train.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)

    y = train["cancel"].astype(int).values
    X = train.drop(columns=["id", "cancel"])

    # Group features (computed on subsample)
    X = add_group_features(X, X)

    # Row-level engineering (converts zip.code to string at the end)
    X = add_engineered_features(X)

    # Pre-compute OOF keys before OHE removes the raw categorical columns
    zip_keys = X["zip.code"].astype(str).reset_index(drop=True)
    age_credit_keys = make_age_credit_key(X).reset_index(drop=True)
    zip_sales_keys = make_zip_sales_key(X).reset_index(drop=True)
    cov_dwell_keys = make_cov_dwell_key(X).reset_index(drop=True)

    # Frequency/count encoding for high-cardinality categoricals
    X = add_frequency_features(X, HIGH_CARD_COLS)

    # One-hot encoding for low-cardinality categoricals
    X = pd.get_dummies(X, columns=LOW_CARD_COLS, dummy_na=True)

    # Drop remaining non-numeric columns
    cols_to_drop = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if cols_to_drop:
        print(f"Dropping non-numeric columns: {cols_to_drop}", flush=True)
        X = X.drop(columns=cols_to_drop)

    bad_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if bad_cols:
        raise ValueError(f"Non-numeric columns remain: {bad_cols}")

    X = X.astype(float)

    # OOF placeholder columns (filled per fold inside each trial)
    for col in OOF_COLS:
        X[col] = np.nan

    X = X.reset_index(drop=True)
    y = np.asarray(y)

    print(f"Features: {X.shape[1]}  Rows: {len(X):,}  Folds: {N_FOLDS}  Trials: {N_TRIALS}", flush=True)
    print(f"Prep done in {time.time() - t0:.0f}s", flush=True)
    print("-" * 100, flush=True)

    results_path = OUTPUT_DIR / "optuna_xgb_results.txt"
    with open(results_path, "w") as f:
        import datetime
        f.write(f"Optuna XGBoost search — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"SAMPLE_N={SAMPLE_N:,}  N_FOLDS={N_FOLDS}  N_TRIALS={N_TRIALS}\n")
        f.write("-" * 100 + "\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    objective = make_objective(X, y, zip_keys, age_credit_keys, zip_sales_keys, cov_dwell_keys, results_path)
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_trial
    print("-" * 100, flush=True)
    print(f"Best trial: {best.number}  tuned_acc={best.value:.5f}  total_time={time.time()-t0:.0f}s", flush=True)
    print(f"Best params: {best.params}", flush=True)

    best_params_full = {**FIXED_PARAMS, **best.params}
    params_path = OUTPUT_DIR / "best_xgb_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params_full, f, indent=2)
    print(f"Saved best params to {params_path}", flush=True)

    with open(results_path, "a") as f:
        f.write("-" * 100 + "\n")
        f.write(f"Best trial: {best.number}  tuned_acc={best.value:.5f}\n")
        f.write(f"Best params: {json.dumps(best.params, indent=2)}\n")


if __name__ == "__main__":
    main()
