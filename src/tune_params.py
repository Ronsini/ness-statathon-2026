"""
NESS Statathon 2026 — Lightweight Optuna hyperparameter search.
Runs 15 trials on a 300k subsample to find good LightGBM parameters,
then prints a ready-to-paste LGB_PARAMS block for train_model.py.

Usage:
  pip install optuna
  python src/tune_params.py
"""

import time
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

SAMPLE_N = 300_000
N_FOLDS = 3      # 3-fold for speed during tuning
N_TRIALS = 15
RANDOM_STATE = 42


def load_and_prep() -> tuple[pd.DataFrame, np.ndarray]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    train = train[train["cancel"].isin([0, 1, 2])].reset_index(drop=True)
    train = train.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)

    y = train["cancel"].astype(int).values
    X = train.drop(columns=["id", "cancel"])

    # Minimal feature engineering matching train_model.py
    X["household_size"] = X["n.adults"].fillna(0) + X["n.children"].fillna(0)
    X["premium_per_sqft"] = X["premium"] / X["square_footage"].replace(0, np.nan)
    X["claim_x_tenure"] = X["claim.ind"].fillna(0) * X["tenure"].fillna(0)
    X["log_premium"] = np.log1p(X["premium"].fillna(0))
    X["log_tenure"] = np.log1p(X["tenure"].fillna(0))
    X["new_customer"] = (X["tenure"].fillna(99) < 3).astype(int)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    return X, y, cat_cols


def objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray, cat_cols: list[str]) -> float:
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 511),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 5,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 1.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "verbose": -1,
        "n_jobs": -1,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)

        model = lgb.train(
            params,
            dtr,
            num_boost_round=1000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        va_pred = model.predict(X_va, num_iteration=model.best_iteration)
        scores.append(accuracy_score(y_va, va_pred.argmax(axis=1)))

    return np.mean(scores)


def main():
    t0 = time.time()
    print(f"Loading {SAMPLE_N:,}-row subsample...")
    X, y, cat_cols = load_and_prep()
    print(f"Running {N_TRIALS} Optuna trials ({N_FOLDS}-fold each)...")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, X, y, cat_cols), n_trials=N_TRIALS)

    best = study.best_trial
    print(f"\nBest trial: {best.number}  accuracy={best.value:.5f}  ({time.time()-t0:.0f}s)")
    print("\n--- Paste into train_model.py LGB_PARAMS ---")
    print("LGB_PARAMS = {")
    print('    "objective": "multiclass",')
    print('    "num_class": 3,')
    print('    "metric": "multi_logloss",')
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f'    "{k}": {v:.6f},')
        else:
            print(f'    "{k}": {v},')
    print('    "bagging_freq": 5,')
    print('    "verbose": -1,')
    print('    "n_jobs": -1,')
    print("}")


if __name__ == "__main__":
    main()
