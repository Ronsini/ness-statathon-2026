"""
NESS Statathon 2026 — attempt-24: LightGBM meta-stacker on 34 meta features.

Uses the same 34 meta features as attempt-22 but trains a regularized LightGBM
meta-model instead of LogisticRegression. Heavy regularization (reg_lambda=20,
min_child_samples=200, small num_leaves) prevents overfitting on the meta-feature space.

Meta features (34 total):
  9  raw probabilities (xgb_0-2, lgb_0-2, cat_0-2)
  3  max probability per model
  3  margin between top-2 classes per model
  3  argmax (predicted class) per model
  4  cross-model agreement indicators
  3  count of models predicting each class
  3  mean class probability across models
  3  std  class probability across models
  3  range class probability across models

Grid search over 3 configs (num_leaves x max_depth).

Outputs:
  output/stack_meta_lgb_results.txt
  output/submission_stack_meta_lgb.csv
  output/stack_meta_lgb_oof_probs.npy
  output/stack_meta_lgb_test_probs.npy

Usage:
  python src/stack_meta_lgb.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"

N_FOLDS = 5
RANDOM_STATE = 42

ATT21_OOF_BASELINE = 0.73856  # attempt-21
ATT23_OOF_BASELINE = 0.73983  # attempt-23 (current best)
ATT21_PUBLIC       = 0.76454
ATT22_PUBLIC       = 0.76315
ATT23_PUBLIC       = 0.76869  # current best public

CONFIGS = [
    {"num_leaves": 7,  "max_depth": 3},
    {"num_leaves": 15, "max_depth": 4},
    {"num_leaves": 31, "max_depth": 5},
]

LGB_BASE_PARAMS = {
    "objective":          "multiclass",
    "num_class":          3,
    "metric":             "multi_logloss",
    "learning_rate":      0.02,
    "n_estimators":       3000,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_lambda":         20,
    "reg_alpha":          5,
    "min_child_samples":  200,
    "random_state":       RANDOM_STATE,
    "n_jobs":             -1,
    "verbose":            -1,
}


# -----------------------------------------------------------------------------
# Meta-feature construction (identical to stack_models_cat_meta.py)
# -----------------------------------------------------------------------------

def build_meta_features(xgb_p: np.ndarray, lgb_p: np.ndarray, cat_p: np.ndarray) -> np.ndarray:
    xgb_pred = xgb_p.argmax(axis=1)
    lgb_pred = lgb_p.argmax(axis=1)
    cat_pred = cat_p.argmax(axis=1)

    def max_prob(p):
        return p.max(axis=1, keepdims=True)

    def top2_margin(p):
        s = np.sort(p, axis=1)[:, ::-1]
        return (s[:, 0] - s[:, 1]).reshape(-1, 1)

    xgb_max    = max_prob(xgb_p)
    lgb_max    = max_prob(lgb_p)
    cat_max    = max_prob(cat_p)

    xgb_margin = top2_margin(xgb_p)
    lgb_margin = top2_margin(lgb_p)
    cat_margin = top2_margin(cat_p)

    xgb_pred_f = xgb_pred.reshape(-1, 1).astype(float)
    lgb_pred_f = lgb_pred.reshape(-1, 1).astype(float)
    cat_pred_f = cat_pred.reshape(-1, 1).astype(float)

    all_agree     = ((xgb_pred == lgb_pred) & (lgb_pred == cat_pred)).astype(float).reshape(-1, 1)
    xgb_lgb_agree = (xgb_pred == lgb_pred).astype(float).reshape(-1, 1)
    xgb_cat_agree = (xgb_pred == cat_pred).astype(float).reshape(-1, 1)
    lgb_cat_agree = (lgb_pred == cat_pred).astype(float).reshape(-1, 1)

    preds_stack = np.stack([xgb_pred, lgb_pred, cat_pred], axis=1)
    n_pred_0 = (preds_stack == 0).sum(axis=1, keepdims=True).astype(float)
    n_pred_1 = (preds_stack == 1).sum(axis=1, keepdims=True).astype(float)
    n_pred_2 = (preds_stack == 2).sum(axis=1, keepdims=True).astype(float)

    probs_stack = np.stack([xgb_p, lgb_p, cat_p], axis=1)  # (n, 3, 3)
    mean_probs  = probs_stack.mean(axis=1)
    std_probs   = probs_stack.std(axis=1)
    range_probs = probs_stack.max(axis=1) - probs_stack.min(axis=1)

    return np.hstack([
        xgb_p, lgb_p, cat_p,
        xgb_max, lgb_max, cat_max,
        xgb_margin, lgb_margin, cat_margin,
        xgb_pred_f, lgb_pred_f, cat_pred_f,
        all_agree, xgb_lgb_agree, xgb_cat_agree, lgb_cat_agree,
        n_pred_0, n_pred_1, n_pred_2,
        mean_probs, std_probs, range_probs,
    ])


# -----------------------------------------------------------------------------
# Multiplier tuning — focused search around the known good region
# -----------------------------------------------------------------------------

def tune_multipliers(oof_probs: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    raw_acc = accuracy_score(y, oof_probs.argmax(axis=1))
    best_acc = raw_acc
    best_mult = (1.0, 1.0, 1.0)

    for t1, t2 in product(np.arange(1.20, 1.91, 0.01), np.arange(0.80, 1.16, 0.01)):
        acc = accuracy_score(y, (oof_probs * np.array([1.0, t1, t2])).argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)

    return best_acc, best_mult


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Loading probability arrays...", flush=True)
    xgb_oof  = np.load(OUTPUT_DIR / "xgb_oof_probs.npy")
    xgb_test = np.load(OUTPUT_DIR / "xgb_test_probs.npy")
    lgb_oof  = np.load(OUTPUT_DIR / "lgb_oof_probs.npy")
    lgb_test = np.load(OUTPUT_DIR / "lgb_test_probs.npy")
    cat_oof  = np.load(OUTPUT_DIR / "cat_oof_probs.npy")
    cat_test = np.load(OUTPUT_DIR / "cat_test_probs.npy")
    y        = np.load(OUTPUT_DIR / "y_train.npy")
    test_ids = np.load(OUTPUT_DIR / "test_ids.npy")

    assert xgb_oof.shape == lgb_oof.shape == cat_oof.shape
    assert xgb_test.shape == lgb_test.shape == cat_test.shape
    assert xgb_oof.shape[0] == len(y)
    assert xgb_test.shape[0] == len(test_ids)

    meta_train = build_meta_features(xgb_oof,  lgb_oof,  cat_oof)
    meta_test  = build_meta_features(xgb_test, lgb_test, cat_test)

    print(f"Meta train: {meta_train.shape}  Meta test: {meta_test.shape}", flush=True)
    print(
        f"Baselines:  att21 OOF={ATT21_OOF_BASELINE}  att23 OOF={ATT23_OOF_BASELINE}  "
        f"att23 public={ATT23_PUBLIC}",
        flush=True,
    )
    print("-" * 105, flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    for cfg in CONFIGS:
        cfg_name = f"leaves={cfg['num_leaves']} depth={cfg['max_depth']}"
        print(f"\n--- Config: {cfg_name} ---", flush=True)

        params = {**LGB_BASE_PARAMS, **cfg}
        oof_meta_probs  = np.zeros((len(y), 3))
        test_meta_probs = np.zeros((len(test_ids), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(meta_train, y), 1):
            X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            clf = LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(500)],
            )

            oof_meta_probs[va_idx] = clf.predict_proba(X_va)
            test_meta_probs += clf.predict_proba(meta_test) / N_FOLDS

            fold_acc = accuracy_score(y_va, clf.predict_proba(X_va).argmax(axis=1))
            best_iter = clf.best_iteration_
            print(f"  fold {fold}  acc={fold_acc:.5f}  best_iter={best_iter}", flush=True)

        raw_acc = accuracy_score(y, oof_meta_probs.argmax(axis=1))
        tuned_acc, best_mult = tune_multipliers(oof_meta_probs, y)

        preds = (oof_meta_probs * np.array(best_mult)).argmax(axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))

        test_preds = (test_meta_probs * np.array(best_mult)).argmax(axis=1)
        tu, tc = np.unique(test_preds, return_counts=True)
        tdist = dict(zip(tu.tolist(), tc.tolist()))

        print(
            f"  raw={raw_acc:.5f}  tuned={tuned_acc:.5f}  "
            f"c1={best_mult[1]:.3f}  c2={best_mult[2]:.3f}",
            flush=True,
        )
        print(
            f"  OOF dist   class-0: {dist.get(0,0)}  class-1: {dist.get(1,0)}  class-2: {dist.get(2,0)}",
            flush=True,
        )
        print(
            f"  Test dist  class-0: {tdist.get(0,0)}  class-1: {tdist.get(1,0)}  class-2: {tdist.get(2,0)}",
            flush=True,
        )

        all_results.append((cfg_name, raw_acc, tuned_acc, best_mult, oof_meta_probs.copy(), test_meta_probs.copy()))

    # Best config
    best = max(all_results, key=lambda r: r[2])
    best_cfg, best_raw, best_tuned, best_mult, best_oof, best_test = best

    multipliers_arr = np.array(best_mult)
    oof_preds  = (best_oof  * multipliers_arr).argmax(axis=1)
    test_preds = (best_test * multipliers_arr).argmax(axis=1)

    cm = confusion_matrix(y, oof_preds)
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    )
    per_class_recall = cm.diagonal() / cm.sum(axis=1)

    tu, tc = np.unique(test_preds, return_counts=True)
    best_tdist = dict(zip(tu.tolist(), tc.tolist()))
    test_dist_str = "  ".join(
        f"class-{k}: {v} ({v / len(test_preds) * 100:.1f}%)" for k, v in best_tdist.items()
    )

    beats_att23 = best_tuned > ATT23_OOF_BASELINE
    beats_att21 = best_tuned > ATT21_OOF_BASELINE
    if beats_att23:
        verdict = f"BEATS attempt-23 OOF by {best_tuned - ATT23_OOF_BASELINE:+.5f}"
    elif beats_att21:
        verdict = (
            f"Beats attempt-21 OOF by {best_tuned - ATT21_OOF_BASELINE:+.5f} "
            f"but not attempt-23 ({best_tuned - ATT23_OOF_BASELINE:+.5f})"
        )
    else:
        verdict = f"Does NOT beat attempt-21 OOF ({best_tuned - ATT21_OOF_BASELINE:+.5f})"

    summary_lines = [
        "=" * 105,
        f"Best config:   {best_cfg}",
        f"Raw OOF:       {best_raw:.5f}",
        f"Tuned OOF:     {best_tuned:.5f}",
        f"Multipliers:   c0x1.00  c1x{best_mult[1]:.3f}  c2x{best_mult[2]:.3f}",
        f"",
        f"Per-class recall:  class-0: {per_class_recall[0]*100:.1f}%  "
        f"class-1: {per_class_recall[1]*100:.1f}%  class-2: {per_class_recall[2]*100:.1f}%",
        f"",
        f"Baselines:",
        f"  attempt-21 OOF  : {ATT21_OOF_BASELINE}  public: {ATT21_PUBLIC}",
        f"  attempt-22       public: {ATT22_PUBLIC}",
        f"  attempt-23 OOF  : {ATT23_OOF_BASELINE}  public: {ATT23_PUBLIC}",
        f"",
        f"Verdict: {verdict}",
        f"",
        f"Test prediction dist: {test_dist_str}",
        f"",
        f"Confusion matrix (rows=true, cols=pred):",
        str(cm_df),
        "=" * 105,
    ]

    print("\n" + "\n".join(summary_lines), flush=True)

    results_path = OUTPUT_DIR / "stack_meta_lgb_results.txt"
    with open(results_path, "w") as f:
        for cfg_name, raw_acc, tuned_acc, mult, _, _ in all_results:
            f.write(
                f"{cfg_name:<25}  raw={raw_acc:.5f}  tuned={tuned_acc:.5f}  "
                f"c1={mult[1]:.3f}  c2={mult[2]:.3f}\n"
            )
        f.write("\n" + "\n".join(summary_lines) + "\n")
    print(f"\nSaved results to {results_path}", flush=True)

    sub = pd.DataFrame({"id": test_ids, "Predicted": test_preds})
    sub_path = OUTPUT_DIR / "submission_stack_meta_lgb.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}", flush=True)

    np.save(OUTPUT_DIR / "stack_meta_lgb_oof_probs.npy",  best_oof)
    np.save(OUTPUT_DIR / "stack_meta_lgb_test_probs.npy", best_test)
    print("Saved stack_meta_lgb probability arrays to output/", flush=True)


if __name__ == "__main__":
    main()
