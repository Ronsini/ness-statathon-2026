"""
NESS Statathon 2026 — attempt-26: LightGBM meta-stacker grid search (6 configs).

Extends attempt-24 by searching a wider regularization grid with higher num_leaves.
Uses the same 34 meta features from attempt-22/24.

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

Outputs:
  output/lgb_meta_grid_results.txt
  output/submission_lgb_meta_grid_rank1.csv
  output/submission_lgb_meta_grid_rank2.csv
  output/submission_lgb_meta_grid_rank3.csv
  output/lgb_meta_grid_rank1_oof_probs.npy
  output/lgb_meta_grid_rank1_test_probs.npy

Usage:
  python src/stack_meta_lgb_grid.py
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

ATT23_OOF_BASELINE = 0.73983  # attempt-23
ATT24_OOF_BASELINE = 0.75623  # attempt-24 (current best OOF)
ATT24_PUBLIC       = 0.77839  # attempt-24 (current best public)

CONFIGS = [
    {"name": "safe_15_d4",           "num_leaves": 15, "max_depth": 4, "reg_lambda": 20, "reg_alpha": 5, "min_child_samples": 200},
    {"name": "less_reg_15_d4",       "num_leaves": 15, "max_depth": 4, "reg_lambda": 10, "reg_alpha": 2, "min_child_samples": 100},
    {"name": "less_reg_31_d5",       "num_leaves": 31, "max_depth": 5, "reg_lambda": 10, "reg_alpha": 2, "min_child_samples": 100},
    {"name": "mid_31_d5",            "num_leaves": 31, "max_depth": 5, "reg_lambda":  5, "reg_alpha": 1, "min_child_samples":  75},
    {"name": "strong_63_d6",         "num_leaves": 63, "max_depth": 6, "reg_lambda": 10, "reg_alpha": 2, "min_child_samples": 100},
    {"name": "strong_63_d6_lessreg", "num_leaves": 63, "max_depth": 6, "reg_lambda":  5, "reg_alpha": 1, "min_child_samples":  75},
]

LGB_BASE_PARAMS = {
    "objective":        "multiclass",
    "num_class":        3,
    "metric":           "multi_logloss",
    "learning_rate":    0.02,
    "n_estimators":     4000,
    "subsample":        0.85,
    "colsample_bytree": 0.9,
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbose":          -1,
}


# -----------------------------------------------------------------------------
# Meta-feature construction (identical to stack_meta_lgb.py)
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
# Multiplier tuning
# -----------------------------------------------------------------------------

def tune_multipliers(oof_probs: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    raw_acc = accuracy_score(y, oof_probs.argmax(axis=1))
    best_acc = raw_acc
    best_mult = (1.0, 1.0, 1.0)

    for t1, t2 in product(np.arange(0.80, 1.31, 0.01), np.arange(0.80, 1.21, 0.01)):
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
        f"Baselines:  att23 OOF={ATT23_OOF_BASELINE}  att24 OOF={ATT24_OOF_BASELINE}  "
        f"att24 public={ATT24_PUBLIC}",
        flush=True,
    )
    print("=" * 110, flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        print(f"\n{'='*110}", flush=True)
        print(f"CONFIG: {cfg_name}  |  leaves={cfg['num_leaves']}  depth={cfg['max_depth']}  "
              f"lambda={cfg['reg_lambda']}  alpha={cfg['reg_alpha']}  min_child={cfg['min_child_samples']}",
              flush=True)
        print("-" * 110, flush=True)

        params = {
            **LGB_BASE_PARAMS,
            "num_leaves":        cfg["num_leaves"],
            "max_depth":         cfg["max_depth"],
            "reg_lambda":        cfg["reg_lambda"],
            "reg_alpha":         cfg["reg_alpha"],
            "min_child_samples": cfg["min_child_samples"],
        }

        oof_meta_probs  = np.zeros((len(y), 3))
        test_meta_probs = np.zeros((len(test_ids), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(meta_train, y), 1):
            X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            clf = LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[early_stopping(150, verbose=False), log_evaluation(1000)],
            )

            oof_meta_probs[va_idx] = clf.predict_proba(X_va)
            test_meta_probs += clf.predict_proba(meta_test) / N_FOLDS

            fold_acc = accuracy_score(y_va, clf.predict_proba(X_va).argmax(axis=1))
            print(f"  fold {fold}  acc={fold_acc:.5f}  best_iter={clf.best_iteration_}", flush=True)

        raw_acc = accuracy_score(y, oof_meta_probs.argmax(axis=1))
        tuned_acc, best_mult = tune_multipliers(oof_meta_probs, y)

        preds = (oof_meta_probs * np.array(best_mult)).argmax(axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        oof_dist = dict(zip(unique.tolist(), counts.tolist()))

        test_preds = (test_meta_probs * np.array(best_mult)).argmax(axis=1)
        tu, tc = np.unique(test_preds, return_counts=True)
        test_dist = dict(zip(tu.tolist(), tc.tolist()))

        delta = tuned_acc - ATT24_OOF_BASELINE
        print(
            f"\n  raw={raw_acc:.5f}  tuned={tuned_acc:.5f}  "
            f"vs att24={delta:+.5f}  c1={best_mult[1]:.3f}  c2={best_mult[2]:.3f}",
            flush=True,
        )
        print(
            f"  OOF  dist  class-0: {oof_dist.get(0,0):,}  "
            f"class-1: {oof_dist.get(1,0):,}  class-2: {oof_dist.get(2,0):,}",
            flush=True,
        )
        print(
            f"  Test dist  class-0: {test_dist.get(0,0):,}  "
            f"class-1: {test_dist.get(1,0):,}  class-2: {test_dist.get(2,0):,}",
            flush=True,
        )

        all_results.append({
            "name":      cfg_name,
            "raw":       raw_acc,
            "tuned":     tuned_acc,
            "mult":      best_mult,
            "oof_probs": oof_meta_probs.copy(),
            "test_probs": test_meta_probs.copy(),
        })

    # Rank by tuned OOF
    all_results.sort(key=lambda r: -r["tuned"])

    print("\n\n" + "=" * 110, flush=True)
    print("FINAL RANKING (by tuned OOF):", flush=True)
    print(f"  {'rank':>4}  {'config':<30}  {'raw':>8}  {'tuned':>8}  {'vs_att24':>10}  {'c1':>6}  {'c2':>6}")
    for rank, r in enumerate(all_results, 1):
        delta = r["tuned"] - ATT24_OOF_BASELINE
        print(
            f"  {rank:>4}  {r['name']:<30}  {r['raw']:>8.5f}  {r['tuned']:>8.5f}  "
            f"{delta:>+10.5f}  {r['mult'][1]:>6.3f}  {r['mult'][2]:>6.3f}",
            flush=True,
        )

    # Best config confusion matrix
    best = all_results[0]
    mult_arr = np.array(best["mult"])
    oof_preds  = (best["oof_probs"]  * mult_arr).argmax(axis=1)
    test_preds = (best["test_probs"] * mult_arr).argmax(axis=1)

    cm = confusion_matrix(y, oof_preds)
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    )
    per_class_recall = cm.diagonal() / cm.sum(axis=1)

    tu, tc = np.unique(test_preds, return_counts=True)
    best_tdist = dict(zip(tu.tolist(), tc.tolist()))

    beats_att24 = best["tuned"] > ATT24_OOF_BASELINE
    if beats_att24:
        verdict = f"BEATS attempt-24 OOF by {best['tuned'] - ATT24_OOF_BASELINE:+.5f}"
    else:
        verdict = f"Does NOT beat attempt-24 OOF ({best['tuned'] - ATT24_OOF_BASELINE:+.5f})"

    summary_lines = [
        "=" * 110,
        f"Best config:   {best['name']}",
        f"Raw OOF:       {best['raw']:.5f}",
        f"Tuned OOF:     {best['tuned']:.5f}",
        f"Multipliers:   c0x1.00  c1x{best['mult'][1]:.3f}  c2x{best['mult'][2]:.3f}",
        f"",
        f"Per-class recall:  class-0: {per_class_recall[0]*100:.1f}%  "
        f"class-1: {per_class_recall[1]*100:.1f}%  class-2: {per_class_recall[2]*100:.1f}%",
        f"",
        f"Test dist (best):  class-0: {best_tdist.get(0,0)}  "
        f"class-1: {best_tdist.get(1,0)}  class-2: {best_tdist.get(2,0)}",
        f"",
        f"Baselines:",
        f"  attempt-23 OOF  : {ATT23_OOF_BASELINE}",
        f"  attempt-24 OOF  : {ATT24_OOF_BASELINE}  public: {ATT24_PUBLIC}",
        f"",
        f"Verdict: {verdict}",
        f"",
        f"Confusion matrix (rows=true, cols=pred):",
        str(cm_df),
        "=" * 110,
    ]
    print("\n" + "\n".join(summary_lines), flush=True)

    # Save results file
    results_path = OUTPUT_DIR / "lgb_meta_grid_results.txt"
    with open(results_path, "w") as f:
        f.write(f"{'rank':>4}  {'config':<30}  {'raw':>8}  {'tuned':>8}  {'vs_att24':>10}  {'c1':>6}  {'c2':>6}\n")
        f.write("-" * 80 + "\n")
        for rank, r in enumerate(all_results, 1):
            delta = r["tuned"] - ATT24_OOF_BASELINE
            f.write(
                f"{rank:>4}  {r['name']:<30}  {r['raw']:>8.5f}  {r['tuned']:>8.5f}  "
                f"{delta:>+10.5f}  {r['mult'][1]:>6.3f}  {r['mult'][2]:>6.3f}\n"
            )
        f.write("\n" + "\n".join(summary_lines) + "\n")
    print(f"\nSaved results to {results_path}", flush=True)

    # Save top-3 submissions
    for rank, r in enumerate(all_results[:3], 1):
        mult_arr = np.array(r["mult"])
        tp = (r["test_probs"] * mult_arr).argmax(axis=1)
        sub = pd.DataFrame({"id": test_ids, "Predicted": tp})
        sub_path = OUTPUT_DIR / f"submission_lgb_meta_grid_rank{rank}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"Saved rank-{rank} submission ({r['name']}) to {sub_path}", flush=True)

    # Save rank-1 probability arrays
    np.save(OUTPUT_DIR / "lgb_meta_grid_rank1_oof_probs.npy",  all_results[0]["oof_probs"])
    np.save(OUTPUT_DIR / "lgb_meta_grid_rank1_test_probs.npy", all_results[0]["test_probs"])
    print("Saved rank-1 probability arrays to output/", flush=True)


if __name__ == "__main__":
    main()
