"""
NESS Statathon 2026 — attempt-22: stacking with XGB + LGB + CatBoost + meta features (34 total).

Extends attempt-21 by enriching the meta-feature space beyond raw probabilities:
  - 9  raw probabilities (xgb_0-2, lgb_0-2, cat_0-2)
  - 3  max probability per model
  - 3  margin between top-2 classes per model
  - 3  argmax (predicted class) per model
  - 4  cross-model agreement indicators (all3, xgb==lgb, xgb==cat, lgb==cat)
  - 3  count of models predicting each class
  - 3  mean class probability across models
  - 3  std  class probability across models
  - 3  range (max-min) class probability across models
  Total: 34 meta features

Finer multiplier search vs attempt-21:
  Stage 1: c1 in [1.20, 2.20] step 0.02, c2 in [0.70, 1.20] step 0.02
  Stage 2: ±0.06 around best, step 0.005

Outputs:
  output/stack_cat_meta_results.txt
  output/submission_stack_cat_meta.csv
  output/stack_cat_meta_oof_probs.npy
  output/stack_cat_meta_test_probs.npy

Usage:
  python src/stack_models_cat_meta.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"

N_FOLDS = 5
RANDOM_STATE = 42

XGB_OOF_BASELINE   = 0.73661  # attempt-14
STACK_OOF_BASELINE = 0.73788  # attempt-20 (XGB+LGB)
CAT_OOF_BASELINE   = 0.73856  # attempt-21 (XGB+LGB+CAT)
CAT_PUBLIC_BEST    = 0.76454  # attempt-21 public

C_VALUES       = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
WEIGHT_OPTIONS = [None, "balanced"]


# -----------------------------------------------------------------------------
# Meta-feature construction
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
    mean_probs  = probs_stack.mean(axis=1)                  # (n, 3)
    std_probs   = probs_stack.std(axis=1)                   # (n, 3)
    range_probs = probs_stack.max(axis=1) - probs_stack.min(axis=1)  # (n, 3)

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
# Finer multiplier tuning — focused on known good region
# -----------------------------------------------------------------------------

def tune_multipliers(oof_probs: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    raw_acc = accuracy_score(y, oof_probs.argmax(axis=1))
    best_acc = raw_acc
    best_mult = (1.0, 1.0, 1.0)

    # Stage 1: c1 in [1.20, 2.20] step 0.02, c2 in [0.70, 1.20] step 0.02
    for t1, t2 in product(np.arange(1.20, 2.21, 0.02), np.arange(0.70, 1.21, 0.02)):
        acc = accuracy_score(y, (oof_probs * np.array([1.0, t1, t2])).argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)

    # Stage 2: ±0.06 around best, step 0.005
    c1_lo = max(0.01, best_mult[1] - 0.06)
    c1_hi = min(3.00, best_mult[1] + 0.061)
    c2_lo = max(0.01, best_mult[2] - 0.06)
    c2_hi = min(3.00, best_mult[2] + 0.061)

    for t1, t2 in product(np.arange(c1_lo, c1_hi, 0.005), np.arange(c2_lo, c2_hi, 0.005)):
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

    assert xgb_oof.shape == lgb_oof.shape == cat_oof.shape, \
        f"OOF shape mismatch: {xgb_oof.shape} / {lgb_oof.shape} / {cat_oof.shape}"
    assert xgb_test.shape == lgb_test.shape == cat_test.shape, \
        f"Test shape mismatch: {xgb_test.shape} / {lgb_test.shape} / {cat_test.shape}"
    assert xgb_oof.shape[0] == len(y)
    assert xgb_test.shape[0] == len(test_ids)
    assert xgb_oof.shape[1] == 3

    meta_train = build_meta_features(xgb_oof,  lgb_oof,  cat_oof)
    meta_test  = build_meta_features(xgb_test, lgb_test, cat_test)

    print(f"Meta train: {meta_train.shape}  Meta test: {meta_test.shape}", flush=True)
    print(
        f"Baselines:  XGB={XGB_OOF_BASELINE}  Stack(XGB+LGB)={STACK_OOF_BASELINE}  "
        f"Stack(+CAT)={CAT_OOF_BASELINE}  Best public={CAT_PUBLIC_BEST}",
        flush=True,
    )
    print("-" * 100, flush=True)

    header = (
        f"{'C':>7}  {'weight':>10}  {'raw':>8}  {'tuned':>8}  "
        f"{'c1':>6}  {'c2':>6}  {'pred_0':>7}  {'pred_1':>7}  {'pred_2':>7}"
    )
    print(header, flush=True)
    print("-" * 100, flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for C, class_weight in product(C_VALUES, WEIGHT_OPTIONS):
        oof_meta_probs  = np.zeros((len(y), 3))
        test_meta_probs = np.zeros((len(test_ids), 3))

        for tr_idx, va_idx in skf.split(meta_train, y):
            X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
            y_tr = y[tr_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)
            X_te = scaler.transform(meta_test)

            clf = LogisticRegression(
                solver="lbfgs",
                C=C,
                class_weight=class_weight,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
            clf.fit(X_tr, y_tr)
            oof_meta_probs[va_idx] = clf.predict_proba(X_va)
            test_meta_probs += clf.predict_proba(X_te) / N_FOLDS

        raw_acc = accuracy_score(y, oof_meta_probs.argmax(axis=1))
        tuned_acc, best_mult = tune_multipliers(oof_meta_probs, y)

        preds = (oof_meta_probs * np.array(best_mult)).argmax(axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        p0, p1, p2 = dist.get(0, 0), dist.get(1, 0), dist.get(2, 0)

        weight_str = "balanced" if class_weight == "balanced" else "None    "
        line = (
            f"{C:>7.4f}  {weight_str:>10}  {raw_acc:>8.5f}  {tuned_acc:>8.5f}  "
            f"{best_mult[1]:>6.3f}  {best_mult[2]:>6.3f}  "
            f"{p0:>7d}  {p1:>7d}  {p2:>7d}"
        )
        print(line, flush=True)
        results.append((C, class_weight, raw_acc, tuned_acc, best_mult, oof_meta_probs, test_meta_probs))

    best = max(results, key=lambda r: r[3])
    best_C, best_weight, best_raw, best_tuned, best_mult, best_oof, best_test = best

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

    unique, counts = np.unique(test_preds, return_counts=True)
    test_dist = dict(zip(unique.tolist(), counts.tolist()))
    test_dist_str = "  ".join(
        f"class-{k}: {v} ({v / len(test_preds) * 100:.1f}%)" for k, v in test_dist.items()
    )

    beats_cat  = best_tuned > CAT_OOF_BASELINE
    beats_stack = best_tuned > STACK_OOF_BASELINE
    if beats_cat:
        verdict = f"BEATS attempt-21 XGB+LGB+CAT OOF by {best_tuned - CAT_OOF_BASELINE:+.5f}"
    elif beats_stack:
        verdict = (
            f"Beats attempt-20 XGB+LGB stack by {best_tuned - STACK_OOF_BASELINE:+.5f} "
            f"but not attempt-21 ({best_tuned - CAT_OOF_BASELINE:+.5f})"
        )
    else:
        verdict = f"Does NOT beat attempt-20 stack ({best_tuned - STACK_OOF_BASELINE:+.5f})"

    summary_lines = [
        "=" * 100,
        f"Best config:   C={best_C}  class_weight={best_weight}",
        f"Raw OOF:       {best_raw:.5f}",
        f"Tuned OOF:     {best_tuned:.5f}",
        f"Multipliers:   c0x1.00  c1x{best_mult[1]:.3f}  c2x{best_mult[2]:.3f}",
        f"Meta features: {meta_train.shape[1]} (9 raw + 25 derived)",
        f"",
        f"Per-class recall:  class-0: {per_class_recall[0]*100:.1f}%  "
        f"class-1: {per_class_recall[1]*100:.1f}%  class-2: {per_class_recall[2]*100:.1f}%",
        f"",
        f"Baselines:",
        f"  attempt-14 XGB tuned OOF               : {XGB_OOF_BASELINE}",
        f"  attempt-20 XGB+LGB stack tuned OOF     : {STACK_OOF_BASELINE}",
        f"  attempt-21 XGB+LGB+CAT stack tuned OOF : {CAT_OOF_BASELINE}",
        f"  attempt-21 public                      : {CAT_PUBLIC_BEST}",
        f"",
        f"Verdict: {verdict}",
        f"",
        f"Test prediction dist: {test_dist_str}",
        f"",
        f"Confusion matrix (rows=true, cols=pred):",
        str(cm_df),
        "=" * 100,
    ]

    print("\n" + "\n".join(summary_lines), flush=True)

    results_path = OUTPUT_DIR / "stack_cat_meta_results.txt"
    with open(results_path, "w") as f:
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        for C, cw, raw_acc, tuned_acc, mult, _, _ in results:
            weight_str = "balanced" if cw == "balanced" else "None    "
            f.write(
                f"{C:>7.4f}  {weight_str:>10}  {raw_acc:>8.5f}  {tuned_acc:>8.5f}  "
                f"{mult[1]:>6.3f}  {mult[2]:>6.3f}\n"
            )
        f.write("\n" + "\n".join(summary_lines) + "\n")
    print(f"\nSaved results to {results_path}", flush=True)

    sub = pd.DataFrame({"id": test_ids, "Predicted": test_preds})
    sub_path = OUTPUT_DIR / "submission_stack_cat_meta.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}", flush=True)

    np.save(OUTPUT_DIR / "stack_cat_meta_oof_probs.npy",  best_oof)
    np.save(OUTPUT_DIR / "stack_cat_meta_test_probs.npy", best_test)
    print("Saved stack_cat_meta probability arrays to output/", flush=True)


if __name__ == "__main__":
    main()
