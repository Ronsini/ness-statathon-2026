"""
NESS Statathon 2026 — attempt-20: stacking meta-model (LogisticRegression).

Loads saved XGB and LGB OOF probabilities from attempt-18 and trains a
multinomial LogisticRegression meta-model via 5-fold CV. No retraining of
XGB or LGB.

Meta features (6 per row):
  xgb_prob_0, xgb_prob_1, xgb_prob_2,
  lgb_prob_0, lgb_prob_1, lgb_prob_2

Searches over C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
and class_weight in [None, "balanced"].

Outputs:
  output/stack_results.txt       per-config results + best summary
  output/submission_stack.csv    Kaggle submission (id, Predicted)
  output/stack_oof_probs.npy     best-config OOF meta-probs  (n_train, 3)
  output/stack_test_probs.npy    best-config test meta-probs (n_test, 3)

Usage:
  python src/stack_models.py
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
BLEND_OOF_BASELINE = 0.73682  # attempt-19 best blend
XGB_PUBLIC_BEST    = 0.75623  # attempt-14 public

C_VALUES      = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
WEIGHT_OPTIONS = [None, "balanced"]


# -----------------------------------------------------------------------------
# Multiplier tuning (same two-stage search as train_model.py)
# -----------------------------------------------------------------------------

def tune_multipliers(oof_probs: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    raw_acc = accuracy_score(y, oof_probs.argmax(axis=1))
    best_acc = raw_acc
    best_mult = (1.0, 1.0, 1.0)

    for t1, t2 in product(np.arange(0.40, 2.01, 0.05), np.arange(0.40, 2.01, 0.05)):
        acc = accuracy_score(y, (oof_probs * np.array([1.0, t1, t2])).argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)

    c1_lo = max(0.01, best_mult[1] - 0.10)
    c1_hi = min(3.00, best_mult[1] + 0.11)
    c2_lo = max(0.01, best_mult[2] - 0.10)
    c2_hi = min(3.00, best_mult[2] + 0.11)

    for t1, t2 in product(np.arange(c1_lo, c1_hi, 0.01), np.arange(c2_lo, c2_hi, 0.01)):
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
    y        = np.load(OUTPUT_DIR / "y_train.npy")
    test_ids = np.load(OUTPUT_DIR / "test_ids.npy")

    assert xgb_oof.shape == lgb_oof.shape,     f"OOF shape mismatch: {xgb_oof.shape} vs {lgb_oof.shape}"
    assert xgb_test.shape == lgb_test.shape,   f"Test shape mismatch: {xgb_test.shape} vs {lgb_test.shape}"
    assert xgb_oof.shape[0] == len(y),         f"OOF rows {xgb_oof.shape[0]} != y {len(y)}"
    assert xgb_test.shape[0] == len(test_ids), f"Test rows {xgb_test.shape[0]} != test_ids {len(test_ids)}"
    assert xgb_oof.shape[1] == 3

    # Build meta features
    meta_train = np.hstack([xgb_oof, lgb_oof])    # (n_train, 6)
    meta_test  = np.hstack([xgb_test, lgb_test])  # (n_test, 6)

    print(f"Meta train: {meta_train.shape}  Meta test: {meta_test.shape}", flush=True)
    print(f"Baselines: XGB OOF={XGB_OOF_BASELINE}  Blend OOF={BLEND_OOF_BASELINE}  XGB public={XGB_PUBLIC_BEST}", flush=True)
    print("-" * 95, flush=True)

    header = (
        f"{'C':>6}  {'weight':>10}  {'raw':>8}  {'tuned':>8}  "
        f"{'c1':>6}  {'c2':>6}  {'pred_0':>7}  {'pred_1':>7}  {'pred_2':>7}"
    )
    print(header, flush=True)
    print("-" * 95, flush=True)

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
                multi_class="multinomial",
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
            f"{C:>6.3f}  {weight_str:>10}  {raw_acc:>8.5f}  {tuned_acc:>8.5f}  "
            f"{best_mult[1]:>6.3f}  {best_mult[2]:>6.3f}  "
            f"{p0:>7d}  {p1:>7d}  {p2:>7d}"
        )
        print(line, flush=True)
        results.append((C, class_weight, raw_acc, tuned_acc, best_mult, oof_meta_probs, test_meta_probs))

    # Best by tuned OOF accuracy
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

    beats_blend = best_tuned > BLEND_OOF_BASELINE
    beats_xgb   = best_tuned > XGB_OOF_BASELINE
    if beats_blend:
        verdict = f"BEATS attempt-19 blend OOF by {best_tuned - BLEND_OOF_BASELINE:+.5f}"
    elif beats_xgb:
        verdict = f"Beats XGB-only OOF by {best_tuned - XGB_OOF_BASELINE:+.5f} but not blend ({best_tuned - BLEND_OOF_BASELINE:+.5f})"
    else:
        verdict = f"Does NOT beat XGB OOF ({best_tuned - XGB_OOF_BASELINE:+.5f} vs baseline)"

    summary_lines = [
        "=" * 95,
        f"Best config:   C={best_C}  class_weight={best_weight}",
        f"Raw OOF:       {best_raw:.5f}",
        f"Tuned OOF:     {best_tuned:.5f}",
        f"Multipliers:   c0x1.00  c1x{best_mult[1]:.3f}  c2x{best_mult[2]:.3f}",
        f"",
        f"Per-class recall:  class-0: {per_class_recall[0]*100:.1f}%  "
        f"class-1: {per_class_recall[1]*100:.1f}%  class-2: {per_class_recall[2]*100:.1f}%",
        f"",
        f"Baselines:",
        f"  attempt-14 XGB tuned OOF  : {XGB_OOF_BASELINE}",
        f"  attempt-19 blend tuned OOF: {BLEND_OOF_BASELINE}",
        f"  attempt-14 public         : {XGB_PUBLIC_BEST}",
        f"",
        f"Verdict: {verdict}",
        f"",
        f"Test prediction dist: {test_dist_str}",
        f"",
        f"Confusion matrix (rows=true, cols=pred):",
        str(cm_df),
        "=" * 95,
    ]

    print("\n" + "\n".join(summary_lines), flush=True)

    results_path = OUTPUT_DIR / "stack_results.txt"
    with open(results_path, "w") as f:
        f.write(header + "\n")
        f.write("-" * 95 + "\n")
        for C, cw, raw_acc, tuned_acc, mult, _, _ in results:
            weight_str = "balanced" if cw == "balanced" else "None    "
            f.write(
                f"{C:>6.3f}  {weight_str:>10}  {raw_acc:>8.5f}  {tuned_acc:>8.5f}  "
                f"{mult[1]:>6.3f}  {mult[2]:>6.3f}\n"
            )
        f.write("\n" + "\n".join(summary_lines) + "\n")
    print(f"\nSaved results to {results_path}", flush=True)

    sub = pd.DataFrame({"id": test_ids, "Predicted": test_preds})
    sub_path = OUTPUT_DIR / "submission_stack.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}", flush=True)

    np.save(OUTPUT_DIR / "stack_oof_probs.npy", best_oof)
    np.save(OUTPUT_DIR / "stack_test_probs.npy", best_test)
    print("Saved stack probability arrays to output/", flush=True)


if __name__ == "__main__":
    main()
