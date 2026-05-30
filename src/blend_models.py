"""
NESS Statathon 2026 — attempt-19: XGBoost + LightGBM soft blend.

Loads saved probability arrays from attempt-18 (no retraining) and searches
for the optimal XGB/LGB blend weight. Tunes class multipliers after blending.

Inputs (from output/):
  xgb_oof_probs.npy    attempt-14 XGBoost OOF probabilities  (n_train, 3)
  xgb_test_probs.npy   attempt-14 XGBoost test probabilities (n_test, 3)
  lgb_oof_probs.npy    attempt-8  LightGBM OOF probabilities (n_train, 3)
  lgb_test_probs.npy   attempt-8  LightGBM test probabilities (n_test, 3)
  y_train.npy          true labels                            (n_train,)
  test_ids.npy         test row ids                           (n_test,)

Outputs (to output/):
  blend_results.txt       full per-weight log + best summary
  submission_blend.csv    Kaggle submission (id, Predicted)
  blend_oof_probs.npy     best-weight blended OOF probs  (n_train, 3)
  blend_test_probs.npy    best-weight blended test probs (n_test, 3)

Usage:
  python src/blend_models.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"

XGB_OOF_BASELINE = 0.73661   # attempt-14 tuned OOF
LGB_OOF_BASELINE = 0.72963   # attempt-8  tuned OOF
XGB_PUBLIC_BEST  = 0.75623   # attempt-14 public leaderboard


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

    # Validate shapes
    assert xgb_oof.shape == lgb_oof.shape,   f"OOF shape mismatch: {xgb_oof.shape} vs {lgb_oof.shape}"
    assert xgb_test.shape == lgb_test.shape, f"Test shape mismatch: {xgb_test.shape} vs {lgb_test.shape}"
    assert xgb_oof.shape[0] == len(y),       f"OOF rows {xgb_oof.shape[0]} != y length {len(y)}"
    assert xgb_test.shape[0] == len(test_ids), f"Test rows {xgb_test.shape[0]} != test_ids length {len(test_ids)}"
    assert xgb_oof.shape[1] == 3,  "OOF probs must have 3 columns"
    assert xgb_test.shape[1] == 3, "Test probs must have 3 columns"

    print(f"OOF shape:    {xgb_oof.shape}  ({len(y):,} rows, 3 classes)", flush=True)
    print(f"Test shape:   {xgb_test.shape}  ({len(test_ids):,} rows, 3 classes)", flush=True)
    print(f"Baselines:    XGB OOF={XGB_OOF_BASELINE}  LGB OOF={LGB_OOF_BASELINE}  XGB public={XGB_PUBLIC_BEST}", flush=True)
    print("-" * 90, flush=True)

    header = (
        f"{'xgb_w':>6}  {'lgb_w':>5}  {'raw':>8}  {'tuned':>8}  "
        f"{'c1':>6}  {'c2':>6}  {'pred_0':>7}  {'pred_1':>7}  {'pred_2':>7}"
    )
    print(header, flush=True)
    print("-" * 90, flush=True)

    results = []
    weights = np.round(np.arange(0.70, 1.01, 0.01), 2)

    for xgb_w in weights:
        lgb_w = round(1.0 - xgb_w, 2)
        blend_oof  = xgb_w * xgb_oof  + lgb_w * lgb_oof
        blend_test = xgb_w * xgb_test + lgb_w * lgb_test

        raw_acc = accuracy_score(y, blend_oof.argmax(axis=1))
        tuned_acc, best_mult = tune_multipliers(blend_oof, y)

        preds = (blend_oof * np.array(best_mult)).argmax(axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        p0 = dist.get(0, 0)
        p1 = dist.get(1, 0)
        p2 = dist.get(2, 0)

        line = (
            f"{xgb_w:>6.2f}  {lgb_w:>5.2f}  {raw_acc:>8.5f}  {tuned_acc:>8.5f}  "
            f"{best_mult[1]:>6.3f}  {best_mult[2]:>6.3f}  "
            f"{p0:>7d}  {p1:>7d}  {p2:>7d}"
        )
        print(line, flush=True)
        results.append((xgb_w, lgb_w, raw_acc, tuned_acc, best_mult, blend_oof, blend_test))

    # Best by tuned OOF accuracy
    best = max(results, key=lambda r: r[3])
    best_xgb_w, best_lgb_w, best_raw, best_tuned, best_mult, best_oof, best_test = best

    multipliers_arr = np.array(best_mult)
    oof_preds  = (best_oof  * multipliers_arr).argmax(axis=1)
    test_preds = (best_test * multipliers_arr).argmax(axis=1)

    cm = confusion_matrix(y, oof_preds)
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    )

    unique, counts = np.unique(test_preds, return_counts=True)
    test_dist = dict(zip(unique.tolist(), counts.tolist()))
    test_dist_str = "  ".join(
        f"class-{k}: {v} ({v / len(test_preds) * 100:.1f}%)" for k, v in test_dist.items()
    )

    beats_xgb = best_tuned > XGB_OOF_BASELINE
    verdict = (
        f"BEATS attempt-14 XGB OOF by +{best_tuned - XGB_OOF_BASELINE:+.5f}"
        if beats_xgb else
        f"DOES NOT beat attempt-14 XGB OOF ({best_tuned - XGB_OOF_BASELINE:+.5f})"
    )

    summary_lines = [
        "=" * 90,
        f"Best blend:    xgb_weight={best_xgb_w:.2f}  lgb_weight={best_lgb_w:.2f}",
        f"Raw OOF:       {best_raw:.5f}",
        f"Tuned OOF:     {best_tuned:.5f}",
        f"Multipliers:   c0x1.00  c1x{best_mult[1]:.3f}  c2x{best_mult[2]:.3f}",
        f"",
        f"Baselines:",
        f"  attempt-14 XGB tuned OOF : {XGB_OOF_BASELINE}",
        f"  attempt-8  LGB tuned OOF : {LGB_OOF_BASELINE}",
        f"  attempt-14 public        : {XGB_PUBLIC_BEST}",
        f"",
        f"Verdict: {verdict}",
        f"",
        f"Test prediction dist: {test_dist_str}",
        f"",
        f"Confusion matrix (rows=true, cols=pred):",
        str(cm_df),
        "=" * 90,
    ]

    print("\n" + "\n".join(summary_lines), flush=True)

    # Save outputs
    results_path = OUTPUT_DIR / "blend_results.txt"
    with open(results_path, "w") as f:
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for xgb_w, lgb_w, raw_acc, tuned_acc, mult, _, _ in results:
            f.write(
                f"{xgb_w:>6.2f}  {lgb_w:>5.2f}  {raw_acc:>8.5f}  {tuned_acc:>8.5f}  "
                f"{mult[1]:>6.3f}  {mult[2]:>6.3f}\n"
            )
        f.write("\n" + "\n".join(summary_lines) + "\n")
    print(f"\nSaved results to {results_path}", flush=True)

    sub = pd.DataFrame({"id": test_ids, "Predicted": test_preds})
    sub_path = OUTPUT_DIR / "submission_blend.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}", flush=True)

    np.save(OUTPUT_DIR / "blend_oof_probs.npy", best_oof)
    np.save(OUTPUT_DIR / "blend_test_probs.npy", best_test)
    print(f"Saved blend probability arrays to output/", flush=True)


if __name__ == "__main__":
    main()
