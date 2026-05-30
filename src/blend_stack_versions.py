"""
NESS Statathon 2026 — attempt-23: blend attempt-21 and attempt-22 stack probabilities.

attempt-21 (stack_cat):      9 raw meta features — better generalization
attempt-22 (stack_cat_meta): 34 meta features   — useful signal but may overfit OOF

final_oof  = w * stack_cat_oof  + (1 - w) * stack_cat_meta_oof
final_test = w * stack_cat_test + (1 - w) * stack_cat_meta_test

Search w in [0.70, 1.00] step 0.01.
Multiplier search: c1 in [1.45, 1.85] step 0.01, c2 in [0.85, 1.05] step 0.01.

Attempt-21 reference test distribution: class-0 1998, class-1 210, class-2 204
Attempt-21 public: 0.76454

Outputs:
  output/stack_blend_21_22_results.txt
  output/submission_stack_blend_21_22.csv

Usage:
  python src/blend_stack_versions.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"

CAT_OOF_BASELINE  = 0.73856   # attempt-21
CAT_PUBLIC_BEST   = 0.76454   # attempt-21
ATT21_TEST_DIST   = {0: 1998, 1: 210, 2: 204}  # attempt-21 reference


def tune_multipliers(oof_probs: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    raw_acc = accuracy_score(y, oof_probs.argmax(axis=1))
    best_acc = raw_acc
    best_mult = (1.0, 1.0, 1.0)

    for t1, t2 in product(np.arange(1.45, 1.86, 0.01), np.arange(0.85, 1.06, 0.01)):
        acc = accuracy_score(y, (oof_probs * np.array([1.0, t1, t2])).argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_mult = (1.0, t1, t2)

    return best_acc, best_mult


def main():
    print("Loading probability arrays...", flush=True)
    cat_oof      = np.load(OUTPUT_DIR / "stack_cat_oof_probs.npy")
    cat_test     = np.load(OUTPUT_DIR / "stack_cat_test_probs.npy")
    meta_oof     = np.load(OUTPUT_DIR / "stack_cat_meta_oof_probs.npy")
    meta_test    = np.load(OUTPUT_DIR / "stack_cat_meta_test_probs.npy")
    y            = np.load(OUTPUT_DIR / "y_train.npy")
    test_ids     = np.load(OUTPUT_DIR / "test_ids.npy")

    assert cat_oof.shape == meta_oof.shape,   f"OOF shape mismatch: {cat_oof.shape} vs {meta_oof.shape}"
    assert cat_test.shape == meta_test.shape, f"Test shape mismatch: {cat_test.shape} vs {meta_test.shape}"
    assert cat_oof.shape[0] == len(y)
    assert cat_test.shape[0] == len(test_ids)

    print(f"OOF shape: {cat_oof.shape}  Test shape: {cat_test.shape}", flush=True)
    print(
        f"Baselines:  attempt-21 OOF={CAT_OOF_BASELINE}  attempt-21 public={CAT_PUBLIC_BEST}",
        flush=True,
    )
    print(
        f"Attempt-21 test dist:  class-0: {ATT21_TEST_DIST[0]}  "
        f"class-1: {ATT21_TEST_DIST[1]}  class-2: {ATT21_TEST_DIST[2]}",
        flush=True,
    )
    print("-" * 100, flush=True)

    header = (
        f"{'w_21':>6}  {'w_22':>6}  {'raw':>8}  {'tuned':>8}  "
        f"{'c1':>6}  {'c2':>6}  {'pred_0':>7}  {'pred_1':>7}  {'pred_2':>7}  {'d0':>5}  {'d1':>5}  {'d2':>5}"
    )
    print(header, flush=True)
    print("-" * 100, flush=True)

    results = []
    w_values = np.round(np.linspace(0.70, 1.00, 31), 2)

    for w in w_values:
        blend_oof  = w * cat_oof  + (1 - w) * meta_oof
        blend_test = w * cat_test + (1 - w) * meta_test

        raw_acc = accuracy_score(y, blend_oof.argmax(axis=1))
        tuned_acc, best_mult = tune_multipliers(blend_oof, y)

        preds = (blend_oof * np.array(best_mult)).argmax(axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        p0, p1, p2 = dist.get(0, 0), dist.get(1, 0), dist.get(2, 0)

        test_preds = (blend_test * np.array(best_mult)).argmax(axis=1)
        tu, tc = np.unique(test_preds, return_counts=True)
        tdist = dict(zip(tu.tolist(), tc.tolist()))
        t0 = tdist.get(0, 0) - ATT21_TEST_DIST[0]
        t1 = tdist.get(1, 0) - ATT21_TEST_DIST[1]
        t2 = tdist.get(2, 0) - ATT21_TEST_DIST[2]

        line = (
            f"{w:>6.2f}  {1-w:>6.2f}  {raw_acc:>8.5f}  {tuned_acc:>8.5f}  "
            f"{best_mult[1]:>6.3f}  {best_mult[2]:>6.3f}  "
            f"{p0:>7d}  {p1:>7d}  {p2:>7d}  "
            f"{t0:>+5d}  {t1:>+5d}  {t2:>+5d}"
        )
        print(line, flush=True)
        results.append((w, raw_acc, tuned_acc, best_mult, blend_oof, blend_test))

    best = max(results, key=lambda r: r[2])
    best_w, best_raw, best_tuned, best_mult, best_oof, best_test = best

    multipliers_arr = np.array(best_mult)
    test_preds = (best_test * multipliers_arr).argmax(axis=1)

    tu, tc = np.unique(test_preds, return_counts=True)
    best_tdist = dict(zip(tu.tolist(), tc.tolist()))
    best_t0 = best_tdist.get(0, 0)
    best_t1 = best_tdist.get(1, 0)
    best_t2 = best_tdist.get(2, 0)

    delta_t0 = best_t0 - ATT21_TEST_DIST[0]
    delta_t1 = best_t1 - ATT21_TEST_DIST[1]
    delta_t2 = best_t2 - ATT21_TEST_DIST[2]

    beats_cat = best_tuned > CAT_OOF_BASELINE
    if beats_cat:
        verdict = f"BEATS attempt-21 OOF by {best_tuned - CAT_OOF_BASELINE:+.5f}"
    else:
        verdict = f"Does NOT beat attempt-21 OOF ({best_tuned - CAT_OOF_BASELINE:+.5f})"

    summary_lines = [
        "=" * 100,
        f"Best blend:    w_21={best_w:.2f}  w_22={1-best_w:.2f}",
        f"Raw OOF:       {best_raw:.5f}",
        f"Tuned OOF:     {best_tuned:.5f}",
        f"Multipliers:   c0x1.00  c1x{best_mult[1]:.3f}  c2x{best_mult[2]:.3f}",
        f"",
        f"Test dist (best blend):  class-0: {best_t0}  class-1: {best_t1}  class-2: {best_t2}",
        f"Attempt-21 test dist:    class-0: {ATT21_TEST_DIST[0]}  class-1: {ATT21_TEST_DIST[1]}  class-2: {ATT21_TEST_DIST[2]}",
        f"Delta vs attempt-21:     class-0: {delta_t0:+d}  class-1: {delta_t1:+d}  class-2: {delta_t2:+d}",
        f"",
        f"Baselines:",
        f"  attempt-21 XGB+LGB+CAT stack tuned OOF : {CAT_OOF_BASELINE}",
        f"  attempt-21 public                      : {CAT_PUBLIC_BEST}",
        f"",
        f"Verdict: {verdict}",
        "=" * 100,
    ]

    print("\n" + "\n".join(summary_lines), flush=True)

    results_path = OUTPUT_DIR / "stack_blend_21_22_results.txt"
    with open(results_path, "w") as f:
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        for w, raw_acc, tuned_acc, mult, _, _ in results:
            blend_oof_r  = w * cat_oof  + (1 - w) * meta_oof
            blend_test_r = w * cat_test + (1 - w) * meta_test
            tp = (blend_test_r * np.array(mult)).argmax(axis=1)
            tu2, tc2 = np.unique(tp, return_counts=True)
            td = dict(zip(tu2.tolist(), tc2.tolist()))
            t0r = td.get(0, 0) - ATT21_TEST_DIST[0]
            t1r = td.get(1, 0) - ATT21_TEST_DIST[1]
            t2r = td.get(2, 0) - ATT21_TEST_DIST[2]
            pp = (blend_oof_r * np.array(mult)).argmax(axis=1)
            pu, pc = np.unique(pp, return_counts=True)
            pd2 = dict(zip(pu.tolist(), pc.tolist()))
            f.write(
                f"{w:>6.2f}  {1-w:>6.2f}  {raw_acc:>8.5f}  {tuned_acc:>8.5f}  "
                f"{mult[1]:>6.3f}  {mult[2]:>6.3f}  "
                f"{pd2.get(0,0):>7d}  {pd2.get(1,0):>7d}  {pd2.get(2,0):>7d}  "
                f"{t0r:>+5d}  {t1r:>+5d}  {t2r:>+5d}\n"
            )
        f.write("\n" + "\n".join(summary_lines) + "\n")
    print(f"\nSaved results to {results_path}", flush=True)

    sub = pd.DataFrame({"id": test_ids, "Predicted": test_preds})
    sub_path = OUTPUT_DIR / "submission_stack_blend_21_22.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}", flush=True)


if __name__ == "__main__":
    main()
