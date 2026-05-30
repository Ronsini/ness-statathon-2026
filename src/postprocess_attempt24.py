"""
NESS Statathon 2026 — attempt-27: postprocess attempt-24 LGB meta-stacker with OOF-optimized rules.

Applies three families of flip rules on top of the attempt-24 argmax predictions,
optimized entirely on OOF to avoid test leakage.

Rule A: flip class-0 → class-2  (high risk-score-class2 + prob constraints)
Rule B: flip class-0 → class-1  (high risk-score-class1 + prob constraints)
Rule C: flip class-2 → class-1  (low risk-score-class2 + prob constraints)

Rules applied in order A → B → C; each row flipped at most once.

Outputs:
  output/postprocess_attempt24_results.txt
  output/submission_postprocess_rank{1-5}.csv

Usage:
  python src/postprocess_attempt24.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "output"

ATT24_OOF_BASELINE = 0.75623
ATT24_PUBLIC       = 0.77839

HIGH_CLASS2_ZIPS = {20110, 20140, 20152, 20129, 20104, 20105, 20109, 20102, 20146, 20115, 20107}


# -----------------------------------------------------------------------------
# Risk feature builder
# -----------------------------------------------------------------------------

def compute_margin(probs: np.ndarray) -> np.ndarray:
    s = np.sort(probs, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def build_risk_features(df: pd.DataFrame, probs: np.ndarray) -> dict:
    def col(name, default=""):
        return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)

    # Zip features
    zip_int = pd.to_numeric(col("zip.code"), errors="coerce").fillna(-1).astype(int)
    zip_str = zip_int.where(zip_int < 0, zip_int).astype(str)
    zip_str = zip_str.where(zip_int >= 0, "")
    zip_prefix_3 = zip_str.str[:3].values
    zip_prefix_2 = zip_str.str[:2].values
    is_201_zip    = (zip_prefix_3 == "201")
    high_class2_zip = zip_int.isin(HIGH_CLASS2_ZIPS).values

    # Credit
    credit          = col("credit").fillna("").astype(str)
    is_low_credit   = (credit == "low").values
    is_medium_credit = (credit == "medium").values

    # Channel
    channel             = col("sales.channel").fillna("").astype(str)
    is_phone            = (channel == "Phone").values
    is_online           = (channel == "Online").values
    is_phone_or_online  = is_phone | is_online
    low_credit_phone    = is_low_credit & is_phone
    low_credit_online   = is_low_credit & is_online
    risky_channel_credit = (is_low_credit | is_medium_credit) & (is_phone | is_online)

    # Age
    age        = pd.to_numeric(col("ni.age"), errors="coerce").fillna(999).values
    young      = (age <= 30)
    very_young = (age <= 26)

    # Model uncertainty
    margin    = compute_margin(probs)
    uncertain = (margin < 0.12)

    # Risk scores
    risk_score_class2 = (
        2 * is_low_credit.astype(int)
        + 1 * is_medium_credit.astype(int)
        + 1 * is_phone.astype(int)
        + 1 * is_online.astype(int)
        + 2 * is_201_zip.astype(int)
        + 2 * high_class2_zip.astype(int)
        + 1 * young.astype(int)
        + 1 * very_young.astype(int)
    )

    risk_score_class1 = (
        1 * is_medium_credit.astype(int)
        + 1 * is_phone.astype(int)
        + 1 * is_online.astype(int)
        + 1 * young.astype(int)
        + 1 * uncertain.astype(int)
    )

    return {
        "zip_prefix_3":       zip_prefix_3,
        "zip_prefix_2":       zip_prefix_2,
        "is_201_zip":         is_201_zip,
        "high_class2_zip":    high_class2_zip,
        "is_low_credit":      is_low_credit,
        "is_medium_credit":   is_medium_credit,
        "is_phone":           is_phone,
        "is_online":          is_online,
        "is_phone_or_online": is_phone_or_online,
        "low_credit_phone":   low_credit_phone,
        "low_credit_online":  low_credit_online,
        "risky_channel_credit": risky_channel_credit,
        "young":              young,
        "very_young":         very_young,
        "risk_score_class2":  risk_score_class2,
        "risk_score_class1":  risk_score_class1,
        "uncertain":          uncertain,
        "margin":             margin,
    }


# -----------------------------------------------------------------------------
# Rule application helpers
# -----------------------------------------------------------------------------

def mask_A(base_pred, probs, rf, threshold, p2_min, margin_max):
    return (
        (base_pred == 0)
        & (rf["risk_score_class2"] >= threshold)
        & (probs[:, 2] >= p2_min)
        & (probs[:, 0] - probs[:, 2] <= margin_max)
    )

def mask_B(base_pred, probs, rf, threshold, p1_min, margin_max):
    return (
        (base_pred == 0)
        & (rf["risk_score_class1"] >= threshold)
        & (probs[:, 1] >= p1_min)
        & (probs[:, 0] - probs[:, 1] <= margin_max)
    )

def mask_C(base_pred, probs, rf, threshold, p1_min, margin_max):
    return (
        (base_pred == 2)
        & (rf["risk_score_class2"] <= threshold)
        & (probs[:, 1] >= p1_min)
        & (probs[:, 2] - probs[:, 1] <= margin_max)
    )


def apply_rules(base_pred, probs, rf, rule_specs):
    """Apply rules in order A→B→C; each row flipped at most once."""
    pred    = base_pred.copy()
    flipped = np.zeros(len(pred), dtype=bool)
    total_flips = 0
    for rtype, params in rule_specs:
        if rtype == "A":
            m = mask_A(base_pred, probs, rf, *params) & ~flipped
            pred[m] = 2
        elif rtype == "B":
            m = mask_B(base_pred, probs, rf, *params) & ~flipped
            pred[m] = 1
        elif rtype == "C":
            m = mask_C(base_pred, probs, rf, *params) & ~flipped
            pred[m] = 1
        flipped |= m
        total_flips += m.sum()
    return pred, total_flips


def test_dist_str(preds):
    u, c = np.unique(preds, return_counts=True)
    d = dict(zip(u.tolist(), c.tolist()))
    return f"c0={d.get(0,0)}  c1={d.get(1,0)}  c2={d.get(2,0)}"


# -----------------------------------------------------------------------------
# Grid searches
# -----------------------------------------------------------------------------

THRESHOLDS_A  = [3, 4, 5, 6, 7]
P2_MINS_A     = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
MARGINS_A     = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18]

THRESHOLDS_B  = [2, 3, 4, 5]
P1_MINS_B     = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
MARGINS_B     = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]

THRESHOLDS_C  = [0, 1, 2, 3]
P1_MINS_C     = [0.08, 0.10, 0.12, 0.15]
MARGINS_C     = [0.03, 0.05, 0.07, 0.10]


def grid_search_A(base_pred, oof_probs, rf_oof, y, base_acc):
    best_acc, best_params, best_flips = base_acc, None, 0
    for thr, p2, mg in product(THRESHOLDS_A, P2_MINS_A, MARGINS_A):
        pred, n_flips = apply_rules(base_pred, oof_probs, rf_oof, [("A", (thr, p2, mg))])
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc, best_params, best_flips = acc, (thr, p2, mg), n_flips
    return best_acc, best_params, best_flips


def grid_search_B(base_pred, oof_probs, rf_oof, y, base_acc):
    best_acc, best_params, best_flips = base_acc, None, 0
    for thr, p1, mg in product(THRESHOLDS_B, P1_MINS_B, MARGINS_B):
        pred, n_flips = apply_rules(base_pred, oof_probs, rf_oof, [("B", (thr, p1, mg))])
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc, best_params, best_flips = acc, (thr, p1, mg), n_flips
    return best_acc, best_params, best_flips


def grid_search_C(base_pred, oof_probs, rf_oof, y, base_acc):
    best_acc, best_params, best_flips = base_acc, None, 0
    for thr, p1, mg in product(THRESHOLDS_C, P1_MINS_C, MARGINS_C):
        pred, n_flips = apply_rules(base_pred, oof_probs, rf_oof, [("C", (thr, p1, mg))])
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc, best_params, best_flips = acc, (thr, p1, mg), n_flips
    return best_acc, best_params, best_flips


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Loading data...", flush=True)
    train    = pd.read_csv(DATA_DIR / "train.csv")
    test     = pd.read_csv(DATA_DIR / "test.csv")
    oof_probs  = np.load(OUTPUT_DIR / "stack_meta_lgb_oof_probs.npy")
    test_probs = np.load(OUTPUT_DIR / "stack_meta_lgb_test_probs.npy")
    y          = np.load(OUTPUT_DIR / "y_train.npy")
    test_ids   = np.load(OUTPUT_DIR / "test_ids.npy")

    train = train[train["cancel"].isin([0, 1, 2])].reset_index(drop=True)
    n = min(len(train), len(y), len(oof_probs))
    train      = train.iloc[:n].reset_index(drop=True)
    oof_probs  = oof_probs[:n]
    y          = y[:n]

    base_pred      = oof_probs.argmax(axis=1)
    test_base_pred = test_probs.argmax(axis=1)
    base_acc       = accuracy_score(y, base_pred)

    print(f"Base OOF accuracy:  {base_acc:.5f}", flush=True)
    print(f"Base test dist:     {test_dist_str(test_base_pred)}", flush=True)
    print(f"Attempt-24 OOF:     {ATT24_OOF_BASELINE}  public: {ATT24_PUBLIC}", flush=True)

    print("\nBuilding risk features...", flush=True)
    rf_oof  = build_risk_features(train, oof_probs)
    rf_test = build_risk_features(test, test_probs)

    # -------------------------------------------------------------------------
    # Print risk score distributions
    # -------------------------------------------------------------------------
    for name, arr in [("risk_score_class2 (OOF)", rf_oof["risk_score_class2"]),
                      ("risk_score_class1 (OOF)", rf_oof["risk_score_class1"])]:
        u, c = np.unique(arr, return_counts=True)
        print(f"\n{name} distribution:")
        for v, cnt in zip(u, c):
            print(f"  score={v}: {cnt:,}  ({cnt/len(arr)*100:.2f}%)")

    # -------------------------------------------------------------------------
    # Individual grid searches
    # -------------------------------------------------------------------------
    print("\n" + "="*90, flush=True)
    print("GRID SEARCH — RULE A (flip class-0 → class-2)", flush=True)
    print("="*90, flush=True)
    acc_A, params_A, flips_A_oof = grid_search_A(base_pred, oof_probs, rf_oof, y, base_acc)
    if params_A:
        pred_A_oof, _ = apply_rules(base_pred, oof_probs, rf_oof, [("A", params_A)])
        pred_A_test, flips_A_test = apply_rules(test_base_pred, test_probs, rf_test, [("A", params_A)])
        print(f"  Best A: thr={params_A[0]}  p2_min={params_A[1]}  margin={params_A[2]}")
        print(f"  OOF acc={acc_A:.5f}  gain={acc_A-base_acc:+.5f}  OOF flips={flips_A_oof:,}  "
              f"test flips={flips_A_test:,}")
        print(f"  Test dist: {test_dist_str(pred_A_test)}")
    else:
        print("  Rule A: no improvement found over base.")
        pred_A_test = test_base_pred.copy()

    print("\n" + "="*90, flush=True)
    print("GRID SEARCH — RULE B (flip class-0 → class-1)", flush=True)
    print("="*90, flush=True)
    acc_B, params_B, flips_B_oof = grid_search_B(base_pred, oof_probs, rf_oof, y, base_acc)
    if params_B:
        pred_B_oof, _ = apply_rules(base_pred, oof_probs, rf_oof, [("B", params_B)])
        pred_B_test, flips_B_test = apply_rules(test_base_pred, test_probs, rf_test, [("B", params_B)])
        print(f"  Best B: thr={params_B[0]}  p1_min={params_B[1]}  margin={params_B[2]}")
        print(f"  OOF acc={acc_B:.5f}  gain={acc_B-base_acc:+.5f}  OOF flips={flips_B_oof:,}  "
              f"test flips={flips_B_test:,}")
        print(f"  Test dist: {test_dist_str(pred_B_test)}")
    else:
        print("  Rule B: no improvement found over base.")
        pred_B_test = test_base_pred.copy()

    print("\n" + "="*90, flush=True)
    print("GRID SEARCH — RULE C (flip class-2 → class-1)", flush=True)
    print("="*90, flush=True)
    acc_C, params_C, flips_C_oof = grid_search_C(base_pred, oof_probs, rf_oof, y, base_acc)
    if params_C:
        pred_C_oof, _ = apply_rules(base_pred, oof_probs, rf_oof, [("C", params_C)])
        pred_C_test, flips_C_test = apply_rules(test_base_pred, test_probs, rf_test, [("C", params_C)])
        print(f"  Best C: thr={params_C[0]}  p1_min={params_C[1]}  margin={params_C[2]}")
        print(f"  OOF acc={acc_C:.5f}  gain={acc_C-base_acc:+.5f}  OOF flips={flips_C_oof:,}  "
              f"test flips={flips_C_test:,}")
        print(f"  Test dist: {test_dist_str(pred_C_test)}")
    else:
        print("  Rule C: no improvement found over base.")
        pred_C_test = test_base_pred.copy()

    # -------------------------------------------------------------------------
    # Combination testing
    # -------------------------------------------------------------------------
    print("\n" + "="*90, flush=True)
    print("COMBINATION TESTING", flush=True)
    print("="*90, flush=True)

    combos = []
    if params_A:
        combos.append(("A only",    [("A", params_A)]))
    if params_B:
        combos.append(("B only",    [("B", params_B)]))
    if params_C:
        combos.append(("C only",    [("C", params_C)]))
    if params_A and params_B:
        combos.append(("A + B",     [("A", params_A), ("B", params_B)]))
    if params_A and params_C:
        combos.append(("A + C",     [("A", params_A), ("C", params_C)]))
    if params_B and params_C:
        combos.append(("B + C",     [("B", params_B), ("C", params_C)]))
    if params_A and params_B and params_C:
        combos.append(("A + B + C", [("A", params_A), ("B", params_B), ("C", params_C)]))

    all_results = []

    # Base always included
    all_results.append({
        "desc":       "BASE (no postprocess)",
        "oof_acc":    base_acc,
        "gain":       0.0,
        "oof_flips":  0,
        "test_flips": 0,
        "test_dist":  test_dist_str(test_base_pred),
        "test_preds": test_base_pred.copy(),
    })

    for desc, rule_specs in combos:
        pred_oof, oof_flips = apply_rules(base_pred, oof_probs, rf_oof, rule_specs)
        pred_test, test_flips = apply_rules(test_base_pred, test_probs, rf_test, rule_specs)
        oof_acc = accuracy_score(y, pred_oof)
        print(f"  {desc:<15}  oof_acc={oof_acc:.5f}  gain={oof_acc-base_acc:+.5f}  "
              f"oof_flips={oof_flips:,}  test_flips={test_flips:,}  {test_dist_str(pred_test)}",
              flush=True)
        all_results.append({
            "desc":       desc,
            "oof_acc":    oof_acc,
            "gain":       oof_acc - base_acc,
            "oof_flips":  oof_flips,
            "test_flips": test_flips,
            "test_dist":  test_dist_str(pred_test),
            "test_preds": pred_test,
        })

    # -------------------------------------------------------------------------
    # Ranking and saving
    # -------------------------------------------------------------------------
    all_results.sort(key=lambda r: -r["oof_acc"])

    print("\n" + "="*90, flush=True)
    print("FINAL RANKING", flush=True)
    print("="*90, flush=True)
    print(f"  {'rank':>4}  {'description':<18}  {'oof_acc':>8}  {'gain':>8}  "
          f"{'oof_flip':>8}  {'tst_flip':>8}  test_dist")
    for rank, r in enumerate(all_results, 1):
        print(f"  {rank:>4}  {r['desc']:<18}  {r['oof_acc']:>8.5f}  {r['gain']:>+8.5f}  "
              f"{r['oof_flips']:>8,}  {r['test_flips']:>8,}  {r['test_dist']}")

    beats_att24 = all_results[0]["oof_acc"] > ATT24_OOF_BASELINE
    verdict = (
        f"BEATS attempt-24 OOF by {all_results[0]['oof_acc'] - ATT24_OOF_BASELINE:+.5f}"
        if beats_att24
        else f"Does NOT beat attempt-24 OOF ({all_results[0]['oof_acc'] - ATT24_OOF_BASELINE:+.5f})"
    )
    print(f"\nVerdict: {verdict}", flush=True)
    print(f"Attempt-24 baseline: OOF={ATT24_OOF_BASELINE}  public={ATT24_PUBLIC}", flush=True)

    # Save results file
    results_path = OUTPUT_DIR / "postprocess_attempt24_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Base OOF accuracy: {base_acc:.5f}\n")
        f.write(f"Base test dist: {test_dist_str(test_base_pred)}\n\n")
        if params_A:
            f.write(f"Best A: thr={params_A[0]}  p2_min={params_A[1]}  margin={params_A[2]}"
                    f"  oof_acc={acc_A:.5f}  gain={acc_A-base_acc:+.5f}\n")
        if params_B:
            f.write(f"Best B: thr={params_B[0]}  p1_min={params_B[1]}  margin={params_B[2]}"
                    f"  oof_acc={acc_B:.5f}  gain={acc_B-base_acc:+.5f}\n")
        if params_C:
            f.write(f"Best C: thr={params_C[0]}  p1_min={params_C[1]}  margin={params_C[2]}"
                    f"  oof_acc={acc_C:.5f}  gain={acc_C-base_acc:+.5f}\n")
        f.write(f"\n{'rank':>4}  {'description':<18}  {'oof_acc':>8}  {'gain':>8}  "
                f"{'oof_flip':>8}  {'tst_flip':>8}  test_dist\n")
        f.write("-" * 90 + "\n")
        for rank, r in enumerate(all_results, 1):
            f.write(f"{rank:>4}  {r['desc']:<18}  {r['oof_acc']:>8.5f}  {r['gain']:>+8.5f}  "
                    f"{r['oof_flips']:>8,}  {r['test_flips']:>8,}  {r['test_dist']}\n")
        f.write(f"\nVerdict: {verdict}\n")
    print(f"\nSaved results to {results_path}", flush=True)

    # Save top-5 submissions
    for rank, r in enumerate(all_results[:5], 1):
        sub = pd.DataFrame({"id": test_ids, "Predicted": r["test_preds"]})
        sub_path = OUTPUT_DIR / f"submission_postprocess_rank{rank}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"Saved rank-{rank} submission ({r['desc']}) to {sub_path}", flush=True)


if __name__ == "__main__":
    main()
