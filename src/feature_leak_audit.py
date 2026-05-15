"""
NESS Statathon 2026 — attempt-25: feature / leak audit (no model training).

Searches for hidden signal, target leakage, distribution drift, and high-risk
class groups across all features. Output saved to output/feature_leak_audit.txt.

Usage:
  python src/feature_leak_audit.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "cancel"
ID_COL     = "id"

CAT_COLS = [
    "zip.code", "house.color", "credit", "coverage.type", "dwelling.type",
    "sales.channel", "ni.gender", "original_quote_weekday", "season_of_renewal",
    "email_domain",
]

NUM_COLS = [
    "premium", "ni.age", "tenure", "len.at.res", "n.adults",
    "n.children", "square_footage", "num_windows_front",
]

COMBOS = [
    ("zip.code",          "sales.channel"),
    ("zip.code",          "credit"),
    ("zip.code",          "coverage.type"),
    ("email_domain",      "sales.channel"),
    ("credit",            "coverage.type"),
    ("coverage.type",     "dwelling.type"),
    ("season_of_renewal", "sales.channel"),
    ("original_quote_weekday", "sales.channel"),
    ("ni.gender",         "credit"),
    ("house.color",       "zip.code"),
]

_output_lines: list[str] = []


def p(*args, **kwargs) -> None:
    line = " ".join(str(a) for a in args)
    print(line, **kwargs)
    _output_lines.append(line)


def section(title: str) -> None:
    p("\n" + "=" * 100)
    p(f"  {title}")
    p("=" * 100)


def sub(title: str) -> None:
    p(f"\n--- {title} ---")


def safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def class_rates(grp: pd.DataFrame, min_count: int = 0) -> pd.DataFrame:
    counts = grp[TARGET_COL].value_counts().reindex([0, 1, 2], fill_value=0)
    total  = counts.sum()
    if total < min_count:
        return None
    return pd.Series({
        "count":      total,
        "c0_rate":    counts[0] / total,
        "c1_rate":    counts[1] / total,
        "c2_rate":    counts[2] / total,
    })


# =============================================================================
# 1. Basic shape and class distribution
# =============================================================================

def audit_basic(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("1. BASIC SHAPE AND CLASS DISTRIBUTION")
    p(f"Train shape: {train.shape}")
    p(f"Test  shape: {test.shape}")
    p(f"Train columns: {list(train.columns)}")
    p(f"\nTest  columns: {list(test.columns)}")

    if safe_col(train, TARGET_COL):
        vc = train[TARGET_COL].value_counts().sort_index()
        p(f"\nClass distribution (train):")
        for cls, cnt in vc.items():
            p(f"  class {cls}: {cnt:>8,}  ({cnt/len(train)*100:.2f}%)")


# =============================================================================
# 2. ID / order leakage check
# =============================================================================

def audit_id_order(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("2. ID / ORDER LEAKAGE CHECK")
    if not safe_col(train, ID_COL):
        p("No id column found — skipping.")
        return

    tr = train[[ID_COL, TARGET_COL]].sort_values(ID_COL).copy()
    tr["id_bin"] = pd.qcut(tr[ID_COL], q=20, labels=False, duplicates="drop")

    p(f"Train id range: {tr[ID_COL].min():,} – {tr[ID_COL].max():,}")
    if safe_col(test, ID_COL):
        p(f"Test  id range: {test[ID_COL].min():,} – {test[ID_COL].max():,}")
        overlap = test[ID_COL].between(tr[ID_COL].min(), tr[ID_COL].max()).sum()
        p(f"Test ids within train id range: {overlap} / {len(test)}")
        p(f"Test ids beyond  train id range: {(test[ID_COL] > tr[ID_COL].max()).sum()}")

    p(f"\nClass rates by id bin (20 equal-count bins, sorted by id):")
    p(f"  {'bin':>4}  {'id_min':>10}  {'id_max':>10}  {'count':>7}  {'c0':>7}  {'c1':>7}  {'c2':>7}")
    for b, grp in tr.groupby("id_bin", observed=True):
        rates = class_rates(grp)
        if rates is not None:
            id_min = grp[ID_COL].min()
            id_max = grp[ID_COL].max()
            p(f"  {b:>4}  {id_min:>10,}  {id_max:>10,}  {int(rates['count']):>7,}  "
              f"{rates['c0_rate']:>7.4f}  {rates['c1_rate']:>7.4f}  {rates['c2_rate']:>7.4f}")


# =============================================================================
# 3. Missingness audit
# =============================================================================

def audit_missingness(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("3. MISSINGNESS AUDIT")

    miss_train = train.isnull().mean()
    miss_test  = test.isnull().mean()
    drift = (miss_test - miss_train).abs().sort_values(ascending=False)

    p(f"\nTop 30 columns by |test_missing_rate - train_missing_rate|:")
    p(f"  {'column':<35}  {'train_miss':>10}  {'test_miss':>10}  {'delta':>10}")
    for col in drift.head(30).index:
        p(f"  {col:<35}  {miss_train.get(col, 0):>10.4f}  {miss_test.get(col, 0):>10.4f}  "
          f"{(miss_test.get(col, 0) - miss_train.get(col, 0)):>+10.4f}")

    p(f"\nTop 30 missingness indicators by class-rate difference when missing vs not:")
    p(f"  {'column':<35}  {'miss_count':>10}  {'c1_miss':>8}  {'c1_obs':>8}  "
      f"{'c1_diff':>8}  {'c2_miss':>8}  {'c2_obs':>8}  {'c2_diff':>8}")

    miss_signals = []
    for col in train.columns:
        if col in [TARGET_COL, ID_COL]:
            continue
        is_missing = train[col].isnull()
        n_miss = is_missing.sum()
        if n_miss < 100 or n_miss >= len(train) - 100:
            continue
        grp_miss = train.loc[is_missing, TARGET_COL]
        grp_obs  = train.loc[~is_missing, TARGET_COL]
        c1_miss = (grp_miss == 1).mean()
        c1_obs  = (grp_obs  == 1).mean()
        c2_miss = (grp_miss == 2).mean()
        c2_obs  = (grp_obs  == 2).mean()
        score = abs(c1_miss - c1_obs) + abs(c2_miss - c2_obs)
        miss_signals.append((col, n_miss, c1_miss, c1_obs, c2_miss, c2_obs, score))

    miss_signals.sort(key=lambda x: -x[6])
    for col, n_miss, c1m, c1o, c2m, c2o, _ in miss_signals[:30]:
        p(f"  {col:<35}  {n_miss:>10,}  {c1m:>8.4f}  {c1o:>8.4f}  "
          f"{c1m-c1o:>+8.4f}  {c2m:>8.4f}  {c2o:>8.4f}  {c2m-c2o:>+8.4f}")


# =============================================================================
# 4. Single-column target-rate audit (categorical)
# =============================================================================

def audit_categorical(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("4. SINGLE-COLUMN CATEGORICAL TARGET-RATE AUDIT")

    for col in CAT_COLS:
        if not safe_col(train, col):
            p(f"\n[SKIP] {col} not in train")
            continue

        sub(col)
        n_train_uniq = train[col].nunique()
        n_test_uniq  = test[col].nunique() if safe_col(test, col) else "n/a"
        p(f"  Unique values — train: {n_train_uniq}  test: {n_test_uniq}")

        grp = train.groupby(col, observed=True).apply(
            lambda g: pd.Series({
                "count":   len(g),
                "c0_rate": (g[TARGET_COL] == 0).mean(),
                "c1_rate": (g[TARGET_COL] == 1).mean(),
                "c2_rate": (g[TARGET_COL] == 2).mean(),
            }),
            include_groups=False,
        ).sort_values("count", ascending=False)

        p(f"\n  Top 30 values by train count:")
        p(f"  {'value':<30}  {'count':>8}  {'c0':>7}  {'c1':>7}  {'c2':>7}")
        for val, row in grp.head(30).iterrows():
            p(f"  {str(val):<30}  {int(row['count']):>8,}  "
              f"{row['c0_rate']:>7.4f}  {row['c1_rate']:>7.4f}  {row['c2_rate']:>7.4f}")

        high_c1 = grp[(grp["count"] >= 200) & (grp["c1_rate"] >= 0.20)]
        if not high_c1.empty:
            p(f"\n  HIGH CLASS-1 GROUPS (count>=200, c1>=0.20):")
            for val, row in high_c1.iterrows():
                p(f"    {str(val):<30}  count={int(row['count']):,}  c1={row['c1_rate']:.4f}")

        high_c2 = grp[(grp["count"] >= 200) & (grp["c2_rate"] >= 0.35)]
        if not high_c2.empty:
            p(f"\n  HIGH CLASS-2 GROUPS (count>=200, c2>=0.35):")
            for val, row in high_c2.iterrows():
                p(f"    {str(val):<30}  count={int(row['count']):,}  c2={row['c2_rate']:.4f}")

        if safe_col(test, col):
            test_vals = set(test[col].dropna().unique())
            train_vals = set(grp.index)
            only_test = test_vals - train_vals
            if only_test:
                p(f"\n  Values in TEST but not TRAIN ({len(only_test)} values): "
                  f"{sorted(str(v) for v in list(only_test)[:20])}")

            test_freq = test[col].value_counts(normalize=True)
            rare_in_train = test_freq[
                test_freq.index.isin(train_vals) &
                (test_freq > grp.reindex(test_freq.index)["count"].fillna(0) / len(train) * 3)
            ]
            if not rare_in_train.empty:
                p(f"\n  Values much more frequent in TEST than TRAIN (test/train rate ratio > 3x):")
                for val, trate in rare_in_train.head(10).items():
                    trate_train = grp.loc[val, "count"] / len(train) if val in grp.index else 0
                    p(f"    {str(val):<30}  test_rate={trate:.4f}  train_rate={trate_train:.4f}")


# =============================================================================
# 5. Numeric bin target-rate audit
# =============================================================================

def audit_numeric(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("5. NUMERIC BIN TARGET-RATE AUDIT")

    for col in NUM_COLS:
        if not safe_col(train, col):
            p(f"\n[SKIP] {col} not in train")
            continue

        sub(col)
        p(f"  Train: mean={train[col].mean():.2f}  std={train[col].std():.2f}  "
          f"median={train[col].median():.2f}  missing={train[col].isnull().sum():,}")
        if safe_col(test, col):
            p(f"  Test:  mean={test[col].mean():.2f}  std={test[col].std():.2f}  "
              f"median={test[col].median():.2f}  missing={test[col].isnull().sum():,}")

        try:
            train_copy = train[[col, TARGET_COL]].dropna(subset=[col])
            train_copy["bin"], bin_edges = pd.qcut(
                train_copy[col], q=20, retbins=True, duplicates="drop"
            )
        except Exception as e:
            p(f"  [qcut failed: {e}]")
            continue

        grp = train_copy.groupby("bin", observed=True).apply(
            lambda g: pd.Series({
                "count":   len(g),
                "c0_rate": (g[TARGET_COL] == 0).mean(),
                "c1_rate": (g[TARGET_COL] == 1).mean(),
                "c2_rate": (g[TARGET_COL] == 2).mean(),
            }),
            include_groups=False,
        )

        if safe_col(test, col):
            test_binned = pd.cut(test[col].dropna(), bins=bin_edges, include_lowest=True)
            test_counts = test_binned.value_counts().reindex(grp.index, fill_value=0)
        else:
            test_counts = pd.Series(0, index=grp.index)

        p(f"\n  {'bin':<30}  {'tr_count':>8}  {'c0':>7}  {'c1':>7}  {'c2':>7}  {'te_count':>8}")
        for b, row in grp.iterrows():
            tc = test_counts.get(b, 0)
            p(f"  {str(b):<30}  {int(row['count']):>8,}  "
              f"{row['c0_rate']:>7.4f}  {row['c1_rate']:>7.4f}  {row['c2_rate']:>7.4f}  {tc:>8,}")


# =============================================================================
# 6. Pair / combo target-rate audit
# =============================================================================

def audit_combos(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("6. PAIR / COMBO TARGET-RATE AUDIT")

    for col_a, col_b in COMBOS:
        if not safe_col(train, col_a) or not safe_col(train, col_b):
            p(f"\n[SKIP] {col_a} + {col_b} — column missing")
            continue

        sub(f"{col_a} × {col_b}")
        train_copy = train[[col_a, col_b, TARGET_COL]].copy()
        train_copy["combo"] = train_copy[col_a].astype(str) + " | " + train_copy[col_b].astype(str)

        grp = train_copy.groupby("combo", observed=True).apply(
            lambda g: pd.Series({
                "count":   len(g),
                "c1_rate": (g[TARGET_COL] == 1).mean(),
                "c2_rate": (g[TARGET_COL] == 2).mean(),
            }),
            include_groups=False,
        )

        high_c1 = grp[(grp["count"] >= 300) & (grp["c1_rate"] >= 0.20)].sort_values("c1_rate", ascending=False)
        if not high_c1.empty:
            p(f"  HIGH CLASS-1 combos (count>=300, c1>=0.20) — top 15:")
            for combo, row in high_c1.head(15).iterrows():
                p(f"    {combo:<55}  count={int(row['count']):,}  c1={row['c1_rate']:.4f}")

        high_c2 = grp[(grp["count"] >= 300) & (grp["c2_rate"] >= 0.35)].sort_values("c2_rate", ascending=False)
        if not high_c2.empty:
            p(f"  HIGH CLASS-2 combos (count>=300, c2>=0.35) — top 15:")
            for combo, row in high_c2.head(15).iterrows():
                p(f"    {combo:<55}  count={int(row['count']):,}  c2={row['c2_rate']:.4f}")

        if safe_col(test, col_a) and safe_col(test, col_b):
            test_copy = test[[col_a, col_b]].copy()
            test_copy["combo"] = test_copy[col_a].astype(str) + " | " + test_copy[col_b].astype(str)
            test_freq = test_copy["combo"].value_counts().head(20)
            p(f"  Top 20 test combos by frequency:")
            p(f"  {'combo':<55}  {'test_count':>10}  {'c1_rate':>8}  {'c2_rate':>8}")
            for combo, cnt in test_freq.items():
                if combo in grp.index:
                    row = grp.loc[combo]
                    p(f"  {combo:<55}  {cnt:>10,}  {row['c1_rate']:>8.4f}  {row['c2_rate']:>8.4f}")
                else:
                    p(f"  {combo:<55}  {cnt:>10,}  {'UNSEEN':>8}  {'UNSEEN':>8}")


# =============================================================================
# 7. Test distribution drift
# =============================================================================

def audit_drift(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("7. TEST DISTRIBUTION DRIFT")

    p("\n--- Categorical drift ---")
    for col in CAT_COLS:
        if not safe_col(train, col) or not safe_col(test, col):
            continue
        train_freq = train[col].value_counts(normalize=True)
        test_freq  = test[col].value_counts(normalize=True)
        all_vals   = set(train_freq.index) | set(test_freq.index)
        drift_rows = []
        for v in all_vals:
            tr = train_freq.get(v, 0.0)
            te = test_freq.get(v, 0.0)
            drift_rows.append((v, tr, te, te - tr))
        drift_df = pd.DataFrame(drift_rows, columns=["value", "train_rate", "test_rate", "delta"])
        drift_df["abs_delta"] = drift_df["delta"].abs()
        drift_df = drift_df.sort_values("abs_delta", ascending=False)

        big = drift_df[drift_df["abs_delta"] >= 0.005].head(10)
        if not big.empty:
            p(f"\n  {col} — top drift values (|delta| >= 0.005):")
            p(f"  {'value':<30}  {'train':>8}  {'test':>8}  {'delta':>8}")
            for _, row in big.iterrows():
                p(f"  {str(row['value']):<30}  {row['train_rate']:>8.4f}  {row['test_rate']:>8.4f}  {row['delta']:>+8.4f}")

        only_test = drift_df[drift_df["train_rate"] == 0]
        if not only_test.empty:
            p(f"  Values in TEST only: {only_test['value'].tolist()[:20]}")

    p("\n--- Numeric drift (standardized mean difference) ---")
    smd_rows = []
    for col in NUM_COLS:
        if not safe_col(train, col) or not safe_col(test, col):
            continue
        tr_mean, tr_std = train[col].mean(), train[col].std()
        te_mean = test[col].mean()
        te_std  = test[col].std()
        smd = (te_mean - tr_mean) / (tr_std + 1e-9)
        smd_rows.append((col, tr_mean, tr_std, train[col].median(),
                         te_mean, te_std, test[col].median(), smd))

    smd_rows.sort(key=lambda x: abs(x[7]), reverse=True)
    p(f"  {'column':<25}  {'tr_mean':>9}  {'tr_std':>9}  {'tr_med':>9}  "
      f"{'te_mean':>9}  {'te_std':>9}  {'te_med':>9}  {'smd':>8}")
    for row in smd_rows:
        p(f"  {row[0]:<25}  {row[1]:>9.2f}  {row[2]:>9.2f}  {row[3]:>9.2f}  "
          f"  {row[4]:>9.2f}  {row[5]:>9.2f}  {row[6]:>9.2f}  {row[7]:>+8.4f}")


# =============================================================================
# 8. Model disagreement row audit
# =============================================================================

def audit_disagreement(train: pd.DataFrame) -> None:
    section("8. MODEL DISAGREEMENT ROW AUDIT")

    paths = {
        "att21": OUTPUT_DIR / "stack_cat_oof_probs.npy",
        "att22": OUTPUT_DIR / "stack_cat_meta_oof_probs.npy",
        "att24": OUTPUT_DIR / "stack_meta_lgb_oof_probs.npy",
        "y":     OUTPUT_DIR / "y_train.npy",
    }

    missing = [k for k, v in paths.items() if not v.exists()]
    if missing:
        p(f"Skipping model disagreement audit — missing files: {missing}")
        return

    att21_probs = np.load(paths["att21"])
    att22_probs = np.load(paths["att22"])
    att24_probs = np.load(paths["att24"])
    y           = np.load(paths["y"])

    att21_pred = att21_probs.argmax(axis=1)
    att22_pred = att22_probs.argmax(axis=1)
    att24_pred = att24_probs.argmax(axis=1)

    n = min(len(y), len(train))
    y        = y[:n]
    att21_pred = att21_pred[:n]
    att22_pred = att22_pred[:n]
    att24_pred = att24_pred[:n]
    train_sub  = train.iloc[:n].reset_index(drop=True)

    att24_right_others_wrong = (att24_pred == y) & (att21_pred != y) & (att22_pred != y)
    att24_wrong_others_right = (att24_pred != y) & (att21_pred == y) & (att22_pred == y)

    p(f"\nRows where att24 correct, att21+att22 both wrong: {att24_right_others_wrong.sum():,}")
    p(f"Rows where att24 wrong,   att21+att22 both right: {att24_wrong_others_right.sum():,}")

    summary_cols = [c for c in CAT_COLS if safe_col(train_sub, c)]

    for label, mask in [
        ("ATT24 correct, att21+att22 wrong", att24_right_others_wrong),
        ("ATT24 wrong,   att21+att22 right", att24_wrong_others_right),
    ]:
        p(f"\n  === {label} ===")
        subset = train_sub[mask]
        if len(subset) == 0:
            p("  (no rows)")
            continue
        for col in summary_cols:
            vc = subset[col].value_counts().head(10)
            p(f"  {col}: {dict(vc)}")


# =============================================================================
# 9. Summary of key findings
# =============================================================================

def print_summary() -> None:
    section("9. SUMMARY — KEY FINDINGS TO INVESTIGATE")
    p("""
  Review the sections above and look for:

  LEAKAGE / ID DRIFT:
    - If class rates trend monotonically by id bin, id may encode time (data drift).
    - If test ids extend beyond train range, test is from a later period.

  HIGH-RISK CATEGORICAL GROUPS:
    - Zip codes / email domains with c1 >= 0.20 or c2 >= 0.35 are strong features.
    - If these groups are also overrepresented in test, they may dominate the score.

  HIGH-RISK COMBOS:
    - Zip × credit or zip × coverage with extreme class rates may be engineerable
      as OOF interaction features in a future attempt.

  TRAIN / TEST DRIFT:
    - Categorical columns where test frequencies differ strongly from train are
      candidates for distributional leakage or temporal shift.
    - Numeric columns with large standardized mean difference (|SMD| > 0.5) suggest
      the test set is from a different distribution than train.

  MISSINGNESS:
    - Columns where test is much more missing than train may be informative.
    - Missingness correlated with class-1 or class-2 can be used as binary flags.

  MODEL DISAGREEMENT:
    - Rows where att24 is uniquely right tell you what signal the LGB meta-model
      learned that LogReg missed. Group by zip, channel, credit for feature ideas.
    - Rows where att24 is uniquely wrong tell you where it overfit to OOF noise.
""")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading data...", flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    if safe_col(train, TARGET_COL):
        train = train[train[TARGET_COL].isin([0, 1, 2])].copy()

    audit_basic(train, test)
    audit_id_order(train, test)
    audit_missingness(train, test)
    audit_categorical(train, test)
    audit_numeric(train, test)
    audit_combos(train, test)
    audit_drift(train, test)
    audit_disagreement(train)
    print_summary()

    out_path = OUTPUT_DIR / "feature_leak_audit.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(_output_lines) + "\n")
    print(f"\nSaved audit to {out_path}", flush=True)


if __name__ == "__main__":
    main()
