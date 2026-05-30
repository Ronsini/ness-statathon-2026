"""
Feature exploration for NESS Statathon 2026.
Run with: .venv/bin/python notebooks/feature_exploration.py
Output: printed to stdout; plots saved to notebooks/plots/
"""

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict

warnings.filterwarnings("ignore")
os.makedirs("notebooks/plots", exist_ok=True)

print("Loading train.csv...")
df = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print(f"Train shape: {df.shape}  |  Test shape: {test.shape}")
print(f"Cancel distribution (raw, including any -1):\n{df['cancel'].value_counts(normalize=True).sort_index().round(4)}\n")

# NOTE: cancel=-1 rows exist — flag them before filtering
neg1_mask = df["cancel"] == -1
print(f"*** FINDING: cancel=-1 rows: {neg1_mask.sum():,} ({neg1_mask.mean()*100:.3f}% of train) ***")
print(f"    These rows are excluded from all downstream analysis.\n")

# Filter to known classes only (0, 1, 2)
df = df[df["cancel"].isin([0, 1, 2])].reset_index(drop=True)
print(f"Train shape after removing cancel=-1: {df.shape}")
print(f"Cancel distribution (clean):\n{df['cancel'].value_counts(normalize=True).sort_index().round(4)}\n")

C0 = df[df.cancel == 0]
C1 = df[df.cancel == 1]
C2 = df[df.cancel == 2]

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("INVESTIGATION 1: id structure — is it sequential? does it encode time?")
print("=" * 72)

df_s = df.sort_values("id").reset_index(drop=True)
print(f"id range: {df.id.min():,} – {df.id.max():,}")
print(f"Unique ids: {df.id.nunique():,} out of {len(df):,} rows  (duplicates: {len(df)-df.id.nunique():,})")

# Correlation between id and year
corr_id_year = df["id"].corr(df["year"])
print(f"\nPearson correlation(id, year): {corr_id_year:.4f}")

# Year-bucket id means
print("\nMean / min / max id by year:")
yid = df.groupby("year")["id"].agg(["min", "max", "mean", "count"]).astype(int)
print(yid.to_string())

# Within-year, is id a monotone counter?
print("\nAre year ranges non-overlapping?")
for y in sorted(df.year.unique()):
    sub = df[df.year == y]
    print(f"  year={y}: id=[{sub.id.min():,}, {sub.id.max():,}]  n={len(sub):,}")

# Rolling cancel rate over id (window=10000)
df_s["cancel_roll"] = df_s["cancel"].rolling(10000, min_periods=500, center=True).mean()
# Compute stats on rolling series
roll = df_s["cancel_roll"].dropna()
print(f"\nRolling cancel rate (window=10000) over sorted id:")
print(f"  min={roll.min():.4f}  max={roll.max():.4f}  std={roll.std():.4f}")

# Are there monotone or oscillation patterns? Check correlation of rolling rate with position
roll_pos = np.arange(len(roll))
roll_corr = np.corrcoef(roll_pos, roll.values)[0, 1]
print(f"  Correlation of rolling rate with id position: {roll_corr:.4f}")

# Class-2 rate within id deciles
df["id_decile"] = pd.qcut(df["id"], 10, labels=False)
id_decile_rates = df.groupby("id_decile")["cancel"].apply(
    lambda x: (x == 2).mean()
).reset_index()
id_decile_rates.columns = ["decile", "cancel2_rate"]
print(f"\nClass-2 cancel rate by id decile (decile 0=lowest ids, 9=highest):")
print(id_decile_rates.to_string(index=False))

# Class-1 rate by id decile
id_decile_c1 = df.groupby("id_decile")["cancel"].apply(
    lambda x: (x == 1).mean()
).reset_index()
id_decile_c1.columns = ["decile", "cancel1_rate"]
id_merged = id_decile_rates.merge(id_decile_c1, on="decile")
print("\nClass-1 cancel rate by id decile:")
print(id_decile_c1.to_string(index=False))

# id_within_year_percentile
df["id_within_year_pct"] = df.groupby("year")["id"].rank(pct=True)
print("\nid_within_year_percentile corr with cancel classes:")
for c in [0, 1, 2]:
    corr = df["id_within_year_pct"].corr((df.cancel == c).astype(int))
    print(f"  cancel=={c}: {corr:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("INVESTIGATION 2: Hidden interactions in categorical pairs")
print("=" * 72)

def interaction_report(df, col1, col2, top_n=5):
    """For each (col1, col2) combination, compute cancel rates and deviation from marginals."""
    grp = df.groupby([col1, col2])["cancel"].value_counts(normalize=False).unstack(fill_value=0)
    grp.columns = [f"n_c{c}" for c in grp.columns]
    grp["n_total"] = grp.sum(axis=1)
    for c in [0, 1, 2]:
        col = f"n_c{c}"
        if col in grp.columns:
            grp[f"rate_c{c}"] = grp[col] / grp["n_total"]
        else:
            grp[f"rate_c{c}"] = 0.0

    # Global marginal rates
    global_rates = {c: (df.cancel == c).mean() for c in [0, 1, 2]}

    # Marginal rates from col1 and col2 independently
    m1 = df.groupby(col1)["cancel"].apply(lambda x: {c: (x == c).mean() for c in [0, 1, 2]})
    m2 = df.groupby(col2)["cancel"].apply(lambda x: {c: (x == c).mean() for c in [0, 1, 2]})

    results = []
    for (v1, v2), row in grp.iterrows():
        if row["n_total"] < 30:
            continue
        for c in [1, 2]:
            rate = row.get(f"rate_c{c}", 0)
            m1_rate = m1.get(v1, {}).get(c, global_rates[c]) if v1 in m1.index else global_rates[c]
            m2_rate = m2.get(v2, {}).get(c, global_rates[c]) if v2 in m2.index else global_rates[c]
            expected = m1_rate * m2_rate / global_rates[c] if global_rates[c] > 0 else 0
            deviation = rate - (m1_rate + m2_rate) / 2
            results.append({
                col1: v1, col2: v2,
                "n": int(row["n_total"]),
                f"rate_c{c}": round(rate, 4),
                f"expected_c{c}": round((m1_rate + m2_rate) / 2, 4),
                f"deviation_c{c}": round(deviation, 4),
                "cancel_class": c
            })
    return pd.DataFrame(results)

# 2a. dwelling.type × credit
print("\n--- 2a. dwelling.type × credit ---")
dw_credit = df.groupby(["dwelling.type", "credit"])["cancel"].value_counts(normalize=True).unstack(fill_value=0)
if 1 in dw_credit.columns and 2 in dw_credit.columns:
    dw_credit.columns = [f"rate_c{c}" for c in dw_credit.columns]
    dw_credit = dw_credit.reset_index()
    print("Top 5 combos by class-1 rate:")
    print(dw_credit.nlargest(5, "rate_c1")[["dwelling.type", "credit", "rate_c1", "rate_c2"]].to_string(index=False))
    print("Top 5 combos by class-2 rate:")
    print(dw_credit.nlargest(5, "rate_c2")[["dwelling.type", "credit", "rate_c1", "rate_c2"]].to_string(index=False))

    # Deviation from marginals
    margin_dw = df.groupby("dwelling.type")["cancel"].apply(lambda x: (x == 2).mean())
    margin_cr = df.groupby("credit")["cancel"].apply(lambda x: (x == 2).mean())
    global_c2 = (df.cancel == 2).mean()
    print("\nInteraction deviation (actual rate_c2 - expected from marginals):")
    dw_credit["expected_c2"] = dw_credit.apply(
        lambda r: margin_dw.get(r["dwelling.type"], global_c2) + margin_cr.get(r["credit"], global_c2) - global_c2,
        axis=1
    )
    dw_credit["deviation_c2"] = dw_credit["rate_c2"] - dw_credit["expected_c2"]
    print(dw_credit.nlargest(5, "deviation_c2")[["dwelling.type", "credit", "rate_c2", "expected_c2", "deviation_c2"]].to_string(index=False))

# 2b. sales.channel × email_domain
print("\n--- 2b. sales.channel × email_domain ---")
sc_em = df.groupby(["sales.channel", "email_domain"])["cancel"].value_counts(normalize=True).unstack(fill_value=0).reset_index()
sc_em.columns = ["sales.channel", "email_domain"] + [f"rate_c{c}" for c in sc_em.columns[2:]]
# Only keep combos with enough data
cnt = df.groupby(["sales.channel", "email_domain"]).size().reset_index(name="n")
sc_em = sc_em.merge(cnt, on=["sales.channel", "email_domain"])
sc_em = sc_em[sc_em["n"] >= 50]
if "rate_c1" in sc_em.columns and "rate_c2" in sc_em.columns:
    print(f"Total combos with n≥50: {len(sc_em)}")
    print("Top 5 by class-1 rate:")
    print(sc_em.nlargest(5, "rate_c1")[["sales.channel", "email_domain", "n", "rate_c1", "rate_c2"]].to_string(index=False))
    print("Top 5 by class-2 rate:")
    print(sc_em.nlargest(5, "rate_c2")[["sales.channel", "email_domain", "n", "rate_c1", "rate_c2"]].to_string(index=False))

# 2c. season_of_renewal × year
print("\n--- 2c. season_of_renewal × year ---")
sy = df.groupby(["season_of_renewal", "year"])["cancel"].value_counts(normalize=True).unstack(fill_value=0).reset_index()
sy.columns = ["season_of_renewal", "year"] + [f"rate_c{c}" for c in sy.columns[2:]]
if "rate_c2" in sy.columns:
    print(sy[["season_of_renewal", "year", "rate_c2"]].pivot(index="season_of_renewal", columns="year", values="rate_c2").round(4).to_string())
    max_dev = sy["rate_c2"].max() - sy["rate_c2"].min()
    print(f"Range of class-2 rates across season×year combos: {max_dev:.4f}")

# 2d. coverage.type × ni.gender
print("\n--- 2d. coverage.type × ni.gender ---")
cg = df.groupby(["coverage.type", "ni.gender"])["cancel"].value_counts(normalize=True).unstack(fill_value=0).reset_index()
cg.columns = ["coverage.type", "ni.gender"] + [f"rate_c{c}" for c in cg.columns[2:]]
cnt2 = df.groupby(["coverage.type", "ni.gender"]).size().reset_index(name="n")
cg = cg.merge(cnt2, on=["coverage.type", "ni.gender"])
if "rate_c1" in cg.columns and "rate_c2" in cg.columns:
    print(cg[["coverage.type", "ni.gender", "n", "rate_c1", "rate_c2"]].to_string(index=False))

# 2e. zip.code × year (zip decile bins)
print("\n--- 2e. zip.code × year (zip deciled by frequency) ---")
zip_freq = df["zip.code"].value_counts()
zip_decile_map = pd.qcut(zip_freq, 10, labels=False, duplicates="drop")
df["zip_freq_decile"] = df["zip.code"].map(zip_decile_map)
zy = df.groupby(["zip_freq_decile", "year"])["cancel"].apply(
    lambda x: (x == 2).mean()
).reset_index()
zy.columns = ["zip_freq_decile", "year", "cancel2_rate"]
pivot_zy = zy.pivot(index="zip_freq_decile", columns="year", values="cancel2_rate").round(4)
print("Class-2 cancel rate by (zip_freq_decile × year):")
print(pivot_zy.to_string())
year_range = zy.groupby("year")["cancel2_rate"].agg(["min","max","std"])
print(f"\nYear-to-year variation in zip-decile cancel2 rates:")
print(year_range.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("INVESTIGATION 3: Premium structure")
print("=" * 72)

# Premium percentile within (zip × year)
df["prem_pct_zip_year"] = df.groupby(["zip.code", "year"])["premium"].rank(pct=True)
print("\nMean premium_pct_within_zip_year by cancel class:")
for c in [0, 1, 2]:
    m = df[df.cancel == c]["prem_pct_zip_year"].mean()
    print(f"  cancel={c}: {m:.4f}")

# Correlation with cancel
df["is_c1"] = (df.cancel == 1).astype(int)
df["is_c2"] = (df.cancel == 2).astype(int)
c1_corr = df["prem_pct_zip_year"].corr(df["is_c1"])
c2_corr = df["prem_pct_zip_year"].corr(df["is_c2"])
print(f"  Corr with class-1: {c1_corr:.4f}")
print(f"  Corr with class-2: {c2_corr:.4f}")

# Premium percentile within (credit × dwelling.type)
df["prem_pct_credit_dw"] = df.groupby(["credit", "dwelling.type"])["premium"].rank(pct=True)
print("\nMean premium_pct_within_credit_dwelling by cancel class:")
for c in [0, 1, 2]:
    m = df[df.cancel == c]["prem_pct_credit_dw"].mean()
    print(f"  cancel={c}: {m:.4f}")
c1_corr2 = df["prem_pct_credit_dw"].corr(df["is_c1"])
c2_corr2 = df["prem_pct_credit_dw"].corr(df["is_c2"])
print(f"  Corr with class-1: {c1_corr2:.4f}  |  class-2: {c2_corr2:.4f}")

# Premium / sqft percentile within zip
df["prem_per_sqft"] = df["premium"] / df["square_footage"].replace(0, np.nan)
df["prem_sqft_pct_zip"] = df.groupby("zip.code")["prem_per_sqft"].rank(pct=True)
print("\nMean premium_per_sqft_pct_within_zip by cancel class:")
for c in [0, 1, 2]:
    m = df[df.cancel == c]["prem_sqft_pct_zip"].mean()
    print(f"  cancel={c}: {m:.4f}")

# log_premium distribution by cancel class
df["log_premium"] = np.log1p(df["premium"])
print("\nlog_premium quantiles by cancel class:")
qp = df.groupby("cancel")["log_premium"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
print(qp.round(4).to_string())

# Bimodal check — std and IQR of log_premium per class
print("\nlog_premium std and IQR by cancel class:")
for c in [0, 1, 2]:
    sub = df[df.cancel == c]["log_premium"]
    print(f"  cancel={c}: std={sub.std():.4f}  IQR={sub.quantile(0.75)-sub.quantile(0.25):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("INVESTIGATION 4: Tenure × claim dynamics")
print("=" * 72)

bins = [0, 1, 3, 5, 10, 100]
labels_t = ["0-1", "1-3", "3-5", "5-10", "10+"]
df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels_t, right=False)

print("\nCancel rate by (tenure_bucket × claim.ind):")
tc = df.groupby(["tenure_bucket", "claim.ind"])["cancel"].value_counts(normalize=True).unstack(fill_value=0)
# Rename only present columns
tc.columns = [f"rate_c{c}" for c in tc.columns]
tc_cnt = df.groupby(["tenure_bucket", "claim.ind"]).size().rename("n")
tc = tc.join(tc_cnt)
print(tc.round(4).to_string())

# For claim.ind=1: cancel rate trajectory with tenure
print("\nFor claim.ind=1 — cancel rates by tenure bucket:")
claim1 = df[df["claim.ind"] == 1]
c1_tenure = claim1.groupby("tenure_bucket")["cancel"].value_counts(normalize=True).unstack(fill_value=0)
c1_tenure.columns = [f"rate_c{c}" for c in c1_tenure.columns]
c1_cnt = claim1.groupby("tenure_bucket").size().rename("n")
c1_tenure = c1_tenure.join(c1_cnt)
print(c1_tenure.round(4).to_string())

# claim_density feature
df["claim_density"] = df["claim.ind"] / df["tenure"].clip(lower=0.1)
print("\nMean claim_density by cancel class:")
for c in [0, 1, 2]:
    m = df[df.cancel == c]["claim_density"].mean()
    print(f"  cancel={c}: {m:.4f}")
cd_c1_corr = df["claim_density"].corr(df["is_c1"])
cd_c2_corr = df["claim_density"].corr(df["is_c2"])
print(f"  Corr with class-1: {cd_c1_corr:.4f}  |  class-2: {cd_c2_corr:.4f}")

# Compare claim_density vs claim.ind alone vs tenure alone
print("\nCorrelations with cancel classes:")
for feat in ["claim.ind", "tenure", "claim_density"]:
    c1 = df[feat].corr(df["is_c1"])
    c2 = df[feat].corr(df["is_c2"])
    print(f"  {feat:20s}  corr_c1={c1:.4f}  corr_c2={c2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("INVESTIGATION 5: Year cohort effects")
print("=" * 72)

print("\nClass balance by year:")
yr_cancel = df.groupby("year")["cancel"].value_counts(normalize=True).unstack(fill_value=0)
yr_cancel.columns = [f"rate_c{c}" for c in yr_cancel.columns]
print(yr_cancel.round(4).to_string())

print("\nMean tenure by year:")
print(df.groupby("year")["tenure"].mean().round(3).to_string())

print("\nCancel-2 rate by (year × credit):")
yc = df.groupby(["year", "credit"])["cancel"].apply(lambda x: (x == 2).mean()).unstack()
print(yc.round(4).to_string())

# Is year itself a strong predictor?
year_dummies = pd.get_dummies(df["year"], prefix="yr")
print("\nCorrelation of year dummies with cancel classes:")
for yr in sorted(df["year"].unique()):
    col = f"yr_{yr}"
    if col in year_dummies.columns:
        c1 = year_dummies[col].corr(df["is_c1"])
        c2 = year_dummies[col].corr(df["is_c2"])
        print(f"  year={yr}  corr_c1={c1:.4f}  corr_c2={c2:.4f}")

# Within-year id percentile corr with cancel
print("\nCancel rate by id_within_year quintile (averaged over all years):")
df["id_wy_quin"] = df.groupby("year")["id"].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
iq = df.groupby("id_wy_quin")["cancel"].value_counts(normalize=True).unstack(fill_value=0)
if 2 in iq.columns:
    print(iq[[c for c in [0, 1, 2] if c in iq.columns]].round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("INVESTIGATION 6: Email domain as behavioral signal")
print("=" * 72)

top_domains = df["email_domain"].value_counts().head(20).index
em_stats = df[df["email_domain"].isin(top_domains)].groupby("email_domain").agg(
    n=("cancel", "count"),
    cancel0_rate=("cancel", lambda x: (x == 0).mean()),
    cancel1_rate=("cancel", lambda x: (x == 1).mean()),
    cancel2_rate=("cancel", lambda x: (x == 2).mean()),
    mean_premium=("premium", "mean"),
    median_tenure=("tenure", "median"),
).sort_values("cancel2_rate", ascending=False)
print("\nTop 20 email domains by class-2 cancel rate:")
print(em_stats.round(4).to_string())

# Free vs paid email
free_domains = {"gmail.com", "yahoo.com", "hotmail.com", "aol.com", "outlook.com",
                "icloud.com", "mail.com", "protonmail.com", "live.com"}
df["free_email"] = df["email_domain"].isin(free_domains).astype(int)
print("\nCancel rates — free email vs other:")
fe = df.groupby("free_email")["cancel"].value_counts(normalize=True).unstack(fill_value=0)
fe.columns = [f"rate_c{c}" for c in fe.columns]
cnt_fe = df.groupby("free_email").size().rename("n")
fe = fe.join(cnt_fe)
print(fe.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("INVESTIGATION 7: Suspicious 'shouldn't predict but does' features")
print("=" * 72)

print("Training quick LightGBM on 50k rows to get feature importances...")
sub50 = df.sample(50000, random_state=42)

# Use id as a feature too
feat_cols = [
    "id", "year", "zip.code", "ni.age", "len.at.res", "premium", "n.adults",
    "n.children", "tenure", "claim.ind", "square_footage", "num_windows_front",
    "ni.marital.status",
    "free_email",  # computed above
]
cat_cols = ["house.color", "credit", "coverage.type", "dwelling.type",
            "sales.channel", "ni.gender", "original_quote_weekday",
            "season_of_renewal", "email_domain"]

for col in cat_cols:
    sub50[col] = sub50[col].astype("category")

feat_cols_all = feat_cols + cat_cols

X50 = sub50[feat_cols_all]
y50 = sub50["cancel"]

ds = lgb.Dataset(X50, label=y50, categorical_feature=cat_cols, free_raw_data=False)
params = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
    "n_jobs": -1, "seed": 42
}
model = lgb.train(params, ds, num_boost_round=200, valid_sets=[ds],
                  callbacks=[lgb.log_evaluation(period=-1)])

imp = pd.DataFrame({
    "feature": feat_cols_all,
    "gain": model.feature_importance(importance_type="gain"),
    "split": model.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)

print("\nFeature importance (gain) — all features including suspicious ones:")
print(imp.to_string(index=False))

# Specifically highlight the suspicious ones
suspicious = ["house.color", "num_windows_front", "original_quote_weekday", "season_of_renewal", "id"]
print("\nSuspicious features:")
print(imp[imp["feature"].isin(suspicious)].to_string(index=False))

# id importance and rank
id_rank = imp[imp["feature"] == "id"].index[0] + 1 if "id" in imp["feature"].values else None
print(f"\n'id' feature rank by gain: #{id_rank} out of {len(imp)}")
id_gain = imp[imp["feature"] == "id"]["gain"].values[0] if "id" in imp["feature"].values else 0
print(f"'id' gain = {id_gain:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("INVESTIGATION 8: Missingness as signal")
print("=" * 72)

print("\nMissing value counts:")
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print(miss.to_string())

print("\nCancel rates by missingness indicator:")
for col in miss.index:
    df[f"{col}_missing"] = df[col].isnull().astype(int)
    rates_miss = df[df[f"{col}_missing"] == 1]["cancel"].value_counts(normalize=True).sort_index()
    rates_present = df[df[f"{col}_missing"] == 0]["cancel"].value_counts(normalize=True).sort_index()
    n_miss = df[f"{col}_missing"].sum()
    print(f"\n  {col}  (n_missing={n_miss:,})")
    print(f"    Missing:  c0={rates_miss.get(0, 0):.4f}  c1={rates_miss.get(1, 0):.4f}  c2={rates_miss.get(2, 0):.4f}")
    print(f"    Present:  c0={rates_present.get(0, 0):.4f}  c1={rates_present.get(1, 0):.4f}  c2={rates_present.get(2, 0):.4f}")
    for c in [0, 1, 2]:
        delta = rates_miss.get(c, 0) - rates_present.get(c, 0)
        print(f"    Delta c{c}: {delta:+.4f}")

# Also check in test set
print("\nMissing value counts in test.csv:")
miss_test = test.isnull().sum()
miss_test = miss_test[miss_test > 0]
print(miss_test.to_string() if len(miss_test) > 0 else "  None")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("BONUS: Within-zip class-1 vs class-2 rate decoupling")
print("=" * 72)

zip_rates = df.groupby("zip.code")["cancel"].agg(
    n="count",
    c1_rate=lambda x: (x == 1).mean(),
    c2_rate=lambda x: (x == 2).mean(),
).reset_index()
zip_rates = zip_rates[zip_rates["n"] >= 20]
print(f"\nZip codes with n≥20: {len(zip_rates)}")
print(f"Corr(c1_rate, c2_rate) within zip: {zip_rates['c1_rate'].corr(zip_rates['c2_rate']):.4f}")
print("(Low correlation → class-1 and class-2 are spatially decoupled → both OOF rates needed)")

# Zip-level class-1 high but class-2 low, and vice versa
print("\nZips with high class-1 but low class-2 (top 5 by c1-c2 spread):")
zip_rates["c1_minus_c2"] = zip_rates["c1_rate"] - zip_rates["c2_rate"]
print(zip_rates.nlargest(5, "c1_minus_c2")[["zip.code", "n", "c1_rate", "c2_rate", "c1_minus_c2"]].round(4).to_string(index=False))
print("\nZips with high class-2 but low class-1 (top 5):")
print(zip_rates.nsmallest(5, "c1_minus_c2")[["zip.code", "n", "c1_rate", "c2_rate", "c1_minus_c2"]].round(4).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("BONUS: ni.age × credit interaction")
print("=" * 72)

df["age_bucket"] = pd.cut(df["ni.age"], bins=[0, 25, 35, 50, 65, 200], labels=["<25", "25-35", "35-50", "50-65", "65+"])
age_credit = df.groupby(["age_bucket", "credit"])["cancel"].value_counts(normalize=True).unstack(fill_value=0)
age_credit.columns = [f"rate_c{c}" for c in age_credit.columns]
cnt_ac = df.groupby(["age_bucket", "credit"]).size().rename("n")
age_credit = age_credit.join(cnt_ac)
print("\nCancel rates by (age_bucket × credit):")
print(age_credit.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("BONUS: is_first_year feature (tenure < 1.1)")
print("=" * 72)

df["is_first_year"] = (df["tenure"] < 1.1).astype(int)
print("\nCancel rates — first year vs established customers:")
fy = df.groupby("is_first_year")["cancel"].value_counts(normalize=True).unstack(fill_value=0)
fy.columns = [f"rate_c{c}" for c in fy.columns]
cnt_fy = df.groupby("is_first_year").size().rename("n")
fy = fy.join(cnt_fy)
print(fy.round(4).to_string())

# First-year × claim interaction
df["first_year_claim"] = ((df["tenure"] < 1.1) & (df["claim.ind"] == 1)).astype(int)
fyc = df.groupby("first_year_claim")["cancel"].value_counts(normalize=True).unstack(fill_value=0)
fyc.columns = [f"rate_c{c}" for c in fyc.columns]
cnt_fyc = df.groupby("first_year_claim").size().rename("n")
fyc = fyc.join(cnt_fyc)
print("\nFirst-year + claim vs all others:")
print(fyc.round(4).to_string())

print("\n" + "=" * 72)
print("DONE — all investigations complete")
print("=" * 72)
