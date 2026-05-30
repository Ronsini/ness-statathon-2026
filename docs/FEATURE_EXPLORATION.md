# Feature Exploration — 2026-05-10

Script: `notebooks/feature_exploration.py`
Data: train.csv only (1,045,123 rows after filtering cancel=-1)
Purpose: find features that explain some of the 7pp gap to the leaderboard leader (0.798 vs 0.729 raw)

---

## Top Findings (ranked by expected impact)

1. **OOF target encoding of `(age_bucket × credit)`** — class-2 rate ranges from 10% to 52% across age×credit cells; the combination contains massively more signal than either feature alone. Must be done as OOF smoothed rates (not a numeric product) to be novel over LightGBM's native splits.

2. **Missingness indicators** — ~1,000 rows missing per column, with consistent and large signal: missing `credit` → -4.7pp c2 rate; missing `n.adults` → -2.8pp c2 rate; missing `coverage.type` → +2.5pp c2 rate; missing `ni.age` → -2.8pp c1 rate. These are free 0/1 flags.

3. **`is_first_year_with_claim`** — customers in their first policy year who also filed a claim have a 31% c2 rate vs 21.9% overall (+9.1pp lift). Small segment (8,595 rows, 0.8%) but the signal is real.

4. **`zip_frequency`** — raw count of rows per zip code. Rare zips (decile 0) have 40–50% c2 rates vs 18–24% for the most common zips. This is a clean monotone signal that complements the OOF zip cancel rates already in the pipeline.

5. **Year cohort interaction** — 2014 is dramatically different (8.9% c1, 26.3% c2) vs 2016 (6.3% c1, 19.6% c2), a 6.7pp swing in c2. The `year` feature already captures this, but OOF zip rates computed per-year might be worth exploring in a future attempt.

**Realistic expectation:** These features are worth maybe +0.3–0.8pp combined. The remaining ~5–6pp to the leaderboard leader is almost certainly stacking / model architecture / ordinal structure exploitation — not feature engineering.

---

## Investigation 1: id structure

id ranges from 1 to 1,048,575. Year ranges fully overlap: all four years 2013–2016 span ids from ~1,000 to ~1,048,000. Year 2017 is clustered at very small ids (7,583–10,000), but that structure is entirely captured by the `year` feature, which is already in the model.

Rolling cancel rate over sorted ids has std=0.014 and essentially no trend (correlation with id position = 0.019). Within-year id percentile correlations with cancel classes are -0.0008 to +0.0007 — indistinguishable from zero. A quick LightGBM on 50k rows ranked id as the #1 feature by gain, but this is overfitting: with ~50k unique ids and 200 training rounds, any tree can memorize id-level patterns that don't generalize.

**Suggested features:** None. See dead-ends.

---

## Investigation 2: Categorical pair interactions

**dwelling.type × credit:** The largest true interaction is `Landlord + medium credit` — actual c2 rate 31.6% vs expected from marginals 21.4%, a +10.2pp deviation. The other high-c2 combos (`Tenant+low`, `Condo+low`) are close to their marginal expectations. The Landlord group is small and the deviation is driven by very few rows (n<40 for Landlord combinations). Not actionable as a new feature; LightGBM already handles this via native categorical splits.

**sales.channel × email_domain:** All top-5 combos by c1 and c2 rate are "Phone" channel — this is because Phone is the dominant channel and email domains are balanced across channels. Rates within Phone differ by <0.004 across domains. No interaction signal worth capturing.

**season_of_renewal × year:** Class-2 rates vary 7.8pp across season×year cells, but this range is entirely explained by year (2014=26.3% c2 vs 2016=19.6% c2). Within any year, season varies by less than 0.5pp. Season×year interaction is not novel over `year` alone.

**coverage.type × ni.gender:** Gender adds essentially nothing on top of coverage type — rates differ by 0.1–0.2pp within each coverage level. Not worth encoding.

**zip.code × year (zip deciled by frequency):** Confirmed that low-frequency zips have much higher AND more volatile cancel rates (decile 0: 40–50% c2) vs high-frequency zips (decile 9: 18–24% c2). The year×zip interaction is real but computing per-zip per-year OOF rates would be noisy given most zips have few rows per year.

**Suggested features:** `zip_frequency` (count of rows per zip code).

---

## Investigation 3: Premium structure

Premium percentile within (zip × year): mean 0.506 for c0, 0.486 for c1, 0.489 for c2. Correlation with cancel classes: -0.014 (c1), -0.022 (c2). Near zero.

Premium percentile within (credit × dwelling.type): almost identical numbers.

log_premium distributions across cancel classes are indistinguishable — identical median, std, and IQR. Premium is already in the model as a raw feature and `premium_per_sqft` is an existing engineered feature. Relative-premium features add nothing.

**Suggested features:** None. Premium is already well-represented.

---

## Investigation 4: Tenure × claim dynamics

Cancel rates by (tenure_bucket × claim.ind) confirm a loyalty effect: within claim.ind=1 customers, c2 rate drops from 32% (tenure 0–1 yr) to 25.4% (10+ yr). Within claim.ind=0, c2 drops from 24.9% to 19.6%.

`claim_density` (= claim.ind / tenure) has weaker correlation with both c1 and c2 than `claim.ind` alone:
- claim.ind: corr_c1=0.036, corr_c2=0.057
- claim_density: corr_c1=0.012, corr_c2=0.024

The interesting segment is `tenure < 1.1 AND claim.ind = 1`: 31% c2 rate vs 21.9% overall (+9.1pp). This is the `is_first_year_with_claim` binary flag.

**Suggested features:** `is_first_year_with_claim` (= `(tenure < 1.1) & (claim.ind == 1)`).

---

## Investigation 5: Year cohort effects

Class balance by year:

| year | c0    | c1    | c2    | n      |
|------|-------|-------|-------|--------|
| 2013 | 73.9% | 6.3%  | 19.8% | 220518 |
| 2014 | 64.8% | 8.9%  | 26.3% | 246242 |
| 2015 | 70.4% | 7.4%  | 22.2% | 274269 |
| 2016 | 74.1% | 6.3%  | 19.6% | 301687 |
| 2017 | 67.9% | 11.2% | 20.9% |   2407 |

2014 is dramatically different — +6.7pp c2 vs 2016 — and this holds across all credit levels (2014 vs 2016: high credit 19.3% vs 14.3%, low credit 48.2% vs 36.2%). Mean tenure is stable at 11.7 years for 2013–2016, then jumps to 12.7 in 2017 (probably a sampling artifact given 2017 has only 2,407 rows).

Year is already a feature in the pipeline. Within-year id percentile has correlation ≈0 with cancel — no additional signal from id.

**Suggested features:** None directly. Year-specific OOF zip rates are a future-attempt idea (see below).

---

## Investigation 6: Email domain as behavioral signal

The dataset has exactly 6 email domains, all free consumer providers (gmail, yahoo, hotmail, outlook, icloud, aol). Every single row has a "free email" by the standard definition — so the `free_email` binary flag is all-ones and has zero split. Cancel rates across the 6 domains differ by at most 0.3pp on any class. email_domain is confirmed noise.

**Suggested features:** None.

---

## Investigation 7: Feature importances on 50k subsample

Full ranking by gain (200 rounds, 50k rows, all features including id):

| Feature | Gain | Notes |
|---|---|---|
| credit | 36,730 | Strong |
| zip.code | 33,869 | Strong |
| sales.channel | 22,282 | Strong |
| ni.age | 20,621 | Strong |
| premium | 16,647 | Strong |
| square_footage | 16,492 | Moderate |
| **id** | **16,100** | **Overfitting — see below** |
| n.children | 15,095 | Moderate |
| len.at.res | 11,016 | Moderate |
| tenure | 10,904 | Moderate |
| year | 10,029 | Meaningful |
| num_windows_front | 7,615 | Suspicious — see below |
| n.adults | 6,943 | Moderate |
| claim.ind | 6,875 | Moderate |
| coverage.type | 3,862 | Lower |
| ni.marital.status | 3,782 | Lower |
| original_quote_weekday | 2,904 | Suspicious |
| email_domain | 2,396 | Noise |
| ni.gender | 1,523 | Near-noise |
| house.color | 1,037 | Near-noise |
| season_of_renewal | 965 | Near-noise |
| dwelling.type | 934 | Near-noise |

**id:** High importance is an artifact of the small training set (50k rows, near-unique ids, 200 rounds = overfitting). Real evidence id is noise: rolling cancel rate over sorted id has std=0.014 and trend correlation=0.019; within-year id percentile has correlation ≈0 with cancel. Do NOT add id as a feature.

**num_windows_front:** Rank 12 with gain 7,615. Not an obvious physical predictor of cancellation. Likely proxying for zip/neighborhood characteristics (older homes have fewer front windows?). Worth checking if removing it hurts — but adding it doesn't help since it's already in the pipeline.

**original_quote_weekday, house.color, season_of_renewal:** Moderate gain but these are likely picking up noise at 200 rounds. At 5000+ rounds with early stopping they contribute near-zero signal (confirmed by the full pipeline's feature importance from prior attempts).

**Suggested features:** None from this investigation.

---

## Investigation 8: Missingness as signal

All columns have roughly ~1,000 missing rows (1 per ~1,000 rows). Key signals:

| Column missing | Delta c1 | Delta c2 | Interpretation |
|---|---|---|---|
| credit | +0.8pp | **-4.7pp** | Missing credit → much lower cancel risk |
| n.adults | -1.3pp | **-2.8pp** | Missing n.adults → lower cancel risk |
| ni.age | **-2.8pp** | -0.5pp | Missing age → much lower class-1 risk |
| coverage.type | -0.8pp | **+2.5pp** | Missing coverage → higher class-2 risk |
| ni.marital.status | -0.3pp | -2.4pp | Missing marital status → lower cancel risk |
| n.children | +0.8pp | +2.3pp | Missing n.children → higher c2 risk |
| len.at.res | +1.4pp | +1.7pp | Missing len → elevated cancel across both |
| sales.channel | -0.6pp | -1.8pp | Missing channel → lower cancel risk |

The largest signals (credit, n.adults, ni.age, coverage.type) are worth encoding as binary `{col}_is_missing` flags. These are trivially cheap to add and the model can't infer them from imputed values.

Test set has a small number of missing values for the same columns, confirming these flags transfer.

**Suggested features:** `credit_is_missing`, `ni.age_is_missing`, `n.adults_is_missing`, `coverage.type_is_missing`, `ni.marital.status_is_missing`, `n.children_is_missing`.

---

## Bonus: ni.age × credit interaction

This is the biggest untapped finding. Class-2 cancel rates by (age_bucket × credit):

| age_bucket | high credit | medium credit | low credit |
|---|---|---|---|
| <25 | 21.2% | 31.1% | **52.0%** |
| 25–35 | 17.2% | 26.4% | 42.8% |
| 35–50 | 16.8% | 25.4% | 41.2% |
| 50–65 | 13.7% | 20.3% | 34.0% |
| 65+ | 10.3% | 15.9% | 25.4% |

The c2 rate ranges from 10.3% (elderly, high credit) to 52.0% (young, low credit) — a 42pp spread. Class-1 rates similarly span 1.1% to 16.4%.

A plain numeric product `age × credit_ordinal` would be attempt-3 redux — LightGBM already finds these splits via trees. The novel form is OOF target-encoded group rates: compute the smoothed cancel1_rate and cancel2_rate per `(age_bucket, credit)` cell across folds, same as zip_cancel2_rate. This gives the model the aggregate signal as a numeric feature it can use across trees, not just through splits on the raw categoricals.

**Suggested features:** `age_credit_cancel2_rate` (OOF), `age_credit_cancel1_rate` (OOF) — smoothed group cancel rates per (age_bucket × credit) cell, computed per fold.

---

## Bonus: cancel=-1 rows

3,452 rows (0.33% of training data) have cancel=-1. These are already excluded by the pipeline. In an insurance context, -1 likely means "mid-term cancel initiated by the insurer" or "policy voided" — distinct from customer-initiated cancellation. These rows probably represent a known risk category that Travelers can identify through processes not visible in this feature set. Not immediately actionable as a feature, but worth a future investigation: do cancel=-1 rows cluster in specific zip codes or credit segments, and does that clustering predict class-2 cancellations among the rows we do train on?

---

## Features to Add in Attempt-7

Ranked by expected impact:

1. **OOF `age_credit_cancel2_rate` and `age_credit_cancel1_rate`** — OOF smoothed cancel rates per (age_bucket × credit) cell, same implementation pattern as `zip_cancel2_rate`. Strongest untapped signal in the data (42pp c2 spread). Expected gain: +0.2–0.5pp.

2. **Missingness indicators** — binary 0/1 flags for missing values in: `credit`, `ni.age`, `n.adults`, `coverage.type`, `ni.marital.status`, `n.children`. Trivially cheap, some have 2–5pp class rate shifts. Expected gain: +0.1–0.2pp.

3. **`zip_frequency`** — count of training rows per zip code (computed on full train, no leakage since it's marginal). Rare zips have systematically high cancel rates; this is a clean monotone signal complementing the OOF rates. Expected gain: +0.05–0.1pp.

4. **`is_first_year_with_claim`** — `(tenure < 1.1) & (claim.ind == 1)`, binary. 31% c2 rate vs 21.9% overall. Small segment but cleanly separable. Expected gain: +0.02–0.05pp.

5. **`(age_bucket × credit)` as a categorical string** — concatenate the two as a native LightGBM categorical (`"<25_low"`, `"35-50_high"`, etc.). This is cheaper to implement than OOF encoding and lets LightGBM do native splits on the combination. Best paired with the OOF rates above; if only doing one, do the OOF version.

---

## Things Investigated but Not Promising

- **id as a feature:** Confirmed noise — zero correlation with cancel after controlling for year, high importance on small subsample is overfitting. Do not add.
- **Premium percentile features** (within zip×year, within credit×dwelling): Near-zero correlation with cancel classes (-0.02). Not worth adding.
- **email_domain / free_email:** Dataset has only 6 email domains, all free consumer providers. free_email is all-ones. email_domain cancel rates differ by <0.3pp. Confirmed noise.
- **claim_density (claim.ind / tenure):** Weaker than claim.ind alone (corr 0.024 vs 0.057 with c2). Not worth replacing existing features.
- **sales.channel × email_domain interaction:** All high-rate combos are Phone channel; within-channel variation by email domain is <0.004. No useful interaction.
- **season_of_renewal × year:** Year explains all the variation; season within year varies <0.5pp. Not useful as an interaction feature.
- **coverage.type × ni.gender:** 0.1–0.2pp difference within coverage levels. Noise.
- **house.color, original_quote_weekday, season_of_renewal:** Low gain in full pipeline (confirmed from prior attempt notes). Not worth investigating further.
- **Year-specific OOF zip rates:** Would be noisy due to thin zip×year cells. Deferred to a future attempt if other features are exhausted.
