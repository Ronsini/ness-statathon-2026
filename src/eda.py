"""
NESS Statathon 2026 — Exploratory Data Analysis
Quick summary of training data: class balance, missingness, basic distributions.

Usage:
  python src/eda.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")

    print("\nClass balance (cancel):")
    print(train["cancel"].value_counts(normalize=True).sort_index().round(4))

    print("\nMissing values per column (train, top 20):")
    miss = train.isna().sum().sort_values(ascending=False)
    print(miss.head(20))

    print("\nDtypes:")
    print(train.dtypes)

    print("\nCancel rate by year (cancel > 0):")
    print(train.groupby("year")["cancel"].apply(lambda x: (x > 0).mean()).round(4))

    print("\nCancel rate by credit tier:")
    print(train.groupby("credit")["cancel"].apply(lambda x: (x > 0).mean()).round(4))

    print("\nCancel rate by claim indicator:")
    print(train.groupby("claim.ind")["cancel"].apply(lambda x: (x > 0).mean()).round(4))

    print("\nMean tenure by cancel class:")
    print(train.groupby("cancel")["tenure"].mean().round(2))


if __name__ == "__main__":
    main()
