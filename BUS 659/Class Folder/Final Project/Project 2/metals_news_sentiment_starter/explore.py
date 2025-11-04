#!/usr/bin/env python3
"""
Quick EDA to summarize sentiment output.
Usage: python explore.py data/run_2025-10-25.csv
"""
import sys, pandas as pd
from datetime import datetime
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python explore.py <csv_path>")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    df["published"] = pd.to_datetime(df["published"], errors="coerce")

    # Overall sentiment
    agg = df.agg({
        "finbert_score": ["mean", "median"],
        "finbert_pos": "mean",
        "finbert_neg": "mean",
        "headline_vader": "mean"
    })
    print("\n=== OVERALL SENTIMENT ===")
    print(agg)

    print("\n=== SENTIMENT BY SOURCE (top 10 by volume) ===")
    by_src = (df.groupby("source")
                .agg(n=("url","count"), score=("finbert_score","mean"))
                .sort_values("n", ascending=False)
                .head(10))
    print(by_src)

    # Metals vs Adjacent flag prevalence
    flag_cols = [c for c in df.columns if c.startswith("match_")]
    print("\n=== MATCH FLAGS (share of articles) ===")
    print(df[flag_cols].mean().sort_values(ascending=False))

    # Time series (daily mean finbert_score)
    ts = df.set_index("published").resample("D")["finbert_score"].mean()
    ts = ts.dropna()
    if len(ts) > 0:
        print("\n=== DAILY MEAN FINBERT SCORE ===")
        print(ts.tail(14))

if __name__ == "__main__":
    main()
