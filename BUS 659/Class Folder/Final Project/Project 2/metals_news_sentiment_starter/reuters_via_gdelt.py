#!/usr/bin/env python3
"""
Reuters News Collection via GDELT
==================================

Since Reuters.com requires authentication/subscription, this script uses
GDELT API to collect Reuters-sourced articles. GDELT aggregates content
from Reuters and many other sources.

This is a compliant, free alternative to direct Reuters API access.

Run:
  python reuters_via_gdelt.py --hours 48 --out data/reuters_via_gdelt.csv
"""
from __future__ import annotations
import os, argparse, logging
from datetime import datetime, timezone
from typing import List, Dict
import pandas as pd
import requests
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

UA = "MetalsResearchBot/1.0 (Educational Research; contact: research@example.com)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def query_gdelt_for_reuters(hours: int, keywords: List[str], max_articles: int = 250) -> List[Dict]:
    """
    Query GDELT API specifically filtering for Reuters-sourced articles.
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Build query with source filter for Reuters
    top_keywords = keywords[:8]

    # GDELT query syntax: only wrap multi-word phrases in quotes
    formatted_kw = [f'"{k}"' if ' ' in k else k for k in top_keywords]
    keyword_query = " OR ".join(formatted_kw)

    # Add source filter for Reuters
    # Note: sourcecommon filter may not work as expected; filter in post-processing instead
    query = f"({keyword_query})"

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_articles),
        "format": "json",
        "timespan": f"{hours}h",
        "sort": "DateDesc"  # Most recent first
    }

    try:
        r = SESSION.get(base, params=params, timeout=300)
        logging.info("GDELT API URL: %s", r.url)

        if r.status_code != 200:
            logging.error("GDELT API returned HTTP %d", r.status_code)
            return []

        if not r.text or len(r.text.strip()) == 0:
            logging.warning("GDELT returned empty response")
            return []

        data = r.json()

        if "articles" not in data:
            logging.warning("GDELT response missing 'articles' key")
            return []

        articles = []
        for a in data.get("articles", []):
            source = a.get("sourceCommonName", "")

            # Double-check it's from Reuters
            if "reuters" not in source.lower():
                continue

            articles.append({
                "source": source,
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "published": a.get("seendate", ""),
                "language": a.get("language", "en"),
                "domain": a.get("domain", ""),
            })

        logging.info("Found %d Reuters articles via GDELT", len(articles))
        return articles

    except requests.exceptions.RequestException as e:
        logging.error("GDELT request failed: %s", e)
        return []
    except ValueError as e:
        logging.error("GDELT JSON parsing failed: %s", e)
        return []
    except Exception as e:
        logging.error("GDELT query failed: %s", e)
        return []

def main():
    ap = argparse.ArgumentParser(description="Collect Reuters articles via GDELT")
    ap.add_argument("--hours", type=int, default=48, help="Lookback window (hours)")
    ap.add_argument("--out", type=str, default="data/reuters_via_gdelt.csv", help="Output CSV")
    ap.add_argument("--max", type=int, default=250, help="Max articles to retrieve")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)

    # Load keywords
    taxonomy = load_yaml("keywords.yaml")
    metals_kw = taxonomy.get("metals", [])
    adjacent_kw = taxonomy.get("adjacent_industries", [])
    keywords = list(set(metals_kw + adjacent_kw))

    logging.info("Querying GDELT for Reuters articles (last %d hours)...", args.hours)

    articles = query_gdelt_for_reuters(args.hours, keywords, args.max)

    if articles:
        df = pd.DataFrame(articles)
        df = df.drop_duplicates(subset=['url'])
        df.to_csv(args.out, index=False)

        print(f"\n{'='*80}")
        print(f"✓ Collected {len(df)} Reuters articles via GDELT")
        print(f"✓ Saved to: {args.out}")
        print(f"{'='*80}")

        # Statistics
        print(f"\nSource breakdown:")
        print(df['source'].value_counts().head(10))

        print(f"\nLanguage breakdown:")
        print(df['language'].value_counts())

        print(f"\nSample articles:")
        for i, row in df.head(5).iterrows():
            print(f"\n  {i+1}. {row['title'][:80]}...")
            print(f"     Source: {row['source']}")
            print(f"     URL: {row['url'][:100]}...")
            print(f"     Published: {row['published']}")

        print(f"\n{'='*80}")
        print(f"Next steps:")
        print(f"  1. Run sentiment analysis: python main.py --hours {args.hours}")
        print(f"  2. Or process these URLs through the existing pipeline")
        print(f"{'='*80}\n")

    else:
        print("\n⚠ No Reuters articles found via GDELT")
        print("\nPossible reasons:")
        print("  1. No Reuters articles matching keywords in time window")
        print("  2. GDELT API temporary issues")
        print("  3. Try expanding time window (--hours 72 or --hours 168)")
        print("\nAlternative approaches:")
        print("  1. Subscribe to official Reuters/LSEG API")
        print("  2. Use the main pipeline which combines GDELT + RSS feeds")
        print("  3. Check GDELT website for advanced query syntax")

if __name__ == "__main__":
    main()
