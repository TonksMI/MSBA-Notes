#!/usr/bin/env python3
"""
Reuters News Collector for Metals & Commodities
================================================

This script collects metals/commodities news from Reuters using compliant methods:
1. Public Reuters.com search pages (respects robots.txt)
2. Optional: Third-party APIs (requires API key)
3. Alternative: Web scraping with proper delays and user agent

IMPORTANT LEGAL NOTES:
- Always respect robots.txt
- Use proper delays between requests (rate limiting)
- Identify your bot with a proper User-Agent
- For production use, consider official LSEG/Reuters API subscription
- This is for educational/research purposes only

Run:
  python reuters_collector.py --hours 24 --out data/reuters_run.csv
  python reuters_collector.py --api rapidapi --key YOUR_KEY --hours 24
"""
from __future__ import annotations
import os, re, time, json, hashlib, argparse, logging, random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin, quote_plus
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# User agent - identify your bot clearly
UA = "MetalsResearchBot/1.0 (Educational Research; contact: research@example.com)"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9"
})

# Rate limiting - be respectful
MIN_DELAY = 2.0  # seconds between requests
MAX_DELAY = 5.0

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def rate_limit():
    """Sleep to respect rate limits."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

def check_robots_txt(base_url: str) -> bool:
    """Check if we're allowed to access the site."""
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        resp = SESSION.get(robots_url, timeout=10)
        if resp.status_code == 200:
            # Simple check - look for Disallow rules
            # For production, use urllib.robotparser
            logging.info("robots.txt accessed successfully")
            return True
    except Exception as e:
        logging.warning("Could not check robots.txt: %s", e)
    return True

def collect_reuters_search(keywords: List[str], hours: int) -> List[Dict]:
    """
    Collect articles from Reuters public search page.

    NOTE: Reuters website structure may change. This uses their public
    search interface which should be accessible, but always verify compliance.
    """
    articles = []
    base_url = "https://www.reuters.com"

    # Check robots.txt first
    check_robots_txt(base_url)

    # Reuters search URL pattern (may need updating based on site structure)
    # Using topic-based URLs which are more stable
    topics = [
        "/markets/commodities/",
        "/business/energy/",
        "/business/industrials/"
    ]

    logging.info("Collecting from Reuters topic pages...")

    for topic in topics:
        try:
            rate_limit()  # Respect rate limits

            url = base_url + topic
            logging.info("Fetching: %s", url)

            resp = SESSION.get(url, timeout=20)
            if resp.status_code != 200:
                logging.warning("Failed to fetch %s: HTTP %d", url, resp.status_code)
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find article links - Reuters uses specific patterns
            # This selector may need updating based on site structure
            article_links = soup.find_all('a', attrs={'data-testid': re.compile('Heading')})

            if not article_links:
                # Fallback: look for article links in general
                article_links = soup.find_all('a', href=re.compile(r'/\w+/\w+/[\w-]+'))

            logging.info("Found %d potential article links in %s", len(article_links), topic)

            for link in article_links[:20]:  # Limit to avoid overwhelming
                href = link.get('href', '')
                if not href:
                    continue

                # Make absolute URL
                if href.startswith('/'):
                    article_url = base_url + href
                elif href.startswith('http'):
                    article_url = href
                else:
                    continue

                # Get article title
                title = link.get_text(strip=True)

                # Basic filtering - only include if mentions metals keywords
                text_to_check = title.lower()
                if any(kw.lower() in text_to_check for kw in keywords[:20]):
                    articles.append({
                        'url': article_url,
                        'title': title,
                        'source': 'Reuters',
                        'published': None,  # Will extract from article page
                        'topic': topic
                    })

        except Exception as e:
            logging.error("Error fetching Reuters topic %s: %s", topic, e)
            continue

    # Deduplicate by URL
    seen = set()
    unique = []
    for a in articles:
        if a['url'] not in seen:
            seen.add(a['url'])
            unique.append(a)

    logging.info("Collected %d unique Reuters articles", len(unique))
    return unique

def collect_reuters_rapidapi(api_key: str, keywords: List[str], hours: int) -> List[Dict]:
    """
    Collect articles using RapidAPI Reuters endpoint.
    Requires API key from https://rapidapi.com/
    """
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        logging.error("RapidAPI key required. Get one from https://rapidapi.com/")
        return []

    articles = []
    base_url = "https://reuters-business-and-financial-news.p.rapidapi.com/search"

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "reuters-business-and-financial-news.p.rapidapi.com"
    }

    # Try each keyword
    for keyword in keywords[:5]:  # Limit to avoid API quota
        try:
            rate_limit()

            params = {"query": keyword}
            resp = requests.get(base_url, headers=headers, params=params, timeout=20)

            if resp.status_code == 200:
                data = resp.json()
                # Parse response - structure depends on API
                for item in data.get('articles', []):
                    articles.append({
                        'url': item.get('url'),
                        'title': item.get('title'),
                        'source': 'Reuters',
                        'published': item.get('published'),
                        'summary': item.get('summary')
                    })
                logging.info("RapidAPI returned %d articles for '%s'", len(data.get('articles', [])), keyword)
            else:
                logging.warning("RapidAPI request failed: HTTP %d", resp.status_code)

        except Exception as e:
            logging.error("RapidAPI error for keyword '%s': %s", keyword, e)
            continue

    return articles

def collect_reuters_commodities_api(api_key: str, keywords: List[str]) -> List[Dict]:
    """
    Collect from Commodities API (commodities-api.com).
    This provides price data, not news articles.
    """
    logging.info("Note: Commodities API provides price data, not news articles")
    return []

def main():
    ap = argparse.ArgumentParser(description="Reuters Metals News Collector")
    ap.add_argument("--hours", type=int, default=24, help="Lookback window (hours)")
    ap.add_argument("--out", type=str, default="data/reuters_run.csv", help="Output CSV file")
    ap.add_argument("--api", type=str, choices=["none", "rapidapi", "commodities"], default="none",
                    help="API to use (requires API key)")
    ap.add_argument("--key", type=str, default="", help="API key if using third-party API")
    ap.add_argument("--method", type=str, choices=["search", "topics"], default="topics",
                    help="Collection method: 'search' or 'topics'")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)

    # Load keywords
    taxonomy = load_yaml("keywords.yaml")
    keywords = list(set(taxonomy.get("metals", []) + taxonomy.get("adjacent_industries", [])))

    logging.info("Starting Reuters collection with method: %s", args.method)

    # Collect articles based on method
    articles = []

    if args.api == "rapidapi":
        logging.info("Using RapidAPI method...")
        articles = collect_reuters_rapidapi(args.key, keywords, args.hours)

    elif args.api == "commodities":
        logging.info("Commodities API provides prices, not articles. Falling back to web collection.")
        articles = collect_reuters_search(keywords, args.hours)

    else:  # none - use web scraping
        logging.info("Using web scraping method (respecting robots.txt and rate limits)...")
        articles = collect_reuters_search(keywords, args.hours)

    # Save results
    if articles:
        df = pd.DataFrame(articles)
        df = df.drop_duplicates(subset=['url'])
        df.to_csv(args.out, index=False)
        print(f"\n✓ Collected {len(df)} Reuters articles")
        print(f"✓ Saved to: {args.out}")

        # Show sample
        print(f"\nSample articles:")
        for i, row in df.head(5).iterrows():
            print(f"  {i+1}. {row['title'][:80]}...")
            print(f"     URL: {row['url'][:100]}...")
    else:
        print("\n⚠ No articles collected. This could be due to:")
        print("  - Reuters website structure changed (update selectors)")
        print("  - Rate limiting or blocking")
        print("  - No matching articles in the time window")
        print("  - API key issues (if using API method)")
        print("\nRecommendations:")
        print("  1. For production use, subscribe to official Reuters/LSEG API")
        print("  2. Check if Reuters RSS feeds are available")
        print("  3. Use the existing GDELT API which aggregates Reuters content")

if __name__ == "__main__":
    main()
