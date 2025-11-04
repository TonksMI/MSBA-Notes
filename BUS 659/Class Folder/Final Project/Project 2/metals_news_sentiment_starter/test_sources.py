#!/usr/bin/env python3
"""
Test script to verify all news sources are working and returning articles.
"""
import feedparser
import requests
from datetime import datetime, timedelta
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def test_rss_feed(url: str, name: str = None):
    """Test a single RSS feed and report status."""
    print(f"\n{'='*80}")
    print(f"Testing: {name or url}")
    print(f"{'='*80}")

    try:
        feed = feedparser.parse(url)

        # Check for parsing errors
        if hasattr(feed, 'bozo') and feed.bozo:
            print(f"⚠️  WARNING: Feed has parsing issues: {feed.get('bozo_exception', 'Unknown error')}")

        # Check feed metadata
        if hasattr(feed, 'feed'):
            print(f"Feed Title: {feed.feed.get('title', 'N/A')}")
            print(f"Feed Description: {feed.feed.get('description', 'N/A')[:100]}...")

        # Check entries
        num_entries = len(feed.entries)
        print(f"\n✓ Found {num_entries} entries")

        if num_entries == 0:
            print("⚠️  WARNING: No articles found in feed")
            return False

        # Show recent entries (last 5)
        print(f"\nRecent articles (showing up to 5):")
        for i, entry in enumerate(feed.entries[:5], 1):
            title = entry.get('title', 'No title')
            link = entry.get('link', 'No link')
            pub_date = entry.get('published', entry.get('updated', 'No date'))
            print(f"\n  {i}. {title[:80]}...")
            print(f"     URL: {link[:100]}...")
            print(f"     Published: {pub_date}")

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to fetch feed: {e}")
        return False

def test_gdelt_api():
    """Test GDELT API with a simple query."""
    print(f"\n{'='*80}")
    print(f"Testing: GDELT API")
    print(f"{'='*80}")

    base = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Simple test query - GDELT requires OR'd terms in parentheses
    params = {
        "query": '("steel" OR "copper" OR "mining")',
        "mode": "ArtList",
        "maxrecords": "10",
        "format": "json",
        "timespan": "24h"
    }

    try:
        r = requests.get(base, params=params, timeout=30)
        print(f"Request URL: {r.url}")
        print(f"Status Code: {r.status_code}")

        if r.status_code == 200:
            if not r.text or len(r.text.strip()) == 0:
                print("⚠️  WARNING: GDELT returned empty response")
                print("This may be due to:")
                print("  - Rate limiting")
                print("  - No articles matching the query in the timespan")
                print("  - Temporary API issues")
                return False

            try:
                data = r.json()
                if "articles" in data:
                    num_articles = len(data["articles"])
                    print(f"\n✓ Found {num_articles} articles")

                    if num_articles > 0:
                        print(f"\nSample articles (showing up to 3):")
                        for i, article in enumerate(data["articles"][:3], 1):
                            print(f"\n  {i}. {article.get('title', 'No title')[:80]}...")
                            print(f"     Source: {article.get('sourceCommonName', 'Unknown')}")
                            print(f"     URL: {article.get('url', 'N/A')[:100]}...")
                        return True
                    else:
                        print("⚠️  WARNING: No articles found in GDELT response")
                        return False
                else:
                    print(f"⚠️  WARNING: Response missing 'articles' key. Keys: {list(data.keys())}")
                    return False
            except ValueError as e:
                print(f"❌ ERROR: Failed to parse JSON response: {e}")
                print(f"Response preview: {r.text[:500]}")
                return False
        else:
            print(f"❌ ERROR: HTTP {r.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("METALS NEWS SENTIMENT - SOURCE VERIFICATION TEST")
    print("="*80)

    # Load sources
    sources = load_yaml("sources.yaml")

    # Test RSS feeds
    rss_urls = sources.get("rss", [])
    print(f"\n\nTesting {len(rss_urls)} RSS feeds...")

    rss_results = {}
    for i, url in enumerate(rss_urls, 1):
        result = test_rss_feed(url, f"RSS Feed #{i}")
        rss_results[url] = result

    # Test GDELT
    print("\n\nTesting GDELT API...")
    gdelt_result = test_gdelt_api()

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\nRSS Feeds:")
    working_rss = sum(1 for v in rss_results.values() if v)
    total_rss = len(rss_results)
    print(f"  Working: {working_rss}/{total_rss}")

    for url, status in rss_results.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {url}")

    print(f"\nGDELT API:")
    print(f"  {'✓ Working' if gdelt_result else '✗ Not working'}")

    print(f"\n{'='*80}")
    total_sources = total_rss + 1
    total_working = working_rss + (1 if gdelt_result else 0)
    print(f"Overall: {total_working}/{total_sources} sources working ({total_working/total_sources*100:.1f}%)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
