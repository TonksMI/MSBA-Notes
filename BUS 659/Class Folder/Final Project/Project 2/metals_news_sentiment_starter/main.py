#!/usr/bin/env python3
"""
Main pipeline: ingest (RSS + GDELT + optional crawl), filter, extract, sentiment, export CSV.
Run:
  python main.py --hours 24 --out data/run.csv --crawl False
"""
from __future__ import annotations
import os, re, time, json, math, hashlib, argparse, logging, random, itertools
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import feedparser
import yaml
import requests
from urllib.parse import urlparse, urljoin
from dateutil import parser as dtparse
from langdetect import detect as lang_detect, LangDetectException
from tqdm import tqdm

# Extraction
import trafilatura
from readability import Document
from bs4 import BeautifulSoup

# Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    _TRANSFORMERS_OK = True
except Exception:
    _TRANSFORMERS_OK = False

# Optional NER
try:
    import spacy
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

UA = "MetalsNewsSentimentBot/1.0 (+contact: research@example.com)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8","ignore")).hexdigest()

def canonicalize_url(u: str) -> str:
    # Basic canonicalization: strip UTM & fragments
    if not u: return u
    pu = urlparse(u)
    qs = "&".join([kv for kv in pu.query.split("&") if not kv.lower().startswith(("utm_", "mc_", "fbclid")) and kv != ""])
    return pu._replace(query=qs, fragment="").geturl()

def within_hours(dt: datetime, hours: int) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt >= cutoff

def guess_published(entry) -> Optional[datetime]:
    for k in ("published", "updated", "created"):
        if k + "_parsed" in entry and entry[k + "_parsed"]:
            try:
                return datetime.fromtimestamp(time.mktime(entry[k + "_parsed"]), tz=timezone.utc)
            except Exception:
                pass
        if k in entry:
            try:
                return dtparse.parse(entry[k]).astimezone(timezone.utc)
            except Exception:
                pass
    return None

def safe_get(url: str, timeout=20) -> Optional[requests.Response]:
    try:
        resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200 and "text" in resp.headers.get("Content-Type",""):
            return resp
    except requests.RequestException:
        return None
    return None

def extract_text(url: str, html: Optional[str] = None) -> Tuple[str, str]:
    """Return (title, text) using trafilatura; fallback to readability-lxml."""
    if html is None:
        r = safe_get(url)
        html = r.text if r else None
    if not html:
        return "", ""

    # Try trafilatura first with metadata extraction
    try:
        metadata = trafilatura.extract_metadata(html)
        title_from_meta = metadata.title if metadata else ""

        # Extract main text body with better settings
        downloaded = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,  # Include tables for better coverage
            favor_recall=True,    # Favor getting all content
            no_fallback=False,
            url=url,
            include_formatting=False,
            output_format='txt'
        )

        if downloaded and len(downloaded.strip()) > 100:
            return title_from_meta or "", downloaded.strip()
    except Exception as e:
        logging.debug("Trafilatura extraction failed for %s: %s", url, e)

    # Fallback to readability-lxml
    try:
        doc = Document(html)
        title = doc.short_title() or ""
        # Better HTML cleaning
        soup = BeautifulSoup(doc.summary(), 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return title, text
    except Exception as e:
        logging.debug("Readability extraction failed for %s: %s", url, e)
        return "", ""

def detect_lang(text: str) -> str:
    try:
        return lang_detect(text or "en")
    except LangDetectException:
        return "unknown"

def load_finbert():
    if not _TRANSFORMERS_OK:
        return None
    model_name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True, truncation=True)
    return clf

def finbert_score(clf, text: str) -> Dict[str, float]:
    """Returns dict with negative/neutral/positive probs and a signed score in [-1,1]."""
    if not clf or not text:
        return {"neg": np.nan, "neu": np.nan, "pos": np.nan, "score": np.nan}
    try:
        res = clf(text[:4000])  # FinBERT is robust; truncate long text
        # res is a list of list of dicts; take first item
        lab = {d["label"].lower(): float(d["score"]) for d in res[0]}
        score = lab.get("positive",0) - lab.get("negative",0)
        return {"neg": lab.get("negative", np.nan), "neu": lab.get("neutral", np.nan), "pos": lab.get("positive", np.nan), "score": score}
    except Exception:
        return {"neg": np.nan, "neu": np.nan, "pos": np.nan, "score": np.nan}

def vader_score(analyzer, text: str) -> float:
    if not text: return np.nan
    try:
        return analyzer.polarity_scores(text).get("compound", np.nan)
    except Exception:
        return np.nan

def compile_keyword_regex(keywords: List[str]) -> re.Pattern:
    esc = [re.escape(k) for k in keywords if k and isinstance(k, str)]
    return re.compile(r"\b(" + "|".join(esc) + r")\b", flags=re.I)

def text_matches_taxonomy(title: str, text: str, taxonomy: Dict[str, List[str]]) -> Dict[str, bool]:
    flags = {}
    for bucket, words in taxonomy.items():
        if not words: 
            flags[bucket] = False
            continue
        rx = compile_keyword_regex(words)
        flags[bucket] = bool(rx.search(title or "") or rx.search(text or ""))
    return flags

def gdelt_query(hours: int, keywords: List[str], max_articles: int = 250) -> List[Dict]:
    """
    Use GDELT v2 API v2/doc/doc?query= ... &mode=ArtList
    Note: This is a simple example; refine queries per your needs.
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Use more focused keywords to avoid overwhelming the API
    top_keywords = keywords[:8]  # Reduce number of keywords

    # GDELT query syntax:
    # - Parentheses ONLY for OR'd terms with 2+ keywords
    # - No quotes needed around simple keywords
    # - Multi-word phrases need quotes
    if len(top_keywords) == 1:
        q = top_keywords[0]
    else:
        # Only wrap multi-word phrases in quotes
        formatted_kw = [f'"{k}"' if ' ' in k else k for k in top_keywords]
        q = "(" + " OR ".join(formatted_kw) + ")"

    params = {
        "query": q,
        "mode": "ArtList",
        "maxrecords": str(max_articles),
        "format": "json",
        "timespan": f"{hours}h"
    }
    try:
        r = SESSION.get(base, params=params, timeout=60)  # Increased timeout for slow GDELT
        logging.info("GDELT API URL: %s", r.url)
        r.raise_for_status()

        # Check if response is empty or not JSON
        if not r.text or len(r.text.strip()) == 0:
            logging.warning("GDELT returned empty response")
            return []

        data = r.json()

        # Check if articles key exists
        if "articles" not in data:
            logging.warning("GDELT response missing 'articles' key. Keys: %s", list(data.keys()))
            return []

        arts = []
        for a in data.get("articles", []):
            arts.append({
                "source": a.get("sourceCommonName", "Unknown"),
                "title": a.get("title", ""),
                "url": canonicalize_url(a.get("url", "")),
                "published": a.get("seendate", ""),
                "language": a.get("language", "en"),
            })
        logging.info("GDELT returned %d articles", len(arts))
        return arts
    except requests.exceptions.RequestException as e:
        logging.warning("GDELT request failed: %s", e)
        return []
    except ValueError as e:
        logging.warning("GDELT JSON parsing failed: %s", e)
        if 'r' in locals():
            logging.warning("Response preview: %s", r.text[:500])
        return []
    except Exception as e:
        logging.warning("GDELT query failed: %s", e)
        if 'r' in locals():
            logging.warning("Response status: %d, preview: %s", r.status_code, r.text[:500])
        return []

def collect_rss(urls: List[str], hours: int) -> List[Dict]:
    rows = []
    for u in urls:
        try:
            feed = feedparser.parse(u)
            for e in feed.entries:
                pub = guess_published(e)
                if pub and not within_hours(pub, hours): continue
                rows.append({
                    "source": feed.feed.get("title", urlparse(u).netloc),
                    "title": e.get("title"),
                    "url": canonicalize_url(e.get("link")),
                    "published": pub.isoformat() if pub else None,
                    "language": None
                })
        except Exception as e:
            logging.warning("RSS error for %s: %s", u, e)
    return rows

def safe_language(text_title, text_body) -> str:
    # Prefer title language if body is short/unknown
    for t in (text_title, text_body):
        lang = detect_lang(t)
        if lang and lang != "unknown":
            return lang
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24, help="Lookback window (hours)")
    ap.add_argument("--out", type=str, default="data/run.csv")
    ap.add_argument("--crawl", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False, help="Enable optional site crawling (respect robots)")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)

    taxonomy = load_yaml("keywords.yaml")
    sources = load_yaml("sources.yaml")
    keywords = list(set(taxonomy.get("metals", []) + taxonomy.get("adjacent_industries", [])))

    logging.info("Collecting RSS within last %sh", args.hours)
    rss_rows = collect_rss(sources.get("rss", []), args.hours)

    logging.info("Collecting GDELT within last %sh", args.hours)
    gdelt_rows = gdelt_query(args.hours, keywords, max_articles=250)

    rows = rss_rows + gdelt_rows

    # De-dup by URL
    seen = set()
    unique = []
    for r in rows:
        u = r.get("url")
        if not u or u in seen: 
            continue
        seen.add(u)
        unique.append(r)

    # Initialize sentiment tools
    vader = SentimentIntensityAnalyzer()
    finbert = load_finbert()

    out_records = []
    for r in tqdm(unique, desc="Processing"):
        url = r["url"]
        # Fetch/Extract
        title_ex, body = extract_text(url)
        title = r.get("title") or title_ex
        if not body or len(body) < 100:
            # Skip very short / likely non-article pages (lowered threshold to capture more content)
            continue
        # Language
        language = safe_language(title or "", body or "")
        if language and language.startswith("en") is False:
            # Keep non-English if you plan to translate; here we skip
            continue

        # Taxonomy match
        flags = text_matches_taxonomy(title or "", body, taxonomy)
        if not (flags.get("metals") or any(flags.values())):
            # Skip if nothing from taxonomy shows up
            continue

        # Sentiment
        headline_vader = vader_score(vader, title or "")
        finbert_body = finbert_score(finbert, body[:3000])

        # Record
        out_records.append({
            "source": r.get("source"),
            "url": url,
            "published": r.get("published"),
            "title": title,
            "headline_vader": headline_vader,
            "finbert_neg": finbert_body["neg"],
            "finbert_neu": finbert_body["neu"],
            "finbert_pos": finbert_body["pos"],
            "finbert_score": finbert_body["score"],
            "lang": language,
            **{f"match_{k}": bool(v) for k, v in flags.items()}
        })

    df = pd.DataFrame(out_records).drop_duplicates(subset=["url"])
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
