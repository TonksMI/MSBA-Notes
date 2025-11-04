# Reuters News Access Guide for Metals & Commodities

**Created**: October 29, 2025
**Purpose**: Guide for accessing Reuters metals/commodities news data

---

## Executive Summary

Reuters news is **not freely accessible** for automated scraping. The Reuters website requires authentication (HTTP 401) for most content pages. However, there are several legitimate ways to access Reuters news for your metals sentiment analysis project.

---

## Access Methods (Ranked by Recommendation)

### ✅ Method 1: GDELT API (FREE - Recommended for Research)

**Status**: FREE, No authentication required
**What it does**: GDELT aggregates Reuters articles along with thousands of other news sources
**Pros**:
- Completely free
- No API key needed
- Already integrated in main.py
- Good coverage of Reuters content
- Includes article metadata

**Cons**:
- Occasional API downtime
- Cannot filter exclusively for Reuters (but can filter by source name)
- May have slight delays in article availability

**How to use**:
```bash
# Already built into main pipeline
python main.py --hours 48 --out data/run.csv

# Reuters-specific collection (when GDELT is working)
python reuters_via_gdelt.py --hours 72 --out data/reuters.csv
```

**GDELT Query Syntax for Reuters**:
```python
# Filter by source
query = '("steel" OR "copper") sourcecommon:Reuters'
```

---

### ✅ Method 2: Official LSEG/Reuters API (PAID - Best for Production)

**Status**: Paid subscription required
**Provider**: LSEG (London Stock Exchange Group)
**What it provides**: Official Reuters news feed with full legal access

**Products**:
1. **LSEG Workspace** - Desktop/mobile interface with Reuters news
2. **LSEG Real-Time Service** - Low-latency API access
3. **Reuters News API** - Direct API access to news content

**Pricing**: Contact LSEG sales (enterprise pricing, typically $$$)

**Pros**:
- Legal, authorized access
- High-quality, real-time data
- Comprehensive coverage
- Structured data with metadata
- Production-grade reliability
- Technical support

**Cons**:
- Expensive (enterprise pricing)
- May require contract/commitment
- Overkill for academic/research projects

**Contact**: https://www.lseg.com/en/data-analytics/financial-data/financial-news-coverage/commodities-news-coverage

---

### ⚠️ Method 3: Third-Party APIs (FREEMIUM)

Several third-party platforms resell Reuters access through APIs:

#### 3a. RapidAPI - Reuters Business News

**URL**: https://rapidapi.com/makingdatameaningful/api/reuters-business-and-financial-news
**Status**: Freemium (limited free tier, paid plans)
**Pricing**:
- Free: Limited requests/month
- Basic: ~$10-50/month
- Pro: ~$100+/month

**How to use**:
```bash
# Get API key from RapidAPI
python reuters_collector.py --api rapidapi --key YOUR_API_KEY --hours 24
```

**Pros**:
- Easier than official LSEG subscription
- Structured API responses
- Free tier for testing
- Pay-as-you-go pricing

**Cons**:
- Not official Reuters API
- Rate limits on free tier
- May have licensing restrictions
- Quality/completeness varies

#### 3b. Zyla Labs - Reuters Market Feed API

**URL**: https://zylalabs.com/api-marketplace/finance/reuters+market+data+api/1361
**Status**: Freemium
**Coverage**: Includes metals, energy, agriculture

**Similar to RapidAPI** - provides structured access to Reuters content through third-party platform.

---

### ❌ Method 4: Direct Web Scraping (NOT RECOMMENDED)

**Status**: Technically possible but **NOT LEGALLY ADVISED**
**Why not**:
- Reuters.com requires authentication (HTTP 401)
- Terms of Service prohibit unauthorized scraping
- robots.txt restrictions
- IP blocking/rate limiting
- Legal risk (copyright violation)
- Unreliable (site structure changes frequently)

**Our testing results**:
```
HTTP 401 - Unauthorized on all topic pages:
- /markets/commodities/
- /business/energy/
- /business/industrials/
```

**Alternative**: If you must scrape, focus on **publicly accessible Reuters blog posts** or **press releases**, but always check robots.txt and ToS first.

---

## Recommended Approach for Your Project

### For Academic/Research Use (FREE)

**Best Option**: Use the existing `main.py` pipeline

```bash
# This already includes GDELT which aggregates Reuters
python main.py --hours 48 --out data/run.csv
```

**Why this works**:
- GDELT aggregates Reuters articles
- You also get Financial Times, WSJ, and global sources
- Completely free and legal
- Good coverage for sentiment analysis research

### For Production/Commercial Use (PAID)

**Best Option**: Subscribe to official LSEG/Reuters API

**Steps**:
1. Contact LSEG sales: https://www.lseg.com/
2. Request access to Reuters News API for commodities
3. Integrate using their official SDK/API
4. Budget for enterprise pricing ($$$)

### For Individual/Startup Use (FREEMIUM)

**Best Option**: RapidAPI Reuters endpoint

```bash
# 1. Sign up at https://rapidapi.com/
# 2. Subscribe to Reuters Business News API
# 3. Get API key
# 4. Run collector
python reuters_collector.py --api rapidapi --key YOUR_KEY --hours 24
```

**Budget**: $10-100/month depending on volume

---

## Alternative News Sources (Free & Accessible)

Instead of focusing exclusively on Reuters, consider these alternatives that are already working in your pipeline:

### ✅ Currently Working Sources

1. **Financial Times Mining** - Excellent metals coverage
   - 25 articles/day
   - High-quality journalism
   - Direct industry focus

2. **Wall Street Journal Markets** - Broad market coverage
   - 20 articles/day
   - Includes commodities
   - Professional reporting

3. **GDELT API** - Global aggregation
   - 250+ articles per query
   - Includes Reuters + many others
   - Comprehensive coverage

### 📰 Additional Free Sources to Consider

**Mining Industry Publications**:
- Mining Technology
- International Mining
- World Coal (for metallurgical coal)

**Industry Associations**:
- World Steel Association (press releases)
- International Copper Study Group
- International Aluminum Institute

**Government Sources**:
- USGS (US Geological Survey) mineral reports
- BGS (British Geological Survey)
- Australian Department of Industry

**Regional News**:
- Mining.com (if RSS is fixed)
- Australian Financial Review (mining section)
- Asian Metal (China/Asia focus)

---

## Implementation Guide

### Option 1: Use Existing Pipeline (Easiest)

The `main.py` script already provides excellent coverage including Reuters content via GDELT:

```bash
# Run the main pipeline
cd "/path/to/metals_news_sentiment_starter"
python main.py --hours 48 --out data/run.csv

# View results
python explore.py data/run.csv
```

**What you get**:
- Reuters articles (via GDELT)
- FT Mining articles (direct RSS)
- WSJ Markets articles (direct RSS)
- Full sentiment analysis
- Metals taxonomy filtering

### Option 2: Reuters-Specific Collection via GDELT

When GDELT API is working properly:

```bash
# Collect Reuters articles via GDELT
python reuters_via_gdelt.py --hours 72 --out data/reuters.csv --max 250
```

### Option 3: Paid API Integration

For RapidAPI integration:

```python
# Add to your config
RAPIDAPI_KEY = "your_key_here"

# Run collector
python reuters_collector.py \
    --api rapidapi \
    --key $RAPIDAPI_KEY \
    --hours 24 \
    --out data/reuters_api.csv
```

---

## Scripts Provided

### 1. `reuters_collector.py`
**Purpose**: Direct Reuters collection (requires API key or web access)
**Status**: Educational reference - Reuters blocks direct web access
**Use**: Template for paid API integration

**Methods**:
- Web scraping (blocked by Reuters - HTTP 401)
- RapidAPI integration (requires API key)
- Extensible for other APIs

### 2. `reuters_via_gdelt.py`
**Purpose**: Collect Reuters articles through GDELT API
**Status**: Free, no API key needed
**Use**: When you specifically want Reuters-sourced articles

**Features**:
- Filters GDELT results for Reuters sources
- No authentication required
- Respects rate limits

### 3. `main.py` (existing)
**Purpose**: Main pipeline with multi-source aggregation
**Status**: Fully operational
**Use**: Primary collection script

**Features**:
- RSS feeds (FT, WSJ)
- GDELT API (includes Reuters)
- Sentiment analysis
- Metals taxonomy filtering

---

## Legal & Ethical Considerations

### ✅ Legal Approaches
1. **Use GDELT API** - They have agreements to aggregate news
2. **Subscribe to official APIs** - Pay for authorized access
3. **Use RSS feeds** - Published for public consumption
4. **Cite sources properly** - Give credit to Reuters

### ❌ Avoid
1. **Scraping Reuters.com** - Violates ToS, requires auth
2. **Circumventing paywalls** - Legal risk
3. **Excessive requests** - Respect rate limits
4. **Reselling data** - Copyright violation

### 📋 Best Practices
- Always identify your bot (User-Agent)
- Respect robots.txt
- Implement rate limiting
- Cache results to minimize requests
- Use data for research/analysis, not republishing
- Attribute sources properly

---

## Troubleshooting

### Problem: "GDELT API returns empty response"

**Solutions**:
1. Check if GDELT is down: https://api.gdeltproject.org/
2. Try different time windows (--hours 72 or --hours 168)
3. Simplify query (fewer keywords)
4. Wait and retry (may be temporary outage)

### Problem: "Reuters returns HTTP 401"

**This is expected** - Reuters requires authentication.

**Solutions**:
1. Use GDELT instead (aggregates Reuters)
2. Subscribe to official LSEG/Reuters API
3. Use RapidAPI third-party access

### Problem: "Not enough articles collected"

**Solutions**:
1. Expand time window: `--hours 168` (1 week)
2. Adjust keywords in keywords.yaml
3. Lower minimum article length in main.py
4. Add more RSS sources
5. Use multiple sources (main.py already does this)

---

## Summary & Recommendations

### For Your Current Project ✅

**Use the existing `main.py` pipeline** - it already provides excellent coverage:
- Working RSS feeds (FT, WSJ)
- GDELT API (aggregates Reuters + global news)
- Complete sentiment analysis pipeline
- Proven to extract 30-40 articles per 48-hour window

**Command**:
```bash
python main.py --hours 48 --out data/run.csv
```

### Future Enhancements 📈

1. **Short term (Free)**:
   - Add more RSS feeds (Mining Technology, industry associations)
   - Optimize GDELT queries for better Reuters coverage
   - Implement caching to avoid re-processing

2. **Medium term (Freemium)**:
   - Trial RapidAPI Reuters endpoint ($10-50/month)
   - Add social media sources (Twitter, LinkedIn)
   - Set up automated scheduling (cron/Airflow)

3. **Long term (Production)**:
   - Subscribe to official LSEG/Reuters API
   - Build comprehensive data warehouse
   - Add real-time monitoring and alerting

---

## Resources

- **LSEG Commodities**: https://www.lseg.com/en/data-analytics/products/workspace/commodities/metals
- **GDELT Project**: https://www.gdeltproject.org/
- **RapidAPI Reuters**: https://rapidapi.com/makingdatameaningful/api/reuters-business-and-financial-news
- **Commodities API**: https://commodities-api.com/ (price data, not news)

---

**Last Updated**: October 29, 2025
**Status**: GDELT experiencing temporary API issues; RSS feeds working normally
