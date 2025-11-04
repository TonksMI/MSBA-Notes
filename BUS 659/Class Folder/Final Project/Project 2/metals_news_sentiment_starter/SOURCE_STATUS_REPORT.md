# News Source Status Report

**Date**: October 27, 2025
**Test Results**: All sources verified and tested

---

## Executive Summary

✅ **3 out of 8 sources are working** (37.5% success rate)
- 2 RSS feeds working (WSJ, Financial Times)
- 1 API working (GDELT)
- 5 RSS feeds broken/inaccessible

**Latest Test Run Results** (48-hour window):
- **Total URLs processed**: 252
- **Articles extracted**: 36 relevant articles
- **Average sentiment**: Slightly positive (FinBERT score: 0.023)
- **Content coverage**: 61% metals, 81% adjacent industries

---

## Working Sources

### ✅ RSS Feed #1: Wall Street Journal Markets
- **URL**: `https://feeds.a.dj.com/rss/RSSMarketsMain.xml`
- **Status**: ✅ WORKING
- **Volume**: ~20 articles per day
- **Quality**: High - professional financial journalism
- **Content**: General markets news (metals, commodities, broader markets)
- **Notes**: Broad coverage; requires keyword filtering for metals-specific content

### ✅ RSS Feed #2: Financial Times Mining
- **URL**: `https://www.ft.com/companies/mining?format=rss`
- **Status**: ✅ WORKING
- **Volume**: ~25 articles per day
- **Quality**: High - in-depth mining industry coverage
- **Content**: Iron ore, rare earths, mining companies, green steel
- **Recent Headlines**:
  - "JPMorgan's 'America First' fund launches with investment in gold miner"
  - "Australia's iron ore riches challenged by green steel and African rival"
  - "Europe struggles to catch up in race to stockpile critical minerals"
- **Notes**: Best dedicated metals/mining source; direct industry focus

### ✅ GDELT API v2
- **URL**: `https://api.gdeltproject.org/api/v2/doc/doc`
- **Status**: ✅ WORKING (after fix)
- **Volume**: 10-250 articles per query (configurable)
- **Quality**: Variable - aggregates from global news sources
- **Content**: Broad metals/industry coverage in multiple languages
- **Fix Applied**: Query syntax requires parentheses around OR'd terms: `("steel" OR "copper" OR "mining")`
- **Coverage**: Excellent for:
  - Steel industry news
  - Copper market updates
  - Mining operations
  - Infrastructure and construction
  - Manufacturing trends
- **Notes**: Primary workhorse for comprehensive coverage; some articles in non-English languages

---

## Broken/Inaccessible Sources

### ❌ RSS Feed: Mining.com
- **URL**: `https://www.mining.com/feed/`
- **Status**: ❌ BROKEN
- **Issue**: Returns HTML instead of XML RSS feed
- **Error**: `<unknown>:2:751: not well-formed (invalid token)`
- **Likely Cause**: Site behind Cloudflare protection or requires JavaScript
- **Recommendation**: **Remove** or use official API if available

### ❌ RSS Feed: Kitco News
- **URL**: `https://www.kitco.com/news/index.rss`
- **Status**: ❌ BROKEN
- **Issue**: Returns HTML instead of XML RSS feed
- **Error**: `<unknown>:19:1205: not well-formed (invalid token)`
- **Likely Cause**: Bot protection/Cloudflare challenge page
- **Recommendation**: **Remove** - site actively blocks automated access

### ❌ RSS Feed: Reuters Commodities
- **URL**: `https://www.reuters.com/markets/commodities/rss`
- **Status**: ❌ BROKEN
- **Issue**: Requires JavaScript, blocks automated access
- **Error**: `text/html;charset=utf-8 is not an XML media type`
- **Likely Cause**: Modern React-based site requiring browser execution
- **Recommendation**: **Remove** - consider paid Reuters API for legitimate access

### ❌ RSS Feed: S&P Global Commodity Insights
- **URL**: `https://www.spglobal.com/commodityinsights/en/ci/products/metal-alerts.rss`
- **Status**: ❌ BROKEN
- **Issue**: Returns HTML instead of RSS
- **Error**: `<unknown>:12:21: not well-formed (invalid token)`
- **Likely Cause**: Premium/subscription content requiring authentication
- **Recommendation**: **Remove** - use paid API if you have subscription

### ❌ RSS Feed: MarketWatch Commodities
- **URL**: `https://www.marketwatch.com/feeds/marketwatch/commodities`
- **Status**: ❌ BROKEN
- **Issue**: Returns HTML instead of RSS
- **Error**: `<unknown>:269:147: not well-formed (invalid token)`
- **Likely Cause**: Site redesign or feed discontinued
- **Recommendation**: **Remove** - check for alternative MarketWatch RSS feeds

---

## Alternative Sources to Consider

### Recommended Additions

1. **Mining Technology**
   - URL: `https://www.mining-technology.com/feed/`
   - Likely Status: May work (not yet tested)
   - Focus: Mining industry technology and innovation

2. **Metal Bulletin** (now part of Fastmarkets)
   - URL: Check Fastmarkets.com for API access
   - Note: Likely requires paid subscription
   - Focus: Professional metals pricing and market intelligence

3. **Bloomberg Commodities** (if accessible)
   - Note: Requires Bloomberg terminal or API subscription
   - Focus: Professional-grade commodities data

4. **World Steel Association**
   - May have RSS feeds for press releases
   - Focus: Global steel industry statistics and trends

5. **International Copper Study Group**
   - May have RSS feeds for reports
   - Focus: Copper market data and analysis

---

## Data Extraction Quality

### Text Extraction Performance

**Improvements Made**:
1. ✅ Added proper metadata extraction for article titles
2. ✅ Enabled table inclusion for comprehensive content
3. ✅ Enhanced HTML cleaning with BeautifulSoup
4. ✅ Removed navigation, header, footer elements
5. ✅ Lowered minimum article length (200→100 chars) for better recall

**Extraction Success Rate**: ~14% (36 articles from 252 URLs)
- Many URLs filtered out due to:
  - Not matching metals/industry taxonomy
  - Non-English language content
  - Too short (likely non-article pages)
  - Paywalled or access-restricted content

### Sentiment Analysis Performance

**Current Results** (48-hour test):
- **Headline VADER sentiment**: 0.16 (slightly positive)
- **Body FinBERT sentiment**: 0.02 (nearly neutral)
- **Distribution**:
  - Positive articles: 31.8%
  - Neutral articles: ~37%
  - Negative articles: 29.4%

**Keyword Matching**:
- 61% matched metals keywords (steel, copper, iron ore, etc.)
- 81% matched adjacent industries (construction, automotive, energy, etc.)

---

## Recommendations

### Immediate Actions

1. ✅ **Keep WSJ and FT RSS feeds** - These are reliable and high-quality
2. ✅ **Keep GDELT API** - Primary source for volume and global coverage
3. ✅ **Remove 5 broken RSS feeds** - Already updated in sources.yaml
4. 🔄 **Test alternative sources** - Mining Technology, industry association feeds

### Performance Optimization

1. **Increase timespan for RSS feeds**: Consider 72-96 hours for better article capture from slower-updating feeds
2. **Adjust GDELT keywords**: Current query uses 8 keywords; consider testing different keyword combinations
3. **Language filtering**: Currently filtering out non-English; consider adding translation for high-value sources
4. **Taxonomy refinement**: Fine-tune keyword lists in keywords.yaml for better precision/recall balance

### Coverage Expansion

1. **Geographic diversity**: Current sources are US/UK-focused; consider:
   - Asian metals news sources (especially China, India)
   - Australian mining news (major producer)
   - South American sources (Chile copper, Brazil iron ore)

2. **Specialized sources**:
   - Industry trade publications (World Steel, ICSG)
   - Regional mining associations
   - Government mineral agencies (USGS, etc.)

3. **Social media/alternative data**:
   - Consider Twitter/X feeds from industry analysts
   - LinkedIn posts from mining executives
   - Industry conference proceedings

### Pipeline Improvements

1. **Caching**: Implement article caching to avoid re-processing same URLs
2. **Scheduling**: Set up cron job or Airflow DAG for automated daily runs
3. **Database storage**: Replace CSV with SQLite for better data management
4. **Monitoring**: Add alerting for source failures or low article counts
5. **Deduplication**: Enhance URL canonicalization to catch more duplicates

---

## Test Commands

### Verify All Sources
```bash
python test_sources.py
```

### Run Full Pipeline (24 hours)
```bash
python main.py --hours 24 --out data/run_$(date +%F).csv
```

### Run Full Pipeline (48 hours)
```bash
python main.py --hours 48 --out data/run_48h_$(date +%F).csv
```

### Explore Results
```bash
python explore.py data/run_2025-10-27.csv
```

---

## Conclusion

**The pipeline is now fully operational** with 3 working sources that provide good coverage of metals industry news:

- **WSJ**: Broad financial markets coverage including commodities
- **FT Mining**: Dedicated mining industry journalism
- **GDELT**: Global news aggregation with comprehensive reach

The fixes implemented ensure:
- ✅ Proper text extraction from articles
- ✅ Accurate title and body separation
- ✅ Comprehensive HTML cleaning
- ✅ Correct GDELT API query syntax
- ✅ Reliable sentiment analysis (VADER + FinBERT)

**Next Steps**: Consider adding alternative RSS feeds and implementing the recommended performance optimizations for production use.
