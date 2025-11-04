# Data Extraction Specification Document
## Reliance Inc. Analytics Project - Data Sources A, B, C, D

**Project**: Reliance Inc. Predictive/Prescriptive Analytics Platform
**Document Version**: 1.0
**Date**: October 23, 2025
**Purpose**: Specification for extracting and processing data from four primary external data sources

---

## Executive Summary

This specification document outlines the data extraction methodology for four critical data sources supporting Reliance Inc.'s analytics initiatives focused on steel price forecasting, demand nowcasting, and supply chain optimization.

**Primary Use Cases:**
- Steel Price Early-Warning Model (1–3 month horizon)
- End-Market Demand Nowcast (volume forecasting)
- Freight Cost & Capacity Monitoring
- Tariff & Policy Impact Analysis

---

## Data Source Architecture

### Source A: FRED Economic Data (Federal Reserve Economic Data)
**Priority**: HIGH
**Update Frequency**: Monthly (some series weekly)
**Access Method**: API + Manual CSV backup

#### Series to Extract

| Series ID | Description | Frequency | Use Case |
|-----------|-------------|-----------|----------|
| WPU1017 | Producer Price Index - Steel Mill Products | Monthly | Price forecasting |
| PCU331331 | Primary Metal Manufacturing PPI | Monthly | Input cost tracking |
| TRUCKD11 | Truck Tonnage Index (ATA) | Monthly | Demand indicator |
| RAILFRTCARLOADSD11 | Rail Carloads of All Freight | Monthly | Logistics monitoring |

#### Technical Specifications
```python
# API Endpoint
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
API_KEY = "[REQUIRED - Register at https://fred.stlouisfed.org/docs/api/api_key.html]"

# Parameters
params = {
    'series_id': 'WPU1017',
    'api_key': API_KEY,
    'file_type': 'json',
    'observation_start': '2020-01-01'
}
```

#### Data Schema
```
fred_series:
  - series_id: VARCHAR(50)
  - date: DATE
  - value: DECIMAL(18,4)
  - realtime_start: DATE
  - realtime_end: DATE
  - source: VARCHAR(10) = 'FRED'
  - extraction_timestamp: TIMESTAMP
```

#### Quality Checks
- ✓ No null values in `value` field
- ✓ Date continuity (no gaps > 45 days for monthly series)
- ✓ Value range validation (e.g., PPI > 0)
- ✓ Compare against prior month (flag changes > 20%)

#### Extraction Schedule
- **Initial Load**: Full historical from 2015-01-01
- **Refresh**: Monthly on 15th (after BLS releases)
- **Retry Logic**: 3 attempts with exponential backoff

---

### Source B: Census Bureau Data
**Priority**: HIGH
**Update Frequency**: Monthly
**Access Method**: API + Bulk CSV Download

#### Datasets to Extract

1. **Construction Spending (VIP)**
   - Series: Non-residential & Public construction
   - Endpoint: `https://api.census.gov/data/timeseries/eits/building`
   - Key for: Demand forecasting in construction segment

2. **Manufacturers' Shipments, Inventories, and Orders (M3)**
   - Series: Primary Metals (NAICS 331)
   - Endpoint: `https://api.census.gov/data/timeseries/eits/m3`
   - Key for: End-market demand signal

3. **Steel Import Data (FT900A Steel Tables)**
   - Source: Census Foreign Trade Division
   - Format: Excel/CSV from https://www.census.gov/foreign-trade/statistics/
   - Key for: Competition analysis

#### Technical Specifications
```python
# Census API Configuration
BASE_URL_CONSTRUCTION = "https://api.census.gov/data/timeseries/eits/building"
BASE_URL_M3 = "https://api.census.gov/data/timeseries/eits/m3"
API_KEY = "[REQUIRED - Register at https://api.census.gov/data/key_signup.html]"

# Example query for construction spending
params_construction = {
    'get': 'cell_value,time_slot_id,category_code',
    'for': 'us:*',
    'time': 'from 2020-01 to 2025-10',
    'category_code': 'TTNONRES,TTPUB'  # Non-residential & Public
}
```

#### Data Schema
```
census_construction:
  - date: DATE
  - category_code: VARCHAR(20)
  - category_name: VARCHAR(100)
  - value_millions: DECIMAL(18,2)
  - seasonally_adjusted: BOOLEAN
  - source: VARCHAR(20) = 'CENSUS_VIP'
  - extraction_timestamp: TIMESTAMP

census_m3:
  - date: DATE
  - naics_code: VARCHAR(10)
  - indicator: VARCHAR(50)  # 'shipments', 'new_orders', 'inventories'
  - value_millions: DECIMAL(18,2)
  - seasonally_adjusted: BOOLEAN
  - source: VARCHAR(20) = 'CENSUS_M3'
  - extraction_timestamp: TIMESTAMP

census_steel_imports:
  - date: DATE
  - country_code: VARCHAR(3)
  - product_code: VARCHAR(20)
  - quantity_metric_tons: DECIMAL(18,2)
  - value_thousands: DECIMAL(18,2)
  - unit_price: DECIMAL(18,2)
  - source: VARCHAR(20) = 'CENSUS_FT900'
  - extraction_timestamp: TIMESTAMP
```

#### Quality Checks
- ✓ Seasonally adjusted vs. non-adjusted consistency
- ✓ Month-over-month changes within historical ranges
- ✓ Import quantities sum correctly by product group
- ✓ Unit prices align with FRED steel PPI (±15%)

#### Extraction Schedule
- **Initial Load**: Full historical from 2015-01-01
- **Refresh**: Monthly on 20th (5 days after Census release)
- **Special Handling**: Import data often revised; store revision history

---

### Source C: Energy Information Administration (EIA)
**Priority**: MEDIUM
**Update Frequency**: Weekly
**Access Method**: API

#### Series to Extract

1. **Diesel Fuel Prices**
   - Series: U.S. No 2 Diesel Retail Prices
   - Endpoint: EIA Petroleum API v2
   - Key for: Freight cost modeling

2. **CO₂ Emissions Data**
   - Series: CO₂ emissions per gallon of diesel
   - Key for: Scope 3 emissions calculations

#### Technical Specifications
```python
# EIA API Configuration
BASE_URL = "https://api.eia.gov/v2/"
API_KEY = "[REQUIRED - Register at https://www.eia.gov/opendata/]"

# Diesel prices endpoint
endpoint_diesel = "petroleum/pri/spt/data/"
params_diesel = {
    'api_key': API_KEY,
    'frequency': 'weekly',
    'data[0]': 'value',
    'facets[product][]': 'EPD2D',  # No. 2 Diesel
    'start': '2020-01',
    'sort[0][column]': 'period',
    'sort[0][direction]': 'desc'
}
```

#### Data Schema
```
eia_diesel_prices:
  - period: DATE
  - product_code: VARCHAR(10)
  - product_name: VARCHAR(100)
  - area_code: VARCHAR(10)
  - area_name: VARCHAR(100)
  - price_dollars_per_gallon: DECIMAL(10,4)
  - units: VARCHAR(20)
  - source: VARCHAR(10) = 'EIA'
  - extraction_timestamp: TIMESTAMP

eia_emissions:
  - fuel_type: VARCHAR(50)
  - co2_kg_per_gallon: DECIMAL(10,6)
  - source: VARCHAR(10) = 'EIA'
  - effective_date: DATE
  - extraction_timestamp: TIMESTAMP
```

#### Quality Checks
- ✓ Weekly prices show logical progression (no jumps > $0.50/gallon)
- ✓ National average aligns with regional averages (weighted)
- ✓ CO₂ factors remain constant (validate against EPA factors)

#### Extraction Schedule
- **Initial Load**: Full historical from 2020-01-01
- **Refresh**: Weekly on Tuesdays (day after EIA update)
- **Backfill**: Automatic for missed weeks

---

### Source D: Freight & Logistics Indices
**Priority**: MEDIUM
**Update Frequency**: Monthly (Cass), Weekly (Freightos)
**Access Method**: Web Scraping + Manual Entry (Cass), API (Freightos if available)

#### Datasets to Extract

1. **Cass Freight Index**
   - Source: https://www.cassinfo.com/freight-audit-indexes
   - Components: Shipments Index, Expenditures Index
   - Format: Manual data entry from monthly report
   - Key for: Freight market conditions

2. **Freightos Baltic Index (FBX)**
   - Source: https://fbx.freightos.com/
   - Focus: Container shipping rates (if relevant for imports)
   - Format: API or CSV download
   - Key for: Import cost modeling

3. **BTS Freight Transportation Services Index (TSI)**
   - Source: Bureau of Transportation Statistics
   - Endpoint: https://data.bts.gov/
   - Key for: Overall freight activity

#### Technical Specifications
```python
# Cass Freight Index (Manual Entry Template)
# Data published monthly around the 25th
# Entry format:
cass_template = {
    'year': 2025,
    'month': 10,
    'shipments_index': 0.00,  # Base year varies
    'expenditures_index': 0.00,
    'yoy_shipments_pct': 0.00,
    'yoy_expenditures_pct': 0.00
}

# BTS TSI (API or Bulk Download)
# Dataset: Freight Transportation Services Index
# https://data.bts.gov/stories/s/Freight-Transportation-Services-Index-TSI-/5pen-s6b6
```

#### Data Schema
```
freight_cass_index:
  - date: DATE  # First day of month
  - shipments_index: DECIMAL(10,2)
  - expenditures_index: DECIMAL(10,2)
  - yoy_shipments_change_pct: DECIMAL(6,2)
  - yoy_expenditures_change_pct: DECIMAL(6,2)
  - source: VARCHAR(20) = 'CASS'
  - extraction_timestamp: TIMESTAMP

freight_fbx:
  - date: DATE
  - route: VARCHAR(100)  # e.g., 'China-US West Coast'
  - container_size: VARCHAR(10)  # '40ft'
  - rate_usd: DECIMAL(10,2)
  - source: VARCHAR(20) = 'FREIGHTOS'
  - extraction_timestamp: TIMESTAMP

freight_bts_tsi:
  - date: DATE
  - index_value: DECIMAL(10,2)
  - seasonally_adjusted: BOOLEAN
  - mode: VARCHAR(50)  # 'All', 'Truck', 'Rail', etc.
  - source: VARCHAR(20) = 'BTS_TSI'
  - extraction_timestamp: TIMESTAMP
```

#### Quality Checks
- ✓ Cass indices should show correlation with FRED truck tonnage (>0.7)
- ✓ TSI values should be > 0 and show reasonable month-over-month changes
- ✓ FBX container rates should align with shipping news (spot check)

#### Extraction Schedule
- **Cass**: Monthly on 26th (day after typical publication)
- **BTS TSI**: Monthly on 15th
- **FBX**: Weekly on Thursdays (if automated)

---

## Cross-Source Integration

### Data Warehouse Design

```sql
-- Fact table for time series
CREATE TABLE fact_economic_indicators (
    indicator_id SERIAL PRIMARY KEY,
    source VARCHAR(20) NOT NULL,
    series_code VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value DECIMAL(18,4) NOT NULL,
    units VARCHAR(50),
    is_seasonally_adjusted BOOLEAN,
    extraction_timestamp TIMESTAMP NOT NULL,
    UNIQUE(source, series_code, date)
);

-- Dimension table for metadata
CREATE TABLE dim_indicator_metadata (
    series_code VARCHAR(50) PRIMARY KEY,
    source VARCHAR(20) NOT NULL,
    full_name VARCHAR(200),
    description TEXT,
    frequency VARCHAR(20),
    units VARCHAR(50),
    base_period VARCHAR(50),
    first_available_date DATE,
    update_schedule VARCHAR(100)
);

-- Audit table
CREATE TABLE audit_data_extraction (
    audit_id SERIAL PRIMARY KEY,
    source VARCHAR(20) NOT NULL,
    extraction_start TIMESTAMP,
    extraction_end TIMESTAMP,
    records_extracted INTEGER,
    status VARCHAR(20),  -- 'SUCCESS', 'PARTIAL', 'FAILED'
    error_message TEXT
);
```

### Master Pipeline Flow

```
1. Source A (FRED) → Extract → Validate → Load → fact_economic_indicators
2. Source B (Census) → Extract → Transform → Validate → Load → fact_economic_indicators
3. Source C (EIA) → Extract → Validate → Load → fact_economic_indicators
4. Source D (Freight) → Extract → Validate → Load → fact_economic_indicators

5. Cross-Validation:
   - Check temporal alignment across sources
   - Validate expected correlations (e.g., steel PPI vs. construction spending)
   - Generate quality report

6. Feature Engineering:
   - Calculate month-over-month changes
   - Compute rolling volatilities
   - Create lag features for modeling
```

---

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1)
- [ ] Register for API keys (FRED, Census, EIA)
- [ ] Set up PostgreSQL database with schema above
- [ ] Create Python virtual environment with dependencies
- [ ] Configure secure credential storage (e.g., `.env` file, secrets manager)

### Phase 2: Source A & C Development (Week 2)
- [ ] Develop FRED extractor (API-based)
- [ ] Develop EIA extractor (API-based)
- [ ] Implement validation logic
- [ ] Build loading functions
- [ ] Create unit tests

### Phase 3: Source B Development (Week 3)
- [ ] Develop Census construction spending extractor
- [ ] Develop Census M3 extractor
- [ ] Develop Census steel imports extractor (CSV-based)
- [ ] Implement validation logic
- [ ] Create integration tests

### Phase 4: Source D Development (Week 4)
- [ ] Create Cass Freight Index manual entry interface
- [ ] Develop BTS TSI extractor
- [ ] Research Freightos FBX access options
- [ ] Implement freight data validation

### Phase 5: Integration & Testing (Week 5)
- [ ] Build master orchestration script
- [ ] Implement cross-source validation
- [ ] Create extraction monitoring dashboard
- [ ] Conduct end-to-end testing with full historical load

### Phase 6: Deployment & Documentation (Week 6)
- [ ] Schedule automated extraction jobs (cron/Airflow)
- [ ] Create operational runbook
- [ ] Document data dictionary
- [ ] Train team on monitoring procedures

---

## Dependencies & Prerequisites

### Python Libraries
```python
# requirements.txt
requests>=2.31.0          # API calls
pandas>=2.0.0             # Data manipulation
numpy>=1.24.0             # Numerical operations
sqlalchemy>=2.0.0         # Database ORM
psycopg2-binary>=2.9.0    # PostgreSQL adapter
python-dotenv>=1.0.0      # Environment variables
schedule>=1.2.0           # Job scheduling
pytest>=7.4.0             # Testing
great-expectations>=0.18.0  # Data validation
```

### API Keys Required
1. **FRED API Key**: https://fred.stlouisfed.org/docs/api/api_key.html
2. **Census API Key**: https://api.census.gov/data/key_signup.html
3. **EIA API Key**: https://www.eia.gov/opendata/

### Infrastructure
- PostgreSQL 14+ or cloud database (AWS RDS, Azure SQL)
- Python 3.9+
- 10GB storage for initial historical data
- Scheduled job runner (cron, Apache Airflow, or cloud scheduler)

---

## Error Handling & Monitoring

### Extraction Error Scenarios

| Error Type | Detection | Response | Notification |
|------------|-----------|----------|--------------|
| API Rate Limit | HTTP 429 | Exponential backoff | Log warning |
| API Down | HTTP 500/503 | Retry 3x, then manual flag | Email ops team |
| Schema Change | Validation failure | Halt load, log error | Email analytics lead |
| Missing Data | Gap detection | Attempt backfill | Log warning |
| Outlier Values | Statistical check | Flag for review | Dashboard alert |

### Monitoring Dashboard Components
1. **Last Successful Extraction** (per source)
2. **Records Extracted This Month** (vs. expected)
3. **Data Quality Score** (0-100, composite metric)
4. **API Health Status** (green/yellow/red)
5. **Outstanding Alerts** (requires manual review)

---

## Data Retention & Archival

- **Raw Extractions**: Retain all raw API responses for 90 days (JSON/CSV backups)
- **Processed Data**: Retain indefinitely in fact tables (disk space permitting)
- **Audit Logs**: Retain for 1 year
- **Revision History**: For Census data, maintain full revision chain (M3 often revised)

---

## Governance & Compliance

### Data Usage Restrictions
- **FRED**: Cite Federal Reserve Bank of St. Louis; no redistribution
- **Census**: Public domain, but attribute to U.S. Census Bureau
- **EIA**: Public domain, attribute to U.S. Energy Information Administration
- **Cass**: Check terms of use; may have redistribution restrictions

### Data Refresh SLA
- **Priority HIGH sources (A, B)**: Within 48 hours of official publication
- **Priority MEDIUM sources (C, D)**: Within 1 week of publication
- **Downtime tolerance**: <5% of scheduled extraction jobs may fail

---

## Success Metrics

### KPIs for Data Extraction Process
1. **Extraction Reliability**: >95% success rate on scheduled runs
2. **Data Freshness**: <5 day lag from publication to availability in warehouse
3. **Data Quality**: <1% of records fail validation checks
4. **Coverage**: 100% of specified series extracted with <2% missing observations

---

## Appendix A: API Code Snippets

### Example: FRED Series Extraction
```python
import requests
import pandas as pd
from datetime import datetime

def extract_fred_series(series_id, api_key, start_date='2015-01-01'):
    """Extract time series data from FRED API."""
    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        observations = data.get('observations', [])

        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['series_id'] = series_id
        df['extraction_timestamp'] = datetime.now()

        return df[['series_id', 'date', 'value', 'extraction_timestamp']]

    except requests.exceptions.RequestException as e:
        print(f"Error extracting {series_id}: {e}")
        return None
```

### Example: Census M3 Extraction
```python
def extract_census_m3(api_key, start_date='2015-01'):
    """Extract Census M3 data for Primary Metals (NAICS 331)."""
    url = "https://api.census.gov/data/timeseries/eits/m3"

    params = {
        'get': 'cell_value,time_slot_id,data_type_code,category_code',
        'for': 'us:*',
        'time': f'from {start_date}',
        'category_code': '331',  # Primary Metals
        'key': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df['extraction_timestamp'] = datetime.now()

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error extracting Census M3: {e}")
        return None
```

---

## Appendix B: Validation Rules Detail

### FRED Steel PPI (WPU1017)
- **Range**: 50 to 500 (index, base year dependent)
- **Month-over-month change**: Typically -10% to +10%
- **Null tolerance**: 0%
- **Alert threshold**: >15% monthly change

### Census Construction Spending
- **Range**: $500M to $3B (monthly, non-residential)
- **Seasonal pattern**: Expected dip in winter months
- **Null tolerance**: 0%
- **Alert threshold**: Negative value, or >30% MoM change

### EIA Diesel Prices
- **Range**: $2.00 to $6.00 per gallon (historically)
- **Weekly change**: Typically -$0.30 to +$0.30
- **Null tolerance**: 0%
- **Alert threshold**: >$0.50 weekly change, or <$1.50 or >$8.00

### Cass Freight Index
- **Range**: 80 to 150 (index, base = 100 in reference year)
- **Month-over-month change**: Typically -5% to +5%
- **Null tolerance**: 0%
- **Alert threshold**: Value = 0 or missing

---

## Contact & Support

**Project Owner**: [Your Name]
**Analytics Team Lead**: [Lead Name]
**Data Engineering Support**: [Team Email]
**Issue Tracking**: [Link to Jira/GitHub Issues]

**Last Updated**: October 23, 2025
