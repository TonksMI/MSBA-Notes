# Enhanced Data Extraction Specification Document
## Reliance Inc. Analytics Project - Expanded Data Sources

**Project**: Reliance Inc. Predictive/Prescriptive Analytics Platform
**Document Version**: 2.0
**Date**: October 23, 2025
**Purpose**: Comprehensive specification for extracting data from multiple Census EITS sources plus FRED and other federal data sources

---

## Executive Summary

This enhanced specification expands the original four data sources to **ten comprehensive data sources** supporting Reliance Inc.'s analytics initiatives focused on steel price forecasting, demand nowcasting, and supply chain optimization.

**Primary Use Cases:**
- Steel Price Early-Warning Model (1–3 month horizon)
- End-Market Demand Nowcast (volume forecasting)
- Freight Cost & Capacity Monitoring
- Tariff & Policy Impact Analysis
- **NEW**: Retail & Wholesale Demand Signals
- **NEW**: Durable Goods Leading Indicators
- **NEW**: International Trade Flow Analysis
- **NEW**: Housing & Construction Market Intelligence

---

## Enhanced Data Source Architecture

### Source A: FRED Economic Data (Federal Reserve Economic Data)
**Priority**: HIGH
**Update Frequency**: Monthly/Weekly
**Access Method**: API

#### Series to Extract

| Series ID | Description | Frequency | Use Case |
|-----------|-------------|-----------|----------|
| WPU1017 | Producer Price Index - Steel Mill Products | Monthly | Price forecasting |
| PCU331331 | Primary Metal Manufacturing PPI | Monthly | Input cost tracking |
| TRUCKD11 | Truck Tonnage Index (ATA) | Monthly | Demand indicator |
| RAILFRTCARLOADSD11 | Rail Carloads of All Freight | Monthly | Logistics monitoring |
| TSIFRGHT | Freight Transportation Services Index | Monthly | Comprehensive freight tracking |
| IPMAN | Industrial Production: Manufacturing | Monthly | Overall manufacturing health |
| INDPRO | Industrial Production Index | Monthly | Economic activity indicator |

**API Endpoint**: `https://api.stlouisfed.org/fred/series/observations`
**Documentation**: https://fred.stlouisfed.org/docs/api/fred/

---

### Source B: Census VIP - Construction Spending (Value In Place)
**Priority**: HIGH
**Update Frequency**: Monthly
**Access Method**: API

**Endpoint**: `https://api.census.gov/data/timeseries/eits/vip`

#### Key Categories to Extract
- Private Non-Residential Construction
- Public Construction
- Manufacturing Construction
- Power Construction
- Commercial Construction

**Use Cases:**
- Construction demand for steel products
- Leading indicator for structural steel demand
- Commercial real estate market health

**Data Available**: 2002-present, monthly

---

### Source C: Census M3 - Manufacturers' Shipments, Inventories, and Orders
**Priority**: HIGH
**Update Frequency**: Monthly
**Access Method**: API

**Endpoint**: `https://api.census.gov/data/timeseries/eits/m3`

#### Key NAICS Categories to Extract

| NAICS Code | Industry | Relevance |
|------------|----------|-----------|
| 331 | Primary Metal Manufacturing | Direct steel production |
| 332 | Fabricated Metal Products | Steel consumption |
| 333 | Machinery Manufacturing | Steel-intensive end market |
| 336 | Transportation Equipment | Major steel consumer |
| 3311 | Iron and Steel Mills | Core steel production |
| 3312 | Steel Product from Purchased Steel | Secondary steel processing |

#### Data Type Codes
- **SM**: Shipments Monthly (demand signal)
- **NO**: New Orders (forward-looking demand)
- **UO**: Unfilled Orders (backlog indicator)
- **TI**: Total Inventories (supply chain health)

**Use Cases:**
- Direct measure of steel manufacturing activity
- New orders as leading indicator
- Inventory-to-shipments ratio for supply/demand balance

---

### Source D: Census ADVM3 - Advance Durable Goods Report
**Priority**: HIGH
**Update Frequency**: Monthly (released ~18 working days after month end)
**Access Method**: API

**Endpoint**: `https://api.census.gov/data/timeseries/eits/advm3`

#### Key Categories
- Durable Goods New Orders
- Fabricated Metal Products
- Machinery
- Transportation Equipment

**Use Cases:**
- Leading indicator (released before full M3)
- Early signal of demand changes
- Quick-turn forecasting input

**Advantage**: Released 5 days before full M3 report

---

### Source E: Census MRTS - Monthly Retail Trade Survey
**Priority**: MEDIUM
**Update Frequency**: Monthly
**Access Method**: API

**Endpoint**: `https://api.census.gov/data/timeseries/eits/mrts`

#### Key Categories to Extract

| Category Code | Description | Steel Relevance |
|---------------|-------------|-----------------|
| 441 | Motor Vehicle and Parts Dealers | Auto steel demand proxy |
| 444 | Building Material & Garden Equipment | Construction materials demand |
| 4441 | Building Material and Supplies Dealers | Direct construction activity |

**Use Cases:**
- Consumer demand for steel-intensive products (autos, appliances)
- Retail building materials as construction activity proxy
- Leading indicator for downstream steel demand

**Data Available**: 1992-present

---

### Source F: Census MARTS - Advance Monthly Retail Sales
**Priority**: MEDIUM
**Update Frequency**: Monthly (advance release)
**Access Method**: API

**Endpoint**: `https://api.census.gov/data/timeseries/eits/marts`

#### Focus Areas
- Advance retail sales totals
- Motor vehicle sales (leading indicator)
- Building materials sales

**Use Cases:**
- Early signal of consumer demand
- Auto sales correlate with auto steel consumption
- Released ~2 weeks before full retail report

---

### Source G: Census MTIS - Manufacturing and Trade Inventories and Sales
**Priority**: MEDIUM
**Update Frequency**: Monthly
**Access Method**: API

**Endpoint**: `https://api.census.gov/data/timeseries/eits/mtis`

#### Key Metrics
- Manufacturing inventories
- Retail inventories
- Wholesale inventories
- Sales by sector

**Use Cases:**
- Comprehensive supply chain inventory tracking
- Inventory-to-sales ratios (leading/lagging indicators)
- Cross-sector validation of demand signals

---

### Source H: Census RES - New Residential Construction
**Priority**: MEDIUM-HIGH
**Update Frequency**: Monthly
**Access Method**: API

**Endpoint**: `https://api.census.gov/data/timeseries/eits/resconst`

#### Key Metrics
- Housing starts (total, single-family, multi-family)
- Building permits
- Housing completions
- Under construction inventory

**Use Cases:**
- Residential construction steel demand
- Rebar and structural steel forecasting
- Leading indicator (permits → starts → completions)

**Data Available**: 1959-present for starts, 1960-present for permits

---

### Source I: Census NHS - New Home Sales
**Priority**: MEDIUM
**Update Frequency**: Monthly
**Access Method**: API

**Endpoint**: `https://api.census.gov/data/timeseries/eits/nhs`

#### Key Metrics
- New home sales (units)
- Median sales price
- Months of supply
- Sales by region

**Use Cases:**
- Housing market health
- Demand for residential construction steel
- Price signals for construction materials

---

### Source J: Census International Trade
**Priority**: MEDIUM
**Update Frequency**: Monthly
**Access Method**: API / Manual CSV

**Endpoint**: `https://api.census.gov/data/timeseries/intltrade/imports/hs`

#### Key Product Codes (HS Codes)
- **7206**: Iron and non-alloy steel ingots
- **7207**: Semi-finished products of iron/steel
- **7208-7216**: Flat-rolled and other steel products
- **7217-7229**: Wire, bars, shapes, sections

**Use Cases:**
- Steel import volumes and pricing
- Competition analysis
- Trade policy impact assessment
- Supply availability forecasting

**Note**: May require manual download from https://www.census.gov/foreign-trade/

---

## Data Integration Schema

### Unified Fact Table Structure

```sql
CREATE TABLE fact_economic_indicators_enhanced (
    indicator_id SERIAL PRIMARY KEY,
    source VARCHAR(20) NOT NULL,  -- 'FRED', 'CENSUS_VIP', 'CENSUS_M3', etc.
    series_code VARCHAR(50) NOT NULL,
    category_code VARCHAR(50),    -- NAICS, product code, etc.
    date DATE NOT NULL,
    value DECIMAL(18,4) NOT NULL,
    units VARCHAR(50),
    data_type VARCHAR(20),        -- 'SM', 'NO', 'TI', etc.
    is_seasonally_adjusted BOOLEAN,
    is_advance_release BOOLEAN,   -- TRUE for MARTS, ADVM3
    extraction_timestamp TIMESTAMP NOT NULL,
    revision_number INTEGER DEFAULT 0,
    UNIQUE(source, series_code, category_code, date, data_type)
);

CREATE INDEX idx_source_date ON fact_economic_indicators_enhanced(source, date);
CREATE INDEX idx_category_date ON fact_economic_indicators_enhanced(category_code, date);
CREATE INDEX idx_series_date ON fact_economic_indicators_enhanced(series_code, date);
```

### Dimension Tables

```sql
-- Industry/Category metadata
CREATE TABLE dim_industry_categories (
    category_code VARCHAR(50) PRIMARY KEY,
    category_name VARCHAR(200),
    naics_level INTEGER,         -- 2-digit, 3-digit, 4-digit
    parent_category VARCHAR(50),
    steel_intensity VARCHAR(20), -- 'HIGH', 'MEDIUM', 'LOW'
    description TEXT
);

-- Series metadata
CREATE TABLE dim_series_metadata (
    series_key VARCHAR(100) PRIMARY KEY,  -- source + series_code
    source VARCHAR(20),
    series_code VARCHAR(50),
    full_name VARCHAR(200),
    description TEXT,
    frequency VARCHAR(20),
    units VARCHAR(50),
    seasonal_adjustment_available BOOLEAN,
    first_available_date DATE,
    typical_release_day INTEGER,
    revision_schedule VARCHAR(100)
);

-- Data type codes
CREATE TABLE dim_data_types (
    data_type_code VARCHAR(20) PRIMARY KEY,
    full_name VARCHAR(100),
    description TEXT,
    source VARCHAR(20)
);
```

---

## API Implementation Guide

### Census EITS Standard Query Pattern

All Census EITS endpoints follow this pattern:

```python
BASE_URL = "https://api.census.gov/data/timeseries/eits/{dataset}"

params = {
    'get': 'cell_value,time_slot_id,category_code,data_type_code,seasonally_adj',
    'for': 'US',  # or empty string '' for some datasets
    'time': f'from+{start_date}',  # Format: YYYY-MM
    'category_code': '{NAICS_or_category}',  # Optional filter
    'key': api_key
}
```

### Datasets and Their Specific Parameters

| Dataset | Endpoint Suffix | `for` Parameter | Key Categories |
|---------|----------------|-----------------|----------------|
| VIP | `/vip` | `''` (empty) | Construction types |
| M3 | `/m3` | `'US'` | NAICS codes |
| ADVM3 | `/advm3` | `'US'` | NAICS codes |
| MRTS | `/mrts` | `'US'` | NAICS retail codes |
| MARTS | `/marts` | `'US'` | Retail categories |
| MTIS | `/mtis` | `'US'` | Sector codes |
| RES | `/resconst` | `''` | Structure types |
| NHS | `/nhs` | `''` | Regional codes |

---

## Extraction Priority and Schedule

### Daily/Weekly Monitoring
- **FRED**: TRUCKD11, TSIFRGHT (weekly updates available)

### Priority 1 - Critical Monthly Indicators (Day-of-Release)
1. **ADVM3** - Advance Durable Goods (~18 working days)
2. **MARTS** - Advance Retail Sales (~13 working days)
3. **FRED Steel PPI** - WPU1017 (~15th of month)

### Priority 2 - Core Monthly Indicators (Within 48 hours)
4. **M3** - Full Manufacturing Report (~23 working days)
5. **VIP** - Construction Spending (1st working day, +2 months)
6. **RES** - Housing Starts (~17th of month)
7. **MRTS** - Retail Trade Full Report (~21 working days)

### Priority 3 - Supporting Indicators (Within 1 week)
8. **MTIS** - Manufacturing & Trade Inventories (~25 working days)
9. **NHS** - New Home Sales (~27 working days)
10. **International Trade** - Monthly (5-7 weeks after reference month)

---

## Feature Engineering Opportunities

### Cross-Source Composite Indicators

#### 1. Steel Demand Index (Composite)
```python
steel_demand_index = (
    0.30 * m3_primary_metals_new_orders +
    0.25 * construction_spending_nonresidential +
    0.20 * durable_goods_new_orders +
    0.15 * housing_starts +
    0.10 * auto_sales
)
```

#### 2. Supply Chain Pressure Index
```python
supply_chain_pressure = (
    inventory_to_sales_ratio * freight_cost_index *
    unfilled_orders_change
)
```

#### 3. Leading vs Lagging Indicators
- **Leading** (3-6 months ahead):
  - New orders (M3, ADVM3)
  - Building permits (RES)
  - New home sales (NHS)

- **Coincident** (current conditions):
  - Shipments (M3)
  - Housing starts (RES)
  - Retail sales (MRTS)

- **Lagging** (confirmation):
  - Inventories (MTIS)
  - Housing completions (RES)
  - Unfilled orders (M3)

#### 4. Sector-Specific Demand Forecasts
```python
# Auto sector steel demand
auto_steel_demand = f(
    motor_vehicle_retail_sales,
    transportation_equipment_new_orders,
    auto_inventories
)

# Construction steel demand
construction_steel_demand = f(
    construction_spending_nonres,
    housing_starts,
    building_permits,
    building_materials_sales
)
```

---

## Implementation Checklist

### Phase 1: Core Data Sources (Week 1-2)
- [x] FRED API integration (7 series)
- [ ] Census VIP - Construction Spending
- [ ] Census M3 - Manufacturing (NAICS 331, 332, 333, 336)
- [ ] Census ADVM3 - Advance Durable Goods

### Phase 2: Demand Indicators (Week 3)
- [ ] Census MRTS - Retail Trade (motor vehicles, building materials)
- [ ] Census MARTS - Advance Retail Sales
- [ ] Census RES - Housing Starts & Permits
- [ ] Census NHS - New Home Sales

### Phase 3: Supply Chain Indicators (Week 4)
- [ ] Census MTIS - Manufacturing & Trade Inventories
- [ ] International Trade - Steel Imports (Manual + API exploration)

### Phase 4: Integration & Analytics (Week 5-6)
- [ ] Unified data warehouse
- [ ] Cross-validation logic
- [ ] Feature engineering pipeline
- [ ] Composite indicator calculations
- [ ] Forecasting model inputs

---

## Expected Data Volume

| Source | Frequency | Records/Month | Annual Records | Storage (CSV) |
|--------|-----------|---------------|----------------|---------------|
| FRED (7 series) | Mixed | ~120 | ~1,440 | ~150 KB |
| VIP | Monthly | ~200 | ~2,400 | ~250 KB |
| M3 (4 NAICS) | Monthly | ~160 | ~1,920 | ~200 KB |
| ADVM3 | Monthly | ~100 | ~1,200 | ~125 KB |
| MRTS | Monthly | ~60 | ~720 | ~75 KB |
| MARTS | Monthly | ~40 | ~480 | ~50 KB |
| MTIS | Monthly | ~100 | ~1,200 | ~125 KB |
| RES | Monthly | ~80 | ~960 | ~100 KB |
| NHS | Monthly | ~40 | ~480 | ~50 KB |
| Trade | Monthly | ~500 | ~6,000 | ~1 MB |
| **TOTAL** | | ~1,400 | ~16,800 | ~2.1 MB |

**Historical Load** (10 years): ~200,000 records, ~25 MB

---

## Success Metrics - Enhanced

### Data Quality KPIs
1. **Extraction Reliability**: >98% success rate (up from 95%)
2. **Data Freshness**: <24 hours for Priority 1 sources
3. **Data Completeness**: >99% of expected records
4. **Cross-Source Validation**: Correlation checks pass >95% of time

### Business Value KPIs
1. **Forecast Accuracy**: Steel price forecasts within ±5% (1-month), ±10% (3-month)
2. **Demand Nowcast**: End-market volume within ±3% of actuals
3. **Early Warning**: Detect price inflections 2-4 weeks in advance
4. **Supply Chain**: Freight cost prediction within ±8%

---

## Risk Mitigation

### API Rate Limits
- **Census**: Unlimited with key, 500/day without
- **FRED**: 120 requests/minute
- **Mitigation**: Implement request queuing, exponential backoff

### Data Revisions
- **High Revision Risk**: VIP, M3, MTIS (2-3 months of revisions)
- **Mitigation**: Store full revision history, track revision patterns

### Source Availability
- **Backup Plan**: Manual CSV downloads for critical sources
- **Monitoring**: Automated alerts for failed extractions

---

## Contact & Support

**Project Owner**: Data Engineering Team
**Analytics Lead**: Steel Price Forecasting Team
**API Support**:
- Census: cedsci.feedback@census.gov / 301-763-1605
- FRED: research@stls.frb.org
- EIA: infoctr@eia.gov

**Last Updated**: October 23, 2025
**Version**: 2.0 - Enhanced with 10 data sources
