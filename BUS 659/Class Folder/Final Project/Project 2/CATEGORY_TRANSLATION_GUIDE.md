# Census Data Category & Indicator Translation Guide
## Easy-to-Read Reference for All Extracted Data

**Last Updated**: October 25, 2025

---

## 1. Manufacturing Shipments, Inventories & Orders (M3)

### NAICS Industry Codes

**Steel-Related Industries** (Primary Focus):
- **331** = Primary Metal Manufacturing (overall)
  - **3311** = Iron and Steel Mills
  - **3312** = Steel Product from Purchased Steel
  - **3313** = Alumina and Aluminum Production
  - **3314** = Nonferrous Metal (except Aluminum)
  - **3315** = Foundries
- **332** = Fabricated Metal Products
- **333** = Machinery Manufacturing
- **336** = Transportation Equipment Manufacturing

**Steel Consumer Industries**:
- **321** = Wood Products
- **322** = Paper Manufacturing
- **323** = Printing and Related Support
- **324** = Petroleum and Coal Products
- **325** = Chemical Manufacturing
- **326** = Plastics and Rubber Products
- **327** = Nonmetallic Mineral Products
- **334** = Computer and Electronic Products
- **335** = Electrical Equipment and Appliances
- **3361** = Motor Vehicle Manufacturing
- **3362** = Motor Vehicle Bodies and Trailers
- **3363** = Motor Vehicle Parts
- **3364** = Aerospace Products and Parts
- **337** = Furniture and Related Products
- **339** = Miscellaneous Manufacturing

**Aggregated Categories**:
- **DUR** = All Durable Goods
- **NDE** = All Nondurable Goods
- **32S** = Nondurable Goods (Seasonally Adjusted)
- **33S** = Durable Goods (Seasonally Adjusted)

### M3 Indicator Codes

**Core Metrics**:
- **NO** = New Orders (in millions of dollars)
  - *Leading indicator* - shows future demand
  - Most important for forecasting
- **SM** = Shipments Monthly (in millions of dollars)
  - *Coincident indicator* - current production
- **UO** = Unfilled Orders (in millions of dollars)
  - *Backlog indicator* - shows order queue
- **TI** = Total Inventories (in millions of dollars)
  - *Supply chain health* - combined inventory

**Inventory Components**:
- **MI** = Materials and Supplies Inventory
  - Raw materials on hand
- **WI** = Work-in-Process Inventory
  - Partially completed goods
- **FI** = Finished Goods Inventory
  - Ready for shipment

**Ratios**:
- **IS** = Inventory-to-Shipments Ratio
  - Measures months of inventory on hand
  - Higher = potential oversupply
- **US** = Unfilled Orders-to-Shipments Ratio
  - Measures backlog relative to production
  - Higher = strong demand

**Monthly Percent Changes** (Month-over-Month):
- **MPCNO** = Monthly % Change in New Orders
- **MPCVS** = Monthly % Change in Shipments
- **MPCUO** = Monthly % Change in Unfilled Orders
- **MPCTI** = Monthly % Change in Total Inventories
- **MPCMI** = Monthly % Change in Materials Inventory
- **MPCWI** = Monthly % Change in Work-in-Process
- **MPCFI** = Monthly % Change in Finished Goods

**Use Cases**:
- **For Demand Forecasting**: Focus on NO (New Orders)
- **For Current Production**: Focus on SM (Shipments)
- **For Supply Chain**: Focus on TI, IS ratio
- **For Pricing Signals**: Track UO (high = tight supply)

---

## 2. Construction Spending (VIP)

### Category Codes

**Main Categories**:
- **00XX** = Total Construction Put in Place
- **01XX** = Private Construction
- **02XX** = Public Construction

**Private Construction Types**:
- **03XX** = Private Residential
- **04XX** = Private Nonresidential

**Public Construction Types**:
- **05XX** = Public Buildings
- **06XX** = Highways and Streets
- **07XX** = Other Public Construction

**Specific Industries** (Steel-Intensive):
- **C30** = Residential Construction
- **C31** = Nonresidential Construction
- **C40** = Manufacturing Facilities
- **C50** = Power Plants
- **C60** = Highway and Street Construction
- **C70** = Sewage and Waste Disposal
- **C80** = Water Supply Facilities
- **C100** = Conservation and Development

### VIP Data Type Codes

**Value Measures**:
- **V** = Value (in millions of dollars)
- **E_V** = Estimated Value
- **T** = Total Value
- **E_T** = Estimated Total
- **P** = Public Value
- **E_P** = Estimated Public Value

**Change Measures**:
- **MPCV** = Monthly Percent Change in Value
- **E_MPCV** = Estimated Monthly % Change in Value
- **MPCT** = Monthly Percent Change in Total
- **E_MPCT** = Estimated Monthly % Change in Total
- **MPCP** = Monthly Percent Change in Public
- **E_MPCP** = Estimated Monthly % Change in Public

**Key Points**:
- **E_** prefix = Preliminary estimate (subject to revision)
- All values in millions of dollars
- Data available seasonally adjusted and not seasonally adjusted

**Steel Demand Relevance**:
- **High Impact**: C31 (Nonresidential), C40 (Manufacturing), C50 (Power)
- **Medium Impact**: C30 (Residential), C60 (Highways)
- **Lower Impact**: C70 (Sewage), C80 (Water)

---

## 3. Residential Construction (RES)

### Category Codes

**By Structure Type**:
- **TOTAL** = All Housing Units
- **SINGLE** = Single-Family Homes
- **MULTI** = Multi-Family Buildings
- **2TO4** = Buildings with 2-4 Units
- **5PLUS** = Buildings with 5+ Units

**By Region**:
- **NORTHEAST** = Northeast U.S.
- **MIDWEST** = Midwest U.S.
- **SOUTH** = Southern U.S.
- **WEST** = Western U.S.

**By Building Status**:
- **ASTARTS** = Authorized Starts (permitted and started)
- **ACOMPLETIONS** = Authorized Completions
- **APERMITS** = Authorized Permits

### RES Data Type Codes

**Primary Metrics**:
- **STARTS** = Housing Starts (units)
  - *Leading indicator* - shows new construction beginning
- **PERMITS** = Building Permits (units)
  - *Leading indicator* - earliest signal (1-3 months before starts)
- **COMP** = Completions (units)
  - *Lagging indicator* - shows finished construction
- **UNDERCONST** = Under Construction (units)
  - *Inventory indicator* - current work in progress

**Regional Indicators**:
- **E_SINGLE** = Estimated Single-Family (by region)
- **E_MULTI** = Estimated Multi-Family (by region)
- **E_TOTAL** = Estimated Total (by region)

**Permitted but Not Started**:
- **AUTHNOTSTD** = Authorized but Not Yet Started
  - Shows pipeline of upcoming construction

**Change Measures**:
- **PCSTART** = Percent Change in Starts
- **PCPERMIT** = Percent Change in Permits
- **PCCOMP** = Percent Change in Completions

**Steel Demand Calculation**:
```
Typical Steel per Unit:
- Single-Family Home: 2-3 tons
- Multi-Family Unit: 1.5-2 tons
- Total Steel Demand = (STARTS × tons per unit)
```

**Leading Indicator Chain**:
1. **PERMITS** (Month 0) → 2. **STARTS** (Month 1-3) → 3. **UNDERCONST** (Month 3-12) → 4. **COMP** (Month 6-18)

---

## 4. Retail Trade (MRTS)

### Category Codes

**Auto Sales** (Major Steel Consumer):
- **441** = Motor Vehicle and Parts Dealers
  - New car sales
  - Used car sales
  - Auto parts and accessories
  - *Steel Relevance*: Each vehicle = 900-1,200 kg steel

**Construction Materials**:
- **444** = Building Material and Garden Equipment
  - Includes hardware stores, garden centers
  - *Steel Relevance*: Indirect (construction activity proxy)
- **4441** = Building Material and Supplies Dealers
  - More focused on construction supplies
  - *Steel Relevance*: Rebar, structural steel, fasteners

**Why These Matter for Steel**:
1. **Auto Sales (441)**:
   - Direct correlation with automotive steel demand
   - 3-month lead time from order to production
   - Each 1% increase in auto sales = ~25,000 tons steel/month

2. **Building Materials (444, 4441)**:
   - Proxy for construction activity
   - Correlates with residential and commercial steel use
   - 1-2 month lag from retail sales to steel orders

### MRTS Data Format

All MRTS data is in **millions of dollars** and available as:
- Seasonally Adjusted
- Not Seasonally Adjusted

**Typical Values**:
- Motor Vehicle Sales: $80,000 - $120,000 million/month
- Building Materials: $30,000 - $40,000 million/month

---

## 5. Advance Durable Goods (ADVM3)

### Category Codes

**Uses Same NAICS Codes as M3** (see Section 1), but:
- Released **5 days earlier** than full M3 report
- Advance estimates (subject to revision)
- Subset of full M3 data

**Key Categories in ADVM3**:
- **NDE** = Nondurable Goods (Total)
- **DUR** = Durable Goods (Total)
- **331** = Primary Metal Manufacturing
- **332** = Fabricated Metal Products
- **333** = Machinery
- **336** = Transportation Equipment
- **3361** = Motor Vehicles

### ADVM3 Indicator Codes

**Available Indicators** (Same as M3):
- **NO** = New Orders
- **SM** = Shipments
- **UO** = Unfilled Orders
- **MPCNO** = Monthly % Change in New Orders
- **MPCVS** = Monthly % Change in Shipments
- **MPCUO** = Monthly % Change in Unfilled Orders

**Use Case**:
- **Early Warning System**: Get durable goods trends 5 days before full report
- **Quick Reactions**: Adjust forecasts immediately
- **Validation**: Compare advance vs. full release for revision patterns

---

## Quick Reference: What to Monitor for Steel Demand

### 🔴 Critical Indicators (Check Daily/Weekly)

1. **M3 New Orders - Primary Metals (NAICS 331, indicator NO)**
   - Direct measure of steel orders
   - 1-2 month lead time

2. **Housing Starts - Total (RES, STARTS)**
   - Each start = 2-3 tons steel
   - 3-6 month lead time

3. **Auto Sales (MRTS, category 441)**
   - Each vehicle = ~1 ton steel
   - 2-3 month lead time

### 🟡 Important Indicators (Check Monthly)

4. **Construction Spending - Nonresidential (VIP, C31)**
   - Commercial/industrial steel demand
   - 2-4 month lead time

5. **Building Permits (RES, PERMITS)**
   - Earliest signal for housing
   - 4-6 month lead time

6. **M3 Unfilled Orders - Primary Metals (NAICS 331, UO)**
   - Backlog indicator
   - Shows supply tightness

### 🟢 Supporting Indicators (Check Monthly)

7. **Manufacturing Shipments (M3, SM)**
   - Current production levels

8. **Inventory-to-Shipments Ratio (M3, IS)**
   - Supply chain balance

9. **Building Materials Sales (MRTS, 444)**
   - Construction activity proxy

10. **Advance Durable Goods New Orders (ADVM3, NO)**
    - Early warning (5-day lead)

---

## Seasonally Adjusted vs. Not Seasonally Adjusted

**Seasonally Adjusted (SA)**:
- Removes predictable seasonal patterns
- Better for trend analysis
- **Use for**: Month-to-month comparisons, forecasting
- **Example**: Housing starts always spike in spring/summer - SA removes this

**Not Seasonally Adjusted (NSA)**:
- Raw data with seasonal patterns intact
- Better for year-over-year comparisons
- **Use for**: Annual planning, historical context

**Recommendation**: Use **Seasonally Adjusted** for forecasting models.

---

## Data Revision Schedule

**High Revision Risk** (2-3 months of revisions):
- VIP (Construction Spending)
- M3 (Manufacturing)
- ADVM3 (Advance Durable Goods)

**Low Revision Risk**:
- RES (Housing) - usually minor revisions
- MRTS (Retail) - stable after first revision

**Best Practice**:
- Use "final" data (3+ months old) for model training
- Use "advance" data for current forecasts
- Track revision patterns for forecast adjustment

---

## Common Abbreviations

- **YoY** = Year-over-Year (e.g., Oct 2025 vs Oct 2024)
- **MoM** = Month-over-Month (e.g., Oct 2025 vs Sep 2025)
- **SA** = Seasonally Adjusted
- **NSA** = Not Seasonally Adjusted
- **NAICS** = North American Industry Classification System
- **EITS** = Economic Indicator Time Series (Census API)
- **VIP** = Value in Place (Construction Spending dataset)
- **M3** = Monthly Survey of Manufacturing
- **RES** = New Residential Construction
- **MRTS** = Monthly Retail Trade Survey
- **ADVM3** = Advance Monthly Report on Durable Goods

---

## Steel Industry Cheat Sheet

### Units to Steel Demand Conversion

**Housing**:
- 1 Single-Family Start = **2.5 tons steel** (average)
- 1 Multi-Family Unit = **1.8 tons steel** (average)
- Monthly Starts = 1.5M units → **3.75M tons steel/year**

**Automotive**:
- 1 Vehicle = **900 kg steel** (average, declining due to aluminum)
- Monthly Sales = 15M vehicles/year → **13.5M tons steel/year**

**Construction Spending**:
- $1B Nonresidential Construction = **~50,000 tons steel**
- $1B Manufacturing Construction = **~75,000 tons steel**

### Price Indicators (Not in Census Data)

To complete steel forecasting, add:
- **FRED WPU1017** = Producer Price Index for Steel Mill Products
- **FRED PCU331331** = Primary Metal Manufacturing PPI
- Import/Export prices from Census International Trade

---

## Example Queries

### "What's happening with steel demand right now?"

Check these in order:
1. M3, NAICS 331, indicator NO (New Orders) - latest month
2. M3, NAICS 331, indicator IS (Inventory/Shipments ratio)
3. RES, STARTS (Housing Starts) - latest month
4. MRTS, 441 (Auto Sales) - latest month

### "What will steel demand be in 3 months?"

Leading indicators (3-month forecast):
1. RES, PERMITS (Building Permits) - 3-month lead
2. ADVM3, NO (Advance New Orders) - 1-month lead
3. M3 331, NO trend (12-month moving average)

### "Is there a supply shortage coming?"

Supply tightness indicators:
1. M3, NAICS 331, UO (Unfilled Orders) - rising = tight
2. M3, NAICS 331, IS ratio - falling = tight
3. Construction Spending growth > 5% YoY = demand surge

---

**This guide is a living document. Update as you discover new patterns in the data!**

**For API documentation, see**: `Enhanced_Data_Extraction_Specification.md`
**For analysis results, see**: `FINAL_ANALYSIS_REPORT.md`
