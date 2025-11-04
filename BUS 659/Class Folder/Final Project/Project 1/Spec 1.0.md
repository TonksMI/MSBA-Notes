It looks like the file creation tool isn’t currently available.

You can manually create a new file in your VS Code directory named:

Spec 1.0 – External Data Source Mapping.md

Then paste the following content:

⸻

Spec 1.0 — External Data Source Mapping & Refresh Strategy

Purpose:
Document, explore, and maintain metadata for all external data sources used in Spec 1.1 – External Data Extraction & Ingestion.
This file acts as a living reference for APIs, download endpoints, frequency, schema, and validation rules.

⸻

1) Objectives
	•	Identify and document reliable commodity and macroeconomic data providers.
	•	Record API endpoints, formats, and update cadence.
	•	Provide test scripts and metadata templates to simplify Spec 1.1 pipeline setup.
	•	Ensure each source has a refresh plan (schedule, authentication, SLA).

⸻

2) Data Source Inventory Template

Source	Dataset	Endpoint / URL	Format	Update Cadence	License / Access	Notes
LME	Daily settlement prices	https://www.lme.com/api/market-data	JSON/CSV	Daily	Free (w/ signup)	Official closing prices
World Bank	“Pink Sheet” commodities	https://api.worldbank.org/v2/en/indicator/	CSV	Monthly	Open Data	Includes copper/aluminum
USGS	Monthly metal stats	https://www.usgs.gov/centers/nmic	XLS	Monthly	Public	Historical only
FRED	CPI, Fed Funds, IPI	https://api.stlouisfed.org/fred/series/observations	JSON	Daily	Free (API key)	Macro indicators
OECD / IMF	Industrial output index	https://stats.oecd.org/	CSV	Monthly	Public	Optional macro series

(Add or edit as more are discovered.)

⸻

3) Example API Mapping

LME JSON structure

{
  "metal": "Aluminium",
  "date": "2025-10-21",
  "price_usd_per_ton": 2289.5,
  "currency": "USD"
}

FRED Example

GET https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key=<API_KEY>&file_type=json

Response keys: date, value

⸻

4) Refresh Strategy

Step	Description	Frequency	Responsible
Fetch	API or CSV download	Daily	Airflow/Prefect task
Validate	Schema & value sanity	Daily	ETL validation step
Transform	Normalize currency, format	Daily	Spec 1.1 transform
Load	Insert into DB (external schema)	Daily	Spec 1.1 load
Audit	Record row count, last refresh	Daily	Auto script


⸻

5) Metadata Schema

YAML/JSON for each source, stored under config/external_sources.yaml:

metals:
  - name: ALUMINUM
    provider: LME
    url: https://www.lme.com/api/market-data/aluminum
    fields: [date, price_usd_per_ton]
    cadence: daily
    license: free
macro:
  - series_id: CPIAUCSL
    provider: FRED
    url: https://api.stlouisfed.org/fred/series/observations
    cadence: daily


⸻

6) Validation Rules
	•	No missing date
	•	price_usd_per_ton > 0
	•	Duplicates (same date/metal) resolved by latest ingestion timestamp
	•	Schema drift → email alert
	•	Anomalous % change > ±15% → log and flag for manual review

⸻

7) Next Steps
	1.	Fill the table with final endpoints and test keys.
	2.	Write short Python notebooks for each provider to verify access.
	3.	Integrate verified sources into Spec 1.1 extraction modules.
