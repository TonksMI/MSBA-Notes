Spec 1.1 – External Data Extraction & Ingestion

Objective: Automatically pull external metal-price and macroeconomic data (e.g., from London Metal Exchange (LME), World Bank, Federal Reserve Bank of St. Louis (FRED)) and stage into a data-lake or database.

Scope / Modules
	1.	Metadata module
	•	Define list of target metals (steel-index, aluminum, copper, nickel) and their external symbols/URLs.
	•	Define update frequency (daily / weekly).
	•	Define required macroeconomic indicators (industrial production index, CPI, interest rates).
	2.	Extraction module
	•	Fetch data via API or flat‐file (CSV/Excel) for each metal + macro variable.
	•	Handle authentication, rate-limiting, retries, error logging.
	•	Store raw extracts (time-stamped) in a “raw” zone.
	3.	Transformation module
	•	Clean raw data: parse dates, unify time‐zones, rename columns, fill missing values.
	•	Convert currencies if needed (USD), adjust units to standard (e.g., USD / ton).
	•	Compute derived features: rolling returns (e.g., 1-day, 7-day, 30-day), volatility (std dev of returns), lagged price features.
	4.	Load module
	•	Save transformed data to a relational database (e.g., PostgreSQL) or data-warehouse table (schema: metal, date, price, return_1d, return_7d, vol_30d, macro_vars…).
	•	Partition by date for incremental loads.
	•	Maintain metadata table (last load time, row-counts).
	5.	Orchestration module
	•	Use scheduler (e.g., Airflow) to run daily/weekly jobs.
	•	On job failure, send alert (email/Slack).
	•	Archive previous raw files, maintain retention protocol.

Inputs / Outputs
	•	Input: External data sources (URLs/APIs)
	•	Output: Cleaned table metal_price_history accessible to modeling.

Tech stack: Python (requests, pandas, sqlalchemy), scheduler (Airflow/Prefect), database (PostgreSQL)
Coding instructions:
	•	Write extraction functions per metal symbol.
	•	Use @st.cache_data style caching not needed here (dashboard side uses caching)
	•	Logging via Python logging module.
	•	Use parameter-file (YAML/JSON) to list metals & sources.

Dependencies: Spec 1.2 (modelling) will consume this table.
Success criteria: Data extraction completes in < 30 mins, <5% missing data, monetised (accessible) table.
