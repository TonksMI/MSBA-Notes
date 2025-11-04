

# Spec 1.2 — Forecast Modeling & Impact Analysis (External-Data Only)

**Project:** #1 Metal Price Volatility Forecasting  
**Scope of this spec:** Modeling *only using external data*. Do **not** read or reference internal Reliance data here. Internal data linkage lives in **Spec 3 — Internal Data Integration**.

---

## 1) Objectives
1. Train and schedule predictive models to forecast **30/60/90‑day** metal **returns** and **volatility** (steel index, aluminum, copper, nickel).
2. Produce a machine‑readable **forecast table** and a **volatility‑risk table** for downstream optimization and dashboards.
3. Log, version, and monitor models with automated backtests.

---

## 2) Inputs & Upstream Contracts
- **Upstream source:** `Spec 1.1 — External Data Extraction & Ingestion` tables
  - `external.metal_price_history` — daily benchmark prices per metal
    - Columns (required): `date` (DATE), `metal` (TEXT), `price_usd_per_ton` (FLOAT)
  - `external.macro_indicators` — daily/weekly macro series
    - Columns (required): `date`, `series_id`, `value`
- **Assumptions:**
  - Timezone normalized to UTC; prices in **USD**; no missing `date` for business days.
  - Metals tracked (configurable): `ALUMINUM, COPPER, NICKEL, STEEL_INDEX`.

---

## 3) Feature Engineering
- **Returns**: `ret_1d = ln(price_t / price_{t-1})`; also compute `ret_5d`, `ret_20d`.
- **Rolling statistics**: `ma_5`, `ma_20`, `vol_20 = std(ret_1d, 20d, annualize)`.
- **Lag features**: `ret_1d_lag{k}`, `vol_20_lag{k}` for k ∈ {1,5,20}.
- **Macro transforms** (joined on `date`): ΔCPI, ΔFedFunds, ΔIPI (industrial production), USD index.
- **Seasonality flags**: month, quarter, end‑of‑month indicator.

**Implementation notes**
- Use **pandas** and **numpy**; ensure forward‑looking leakage is avoided (shift prior to rolling where needed).
- Parameterize window sizes via `config.yaml`.

---

## 4) Modelling Tasks
### 4.1 Price/Return Forecasts
- **Candidates:**
  - Classical: ARIMA/SARIMA; VAR for multi‑metal interactions.
  - ML: Gradient Boosting (XGBoost/LightGBM) on tabular lag features.
  - DL (optional): LSTM with walk‑forward validation.
- **Target options (choose one per metal per horizon):**
  - `target_ret_20d` (sum of daily returns next 20 trading days)
  - `target_price_30d` (level forecast via transformed model)

### 4.2 Volatility Forecasts
- **Candidates:** ARCH/GARCH/GJR‑GARCH (via `arch` Python package).  
- **Target:** `sigma_20d_forward` (annualized).

### 4.3 Model Selection & Validation
- **Cross‑validation:** Rolling/walk‑forward split with expanding window.  
- **Metrics:** MAE/RMSE for returns/levels; QLIKE for volatility; directional hit‑rate.
- **Champion/Challenger:** keep top‑2 per metal/horizon; record metadata.

---

## 5) Outputs & Downstream Contracts
- **Table:** `model.metal_price_forecasts`
  - `as_of_date` (DATE), `metal` (TEXT), `horizon_days` (INT), `forecast_return` (FLOAT), `forecast_price` (FLOAT, nullable), `lower_ci` (FLOAT), `upper_ci` (FLOAT), `model_id` (TEXT), `train_window` (TEXT), `cv_mae` (FLOAT)
- **Table:** `model.metal_vol_forecasts`
  - `as_of_date`, `metal`, `horizon_days`, `forecast_vol_annualized`, `lower_ci`, `upper_ci`, `model_id`, `cv_qlike`
- **Artifacts:** serialized models in `/models/spec1_2/{metal}/{horizon}/{model_id}.pkl` with `model_card.json`.
- **Logs:** `/logs/spec1_2/*.log` (Python `logging`).

---

## 6) Orchestration & Schedule
- **Frequency:** daily after Spec 1.1 completes.  
- **Order:** feature build → train/refresh (if retrain_day) → predict → write tables → push metric summary.
- **CLI Entrypoints:**
  - `python -m spec1_2.train --metal ALUMINUM --horizon 30`
  - `python -m spec1_2.predict --as_of 2025-10-22`

---

## 7) Python Package Layout (suggested)
```
src/
  spec1_2/
    __init__.py
    config.py              # load YAML
    data.py                # read from Spec 1.1 tables
    features.py            # create features safely (no leakage)
    models/
      arima.py
      var.py
      garch.py
      lgbm.py
      lstm.py
    train.py               # CV, selection, model registry write
    predict.py             # batch predict write -> tables
    metrics.py             # MAE/RMSE/QLIKE + backtest
    io.py                  # db i/o with SQLAlchemy
    utils.py
config/
  spec1_2_config.yaml
models/
logs/
```

---

## 8) Database Schemas (DDL examples)
```sql
CREATE TABLE IF NOT EXISTS model.metal_price_forecasts (
  as_of_date DATE NOT NULL,
  metal TEXT NOT NULL,
  horizon_days INT NOT NULL,
  forecast_return DOUBLE PRECISION,
  forecast_price DOUBLE PRECISION,
  lower_ci DOUBLE PRECISION,
  upper_ci DOUBLE PRECISION,
  model_id TEXT,
  train_window TEXT,
  cv_mae DOUBLE PRECISION,
  PRIMARY KEY (as_of_date, metal, horizon_days)
);

CREATE TABLE IF NOT EXISTS model.metal_vol_forecasts (
  as_of_date DATE NOT NULL,
  metal TEXT NOT NULL,
  horizon_days INT NOT NULL,
  forecast_vol_annualized DOUBLE PRECISION,
  lower_ci DOUBLE PRECISION,
  upper_ci DOUBLE PRECISION,
  model_id TEXT,
  cv_qlike DOUBLE PRECISION,
  PRIMARY KEY (as_of_date, metal, horizon_days)
);
```

---

## 9) Code Stubs (for code‑gen tools)
```python
# features.py
import pandas as pd
import numpy as np

def make_features(df_prices: pd.DataFrame, df_macro: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df_prices.sort_values(["metal", "date"]).copy()
    df["ret_1d"] = np.log(df["price_usd_per_ton"]).diff()
    for w in [5, 20]:
        df[f"ma_{w}"] = df.groupby("metal")["price_usd_per_ton"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        df[f"vol_{w}"] = df.groupby("metal")["ret_1d"].transform(lambda s: s.rolling(w, min_periods=w).std() * np.sqrt(252))
    # Join macro and add calendar features here ...
    return df.dropna()
```

```python
# train.py
from .data import load_price_macro
from .features import make_features
from .models.arima import fit_arima
from .models.garch import fit_garch

# pseudo-code for rolling CV and model selection
```

---

## 10) Monitoring & Alerts
- **Backtest report:** rolling window performance, saved daily.
- **Drift checks:** PSI on key features; alert when thresholds breached.
- **Alerting hooks:** email/Slack webhooks when MAE worsens > X% week‑over‑week.

---

## 11) Non‑Goals (kept out of this spec)
- No internal PO/COGS/margin joins (see **Spec 3**).
- No prescriptive optimization (see **Spec 6.2**).

---

## 12) Links to Related Specs
- Upstream data: **Spec 1.1 — External Data Extraction & Ingestion**
- Downstream prescriptive: **Spec 6.2 — Optimization Engine & Streamlit Dashboard**
- Internal integration: **Spec 3 — Internal Data Integration**
