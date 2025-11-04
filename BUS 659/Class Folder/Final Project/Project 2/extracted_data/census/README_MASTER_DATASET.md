# M3 Primary Metals - Master Dataset Documentation

**Created**: October 29, 2025
**Purpose**: Aggregated and feature-engineered dataset for predicting M3 Primary Metals indicators

---

## Files

### Main Dataset
- **`master_m3_metals_features.csv`** (2.4 MB)
  - Master dataset with all features
  - 127 rows (months) × 1,960 columns
  - Date range: January 2015 - July 2025

### Documentation
- **`master_dataset_metadata.json`**
  - Dataset statistics and metadata
- **`column_documentation.txt`**
  - Complete column listing by category
- **`create_master_dataset.py`**
  - Script to regenerate the dataset

---

## Dataset Overview

### Dimensions
- **Rows**: 127 monthly observations (2015-01 to 2025-07)
- **Columns**: 1,960 total
  - **Date**: 1 column
  - **Target Variables**: 16 M3 Primary Metals indicators
  - **Features**: 1,943 predictor columns

### Target Variables (M3 Primary Metals)

The dataset includes 16 key indicators from the M3 Primary Metals survey:

| Indicator | Description |
|-----------|-------------|
| `m3_FI` | Finished goods inventory |
| `m3_IS` | Inventories to shipments ratio |
| `m3_MI` | Materials and supplies inventory |
| `m3_MPCFI` | Finished goods inventory percent change |
| `m3_MPCMI` | Materials inventory percent change |
| `m3_MPCNO` | New orders percent change |
| `m3_MPCTI` | Total inventories percent change |
| `m3_MPCUO` | Unfilled orders percent change |
| `m3_MPCVS` | Value of shipments percent change |
| `m3_MPCWI` | Work-in-process inventory percent change |
| `m3_NO` | New orders |
| `m3_TI` | Total inventories |
| `m3_UO` | Unfilled orders |
| `m3_US` | Unfilled orders to shipments ratio |
| `m3_VS` | Value of shipments |
| `m3_WI` | Work-in-process inventory |

---

## Feature Categories

### 1. Original Features (176 columns)

#### Advance M3 (8 columns)
- Leading indicators from advance manufacturing survey
- Prefix: `advm3_`
- Examples: `advm3_MPCNO`, `advm3_MPCTI`, `advm3_MPCUO`

#### Construction (144 columns)
- Construction spending and permits
- Prefix: `const_`
- Format: `const_{category}_{data_type}`
- Examples: `const_A07XX_P`, `const_20IX_MPCT`

#### Retail Sales (3 columns)
- Selected retail trade categories
- Prefix: `retail_`
- Examples: `retail_441` (Motor Vehicle and Parts Dealers)

#### Residential (21 columns)
- Housing starts, completions, permits
- Prefix: `res_`
- Format: `res_{category}_{type}`
- Examples: `res_ACOMPLETIONS_TOTAL`, `res_APERMITS_TOTAL`

### 2. Engineered Features (1,767 columns)

#### Lagged Features (880 columns)
Captures temporal dependencies with multiple time horizons:
- **Lag 1**: Previous month (t-1)
- **Lag 2**: 2 months ago (t-2)
- **Lag 3**: 3 months ago (t-3)
- **Lag 6**: 6 months ago (t-6)
- **Lag 12**: 12 months ago / year-ago (t-12)

Format: `{feature}_lag{period}`
- Example: `advm3_MPCNO_lag1`, `retail_441_lag12`

**Use cases**:
- Short-term lags (1-3): Capture immediate momentum
- Medium-term lag (6): Capture semi-annual trends
- Long-term lag (12): Capture year-over-year patterns and seasonality

#### Rolling Average Features (528 columns)
Smoothed trends using moving averages:
- **3-month MA**: Short-term trend
- **6-month MA**: Medium-term trend
- **12-month MA**: Long-term trend / annual average

Format: `{feature}_ma{window}`
- Example: `const_A07XX_P_ma3`, `retail_441_ma12`

**Use cases**:
- Reduce noise and volatility
- Identify underlying trends
- Detect trend changes

#### Rate of Change Features (352 columns)
Percentage changes to capture growth dynamics:
- **Month-over-Month (MoM)**: `{feature}_mom_change`
  - Formula: `(current - previous) / previous`
  - Captures short-term growth rates

- **Year-over-Year (YoY)**: `{feature}_yoy_change`
  - Formula: `(current - year_ago) / year_ago`
  - Captures annual growth rates, eliminates seasonality

Examples:
- `advm3_MPCNO_mom_change`: Monthly change in advance new orders
- `retail_441_yoy_change`: Annual change in motor vehicle sales

**Use cases**:
- Growth trend analysis
- Seasonality adjustment (YoY)
- Momentum indicators

#### Time Features (8 columns)
Calendar and seasonal indicators:
- `year`: Calendar year (2015-2025)
- `month`: Month (1-12)
- `quarter`: Quarter (1-4)
- `is_q1`, `is_q2`, `is_q3`, `is_q4`: Quarter dummy variables

**Use cases**:
- Capture seasonal patterns
- Model year effects
- Quarter-specific behavior

---

## Data Quality

### Completeness
- **Total observations**: 127 months
- **Missing data**: Minimal (<1% for most columns)
- **High missing columns**: 2 columns with >50% missing (construction category with no variation)

### Temporal Coverage
- **Start**: January 2015
- **End**: July 2025
- **Frequency**: Monthly
- **Gaps**: None (continuous monthly data)

### Seasonality
- All data is **seasonally adjusted** (where available)
- Additional seasonal indicators provided via time features
- Original non-adjusted data available in source files

---

## Usage Guide

### Loading the Dataset

```python
import pandas as pd

# Load master dataset
df = pd.read_csv('master_m3_metals_features.csv')
df['date'] = pd.to_datetime(df['date'])

# Set date as index
df = df.set_index('date')

print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
```

### Selecting Target Variables

```python
# Get all M3 target columns
m3_targets = [col for col in df.columns if col.startswith('m3_')]

# Extract target data
y = df[m3_targets]

# Select specific target (e.g., New Orders percent change)
y_target = df['m3_MPCNO']
```

### Selecting Features

```python
# Get all feature columns (exclude targets)
features = [col for col in df.columns if not col.startswith('m3_')]

# Extract feature data
X = df[features]

# Select specific feature types
lagged_features = [col for col in df.columns if '_lag' in col]
rolling_features = [col for col in df.columns if '_ma' in col]
change_features = [col for col in df.columns if '_change' in col]
```

### Train/Test Split

```python
# Time-based split (no shuffling for time series!)
train_cutoff = '2023-12-01'

train = df[df.index < train_cutoff]
test = df[df.index >= train_cutoff]

print(f"Train: {len(train)} months")
print(f"Test: {len(test)} months")
```

### Handling Missing Values

```python
# Check missing values
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# Forward fill lagged features (common for time series)
df_filled = df.fillna(method='ffill')

# Or drop rows with missing targets
df_clean = df.dropna(subset=m3_targets)
```

---

## Modeling Recommendations

### Feature Selection
With 1,943 features and 127 observations, feature selection is critical:

1. **Start with domain knowledge**
   - Use construction indicators for manufacturing
   - Use retail indicators for consumer-driven metals demand
   - Use residential indicators for housing-related metals

2. **Correlation analysis**
   ```python
   # Find features most correlated with target
   target = 'm3_MPCNO'
   correlations = df.corr()[target].abs().sort_values(ascending=False)
   top_features = correlations.head(20).index.tolist()
   ```

3. **Regularization methods**
   - Lasso (L1): Automatic feature selection
   - Ridge (L2): Handle multicollinearity
   - ElasticNet: Combination of both

4. **Tree-based importance**
   ```python
   from sklearn.ensemble import RandomForestRegressor

   rf = RandomForestRegressor(n_estimators=100)
   rf.fit(X_train, y_train)

   importances = pd.Series(rf.feature_importances_, index=X_train.columns)
   top_features = importances.nlargest(20)
   ```

### Model Approaches

#### 1. Single-Output Models
Predict one M3 indicator at a time:
```python
from sklearn.linear_model import Ridge

# Predict new orders percent change
y = df['m3_MPCNO']
X = df[features]

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

#### 2. Multi-Output Models
Predict multiple M3 indicators simultaneously:
```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

y = df[m3_targets]
X = df[features]

model = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(X_train, y_train)
```

#### 3. Time Series Models
Leverage temporal structure:
```python
from statsmodels.tsa.api import VAR

# Vector Autoregression
model = VAR(df[m3_targets])
results = model.fit(maxlags=12)
```

### Cross-Validation Strategy

**Use time series cross-validation** (expanding window):
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train and evaluate model
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

**Do NOT use regular k-fold** (would leak future information)!

---

## Feature Engineering Rationale

### Why Lagged Features?
- Manufacturing has **lead times** (orders today → production next month)
- **Economic momentum** persists across months
- **Seasonal patterns** repeat year-over-year
- Example: Construction in spring → metals demand in summer

### Why Rolling Averages?
- **Smooth volatility** in monthly data
- **Identify trends** vs. noise
- **Different time horizons** capture different dynamics:
  - 3-month: Recent trend
  - 12-month: Annual cycle

### Why Rate of Change?
- **Growth rates** often more predictive than levels
- **Normalize** different scales across industries
- **YoY removes seasonality** better than SA alone
- Example: 5% MoM growth + 10% YoY growth = strong acceleration

### Why Multiple Lags?
- **Different indicators lead by different amounts**:
  - Advance orders lead final metals by 1-2 months
  - Housing starts lead metals by 3-6 months
  - Retail sales are coincident or lag by 1 month
- **Model can learn** optimal lag structure

---

## Example Analysis

### Quick Start: Predict New Orders % Change

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
df = pd.read_csv('master_m3_metals_features.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Define target and features
target = 'm3_MPCNO'  # New orders percent change
features = [col for col in df.columns if not col.startswith('m3_')]

# Remove rows with missing target
df_clean = df.dropna(subset=[target])

# Split data
X = df_clean[features]
y = df_clean[target]

# Handle missing features (forward fill then drop remaining)
X = X.fillna(method='ffill').dropna()
y = y.loc[X.index]

# Train/test split (last 12 months for testing)
train_size = len(X) - 12
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train model with Lasso (automatic feature selection)
model = LassoCV(cv=TimeSeriesSplit(n_splits=5))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.3f}")
print(f"Test R²: {r2:.3f}")

# Get important features
feature_importance = pd.Series(
    model.coef_,
    index=X_train.columns
).abs().sort_values(ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10))
```

---

## Data Sources

All data sourced from U.S. Census Bureau:
- **M3 Primary Metals**: Manufacturers' Shipments, Inventories, and Orders (M3)
- **Advance M3**: Advance Report on Durable Goods
- **Construction**: Value in Place Survey (VIP)
- **Retail Sales**: Monthly Retail Trade Survey (MRTS)
- **Residential**: New Residential Construction

**Data Retrieval Date**: October 25, 2025

---

## Regenerating the Dataset

To recreate or update the dataset:

```bash
cd "/path/to/census/data"
python create_master_dataset.py
```

The script will:
1. Load all source CSV files
2. Filter for seasonally adjusted data
3. Pivot to wide format
4. Merge all sources
5. Create lagged features (1, 2, 3, 6, 12 months)
6. Create rolling averages (3, 6, 12 months)
7. Create rate of change features (MoM, YoY)
8. Add time features
9. Save master dataset and documentation

**Runtime**: ~10-15 seconds

---

## Notes and Caveats

### Data Limitations
1. **Small sample size**: Only 127 observations
   - Use regularization
   - Limit feature count through selection
   - Consider ensemble methods

2. **High dimensionality**: 1,943 features
   - Risk of overfitting
   - Feature selection critical
   - Domain knowledge helps

3. **Missing data**: Some lags/rolling averages have NaN for early periods
   - First 12 rows have missing 12-month lags
   - Handle appropriately (drop, fill, or use subset)

4. **Multicollinearity**: Many features highly correlated
   - Lagged versions of same feature
   - Use regularization or PCA
   - Tree-based models handle well

### Best Practices
- ✅ Use time series cross-validation
- ✅ Feature selection before modeling
- ✅ Domain knowledge to guide feature choice
- ✅ Regularization to prevent overfitting
- ✅ Ensemble methods for robustness
- ❌ Don't shuffle data (time series!)
- ❌ Don't use regular k-fold CV
- ❌ Don't include future information in features

---

## Contact & Updates

For questions or issues with the dataset:
1. Check documentation files
2. Review source CSV files
3. Re-run `create_master_dataset.py` to verify

**Last Updated**: October 29, 2025
