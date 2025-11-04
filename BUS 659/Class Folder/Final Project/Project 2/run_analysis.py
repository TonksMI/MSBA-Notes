#!/usr/bin/env python3
"""
Advanced Census Data Analysis
Runs comprehensive modeling and generates performance report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical and time series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try advanced libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
np.random.seed(42)

# Directories
DATA_DIR = Path('extracted_data/census')
OUTPUT_DIR = Path('analysis_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("ADVANCED CENSUS DATA ANALYSIS")
print("="*80)
print(f"Data directory: {DATA_DIR.resolve()}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
print(f"XGBoost available: {XGB_AVAILABLE}")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. LOADING DATA")
print("-"*80)

datasets = {}

try:
    datasets['construction'] = pd.read_csv(DATA_DIR / 'census_construction.csv', parse_dates=['date'])
    print(f"✓ Construction (VIP): {len(datasets['construction']):,} records")
except FileNotFoundError:
    print("✗ Construction data not found")

try:
    datasets['m3'] = pd.read_csv(DATA_DIR / 'census_m3_primary_metals.csv', parse_dates=['date'])
    print(f"✓ Manufacturing (M3): {len(datasets['m3']):,} records")
except FileNotFoundError:
    print("✗ M3 data not found")

try:
    datasets['residential'] = pd.read_csv(DATA_DIR / 'census_res.csv', parse_dates=['date'])
    print(f"✓ Residential (RES): {len(datasets['residential']):,} records")
except FileNotFoundError:
    print("✗ Residential data not found")

try:
    datasets['retail'] = pd.read_csv(DATA_DIR / 'census_mrts_selected.csv', parse_dates=['date'])
    print(f"✓ Retail (MRTS): {len(datasets['retail']):,} records")
except FileNotFoundError:
    print("✗ Retail data not found")

try:
    datasets['durable'] = pd.read_csv(DATA_DIR / 'census_advm3.csv', parse_dates=['date'])
    print(f"✓ Durable Goods (ADVM3): {len(datasets['durable']):,} records")
except FileNotFoundError:
    print("✗ Durable goods data not found")

print(f"\n✓ Loaded {len(datasets)} datasets")

# ============================================================================
# 2. CREATE TIME SERIES
# ============================================================================
print("\n2. CREATING TIME SERIES")
print("-"*80)

time_series = {}

# Manufacturing New Orders (primary indicator)
if 'm3' in datasets:
    m3_no = datasets['m3'][
        (datasets['m3']['indicator'] == 'NO') &
        (datasets['m3']['seasonally_adjusted'] == True)
    ]
    if len(m3_no) > 0:
        ts = m3_no.groupby('date')['value_millions'].sum().sort_index()
        time_series['manufacturing_orders'] = ts
        print(f"✓ Manufacturing New Orders: {len(ts)} months ({ts.index[0].strftime('%Y-%m')} to {ts.index[-1].strftime('%Y-%m')})")

# Construction Spending
if 'construction' in datasets:
    const_sa = datasets['construction'][datasets['construction']['seasonally_adjusted'] == True]
    if len(const_sa) > 0:
        ts = const_sa.groupby('date')['value_millions'].sum().sort_index()
        time_series['construction_spending'] = ts
        print(f"✓ Construction Spending: {len(ts)} months")

# Housing Units
if 'residential' in datasets:
    res_sa = datasets['residential'][datasets['residential']['seasonally_adjusted'] == True]
    if len(res_sa) > 0:
        ts = res_sa.groupby('date')['value_units'].sum().sort_index()
        time_series['housing_units'] = ts
        print(f"✓ Housing Units: {len(ts)} months")

# Auto Sales
if 'retail' in datasets:
    auto = datasets['retail'][
        (datasets['retail']['category_code'] == '441') &
        (datasets['retail']['seasonally_adjusted'] == True)
    ]
    if len(auto) > 0:
        ts = auto.groupby('date')['value_millions'].sum().sort_index()
        time_series['auto_sales'] = ts
        print(f"✓ Auto Sales: {len(ts)} months")

print(f"\n✓ Created {len(time_series)} time series")

if len(time_series) == 0:
    print("\n✗ No time series data available. Exiting.")
    exit(1)

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n3. FEATURE ENGINEERING")
print("-"*80)

def create_features(series, lags=[1, 2, 3, 6, 12]):
    """Create lagged features and rolling statistics"""
    df = pd.DataFrame({'y': series})

    # Lagged values
    for lag in lags:
        df[f'lag_{lag}'] = series.shift(lag)

    # Rolling statistics
    df['rolling_mean_3'] = series.shift(1).rolling(window=3).mean()
    df['rolling_mean_6'] = series.shift(1).rolling(window=6).mean()
    df['rolling_std_3'] = series.shift(1).rolling(window=3).std()
    df['rolling_std_6'] = series.shift(1).rolling(window=6).std()

    # Time features
    df['month'] = series.index.month
    df['quarter'] = series.index.quarter
    df['year'] = series.index.year
    df['trend'] = np.arange(len(series))

    return df.dropna()

# Use primary series for modeling
primary_series_name = list(time_series.keys())[0]
primary_series = time_series[primary_series_name]

print(f"Primary series: {primary_series_name}")
print(f"  Length: {len(primary_series)} months")
print(f"  Range: {primary_series.index[0].strftime('%Y-%m')} to {primary_series.index[-1].strftime('%Y-%m')}")
print(f"  Mean: ${primary_series.mean():,.0f}M")
print(f"  Std: ${primary_series.std():,.0f}M")

feature_df = create_features(primary_series)
print(f"\n✓ Created {len(feature_df.columns)-1} features")

# ============================================================================
# 4. STATIONARITY TEST
# ============================================================================
print("\n4. STATIONARITY TEST (Augmented Dickey-Fuller)")
print("-"*80)

result = adfuller(primary_series.dropna())
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"  {key}: {value:.4f}")

if result[1] < 0.05:
    print("✓ Series is STATIONARY (reject null hypothesis)")
    stationary = True
else:
    print("✗ Series is NON-STATIONARY (fail to reject null hypothesis)")
    stationary = False

# ============================================================================
# 5. TRAIN/TEST SPLIT
# ============================================================================
print("\n5. TRAIN/TEST SPLIT")
print("-"*80)

train_size = int(len(feature_df) * 0.8)
train_df = feature_df[:train_size]
test_df = feature_df[train_size:]

X_train = train_df.drop('y', axis=1)
y_train = train_df['y']
X_test = test_df.drop('y', axis=1)
y_test = test_df['y']

print(f"Training set: {len(X_train)} samples ({train_df.index[0].strftime('%Y-%m')} to {train_df.index[-1].strftime('%Y-%m')})")
print(f"Test set: {len(X_test)} samples ({test_df.index[0].strftime('%Y-%m')} to {test_df.index[-1].strftime('%Y-%m')})")

# For SARIMA, use the original series split
train_series = primary_series[:train_size]
test_series = primary_series[train_size:]

# ============================================================================
# 6. MODEL 1: SARIMA
# ============================================================================
print("\n6. MODEL 1: SARIMA")
print("-"*80)

try:
    model = SARIMAX(train_series,
                   order=(1, 1, 1),
                   seasonal_order=(1, 1, 1, 12),
                   enforce_stationarity=False,
                   enforce_invertibility=False)

    results = model.fit(disp=False)

    print(f"✓ SARIMAX(1,1,1)(1,1,1)[12] fitted")
    print(f"  AIC: {results.aic:.2f}")
    print(f"  BIC: {results.bic:.2f}")

    # Forecast
    forecast = results.forecast(steps=len(test_series))

    # Metrics
    sarima_mape = np.mean(np.abs((test_series - forecast) / test_series)) * 100
    sarima_rmse = np.sqrt(mean_squared_error(test_series, forecast))
    sarima_mae = mean_absolute_error(test_series, forecast)
    sarima_r2 = r2_score(test_series, forecast)

    # Directional accuracy
    actual_direction = np.sign(test_series.diff().dropna())
    forecast_direction = np.sign(pd.Series(forecast.values, index=test_series.index).diff().dropna())
    sarima_directional = (actual_direction == forecast_direction).sum() / len(actual_direction) * 100

    print(f"\n  MAPE: {sarima_mape:.2f}%")
    print(f"  RMSE: {sarima_rmse:,.2f}")
    print(f"  MAE: {sarima_mae:,.2f}")
    print(f"  R²: {sarima_r2:.4f}")
    print(f"  Directional Accuracy: {sarima_directional:.1f}%")

    sarima_results = {
        'model': 'SARIMA',
        'mape': sarima_mape,
        'rmse': sarima_rmse,
        'mae': sarima_mae,
        'r2': sarima_r2,
        'directional_accuracy': sarima_directional
    }

except Exception as e:
    print(f"✗ SARIMA failed: {e}")
    sarima_results = None

# ============================================================================
# 7. MODEL 2: RANDOM FOREST
# ============================================================================
print("\n7. MODEL 2: RANDOM FOREST")
print("-"*80)

try:
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Metrics
    rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    # Directional accuracy
    actual_direction = np.sign(y_test.diff().dropna())
    pred_direction = np.sign(pd.Series(rf_pred, index=y_test.index).diff().dropna())
    rf_directional = (actual_direction == pred_direction).sum() / len(actual_direction) * 100

    print(f"✓ Random Forest trained")
    print(f"\n  MAPE: {rf_mape:.2f}%")
    print(f"  RMSE: {rf_rmse:,.2f}")
    print(f"  MAE: {rf_mae:,.2f}")
    print(f"  R²: {rf_r2:.4f}")
    print(f"  Directional Accuracy: {rf_directional:.1f}%")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 5 Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"    {row['feature']:20s}: {row['importance']:.4f}")

    rf_results = {
        'model': 'Random Forest',
        'mape': rf_mape,
        'rmse': rf_rmse,
        'mae': rf_mae,
        'r2': rf_r2,
        'directional_accuracy': rf_directional
    }

    # Save feature importance
    feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

except Exception as e:
    print(f"✗ Random Forest failed: {e}")
    rf_results = None

# ============================================================================
# 8. MODEL 3: XGBOOST (IF AVAILABLE)
# ============================================================================
print("\n8. MODEL 3: XGBOOST")
print("-"*80)

if XGB_AVAILABLE:
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        # Metrics
        xgb_mape = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)

        actual_direction = np.sign(y_test.diff().dropna())
        pred_direction = np.sign(pd.Series(xgb_pred, index=y_test.index).diff().dropna())
        xgb_directional = (actual_direction == pred_direction).sum() / len(actual_direction) * 100

        print(f"✓ XGBoost trained")
        print(f"\n  MAPE: {xgb_mape:.2f}%")
        print(f"  RMSE: {xgb_rmse:,.2f}")
        print(f"  MAE: {xgb_mae:,.2f}")
        print(f"  R²: {xgb_r2:.4f}")
        print(f"  Directional Accuracy: {xgb_directional:.1f}%")

        xgb_results = {
            'model': 'XGBoost',
            'mape': xgb_mape,
            'rmse': xgb_rmse,
            'mae': xgb_mae,
            'r2': xgb_r2,
            'directional_accuracy': xgb_directional
        }

    except Exception as e:
        print(f"✗ XGBoost failed: {e}")
        xgb_results = None
else:
    print("⚠ XGBoost not available")
    xgb_results = None

# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================
print("\n9. MODEL COMPARISON")
print("-"*80)

all_results = []
if sarima_results:
    all_results.append(sarima_results)
if rf_results:
    all_results.append(rf_results)
if xgb_results:
    all_results.append(xgb_results)

if all_results:
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.set_index('model')

    print("\nModel Performance Summary:")
    print(comparison_df.to_string())

    # Save comparison
    comparison_df.to_csv(OUTPUT_DIR / 'model_comparison.csv')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'model_comparison.csv'}")

    # Best models
    best_mape = comparison_df['mape'].idxmin()
    best_r2 = comparison_df['r2'].idxmax()
    best_directional = comparison_df['directional_accuracy'].idxmax()

    print(f"\n🏆 BEST MODELS:")
    print(f"  Lowest MAPE: {best_mape} ({comparison_df.loc[best_mape, 'mape']:.2f}%)")
    print(f"  Highest R²: {best_r2} ({comparison_df.loc[best_r2, 'r2']:.4f})")
    print(f"  Best Directional: {best_directional} ({comparison_df.loc[best_directional, 'directional_accuracy']:.1f}%)")

else:
    print("✗ No successful models")

# ============================================================================
# 10. GRANGER CAUSALITY (IF MULTIPLE SERIES)
# ============================================================================
print("\n10. GRANGER CAUSALITY ANALYSIS")
print("-"*80)

if len(time_series) >= 2:
    combined_df = pd.DataFrame(time_series).dropna()

    if len(combined_df) > 24:
        print(f"Testing {len(combined_df.columns)} series with {len(combined_df)} overlapping months\n")

        series_names = list(combined_df.columns)
        significant_relationships = []

        for i, cause in enumerate(series_names):
            for j, effect in enumerate(series_names):
                if i != j:
                    try:
                        test_result = grangercausalitytests(
                            combined_df[[effect, cause]],
                            maxlag=3,
                            verbose=False
                        )

                        p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, 4)]
                        min_p = min(p_values)
                        best_lag = p_values.index(min_p) + 1

                        if min_p < 0.05:
                            significant_relationships.append({
                                'Cause': cause.replace('_', ' ').title(),
                                'Effect': effect.replace('_', ' ').title(),
                                'Lag (months)': best_lag,
                                'p-value': min_p
                            })
                    except:
                        pass

        if significant_relationships:
            granger_df = pd.DataFrame(significant_relationships).sort_values('p-value')
            print("Significant Granger Causality Relationships (p < 0.05):\n")
            print(granger_df.to_string(index=False))

            granger_df.to_csv(OUTPUT_DIR / 'granger_causality.csv', index=False)
            print(f"\n✓ Saved: {OUTPUT_DIR / 'granger_causality.csv'}")
        else:
            print("No significant Granger causality relationships found")
    else:
        print("Insufficient overlapping data")
else:
    print("Need at least 2 time series")

# ============================================================================
# 11. SAVE TIME SERIES DATA
# ============================================================================
print("\n11. SAVING TIME SERIES DATA")
print("-"*80)

if time_series:
    ts_df = pd.DataFrame(time_series)
    ts_df.to_csv(OUTPUT_DIR / 'time_series_data.csv')
    print(f"✓ Saved: {OUTPUT_DIR / 'time_series_data.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")
print("\nFiles created:")
print("  - model_comparison.csv")
print("  - feature_importance.csv")
print("  - time_series_data.csv")
if len(time_series) >= 2 and 'granger_df' in locals():
    print("  - granger_causality.csv")

print("\n" + "="*80)
