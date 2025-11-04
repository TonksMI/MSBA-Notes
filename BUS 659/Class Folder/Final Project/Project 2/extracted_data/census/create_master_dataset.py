#!/usr/bin/env python3
"""
Census Data Aggregation and Feature Engineering
===============================================

This script creates a master dataset for predicting M3 Primary Metals values
by aggregating all census data sources with engineered lagged features.

Output: master_m3_metals_features.csv

Features:
- Target: M3 Primary Metals indicators
- Predictors: Lagged features from construction, retail, residential, and advance manufacturing
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("M3 PRIMARY METALS - MASTER DATASET CREATION")
print("="*80)

# Load all datasets
print("\n1. Loading all census datasets...")

m3_metals = pd.read_csv('census_m3_primary_metals.csv')
advm3 = pd.read_csv('census_advm3.csv')
construction = pd.read_csv('census_construction.csv')
mrts = pd.read_csv('census_mrts_selected.csv')
residential = pd.read_csv('census_res.csv')

print(f"   ✓ M3 Primary Metals: {len(m3_metals):,} rows")
print(f"   ✓ Advance M3: {len(advm3):,} rows")
print(f"   ✓ Construction: {len(construction):,} rows")
print(f"   ✓ Retail Sales: {len(mrts):,} rows")
print(f"   ✓ Residential: {len(residential):,} rows")

# Convert dates
print("\n2. Converting dates...")
for df in [m3_metals, advm3, construction, mrts, residential]:
    df['date'] = pd.to_datetime(df['date'])

# Focus on seasonally adjusted data
print("\n3. Filtering for seasonally adjusted data...")
m3_metals_sa = m3_metals[m3_metals['seasonally_adjusted'] == True].copy()
print(f"   ✓ M3 Metals SA: {len(m3_metals_sa):,} rows")

# Get unique indicators
m3_indicators = m3_metals_sa['indicator'].unique()
print(f"   ✓ Found {len(m3_indicators)} indicators in M3 Primary Metals")

# Pivot M3 metals to wide format (one column per indicator)
print("\n4. Pivoting M3 Primary Metals data...")
m3_pivot = m3_metals_sa.pivot_table(
    index='date',
    columns='indicator',
    values='value_millions',
    aggfunc='first'
).reset_index()

# Rename columns with m3_ prefix
m3_pivot.columns = ['date'] + [f'm3_{col}' for col in m3_pivot.columns[1:]]
print(f"   ✓ Created {len(m3_pivot.columns)-1} M3 target columns")

# Process Advance M3 data
print("\n5. Processing Advance M3 data...")
advm3_sa = advm3[advm3['seasonally_adjusted'] == True].copy()
advm3_pivot = advm3_sa.pivot_table(
    index='date',
    columns='indicator',
    values='value_millions',
    aggfunc='first'
).reset_index()
advm3_pivot.columns = ['date'] + [f'advm3_{col}' for col in advm3_pivot.columns[1:]]
print(f"   ✓ Created {len(advm3_pivot.columns)-1} Advance M3 columns")

# Process Construction data
print("\n6. Processing Construction data...")
construction_sa = construction[construction['seasonally_adjusted'] == True].copy()
construction_pivot = construction_sa.pivot_table(
    index='date',
    columns=['category_code', 'data_type_code'],
    values='value_millions',
    aggfunc='first'
).reset_index()
construction_pivot.columns = ['date'] + [f'const_{c[0]}_{c[1]}' for c in construction_pivot.columns[1:]]
print(f"   ✓ Created {len(construction_pivot.columns)-1} Construction columns")

# Process Retail Sales data
print("\n7. Processing Retail Sales data...")
mrts_sa = mrts[mrts['seasonally_adjusted'] == True].copy()
mrts_pivot = mrts_sa.pivot_table(
    index='date',
    columns='category_code',
    values='value_millions',
    aggfunc='first'
).reset_index()
mrts_pivot.columns = ['date'] + [f'retail_{col}' for col in mrts_pivot.columns[1:]]
print(f"   ✓ Created {len(mrts_pivot.columns)-1} Retail columns")

# Process Residential data
print("\n8. Processing Residential data...")
residential_sa = residential[residential['seasonally_adjusted'] == True].copy()
residential_pivot = residential_sa.pivot_table(
    index='date',
    columns=['category_code', 'data_type_code'],
    values='value_units',
    aggfunc='first'
).reset_index()
residential_pivot.columns = ['date'] + [f'res_{c[0]}_{c[1]}' for c in residential_pivot.columns[1:]]
print(f"   ✓ Created {len(residential_pivot.columns)-1} Residential columns")

# Merge all datasets
print("\n9. Merging all datasets...")
master = m3_pivot.copy()
master = master.merge(advm3_pivot, on='date', how='left')
master = master.merge(construction_pivot, on='date', how='left')
master = master.merge(mrts_pivot, on='date', how='left')
master = master.merge(residential_pivot, on='date', how='left')

print(f"   ✓ Master dataset shape: {master.shape}")
print(f"   ✓ Date range: {master['date'].min()} to {master['date'].max()}")

# Create lagged features
print("\n10. Creating lagged features...")

# Get feature columns (everything except date and m3_ columns)
m3_cols = [col for col in master.columns if col.startswith('m3_')]
feature_cols = [col for col in master.columns if not col.startswith('m3_') and col != 'date']

print(f"   ✓ Target columns (M3): {len(m3_cols)}")
print(f"   ✓ Feature columns: {len(feature_cols)}")

# Sort by date
master = master.sort_values('date').reset_index(drop=True)

# Create lags (1, 2, 3, 6, 12 months)
lag_periods = [1, 2, 3, 6, 12]

for lag in lag_periods:
    print(f"   Creating lag-{lag} features...")
    for col in feature_cols:
        master[f'{col}_lag{lag}'] = master[col].shift(lag)

print(f"   ✓ Total lagged features created: {len(feature_cols) * len(lag_periods)}")

# Create rolling averages (3, 6, 12 months)
print("\n11. Creating rolling average features...")
rolling_windows = [3, 6, 12]

for window in rolling_windows:
    print(f"   Creating {window}-month rolling averages...")
    for col in feature_cols:
        master[f'{col}_ma{window}'] = master[col].rolling(window=window).mean()

print(f"   ✓ Total rolling features created: {len(feature_cols) * len(rolling_windows)}")

# Create rate of change features
print("\n12. Creating rate of change features...")
for col in feature_cols:
    # Month-over-month change
    master[f'{col}_mom_change'] = master[col].pct_change()
    # Year-over-year change
    master[f'{col}_yoy_change'] = master[col].pct_change(periods=12)

print(f"   ✓ Created MoM and YoY change features")

# Add time-based features
print("\n13. Adding time-based features...")
master['year'] = master['date'].dt.year
master['month'] = master['date'].dt.month
master['quarter'] = master['date'].dt.quarter
master['is_q1'] = (master['quarter'] == 1).astype(int)
master['is_q2'] = (master['quarter'] == 2).astype(int)
master['is_q3'] = (master['quarter'] == 3).astype(int)
master['is_q4'] = (master['quarter'] == 4).astype(int)

print(f"   ✓ Added year, month, quarter indicators")

# Data quality check
print("\n14. Data Quality Summary...")
print(f"   Total rows: {len(master):,}")
print(f"   Total columns: {len(master.columns):,}")
print(f"   M3 target columns: {len(m3_cols)}")
print(f"   Feature columns (including engineered): {len(master.columns) - len(m3_cols) - 1}")

# Check missing values
missing_pct = (master.isnull().sum() / len(master) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 50].head(10)

if len(high_missing) > 0:
    print(f"\n   ⚠  Columns with >50% missing data: {len(missing_pct[missing_pct > 50])}")
    print("   Top 10:")
    for col, pct in high_missing.items():
        print(f"      - {col}: {pct:.1f}%")
else:
    print(f"   ✓ No columns with >50% missing data")

# Save master dataset
print("\n15. Saving master dataset...")
output_file = 'master_m3_metals_features.csv'
master.to_csv(output_file, index=False)

print(f"   ✓ Saved to: {output_file}")
print(f"   ✓ File size: {master.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Create metadata file
print("\n16. Creating metadata file...")
metadata = {
    'creation_date': datetime.now().isoformat(),
    'total_rows': len(master),
    'total_columns': len(master.columns),
    'date_range_start': str(master['date'].min()),
    'date_range_end': str(master['date'].max()),
    'target_columns': len(m3_cols),
    'feature_columns_original': len(feature_cols),
    'lagged_features': len(feature_cols) * len(lag_periods),
    'rolling_features': len(feature_cols) * len(rolling_windows),
    'change_features': len(feature_cols) * 2,
    'time_features': 8,
    'total_engineered_features': len(master.columns) - len(m3_cols) - 1 - len(feature_cols)
}

import json
with open('master_dataset_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✓ Metadata saved to: master_dataset_metadata.json")

# Create column documentation
print("\n17. Creating column documentation...")

column_types = {
    'Target Variables (M3 Primary Metals)': [col for col in master.columns if col.startswith('m3_')],
    'Original Features - Advance M3': [col for col in master.columns if col.startswith('advm3_') and '_lag' not in col and '_ma' not in col and '_change' not in col],
    'Original Features - Construction': [col for col in master.columns if col.startswith('const_') and '_lag' not in col and '_ma' not in col and '_change' not in col],
    'Original Features - Retail': [col for col in master.columns if col.startswith('retail_') and '_lag' not in col and '_ma' not in col and '_change' not in col],
    'Original Features - Residential': [col for col in master.columns if col.startswith('res_') and '_lag' not in col and '_ma' not in col and '_change' not in col],
    'Lagged Features': [col for col in master.columns if '_lag' in col],
    'Rolling Average Features': [col for col in master.columns if '_ma' in col],
    'Rate of Change Features': [col for col in master.columns if '_change' in col],
    'Time Features': ['year', 'month', 'quarter', 'is_q1', 'is_q2', 'is_q3', 'is_q4']
}

with open('column_documentation.txt', 'w') as f:
    f.write("M3 PRIMARY METALS - MASTER DATASET COLUMN DOCUMENTATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for category, cols in column_types.items():
        f.write(f"\n{category} ({len(cols)} columns)\n")
        f.write("-" * 80 + "\n")
        for col in sorted(cols)[:50]:  # Limit to 50 per category for readability
            f.write(f"  - {col}\n")
        if len(cols) > 50:
            f.write(f"  ... and {len(cols) - 50} more\n")

print(f"   ✓ Column documentation saved to: column_documentation.txt")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nMaster Dataset Created Successfully!")
print(f"  📊 File: {output_file}")
print(f"  📅 Date Range: {master['date'].min().strftime('%Y-%m')} to {master['date'].max().strftime('%Y-%m')}")
print(f"  📏 Dimensions: {master.shape[0]:,} rows × {master.shape[1]:,} columns")
print(f"\nColumn Breakdown:")
print(f"  🎯 Target variables (M3 metals): {len(m3_cols)}")
print(f"  📈 Original features: {len(feature_cols)}")
print(f"  ⏱️  Lagged features: {len(feature_cols) * len(lag_periods)}")
print(f"  📊 Rolling averages: {len(feature_cols) * len(rolling_windows)}")
print(f"  📉 Change features: {len(feature_cols) * 2}")
print(f"  📅 Time features: 8")
print(f"  ✨ Total engineered features: {metadata['total_engineered_features']}")
print("\n" + "="*80)
print("Ready for modeling!")
print("="*80 + "\n")
