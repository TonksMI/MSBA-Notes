#!/usr/bin/env python3
"""
Census Data Aggregation with Readable Indicator Names
=====================================================

Creates a master dataset with:
1. Aggregated M3 indicators by name (not code)
2. Human-readable column names
3. Simplified structure for easier analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("M3 PRIMARY METALS - READABLE MASTER DATASET")
print("="*80)

# Define indicator translations
M3_TRANSLATIONS = {
    'FI': 'Finished_Goods_Inventory',
    'IS': 'Inventory_Shipments_Ratio',
    'MI': 'Materials_Supplies_Inventory',
    'MPCFI': 'Pct_Change_Finished_Inventory',
    'MPCMI': 'Pct_Change_Materials_Inventory',
    'MPCNO': 'Pct_Change_New_Orders',
    'MPCTI': 'Pct_Change_Total_Inventory',
    'MPCUO': 'Pct_Change_Unfilled_Orders',
    'MPCVS': 'Pct_Change_Shipments_Value',
    'MPCWI': 'Pct_Change_WIP_Inventory',
    'NO': 'New_Orders',
    'TI': 'Total_Inventories',
    'UO': 'Unfilled_Orders',
    'US': 'Unfilled_Shipments_Ratio',
    'VS': 'Value_of_Shipments',
    'WI': 'Work_In_Process_Inventory'
}

ADVANCE_M3_TRANSLATIONS = {
    'MPCNO': 'Adv_Pct_Change_New_Orders',
    'MPCTI': 'Adv_Pct_Change_Total_Inventory',
    'MPCUO': 'Adv_Pct_Change_Unfilled_Orders',
    'MPCVS': 'Adv_Pct_Change_Shipments_Value',
    'NO': 'Adv_New_Orders',
    'TI': 'Adv_Total_Inventories',
    'UO': 'Adv_Unfilled_Orders',
    'VS': 'Adv_Value_of_Shipments'
}

# Construction category translations (simplified)
CONSTRUCTION_CATEGORIES = {
    'A07XX': 'Total_Construction',
    '20IX': 'Residential_Construction',
    'A10XX': 'Private_Construction',
    'A20XX': 'Public_Construction'
}

# Retail category translations
RETAIL_CATEGORIES = {
    '441': 'Motor_Vehicle_Parts_Dealers',
    '4411': 'Auto_Dealers',
    '4412': 'Other_Motor_Vehicle_Dealers'
}

# Residential category translations
RESIDENTIAL_CATEGORIES = {
    'ACOMPLETIONS': 'Housing_Completions',
    'APERMITS': 'Housing_Permits',
    'ASTARTS': 'Housing_Starts',
    'AUNDER': 'Housing_Under_Construction'
}

print("\n1. Loading all census datasets...")
m3_metals = pd.read_csv('census_m3_primary_metals.csv')
advm3 = pd.read_csv('census_advm3.csv')
construction = pd.read_csv('census_construction.csv')
mrts = pd.read_csv('census_mrts_selected.csv')
residential = pd.read_csv('census_res.csv')

print(f"   ✓ Loaded {len(m3_metals):,} M3 records")

# Convert dates
for df in [m3_metals, advm3, construction, mrts, residential]:
    df['date'] = pd.to_datetime(df['date'])

# Process M3 Primary Metals with aggregation
print("\n2. Processing M3 Primary Metals (aggregated by indicator)...")
m3_sa = m3_metals[m3_metals['seasonally_adjusted'] == True].copy()

# Aggregate by date and indicator (sum across all NAICS codes)
m3_agg = m3_sa.groupby(['date', 'indicator']).agg({
    'value_millions': 'sum'  # Sum across all subcategories
}).reset_index()

# Translate indicator names
m3_agg['indicator_readable'] = m3_agg['indicator'].map(M3_TRANSLATIONS)

# Pivot to wide format
m3_pivot = m3_agg.pivot_table(
    index='date',
    columns='indicator_readable',
    values='value_millions',
    aggfunc='first'
).reset_index()

print(f"   ✓ Created {len(m3_pivot.columns)-1} aggregated M3 columns")

# Process Advance M3
print("\n3. Processing Advance M3 data...")
advm3_sa = advm3[advm3['seasonally_adjusted'] == True].copy()

# Aggregate by date and indicator
advm3_agg = advm3_sa.groupby(['date', 'indicator']).agg({
    'value_millions': 'sum'
}).reset_index()

advm3_agg['indicator_readable'] = advm3_agg['indicator'].map(ADVANCE_M3_TRANSLATIONS)

advm3_pivot = advm3_agg.pivot_table(
    index='date',
    columns='indicator_readable',
    values='value_millions',
    aggfunc='first'
).reset_index()

print(f"   ✓ Created {len(advm3_pivot.columns)-1} advance M3 columns")

# Process Construction (keep major categories only)
print("\n4. Processing Construction data (major categories)...")
construction_sa = construction[construction['seasonally_adjusted'] == True].copy()

# Filter for major categories
major_categories = list(CONSTRUCTION_CATEGORIES.keys())
construction_major = construction_sa[construction_sa['category_code'].isin(major_categories)].copy()

# Create readable names
construction_major['category_readable'] = construction_major['category_code'].map(CONSTRUCTION_CATEGORIES)
construction_major['column_name'] = (
    construction_major['category_readable'] + '_' +
    construction_major['data_type_code']
)

construction_pivot = construction_major.pivot_table(
    index='date',
    columns='column_name',
    values='value_millions',
    aggfunc='first'
).reset_index()

print(f"   ✓ Created {len(construction_pivot.columns)-1} construction columns")

# Process Retail
print("\n5. Processing Retail Sales...")
mrts_sa = mrts[mrts['seasonally_adjusted'] == True].copy()

# Use category names directly or map codes
mrts_sa['category_readable'] = mrts_sa.apply(
    lambda x: RETAIL_CATEGORIES.get(str(x['category_code']), x['category_name'].replace(' ', '_')),
    axis=1
)

mrts_pivot = mrts_sa.pivot_table(
    index='date',
    columns='category_readable',
    values='value_millions',
    aggfunc='first'
).reset_index()

print(f"   ✓ Created {len(mrts_pivot.columns)-1} retail columns")

# Process Residential (major categories only)
print("\n6. Processing Residential data...")
residential_sa = residential[residential['seasonally_adjusted'] == True].copy()

# Filter for major categories
residential_major = residential_sa[
    residential_sa['category_code'].isin(RESIDENTIAL_CATEGORIES.keys())
].copy()

residential_major['category_readable'] = residential_major['category_code'].map(RESIDENTIAL_CATEGORIES)
residential_major['column_name'] = (
    residential_major['category_readable'] + '_' +
    residential_major['data_type_code']
)

residential_pivot = residential_major.pivot_table(
    index='date',
    columns='column_name',
    values='value_units',
    aggfunc='first'
).reset_index()

print(f"   ✓ Created {len(residential_pivot.columns)-1} residential columns")

# Merge all datasets
print("\n7. Merging all datasets...")
master = m3_pivot.copy()
master = master.merge(advm3_pivot, on='date', how='left')
master = master.merge(construction_pivot, on='date', how='left')
master = master.merge(mrts_pivot, on='date', how='left')
master = master.merge(residential_pivot, on='date', how='left')

print(f"   ✓ Master dataset shape: {master.shape}")

# Get column categories
m3_cols = [col for col in master.columns if col in M3_TRANSLATIONS.values()]
feature_cols = [col for col in master.columns if col not in m3_cols and col != 'date']

print(f"   ✓ Target columns: {len(m3_cols)}")
print(f"   ✓ Feature columns: {len(feature_cols)}")

# Sort by date
master = master.sort_values('date').reset_index(drop=True)

# Create lagged features
print("\n8. Creating lagged features (1, 2, 3, 6, 12 months)...")
lag_periods = [1, 2, 3, 6, 12]

for lag in lag_periods:
    for col in feature_cols:
        master[f'{col}_Lag{lag}mo'] = master[col].shift(lag)

# Create rolling averages
print("\n9. Creating rolling average features...")
rolling_windows = [3, 6, 12]

for window in rolling_windows:
    for col in feature_cols:
        master[f'{col}_MA{window}mo'] = master[col].rolling(window=window).mean()

# Create growth rates
print("\n10. Creating growth rate features...")
for col in feature_cols:
    master[f'{col}_MoM_Growth'] = master[col].pct_change() * 100  # As percentage
    master[f'{col}_YoY_Growth'] = master[col].pct_change(periods=12) * 100

# Add time features
print("\n11. Adding time features...")
master['Year'] = master['date'].dt.year
master['Month'] = master['date'].dt.month
master['Quarter'] = master['date'].dt.quarter
master['Is_Q1'] = (master['Quarter'] == 1).astype(int)
master['Is_Q2'] = (master['Quarter'] == 2).astype(int)
master['Is_Q3'] = (master['Quarter'] == 3).astype(int)
master['Is_Q4'] = (master['Quarter'] == 4).astype(int)

# Rename date column for clarity
master = master.rename(columns={'date': 'Date'})

# Save master dataset
print("\n12. Saving master dataset...")
output_file = 'master_m3_readable.csv'
master.to_csv(output_file, index=False)

print(f"   ✓ Saved to: {output_file}")

# Create data dictionary
print("\n13. Creating data dictionary...")
data_dict = []

# Target variables
for indicator, readable in M3_TRANSLATIONS.items():
    data_dict.append({
        'Column': readable,
        'Category': 'M3 Primary Metals (Target)',
        'Description': f'M3 Primary Metals - {readable.replace("_", " ")}',
        'Units': 'Millions USD' if not indicator.startswith('MPC') else 'Percent Change',
        'Type': 'Target Variable'
    })

# Feature variables
for indicator, readable in ADVANCE_M3_TRANSLATIONS.items():
    data_dict.append({
        'Column': readable,
        'Category': 'Advance M3',
        'Description': f'Advance Manufacturing - {readable.replace("_", " ")}',
        'Units': 'Millions USD' if not indicator.startswith('MPC') else 'Percent Change',
        'Type': 'Predictor'
    })

for code, readable in CONSTRUCTION_CATEGORIES.items():
    data_dict.append({
        'Column': readable,
        'Category': 'Construction',
        'Description': f'Construction - {readable.replace("_", " ")}',
        'Units': 'Millions USD',
        'Type': 'Predictor'
    })

for code, readable in RETAIL_CATEGORIES.items():
    data_dict.append({
        'Column': readable,
        'Category': 'Retail Sales',
        'Description': f'Retail Sales - {readable.replace("_", " ")}',
        'Units': 'Millions USD',
        'Type': 'Predictor'
    })

for code, readable in RESIDENTIAL_CATEGORIES.items():
    data_dict.append({
        'Column': readable,
        'Category': 'Residential',
        'Description': f'Residential - {readable.replace("_", " ")}',
        'Units': 'Units (thousands)',
        'Type': 'Predictor'
    })

# Engineered features
data_dict.append({
    'Column': '{Feature}_Lag{N}mo',
    'Category': 'Engineered - Lagged',
    'Description': 'Lagged value N months ago (1, 2, 3, 6, 12)',
    'Units': 'Same as base feature',
    'Type': 'Predictor'
})

data_dict.append({
    'Column': '{Feature}_MA{N}mo',
    'Category': 'Engineered - Rolling Average',
    'Description': 'N-month moving average (3, 6, 12)',
    'Units': 'Same as base feature',
    'Type': 'Predictor'
})

data_dict.append({
    'Column': '{Feature}_MoM_Growth',
    'Category': 'Engineered - Growth Rate',
    'Description': 'Month-over-month growth rate',
    'Units': 'Percentage',
    'Type': 'Predictor'
})

data_dict.append({
    'Column': '{Feature}_YoY_Growth',
    'Category': 'Engineered - Growth Rate',
    'Description': 'Year-over-year growth rate',
    'Units': 'Percentage',
    'Type': 'Predictor'
})

dict_df = pd.DataFrame(data_dict)
dict_df.to_csv('data_dictionary.csv', index=False)

print(f"   ✓ Data dictionary saved to: data_dictionary.csv")

# Create summary
print("\n" + "="*80)
print("SUMMARY - READABLE MASTER DATASET")
print("="*80)
print(f"\n📊 File: {output_file}")
print(f"📅 Date Range: {master['Date'].min().strftime('%Y-%m')} to {master['Date'].max().strftime('%Y-%m')}")
print(f"📏 Dimensions: {master.shape[0]:,} rows × {master.shape[1]:,} columns")

print(f"\n🎯 Target Variables (M3 Primary Metals):")
for indicator, readable in sorted(M3_TRANSLATIONS.items()):
    if readable in master.columns:
        print(f"   • {readable}")

print(f"\n📈 Feature Categories:")
print(f"   • Advance M3: {len(ADVANCE_M3_TRANSLATIONS)} indicators")
print(f"   • Construction: {len([c for c in master.columns if 'Construction' in c and 'Lag' not in c and 'MA' not in c and 'Growth' not in c])} series")
print(f"   • Retail Sales: {len([c for c in master.columns if 'Dealers' in c and 'Lag' not in c and 'MA' not in c and 'Growth' not in c])} series")
print(f"   • Residential: {len([c for c in master.columns if 'Housing' in c and 'Lag' not in c and 'MA' not in c and 'Growth' not in c])} series")

print(f"\n✨ Engineered Features:")
print(f"   • Lagged features: {len([c for c in master.columns if 'Lag' in c])}")
print(f"   • Moving averages: {len([c for c in master.columns if 'MA' in c and 'Lag' not in c])}")
print(f"   • Growth rates: {len([c for c in master.columns if 'Growth' in c])}")
print(f"   • Time features: {len([c for c in master.columns if c in ['Year', 'Month', 'Quarter', 'Is_Q1', 'Is_Q2', 'Is_Q3', 'Is_Q4']])}")

print(f"\n📖 Documentation:")
print(f"   • Data dictionary: data_dictionary.csv")
print(f"   • Column descriptions and units included")

print("\n" + "="*80)
print("✅ Readable master dataset created successfully!")
print("="*80 + "\n")
