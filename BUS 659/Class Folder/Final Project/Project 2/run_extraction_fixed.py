#!/usr/bin/env python3
"""
Census Data Extraction Script - FIXED VERSION
Extracts data from Census Bureau economic indicators with corrected date parsing
"""

import requests
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import time
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_KEY = '87f911d32c71d4325b27bea0d9d72358be85825c'
START_DATE = '2015-01-01'
OUTPUT_DIR = Path('extracted_data/census')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG = []

def log_extraction(source, status, records=0, error_msg=None):
    AUDIT_LOG.append({
        'timestamp': datetime.now(),
        'source': source,
        'status': status,
        'records': records,
        'error': error_msg
    })

def retry_request(url, params=None, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            wait = backoff_factor ** attempt
            print(f"  Retry attempt {attempt + 1} after {wait}s...")
            time.sleep(wait)

def save_data(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"  ✓ Saved to {path}")
    return path

def parse_date_from_api(df):
    """
    Parse date from Census API response.
    The API returns a 'time' column with format 'YYYY-MM'
    """
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'], format='%Y-%m', errors='coerce')
    elif 'time_slot_id' in df.columns:
        # Fallback to time_slot_id if time not available
        df['date'] = pd.to_datetime(df['time_slot_id'], format='%Y%m', errors='coerce')
    else:
        print("  ⚠ Warning: No time column found")
        df['date'] = pd.NaT
    return df

# Extract Construction Spending (VIP)
def extract_census_construction(api_key, start_date=START_DATE):
    base_url = "https://api.census.gov/data/timeseries/eits/vip"
    start_ym = pd.to_datetime(start_date).strftime('%Y-%m')
    params = {
        'get': 'cell_value,time_slot_id,category_code,data_type_code,seasonally_adj',
        'time': f'from {start_ym}',
        'key': api_key
    }
    request_url = f"{base_url}?{urlencode(params, doseq=True)}"
    try:
        print(f"\nExtracting Census Construction (VIP)")
        print(f"  Request URL: {request_url}")
        response = retry_request(request_url)
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])

        # Use corrected date parsing
        df = parse_date_from_api(df)

        df['value_millions'] = pd.to_numeric(df['cell_value'], errors='coerce')
        df['source'] = 'CENSUS_VIP'
        df['extraction_timestamp'] = datetime.now()
        if 'seasonally_adj' in df.columns:
            df['seasonally_adjusted'] = df['seasonally_adj'] == 'yes'
        df = df[['date', 'category_code', 'data_type_code', 'value_millions',
                 'seasonally_adjusted', 'source', 'extraction_timestamp']]
        df = df.dropna(subset=['value_millions'])
        print(f"  ✓ {len(df)} rows | {df['date'].min()} → {df['date'].max()}")
        log_extraction('CENSUS_VIP', 'SUCCESS', len(df))
        return df
    except Exception as err:
        print(f"  ✗ VIP extraction failed: {err}")
        log_extraction('CENSUS_VIP', 'FAILED', 0, str(err))
        return None

# Extract M3 Manufacturing
def extract_census_m3(api_key, start_date=START_DATE):
    base_url = "https://api.census.gov/data/timeseries/eits/m3"
    start_ym = pd.to_datetime(start_date).strftime('%Y-%m')
    params = {
        'get': 'cell_value,time_slot_id,data_type_code,category_code,seasonally_adj',
        'for': 'US',
        'time': f'from {start_ym}',
        'key': api_key
    }
    request_url = f"{base_url}?{urlencode(params, doseq=True)}"
    try:
        print(f"\nExtracting Census M3 (Manufacturing)")
        print(f"  Request URL: {request_url}")
        response = retry_request(request_url)
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])

        # Use corrected date parsing
        df = parse_date_from_api(df)

        df['value_millions'] = pd.to_numeric(df['cell_value'], errors='coerce')
        df['naics_code'] = df['category_code']
        df['indicator'] = df['data_type_code']
        df['source'] = 'CENSUS_M3'
        df['extraction_timestamp'] = datetime.now()

        data_type_map = {
            'SM': 'Shipments Monthly',
            'NO': 'New Orders',
            'UO': 'Unfilled Orders',
            'TI': 'Total Inventories'
        }
        df['indicator_name'] = df['indicator'].map(data_type_map).fillna(df['indicator'])

        if 'seasonally_adj' in df.columns:
            df['seasonally_adjusted'] = df['seasonally_adj'] == 'yes'

        df = df[['date', 'naics_code', 'indicator', 'indicator_name', 'value_millions',
                 'seasonally_adjusted', 'source', 'extraction_timestamp']]
        df = df.dropna(subset=['value_millions'])
        print(f"  ✓ {len(df)} rows | {df['date'].min()} → {df['date'].max()}")
        log_extraction('CENSUS_M3', 'SUCCESS', len(df))
        return df
    except Exception as err:
        print(f"  ✗ M3 extraction failed: {err}")
        log_extraction('CENSUS_M3', 'FAILED', 0, str(err))
        return None

# Extract Residential Construction
def extract_census_res(api_key, start_date=START_DATE):
    base_url = "https://api.census.gov/data/timeseries/eits/resconst"
    start_ym = pd.to_datetime(start_date).strftime('%Y-%m')
    params = {
        'get': 'cell_value,time_slot_id,category_code,data_type_code,seasonally_adj',
        'time': f'from {start_ym}',
        'key': api_key
    }
    request_url = f"{base_url}?{urlencode(params, doseq=True)}"
    try:
        print(f"\nExtracting Census RES (Residential Construction)")
        print(f"  Request URL: {request_url}")
        response = retry_request(request_url)
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])

        # Use corrected date parsing
        df = parse_date_from_api(df)

        df['value_units'] = pd.to_numeric(df['cell_value'], errors='coerce')
        df['source'] = 'CENSUS_RES'
        df['extraction_timestamp'] = datetime.now()

        if 'seasonally_adj' in df.columns:
            df['seasonally_adjusted'] = df['seasonally_adj'] == 'yes'

        df = df[['date', 'category_code', 'data_type_code', 'value_units',
                 'seasonally_adjusted', 'source', 'extraction_timestamp']]
        df = df.dropna(subset=['value_units'])
        print(f"  ✓ {len(df)} rows | {df['date'].min()} → {df['date'].max()}")
        log_extraction('CENSUS_RES', 'SUCCESS', len(df))
        return df
    except Exception as err:
        print(f"  ✗ RES extraction failed: {err}")
        log_extraction('CENSUS_RES', 'FAILED', 0, str(err))
        return None

# Extract Retail Trade
def extract_census_mrts(api_key, start_date=START_DATE, categories=['441', '444', '4441']):
    base_url = "https://api.census.gov/data/timeseries/eits/mrts"
    start_ym = pd.to_datetime(start_date).strftime('%Y-%m')
    frames = []

    for category in categories:
        params = {
            'get': 'cell_value,time_slot_id,data_type_code,category_code,seasonally_adj',
            'for': 'US',
            'time': f'from {start_ym}',
            'category_code': category,
            'key': api_key
        }
        request_url = f"{base_url}?{urlencode(params, doseq=True)}"
        try:
            print(f"\n  Extracting MRTS category {category}")
            response = retry_request(request_url)
            data = response.json()
            frames.append(pd.DataFrame(data[1:], columns=data[0]))
        except Exception as err:
            print(f"    ✗ Category {category} failed: {err}")

    if not frames:
        log_extraction('CENSUS_MRTS', 'FAILED', 0, 'All category requests failed')
        return None

    df = pd.concat(frames, ignore_index=True)

    # Remove duplicate columns if they exist
    df = df.loc[:, ~df.columns.duplicated()]

    # Use corrected date parsing
    df = parse_date_from_api(df)

    df['value_millions'] = pd.to_numeric(df['cell_value'], errors='coerce')
    df['source'] = 'CENSUS_MRTS'
    df['extraction_timestamp'] = datetime.now()

    category_names = {
        '441': 'Motor Vehicle and Parts Dealers',
        '444': 'Building Material & Garden Equipment',
        '4441': 'Building Material and Supplies Dealers'
    }

    # Create category_name column
    if 'category_code' in df.columns:
        df['category_name'] = df['category_code'].astype(str).map(category_names)
        df['category_name'] = df['category_name'].fillna(df['category_code'].astype(str))
    else:
        df['category_name'] = 'Unknown'

    if 'seasonally_adj' in df.columns:
        df['seasonally_adjusted'] = df['seasonally_adj'] == 'yes'
    else:
        df['seasonally_adjusted'] = False

    # Select specific columns
    df = df[['date', 'category_code', 'category_name', 'value_millions',
             'seasonally_adjusted', 'source', 'extraction_timestamp']]
    df = df.dropna(subset=['value_millions'])
    print(f"\n✓ MRTS total rows: {len(df)}")
    log_extraction('CENSUS_MRTS', 'SUCCESS', len(df))
    return df

# Extract Advance Durable Goods
def extract_census_advm3(api_key, start_date=START_DATE):
    base_url = "https://api.census.gov/data/timeseries/eits/advm3"
    start_ym = pd.to_datetime(start_date).strftime('%Y-%m')
    params = {
        'get': 'cell_value,time_slot_id,data_type_code,category_code,seasonally_adj',
        'for': 'US',
        'time': f'from {start_ym}',
        'key': api_key
    }
    request_url = f"{base_url}?{urlencode(params, doseq=True)}"
    try:
        print(f"\nExtracting Census ADVM3 (Advance Durable Goods)")
        print(f"  Request URL: {request_url}")
        response = retry_request(request_url)
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])

        # Use corrected date parsing
        df = parse_date_from_api(df)

        df['value_millions'] = pd.to_numeric(df['cell_value'], errors='coerce')
        df['indicator'] = df['data_type_code']
        df['source'] = 'CENSUS_ADVM3'
        df['is_advance_release'] = True
        df['extraction_timestamp'] = datetime.now()

        data_type_map = {
            'SM': 'Shipments Monthly',
            'NO': 'New Orders',
            'UO': 'Unfilled Orders'
        }
        df['indicator_name'] = df['indicator'].map(data_type_map).fillna(df['indicator'])

        if 'seasonally_adj' in df.columns:
            df['seasonally_adjusted'] = df['seasonally_adj'] == 'yes'

        df = df[['date', 'category_code', 'indicator', 'indicator_name', 'value_millions',
                 'seasonally_adjusted', 'is_advance_release', 'source', 'extraction_timestamp']]
        df = df.dropna(subset=['value_millions'])
        print(f"  ✓ {len(df)} rows | {df['date'].min()} → {df['date'].max()}")
        log_extraction('CENSUS_ADVM3', 'SUCCESS', len(df))
        return df
    except Exception as err:
        print(f"  ✗ ADVM3 extraction failed: {err}")
        log_extraction('CENSUS_ADVM3', 'FAILED', 0, str(err))
        return None

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("CENSUS DATA EXTRACTION - FIXED VERSION")
    print("="*80)
    print(f"Start Date: {START_DATE}")
    print(f"Output Directory: {OUTPUT_DIR.resolve()}")
    print(f"Fix Applied: Using 'time' column with format 'YYYY-MM'")
    print("="*80)

    # Extract all datasets
    construction = extract_census_construction(API_KEY)
    if construction is not None:
        save_data(construction, 'census_construction.csv')

    m3 = extract_census_m3(API_KEY)
    if m3 is not None:
        save_data(m3, 'census_m3_primary_metals.csv')

    residential = extract_census_res(API_KEY)
    if residential is not None:
        save_data(residential, 'census_res.csv')

    retail = extract_census_mrts(API_KEY)
    if retail is not None:
        save_data(retail, 'census_mrts_selected.csv')

    durable = extract_census_advm3(API_KEY)
    if durable is not None:
        save_data(durable, 'census_advm3.csv')

    # Save audit log
    audit_df = pd.DataFrame(AUDIT_LOG)
    save_data(audit_df, 'audit_log.csv')

    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(audit_df.groupby(['source', 'status']).agg({'records': 'sum'}))
    print("\n✅ Extraction complete with corrected date parsing!")
