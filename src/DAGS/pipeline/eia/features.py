"""
EIA Data Feature Engineering
Standardizes formats and calculates derived metrics
"""
import pandas as pd

def engineer_eia_features(df, dataset_name):
    """
    Clean and engineer features for EIA data

    Args:
        df: Raw EIA dataframe
        dataset_name: Name of dataset for metadata
    """
    if df.empty:
        return df

    print(f"    Engineering {dataset_name}: {len(df)} raw records")

    # Standardize column names
    df = df.copy()

    # Ensure date column exists
    if 'period' in df.columns:
        df['report_date'] = pd.to_datetime(df['period'], errors='coerce')
        # Drop rows with invalid dates
        invalid_dates = df['report_date'].isna().sum()
        if invalid_dates > 0:
            print(f"    ⚠ Dropping {invalid_dates} records with invalid dates")
        df = df.dropna(subset=['report_date'])
        df['report_date'] = df['report_date'].dt.strftime('%Y-%m-%d')

    # Convert value to numeric
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Sort by date
    if 'report_date' in df.columns:
        df = df.sort_values('report_date')

    # Calculate change from previous period
    if 'value' in df.columns:
        df['change_from_previous'] = df['value'].diff()
        df['pct_change'] = df['value'].pct_change(fill_method=None) * 100

    # Add metadata
    df['ingested_at'] = pd.Timestamp.now().isoformat()

    # Drop nulls in key fields
    if 'report_date' in df.columns and 'value' in df.columns:
        before_drop = len(df)
        df = df.dropna(subset=['report_date', 'value'])
        after_drop = len(df)
        if before_drop != after_drop:
            print(f"    ⚠ Dropped {before_drop - after_drop} records with null date/value")

    print(f"    Final {dataset_name}: {len(df)} clean records")
    return df
