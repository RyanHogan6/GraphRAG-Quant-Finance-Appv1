"""
FRED Data Feature Engineering
Pivots long format to wide format (one row per date)
"""
import pandas as pd

def engineer_fred_features(df):
    """
    Transform FRED data from long to wide format
    One document per date with all indicators

    Args:
        df: Long-format DataFrame (date, series_id, value)
    """
    if df.empty:
        return df

    print(f"  Engineering FRED data: {len(df)} raw records")

    # Convert to wide format (pivot)
    # Each date becomes one row with all indicators as columns
    pivot_df = df.pivot_table(
        index='date',
        columns='series_name',
        values='value',
        aggfunc='last'  # If multiple values per date, take last
    )

    # Reset index to make date a column
    pivot_df = pivot_df.reset_index()

    # Clean column names for ArangoDB
    pivot_df.columns = [
        col.replace(' ', '_')
           .replace('(', '')
           .replace(')', '')
           .replace('/', '_')
           .replace('-', '_')
           .replace('&', 'and')
           .replace("'", '')
           .lower()
        for col in pivot_df.columns
    ]

    # Ensure date column
    if 'date' in pivot_df.columns:
        pivot_df['date'] = pd.to_datetime(pivot_df['date'], errors='coerce')
        pivot_df = pivot_df.dropna(subset=['date'])
        pivot_df['date'] = pivot_df['date'].dt.strftime('%Y-%m-%d')

    # Calculate derived features
    # Yield curve slope (10Y - 2Y)
    if '10_year_treasury_yield' in pivot_df.columns and '2_year_treasury_yield' in pivot_df.columns:
        pivot_df['yield_curve_slope'] = (
            pivot_df['10_year_treasury_yield'] - pivot_df['2_year_treasury_yield']
        )
        pivot_df['yield_curve_inverted'] = (pivot_df['yield_curve_slope'] < 0).astype(int)

    print(f"  Final FRED data: {len(pivot_df)} dates with {len(pivot_df.columns)} indicators")

    return pivot_df
