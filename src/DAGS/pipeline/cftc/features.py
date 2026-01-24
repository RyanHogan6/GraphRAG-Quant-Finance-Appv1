"""
CFTC Data Feature Engineering
"""
import pandas as pd

KEY_FIELDS = [
    "Market and Exchange Names",
    "As of Date in Form YYYY-MM-DD",
    "CFTC Commodity Code",
    "Open Interest (All)",
    "Noncommercial Positions-Long (All)",
    "Noncommercial Positions-Short (All)",
    "Commercial Positions-Long (All)",
    "Commercial Positions-Short (All)",
    "Change in Open Interest (All)",
    "Change in Noncommercial-Long (All)",
    "Change in Noncommercial-Short (All)",
    "% of OI-Noncommercial-Long (All)",
    "% of OI-Noncommercial-Short (All)",
    "% of OI-Commercial-Long (All)",
    "% of OI-Commercial-Short (All)",
]

def engineer_cftc_features(df):
    """Clean and engineer features from CFTC data"""
    if df.empty:
        return df

    # Keep only key fields
    available_fields = [f for f in KEY_FIELDS if f in df.columns]
    df = df[available_fields].copy()

    # Convert date
    date_col = "As of Date in Form YYYY-MM-DD"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')

    # Convert numeric fields
    for col in df.columns:
        if col not in ["Market and Exchange Names", date_col, "CFTC Commodity Code"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate derived features
    if all(c in df.columns for c in ["Noncommercial Positions-Long (All)", "Noncommercial Positions-Short (All)"]):
        df["Net_Noncommercial"] = df["Noncommercial Positions-Long (All)"] - df["Noncommercial Positions-Short (All)"]

    if all(c in df.columns for c in ["Commercial Positions-Long (All)", "Commercial Positions-Short (All)"]):
        df["Net_Commercial"] = df["Commercial Positions-Long (All)"] - df["Commercial Positions-Short (All)"]

    # Drop nulls in key fields
    df = df.dropna(subset=["CFTC Commodity Code", date_col])

    return df
