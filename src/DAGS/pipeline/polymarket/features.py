"""
Polymarket Feature Engineering Module
Calculates derived features for markets and traders
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

# ============================================================================
# MARKET-LEVEL FEATURES
# ============================================================================

def engineer_market_features(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for prediction markets.

    Features added:
    - days_until_end: Days remaining until market closes
    - volume_per_day: Average daily volume
    - liquidity_score: Liquidity relative to volume
    - activity_score: Composite score of volume + liquidity
    - category_encoded: Numeric encoding of categories
    - outcome_count: Number of possible outcomes

    Args:
        markets_df: DataFrame with raw market data

    Returns:
        DataFrame with engineered features added
    """

    print("\n[FEATURES] Engineering market-level features...")
    print("-" * 80)

    if len(markets_df) == 0:
        print("  [WARN] No markets to engineer features for")
        return markets_df

    df = markets_df.copy()

    # --------------------------------------------------
    # Feature 1: Days until expiry
    # --------------------------------------------------
    def calculate_days_until_end(end_date_str):
        try:
            if pd.isna(end_date_str):
                return 999  # Default for markets with no end date

            end_date = pd.to_datetime(end_date_str)
            if end_date.tzinfo is not None:
                end_date = end_date.tz_localize(None)

            now = pd.Timestamp.now()
            if now.tzinfo is not None:
                now = now.tz_localize(None)

            days = (end_date - now).days
            return max(days, 1)  # Minimum 1 day
        except:
            return 999

    df['days_until_end'] = df['end_date'].apply(calculate_days_until_end)

    # --------------------------------------------------
    # Feature 2: Volume per day
    # --------------------------------------------------
    df['volume_per_day'] = df['volume'] / df['days_until_end']
    df['volume_per_day'] = df['volume_per_day'].replace([np.inf, -np.inf], 0).fillna(0)

    # --------------------------------------------------
    # Feature 3: Liquidity score
    # --------------------------------------------------
    # Higher liquidity relative to volume = better market quality
    df['liquidity_score'] = df['liquidity'] / (df['volume_24h'] + 1)
    df['liquidity_score'] = df['liquidity_score'].replace([np.inf, -np.inf], 0).fillna(0)

    # --------------------------------------------------
    # Feature 4: Activity score (composite)
    # --------------------------------------------------
    # Normalize volume and liquidity, then combine
    vol_max = df['volume_24h'].max()
    liq_max = df['liquidity'].max()

    df['activity_score'] = (
        (df['volume_24h'] / vol_max if vol_max > 0 else 0) * 0.6 +
        (df['liquidity'] / liq_max if liq_max > 0 else 0) * 0.4
    )
    df['activity_score'] = df['activity_score'].fillna(0)

    # --------------------------------------------------
    # Feature 5: Category encoding
    # --------------------------------------------------
    df['category_encoded'] = pd.Categorical(df['category']).codes

    # --------------------------------------------------
    # Feature 6: Outcome count
    # --------------------------------------------------
    def count_outcomes(outcomes_str):
        try:
            if pd.isna(outcomes_str):
                return 0
            outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
            return len(outcomes) if isinstance(outcomes, list) else 0
        except:
            return 0

    df['outcome_count'] = df['outcomes'].apply(count_outcomes)

    # --------------------------------------------------
    # Feature 7: Market age (if game_start_time exists)
    # --------------------------------------------------
    if 'game_start_time' in df.columns:
        def calculate_market_age(start_time_str):
            try:
                if pd.isna(start_time_str):
                    return 0
                start_time = pd.to_datetime(start_time_str)
                if start_time.tzinfo is not None:
                    start_time = start_time.tz_localize(None)
                now = pd.Timestamp.now()
                if now.tzinfo is not None:
                    now = now.tz_localize(None)
                age = (now - start_time).days
                return max(age, 0)
            except:
                return 0

        df['market_age_days'] = df['game_start_time'].apply(calculate_market_age)
    else:
        df['market_age_days'] = 0

    # --------------------------------------------------
    # Feature 8: Probability confidence
    # --------------------------------------------------
    # How far from 50/50 is the market? (higher = more confident)
    if 'yes_probability' in df.columns:
        df['probability_confidence'] = abs(df['yes_probability'] - 0.5)
        df['probability_confidence'] = df['probability_confidence'].fillna(0)
    else:
        df['probability_confidence'] = 0

    print(f"  [OK] Engineered {8} market-level features")
    print(f"    - days_until_end, volume_per_day, liquidity_score")
    print(f"    - activity_score, category_encoded, outcome_count")
    print(f"    - market_age_days, probability_confidence")

    return df


# ============================================================================
# TRADER-LEVEL FEATURES
# ============================================================================

def engineer_trader_features(traders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for traders.

    Features added:
    - volume_rank: Rank by total volume (1 = highest)
    - avg_position_size: Average $ per position
    - activity_level: Categorical (casual/regular/active/power_user)

    Args:
        traders_df: DataFrame with trader data

    Returns:
        DataFrame with engineered features added
    """

    print("\n[FEATURES] Engineering trader-level features...")
    print("-" * 80)

    if len(traders_df) == 0:
        print("  [WARN] No traders to engineer features for")
        return traders_df

    df = traders_df.copy()

    # --------------------------------------------------
    # Feature 1: Volume rank
    # --------------------------------------------------
    df['volume_rank'] = df['total_volume'].rank(ascending=False, method='min')

    # --------------------------------------------------
    # Feature 2: Average position size
    # --------------------------------------------------
    df['avg_position_size'] = df['total_volume'] / df['total_trades'].replace(0, 1)
    df['avg_position_size'] = df['avg_position_size'].replace([np.inf, -np.inf], 0).fillna(0)

    # --------------------------------------------------
    # Feature 3: Activity level (categorical)
    # --------------------------------------------------
    df['activity_level'] = pd.cut(
        df['total_trades'],
        bins=[0, 5, 20, 100, float('inf')],
        labels=['casual', 'regular', 'active', 'power_user']
    ).astype(str)

    # --------------------------------------------------
    # Feature 4: Profitability ratio
    # --------------------------------------------------
    df['profit_ratio'] = df['total_profit'] / df['total_volume'].replace(0, 1)
    df['profit_ratio'] = df['profit_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

    # --------------------------------------------------
    # Feature 5: Win rate estimation (if profitable)
    # --------------------------------------------------
    df['is_profitable'] = df['total_profit'] > 0

    print(f"  [OK] Engineered {5} trader-level features")
    print(f"    - volume_rank, avg_position_size, activity_level")
    print(f"    - profit_ratio, is_profitable")

    # Print distribution stats
    if len(df) > 0:
        print(f"\n  Activity Level Distribution:")
        activity_counts = df['activity_level'].value_counts()
        for level, count in activity_counts.items():
            print(f"    - {level}: {count} ({count/len(df)*100:.1f}%)")

        profitable_count = df['is_profitable'].sum()
        print(f"\n  Profitable traders: {profitable_count} ({profitable_count/len(df)*100:.1f}%)")

    return df


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def engineer_all_features(
    markets_df: pd.DataFrame,
    traders_df: pd.DataFrame = None
) -> tuple:
    """
    Convenience function to engineer features for both markets and traders.

    Args:
        markets_df: Market data
        traders_df: Trader data (optional)

    Returns:
        Tuple of (markets_with_features_df, traders_with_features_df)
    """

    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    # Engineer market features
    markets_engineered = engineer_market_features(markets_df)

    # Engineer trader features (if provided)
    if traders_df is not None and len(traders_df) > 0:
        traders_engineered = engineer_trader_features(traders_df)
    else:
        print("\n  [WARN] No trader data provided, skipping trader features")
        traders_engineered = pd.DataFrame()

    print("\n" + "="*80)
    print("[OK] FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"  Markets with features: {len(markets_engineered):,}")
    if traders_engineered is not None and len(traders_engineered) > 0:
        print(f"  Traders with features: {len(traders_engineered):,}")
    print("="*80 + "\n")

    return markets_engineered, traders_engineered


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    # Test feature engineering with mock data
    print("Testing feature engineering with mock data...")

    # Mock market data
    markets_mock = pd.DataFrame({
        'market_id': ['1', '2', '3'],
        'question': ['Test market 1', 'Test market 2', 'Test market 3'],
        'end_date': [
            (datetime.now() + pd.Timedelta(days=10)).isoformat(),
            (datetime.now() + pd.Timedelta(days=30)).isoformat(),
            (datetime.now() + pd.Timedelta(days=5)).isoformat()
        ],
        'volume': [100000, 50000, 200000],
        'volume_24h': [10000, 5000, 25000],
        'liquidity': [5000, 2500, 15000],
        'category': ['Politics', 'Sports', 'Politics'],
        'outcomes': ['["Yes", "No"]', '["Yes", "No"]', '["Yes", "No"]'],
        'yes_probability': [0.65, 0.45, 0.80]
    })

    # Mock trader data
    traders_mock = pd.DataFrame({
        'trader_key': ['trader1', 'trader2', 'trader3'],
        'address': ['0x123...', '0x456...', '0x789...'],
        'total_volume': [100000, 50000, 200000],
        'total_trades': [50, 10, 150],
        'total_profit': [5000, -1000, 25000],
        'is_whale': [False, False, True]
    })

    # Engineer features
    markets_eng, traders_eng = engineer_all_features(markets_mock, traders_mock)

    print(f"\nMarket features sample:")
    print(markets_eng[['question', 'days_until_end', 'activity_score', 'probability_confidence']].head())

    print(f"\nTrader features sample:")
    print(traders_eng[['trader_key', 'volume_rank', 'activity_level', 'profit_ratio']].head())
