"""
Polymarket Feature Engineering Module
Calculates derived features for markets and traders
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

# ============================================================================
# INTELLIGENT CATEGORIZATION
# ============================================================================

def categorize_market(row) -> str:
    """
    Intelligently categorize a market based on question, tags, and outcomes.

    Categories:
    - Sports: NBA, NFL, NHL, MLB, soccer, MMA, boxing, tennis, etc.
    - Politics: Elections, Trump, government, policy
    - Crypto: Bitcoin, Ethereum, DeFi, crypto markets
    - Entertainment: Movies, TV, music, celebrities, awards
    - Business: Stocks, companies, earnings, IPOs
    - Climate: Weather, temperature, climate change
    - Science: Space, technology, research, discoveries
    - World Events: International news, conflicts, geopolitics
    - Other: Fallback category
    """

    question = str(row.get('question', '')).lower()
    description = str(row.get('description', '')).lower()
    tags = row.get('tags', '[]')

    # Parse tags if it's a JSON string
    try:
        if isinstance(tags, str):
            tags_list = json.loads(tags)
        else:
            tags_list = tags if isinstance(tags, list) else []
        tags_str = ' '.join([str(t).lower() for t in tags_list])
    except:
        tags_str = ''

    # Combine text for analysis
    text = f"{question} {description} {tags_str}"

    # PRIORITY 1: World Events (check first - most specific)
    world_keywords = [
        'russia', 'ukraine', 'china', 'war', 'conflict', 'military', 'nato',
        'peace', 'treaty', 'sanctions', 'israel', 'palestine', 'iran',
        'north korea', 'taiwan', 'middle east', 'europe', 'asia', 'strike',
        'invasion', 'ceasefire', 'troops', 'attack', 'weapon', 'nuclear',
        'missile', 'army', 'navy', 'defense', 'offensive', 'combat'
    ]
    if any(kw in text for kw in world_keywords):
        return 'World Events'

    # PRIORITY 2: Politics (check second)
    politics_keywords = [
        'trump', 'biden', 'harris', 'election', 'president', 'senate', 'congress',
        'democrat', 'republican', 'gop', 'vote', 'political', 'governor', 'mayor',
        'legislation', 'bill', 'law', 'policy', 'white house', 'cabinet',
        'impeachment', 'nomination', 'campaign', 'primary', 'ballot', 'deport',
        'immigration', 'executive order', 'veto', 'supreme court', 'scotus'
    ]
    if any(kw in text for kw in politics_keywords):
        return 'Politics'

    # Sports keywords
    sports_keywords = [
        'nba', 'nfl', 'nhl', 'mlb', 'mls', 'ufc', 'boxing', 'soccer', 'football',
        'basketball', 'baseball', 'hockey', 'tennis', 'golf', 'championship',
        'super bowl', 'world cup', 'playoffs', 'finals', 'game', 'match',
        'lakers', 'celtics', 'warriors', 'knicks', 'bulls', 'heat', 'nets',
        'eagles', 'cowboys', 'chiefs', 'patriots', '49ers', 'packers', 'steelers',
        ' vs ', ' @ ', 'defeat', 'win the', 'score', 'points', 'team',
        'player', 'mvp', 'rookie', 'draft', 'season', 'league', 'tournament',
        'premier league', 'la liga', 'champions league', 'euros', 'world series'
    ]

    # Crypto keywords
    crypto_keywords = [
        'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency',
        'blockchain', 'defi', 'nft', 'dogecoin', 'solana', 'cardano',
        'binance', 'coinbase', 'wallet', 'token', 'altcoin', 'satoshi',
        'web3', 'memecoin', 'stablecoin', 'mining'
    ]

    # Entertainment keywords
    entertainment_keywords = [
        'movie', 'film', 'box office', 'oscar', 'emmy', 'grammy', 'award',
        'actor', 'actress', 'director', 'celebrity', 'taylor swift', 'drake',
        'netflix', 'disney', 'marvel', 'star wars', 'streaming', 'album',
        'concert', 'tour', 'billboard', 'spotify', 'youtube', 'tiktok',
        'influencer', 'podcast', 'series', 'episode', 'premiere', 'sequel'
    ]

    # Business keywords
    business_keywords = [
        'stock', 'market cap', 'earnings', 'revenue', 'ipo', 'acquisition',
        'merger', 'ceo', 'company', 'corporation', 'shares', 'investor',
        'tesla', 'apple', 'amazon', 'google', 'microsoft', 'meta',
        'nvidia', 'dow', 's&p', 'nasdaq', 'wall street', 'startup',
        'unicorn', 'valuation', 'quarterly', 'profit', 'loss'
    ]

    # Climate keywords
    climate_keywords = [
        'temperature', 'climate', 'weather', 'hottest', 'coldest', 'hurricane',
        'tornado', 'flood', 'drought', 'global warming', 'celsius', 'fahrenheit',
        'ice', 'arctic', 'antarctic', 'sea level', 'carbon', 'emissions',
        'wildfire', 'storm', 'el niño', 'la niña', 'precipitation'
    ]

    # Science keywords
    science_keywords = [
        'nasa', 'space', 'rocket', 'satellite', 'mars', 'moon', 'asteroid',
        'spacex', 'blue origin', 'telescope', 'research', 'study', 'discovery',
        'nobel', 'scientist', 'laboratory', 'experiment', 'breakthrough',
        'vaccine', 'cure', 'disease', 'pandemic', 'covid', 'ai', 'artificial intelligence'
    ]

    # Count keyword matches for remaining categories
    sports_count = sum(1 for kw in sports_keywords if kw in text)
    crypto_count = sum(1 for kw in crypto_keywords if kw in text)
    entertainment_count = sum(1 for kw in entertainment_keywords if kw in text)
    business_count = sum(1 for kw in business_keywords if kw in text)
    climate_count = sum(1 for kw in climate_keywords if kw in text)
    science_count = sum(1 for kw in science_keywords if kw in text)

    # Determine category based on highest match count
    counts = {
        'Sports': sports_count,
        'Crypto': crypto_count,
        'Entertainment': entertainment_count,
        'Business': business_count,
        'Climate': climate_count,
        'Science': science_count,
    }

    max_count = max(counts.values())

    # Require at least 1 match to assign a category
    if max_count >= 1:
        return max(counts, key=counts.get)
    else:
        return 'Other'


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
    # Step 0: Intelligent Categorization (override API's 'Other' with smart classification)
    # --------------------------------------------------
    print("  [1/9] Applying intelligent categorization...")

    # Show category distribution from API/database BEFORE categorization
    category_counts_before = df['category'].value_counts()
    other_count_before = category_counts_before.get('Other', 0)
    print(f"  [INFO] Before categorization: {other_count_before:,} 'Other' markets ({other_count_before/len(df)*100:.1f}%)")

    # Apply intelligent categorization to all markets
    # This will override API's 'Other' with keyword-based classification
    df['category'] = df.apply(categorize_market, axis=1)

    # Show category distribution AFTER categorization
    category_counts_after = df['category'].value_counts()
    other_count_after = category_counts_after.get('Other', 0)
    print(f"  [OK] After categorization: {other_count_after:,} 'Other' markets ({other_count_after/len(df)*100:.1f}%)")
    print(f"  [OK] Improved categorization for {other_count_before - other_count_after:,} markets")

    print(f"  [OK] Category distribution:")
    for cat, count in category_counts_after.head(10).items():
        print(f"    - {cat}: {count:,} markets ({count/len(df)*100:.1f}%)")
    if len(category_counts_after) > 10:
        print(f"    ... and {len(category_counts_after) - 10} more")

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

    print(f"  [OK] Engineered {9} market-level features")
    print(f"    - category (intelligently classified from question/description)")
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
