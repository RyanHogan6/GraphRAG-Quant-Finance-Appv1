"""
Polymarket Pipeline Configuration
API endpoints, database settings, and pipeline parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
# This file should be in .gitignore and NEVER committed!
load_dotenv()

# ============================================================================
# API ENDPOINTS
# ============================================================================

# Polymarket API endpoints
POLYMARKET_BASE_URL = "https://clob.polymarket.com"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com/v1"  # New unified data API
LEADERBOARD_URL = f"{DATA_API_URL}/leaderboard"
POSITIONS_URL = "https://data-api.polymarket.com/positions"  # User positions endpoint
SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket"  # DEPRECATED - kept for reference

# ============================================================================
# ARANGODB CONFIGURATION
# ============================================================================

DB_NAME = os.getenv('ARANGO_DATABASE', 'QUANT_v3')
USERNAME = os.getenv('ARANGO_USERNAME', 'root')

# CRITICAL: Load password from environment variable
# Set via .env file (local dev) or Airflow secrets (production)
PASSWORD = os.getenv('ARANGO_PASSWORD')

if not PASSWORD:
    raise ValueError(
        "ARANGO_PASSWORD not set! Please create a .env file with:\n"
        "ARANGO_PASSWORD=your_password_here\n"
        "Or set the environment variable in your system."
    )

# Support both ARANGO_HOST and ARANGO_URL (Railway uses ARANGO_URL)
ARANGO_HOST = os.getenv('ARANGO_URL') or os.getenv('ARANGO_HOST')
GRAPH_NAME = os.getenv('ARANGO_GRAPH_NAME', 'QUANT_v3_FinanceGraph')

# ============================================================================
# COLLECTION NAMES
# ============================================================================

# Document collections
COMPANY_COL = "Company"
MARKET_COL = "prediction_markets_polymarket"
TRADER_COL = "polymarket_traders"
POSITION_COL = "polymarket_positions"

# Edge collections
EDGE_DIRECT = "market_mentions_company_polymarket"
EDGE_SECTOR = "market_related_to_sector_polymarket"
EDGE_MACRO = "market_affects_company_polymarket"
EDGE_TRADER_POSITION = "trader_has_position"
EDGE_POSITION_MARKET = "position_in_market"

# ============================================================================
# DATA PATHS
# ============================================================================

# Output directories (relative paths for Railway compatibility)
# Use /tmp on Railway (ephemeral storage) or configure persistent storage
DATA_BASE = os.getenv('PIPELINE_DATA_DIR', '/tmp/polymarket_data/')
DATA_RAW = os.path.join(DATA_BASE, "raw")
DATA_PROCESSED = os.path.join(DATA_BASE, "processed")

# ============================================================================
# FETCHING PARAMETERS
# ============================================================================

# Batch processing
BATCH_SIZE = 100          # Markets to fetch per API call
INSERT_BATCH_SIZE = 1000  # Documents to insert per database batch

# Trader tracking
MIN_TRADER_VOLUME = 1000  # $1k minimum volume for trader tracking
TOP_MARKETS_LIMIT = 200   # Only fetch prices for top N markets by volume
WHALE_THRESHOLD = 50000   # $50k+ volume = whale status

# Rate limiting
RATE_LIMIT_DELAY = 0.3    # Seconds between API calls
API_TIMEOUT = 30          # Seconds to wait for API response
MAX_RETRIES = 3           # Max retry attempts for failed API calls

# ============================================================================
# COMPANY KEYWORDS FOR EDGE MATCHING
# ============================================================================

# Comprehensive keyword mapping for direct company mentions
COMPANY_KEYWORDS = {
    # Tech Giants
    'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'ios', 'app store', 'tim cook', 'macbook', 'airpods'],
    'MSFT': ['microsoft', 'windows', 'azure', 'office 365', 'xbox', 'satya nadella', 'bing', 'copilot'],
    'GOOGL': ['google', 'alphabet', 'android', 'youtube', 'chrome', 'search engine', 'sundar pichai', 'gmail'],
    'GOOG': ['google', 'alphabet', 'android', 'youtube'],
    'AMZN': ['amazon', 'aws', 'prime', 'alexa', 'jeff bezos', 'andy jassy', 'kindle', 'whole foods'],
    'TSLA': ['tesla', 'elon musk', 'musk', 'electric vehicle', 'model s', 'model 3', 'model y', 'model x', 'cybertruck', 'ev'],
    'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'mark zuckerberg', 'zuckerberg', 'oculus', 'metaverse'],
    'NVDA': ['nvidia', 'gpu', 'graphics card', 'ai chip', 'cuda', 'jensen huang', 'geforce', 'rtx'],
    'NFLX': ['netflix', 'streaming service'],
    'AMD': ['amd', 'ryzen', 'radeon', 'epyc'],
    'INTC': ['intel', 'chip maker', 'core processor'],

    # Defense
    'LMT': ['lockheed', 'lockheed martin', 'f-35', 'f-22', 'f-16', 'missile defense', 'aegis'],
    'RTX': ['raytheon', 'raytheon technologies', 'patriot missile', 'pratt whitney', 'pratt & whitney'],
    'BA': ['boeing', '737', '787', 'dreamliner', 'aircraft manufacturer'],
    'NOC': ['northrop', 'northrop grumman', 'b-21', 'b-2', 'global hawk', 'stealth bomber'],
    'GD': ['general dynamics', 'submarine', 'tank', 'gulfstream', 'virginia class'],
    'LHX': ['l3harris', 'harris', 'defense electronics'],

    # Finance
    'JPM': ['jpmorgan', 'jp morgan', 'chase', 'jamie dimon'],
    'GS': ['goldman sachs', 'goldman'],
    'MS': ['morgan stanley'],
    'BAC': ['bank of america', 'bofa', 'merrill lynch'],
    'WFC': ['wells fargo'],
    'C': ['citigroup', 'citibank', 'citi'],
    'BLK': ['blackrock', 'larry fink'],

    # Healthcare
    'JNJ': ['johnson & johnson', 'johnson and johnson', 'j&j'],
    'PFE': ['pfizer', 'covid vaccine', 'pfizer vaccine'],
    'MRNA': ['moderna', 'mrna vaccine', 'moderna vaccine'],
    'UNH': ['unitedhealth', 'united health', 'optum'],

    # Energy
    'XOM': ['exxon', 'exxonmobil', 'exxon mobil'],
    'CVX': ['chevron'],
    'COP': ['conocophillips', 'conoco'],

    # Retail
    'WMT': ['walmart', 'wal-mart', 'sam walton'],
    'TGT': ['target', 'target stores'],
    'COST': ['costco'],
    'HD': ['home depot'],

    # Consumer Brands
    'NKE': ['nike', 'swoosh'],

    # Automotive
    'F': ['ford', 'ford motor', 'f-150'],
    'GM': ['general motors', 'gm', 'chevy', 'chevrolet'],
    'RIVN': ['rivian', 'r1t'],
    'LCID': ['lucid', 'lucid motors', 'lucid air'],

    # Media & Entertainment
    'DIS': ['disney', 'marvel', 'pixar', 'star wars', 'espn', 'disney+'],
    'CMCSA': ['comcast', 'nbc', 'universal'],
}

# ============================================================================
# SECTOR KEYWORDS FOR SECTOR-LEVEL MATCHING
# ============================================================================

SECTOR_KEYWORDS = {
    'technology': [
        'tech company', 'software', 'ai', 'artificial intelligence',
        'cloud computing', 'semiconductor', 'chip', 'saas', 'machine learning',
        'data center', 'software developer', 'tech sector'
    ],
    'defense': [
        'defense', 'military', 'weapon', 'missile', 'fighter jet',
        'aircraft carrier', 'pentagon', 'dod', 'department of defense',
        'defense contractor', 'military equipment', 'national security'
    ],
    'finance': [
        'bank', 'financial institution', 'wall street', 'trading',
        'investment bank', 'hedge fund', 'asset management', 'private equity',
        'commercial bank', 'investment management'
    ],
    'healthcare': [
        'healthcare', 'pharma', 'pharmaceutical', 'drug', 'vaccine',
        'medical device', 'hospital', 'fda approval', 'clinical trial',
        'biotech', 'health insurance'
    ],
    'energy': [
        'oil', 'gas', 'petroleum', 'energy sector', 'opec', 'crude oil',
        'natural gas', 'oil company', 'energy producer', 'fossil fuel',
        'oil exploration'
    ],
    'automotive': [
        'auto', 'car manufacturer', 'electric vehicle', 'ev maker',
        'automobile', 'car company', 'vehicle manufacturer', 'auto industry'
    ],
    'retail': [
        'retail', 'retailer', 'store chain', 'shopping', 'ecommerce',
        'e-commerce', 'online retail', 'brick and mortar', 'consumer retail'
    ],
}

# ============================================================================
# MACRO EVENT KEYWORDS
# ============================================================================

MACRO_EVENTS = {
    'fed_rate': {
        'keywords': [
            'fed', 'federal reserve', 'interest rate', 'rate cut', 'rate hike',
            'fomc', 'powell', 'jerome powell', 'monetary policy', 'fed decision',
            'rate increase', 'rate decrease'
        ],
        'sectors': ['Financials', 'Real Estate', 'Utilities', 'Consumer Discretionary']
    },
    'inflation': {
        'keywords': [
            'inflation', 'cpi', 'pce', 'price increase', 'consumer prices',
            'inflation rate', 'rising prices', 'cost of living', 'price index'
        ],
        'sectors': ['Consumer Staples', 'Consumer Discretionary', 'Materials']
    },
    'recession': {
        'keywords': [
            'recession', 'economic downturn', 'gdp decline', 'economic crisis',
            'market crash', 'bear market', 'economic contraction'
        ],
        'sectors': ['Financials', 'Consumer Discretionary', 'Industrials']
    },
    'unemployment': {
        'keywords': [
            'unemployment', 'jobs report', 'job losses', 'labor market',
            'employment rate', 'jobless claims', 'hiring freeze'
        ],
        'sectors': ['Consumer Discretionary', 'Industrials', 'Financials']
    },
}
