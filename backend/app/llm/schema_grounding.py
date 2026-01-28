"""
Schema Grounding Module - Industry Standard Pattern
Retrieves only relevant schema portions per query (not all 18 collections)
Based on Microsoft GraphRAG + LangChain ArangoGraphQAChain patterns
"""
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from app.database.connection import get_db


# Collection keyword mapping for intent detection
COLLECTION_KEYWORDS = {
    'options_flow': {
        'keywords': ['options', 'calls', 'puts', 'unusual', 'sweep', 'iv', 'implied volatility',
                     'put/call', 'option activity', 'option volume'],
        'priority': 1  # High priority for options-specific queries
    },
    'futures_prices': {
        'keywords': ['futures', 'commodity', 'crude oil', 'natural gas', 'gold', 'silver',
                     'copper', 'wheat', 'corn', 'commodities'],
        'priority': 1
    },
    'eia_crude_inventory': {
        'keywords': ['crude inventory', 'oil inventory', 'crude stocks', 'oil stocks',
                     'inventory levels', 'crude storage', 'inventory build', 'inventory draw',
                     'cushing', 'refinery utilization', 'crude oil inventory'],
        'priority': 1
    },
    'eia_natgas_storage': {
        'keywords': ['natural gas storage', 'gas storage', 'natgas storage', 'gas stocks',
                     'storage levels', '5-year average', 'below average', 'above average'],
        'priority': 1
    },
    'eia_natgas_production': {
        'keywords': ['natural gas production', 'gas production', 'natgas production',
                     'production trends', 'monthly production'],
        'priority': 1
    },
    'eia_lng_exports': {
        'keywords': ['lng exports', 'lng', 'liquified natural gas'],
        'priority': 1
    },
    'Award': {
        'keywords': ['government contracts', 'awards', 'defense contracts', 'contractor',
                     'usaspending', 'government spending', 'contract value'],
        'priority': 2
    },
    'commodity_positions': {
        'keywords': ['cftc', 'commitments of traders', 'speculator', 'commercial',
                     'positioning', 'trader positions', 'crude oil positions',
                     'oil positions', 'commodity positions', 'gold positions',
                     'copper positions', 'futures positions', 'long positions',
                     'short positions', 'net positions'],
        'priority': 1  # High priority - very specific queries
    },
    'Company': {
        'keywords': ['company', 'ticker', 'sector', 'industry', 'market cap', 's&p 500'],
        'priority': 3  # Low priority - often needed with other collections
    },
    'MarketData': {
        'keywords': ['stock price', 'closing price', 'volume', 'ohlc', 'technical',
                     'rsi', 'sma', 'golden cross', 'death cross'],
        'priority': 2
    },
    'sec_filings': {
        'keywords': ['sec filing', '10-k', '10-q', '8-k', 'form 4', 'form 5', '13f',
                     'sec', 'filing', 'annual report', 'quarterly report', 'insider',
                     'insider trading', 'insider buying', 'insider selling', 'sc 13d',
                     'sc 13g', 'activist', 'institutional', 'proxy', 's-1', 'ipo',
                     'material event', 'earnings release', 'sentiment', 'negative sentiment',
                     'risk factors', 'supply chain', 'regulation', 'litigation'],
        'priority': 1  # High priority - rich dataset
    },
    'sec_sentences': {
        'keywords': ['sec text', 'filing text', 'sec search', 'find in filings',
                     'risk factors', 'md&a', 'management discussion', 'supply chain',
                     'regulation', 'litigation', 'material', 'uncertainty'],
        'priority': 2
    },
    'prediction_markets_polymarket': {
        'keywords': ['polymarket', 'prediction market', 'betting market',
                     'market sentiment', 'probability'],
        'priority': 2
    },
    'prediction_markets_kalshi': {
        'keywords': ['kalshi', 'prediction market'],
        'priority': 2
    },
    'EconomicData': {
        'keywords': ['unemployment', 'inflation', 'gdp', 'fed rate', 'treasury',
                     'federal funds', 'economic', 'macro'],
        'priority': 2
    }
}


def detect_relevant_collections(question: str, max_collections: int = 3) -> List[str]:
    """
    Detect which collections are relevant for a given question.
    Returns max 3 collections to keep prompt focused.

    Industry standard: Schema grounding reduces context from 15k → 3k tokens
    """
    question_lower = question.lower()

    # Score each collection based on keyword matches
    collection_scores = {}

    for collection, metadata in COLLECTION_KEYWORDS.items():
        score = 0
        keywords = metadata['keywords']
        priority = metadata['priority']

        # Count keyword matches
        for keyword in keywords:
            if keyword in question_lower:
                # Weight by priority (priority 1 = high, 3 = low)
                score += (4 - priority)

        if score > 0:
            collection_scores[collection] = score

    # If no matches, use default collections
    if not collection_scores:
        return ['Company', 'MarketData', 'Award']

    # Sort by score and return top N
    sorted_collections = sorted(
        collection_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    relevant = [coll for coll, score in sorted_collections[:max_collections]]

    print(f"\n[SCHEMA GROUNDING] Detected relevant collections for query:")
    print(f"  Question: {question[:80]}...")
    print(f"  Relevant: {relevant}")

    return relevant


def get_collection_schema_focused(collection_name: str) -> Dict[str, Any]:
    """
    Get focused schema for a single collection with value samples.

    Industry standard: Include actual value samples (not just field types)
    From Text2Cypher research: "Schema enhanced with value samples improves value translation"
    """
    db = get_db()

    try:
        collection = db.collection(collection_name)

        # Get a sample document (ArangoDB has random() method!)
        sample_docs = list(db.aql.execute(f'''
            FOR doc IN {collection_name}
            LIMIT 1
            RETURN doc
        '''))

        if not sample_docs:
            return {
                'collection': collection_name,
                'fields': [],
                'sample': None,
                'note': 'Collection empty'
            }

        sample = sample_docs[0]

        # Get field types and sample values
        fields = {}
        for key, value in sample.items():
            if key.startswith('_'):
                continue  # Skip internal fields

            fields[key] = {
                'type': type(value).__name__,
                'sample_value': value if not isinstance(value, (list, dict)) else str(type(value).__name__)
            }

        # Get indexes (important for query optimization)
        indexes = collection.indexes()
        indexed_fields = []
        for idx in indexes:
            if idx['type'] != 'primary':
                indexed_fields.extend(idx.get('fields', []))

        # Special handling for critical collections with enum values
        enum_values = {}
        if collection_name == 'futures_prices':
            # Get unique commodity values
            commodities = list(db.aql.execute('''
                FOR doc IN futures_prices
                COLLECT commodity = doc.commodity
                RETURN commodity
            '''))
            enum_values['commodity'] = {
                'CRITICAL': 'MUST be UPPERCASE with underscores',
                'allowed_values': commodities[:20]  # Limit to avoid huge prompts
            }

        return {
            'collection': collection_name,
            'fields': fields,
            'sample': sample,
            'indexed_fields': indexed_fields,
            'enum_values': enum_values,
            'total_documents': collection.count()
        }

    except Exception as e:
        print(f"[SCHEMA GROUNDING] Error getting schema for {collection_name}: {e}")
        return {
            'collection': collection_name,
            'error': str(e)
        }


def build_focused_schema_prompt(question: str) -> str:
    """
    Build focused schema prompt with only relevant collections.

    Industry standard: 2-3 relevant collections instead of all 18
    Reduces prompt size by 75% (15k → 3k tokens)
    """
    # Detect relevant collections
    relevant_collections = detect_relevant_collections(question, max_collections=3)

    # Build focused schema description
    schema_parts = []
    schema_parts.append("## RELEVANT DATABASE SCHEMA (Focused for Your Query)\n")
    schema_parts.append(f"Query: {question}\n")
    schema_parts.append(f"Relevant Collections: {', '.join(relevant_collections)}\n\n")

    for collection_name in relevant_collections:
        schema_info = get_collection_schema_focused(collection_name)

        if 'error' in schema_info:
            continue

        schema_parts.append(f"### Collection: {collection_name}\n")
        schema_parts.append(f"Total documents: {schema_info.get('total_documents', 'unknown')}\n")

        # Show fields with types and sample values
        if schema_info.get('fields'):
            schema_parts.append("\n**Fields:**\n")
            for field_name, field_info in list(schema_info['fields'].items())[:15]:  # Limit to 15 fields
                field_type = field_info['type']
                sample = field_info.get('sample_value', '')

                # Truncate long sample values
                if isinstance(sample, str) and len(sample) > 50:
                    sample = sample[:50] + '...'

                schema_parts.append(f"  - `{field_name}` ({field_type})")
                if sample:
                    schema_parts.append(f" - Example: `{sample}`")
                schema_parts.append("\n")

        # Show indexed fields (important for performance)
        if schema_info.get('indexed_fields'):
            schema_parts.append(f"\n**Indexed fields (use these for filtering!):** {', '.join(schema_info['indexed_fields'])}\n")

        # Show enum values with critical warnings
        if schema_info.get('enum_values'):
            for field, enum_info in schema_info['enum_values'].items():
                schema_parts.append(f"\n**CRITICAL: {field} field:**\n")
                schema_parts.append(f"  {enum_info['CRITICAL']}\n")
                schema_parts.append(f"  Allowed values: {enum_info['allowed_values']}\n")

        schema_parts.append("\n---\n\n")

    focused_schema = ''.join(schema_parts)

    print(f"[SCHEMA GROUNDING] Generated focused schema: {len(focused_schema)} chars")
    print(f"[SCHEMA GROUNDING] vs Full schema would be: ~15000 chars (75% reduction)")

    return focused_schema


def get_collection_specific_rules(collections: List[str]) -> str:
    """
    Get collection-specific query rules.
    Replaces generic rules with targeted guidance.
    """
    rules = []

    if 'options_flow' in collections:
        rules.append("""
**OPTIONS FLOW COLLECTION RULES:**
- Use options_flow collection, NOT MarketData for options queries
- Field: unusual_call_activity (1 = unusual, 0 = normal)
- Field: potential_call_sweep (1 = sweep detected)
- Field: put_call_volume_ratio (numeric)
- Field: iv_avg (implied volatility average)
""")

    if 'futures_prices' in collections:
        rules.append("""
**FUTURES PRICES COLLECTION RULES:**
- CRITICAL: commodity field MUST be UPPERCASE with underscores
- Examples: 'CRUDE_OIL', 'NATURAL_GAS', 'GOLD', 'SILVER', 'COPPER'
- WRONG: 'crude oil', 'natural gas' (lowercase will return 0 results)
- Use indexed field 'commodity' + 'date' for best performance

Available technical indicators:
- Price: open, high, low, close, volume
- Moving averages: sma_5, sma_10, sma_20, sma_50, sma_200
- Distance from MAs: dist_from_sma20, dist_from_sma50
- MACD: macd, macd_signal, macd_histogram
- 52-week: high_52w, low_52w, at_52w_high, at_52w_low, pct_off_high, pct_off_low
- Volatility: daily_range_pct
- Indicators: above_sma20, above_sma50, above_sma200 (boolean 0/1)

⚠️ NOT available: rsi, rsi_14, stochastic, atr (use MACD/SMA distance instead)
""")

    if any('eia_' in c for c in collections):
        rules.append("""
**EIA COLLECTIONS RULES:**
- Use eia_crude_inventory, eia_natgas_storage, eia_natgas_production (NOT MarketData!)
- NEVER use MarketData with ticker='NATGAS' or ticker='CRUDE' (these don't exist)
- CRITICAL: Hyphenated field names MUST use backticks in AQL!
  - CORRECT: doc.`product-name`, doc.`area-name`, doc.`series-description`
  - WRONG: doc.product-name (AQL interprets as subtraction: doc.product - name)
- Field: report_date (for date filtering, no backticks needed)
- Data frequency: WEEKLY (reported Wednesdays) - expect nulls on other days

⚠️ CRITICAL: EIA has MULTIPLE records per date (different products/regions/processes)!
When joining with futures_prices, you MUST filter to specific series:

✅ CORRECT pattern for crude oil + inventory (use this!):
FOR price IN futures_prices
  FILTER price.commodity == "CRUDE_OIL"
  FILTER price.date >= DATE_SUBTRACT(DATE_NOW(), 90, "day")
  SORT price.date DESC

  LET total_stocks = FIRST(
    FOR inv IN eia_crude_inventory
      FILTER inv.report_date == price.date
        AND CONTAINS(inv.`series-description`, "U.S. Ending Stocks of Crude Oil")
        AND inv.`product-name` == "Crude Oil"
      LIMIT 1
      RETURN {
        value: inv.value,
        change: inv.change_from_previous,
        pct_change: inv.pct_change
      }
  )

  RETURN {
    date: price.date,
    crude_price: price.close,
    macd: price.macd,
    dist_from_sma20: price.dist_from_sma20,
    total_stocks: total_stocks.value,
    weekly_change: total_stocks.change,
    inventory_signal: total_stocks.change > 0 ? "Build" : "Draw"
  }

❌ WRONG pattern (creates cartesian product with 10+ rows per date):
  FOR inventory IN eia_crude_inventory
    FILTER inventory.report_date == price.date  // Matches ethanol, gasoline, regional data!

Key series filters:
- U.S. Crude Stocks: CONTAINS(`series-description`, "U.S. Ending Stocks of Crude Oil") AND `product-name` == "Crude Oil"
- Natural Gas Storage: CONTAINS(`series-description`, "Lower 48 States Natural Gas Working")

Available fields in eia_crude_inventory:
- value: Stock level (thousands of barrels)
- change_from_previous: Weekly change (positive = build, negative = draw)
- pct_change: Percent change from previous week
- report_date: Date of report

⚠️ Data NOT available: Cushing storage, refinery utilization (not in collection)
""")

    if 'Award' in collections:
        rules.append("""
**AWARD COLLECTION RULES:**
- Use CONTAINS(LOWER(doc.recipient_name), 'keyword') for company search
- NEVER use exact match == on recipient_name (names have variations)
- Use award_amount_float (NOT award_amount) for math operations
- Indexed fields: ticker, start_date, award_amount_float
""")

    if 'sec_filings' in collections:
        rules.append("""
**SEC FILINGS COLLECTION RULES:**
- 12 form types available: "10-K", "10-Q", "8-K", "4", "5", "SC 13D", "SC 13G", "13F-HR", "6-K", "S-1", "DEF 14A", "424B4"
- INSIDER TRADING SIGNALS:
  - Form 4/5 have `trades` array with buy/sell data
  - Check: doc.trades[? ANY.code == "P"] for insider BUYING (bullish)
  - Check: doc.trades[? ANY.code == "S"] for insider SELLING (bearish)
  - Filter: doc.trades[? ANY.is_informed == true] to exclude tax withholding
- SENTIMENT ANALYSIS:
  - Use avg_finbert (-1 to +1) for overall filing sentiment
  - Use avg_negative, avg_uncertainty for specific tone metrics
- CRITICAL: sec_filings has NO text content - only metadata/sentiment
  - For text search, use sec_sentences collection with CONTAINS()
- Indexed fields: ticker, type, filing_date
""")

    if 'sec_sentences' in collections:
        rules.append("""
**SEC SENTENCES COLLECTION RULES:**
- This is where actual SEC filing TEXT lives
- Use CONTAINS(LOWER(doc.text), 'keyword') for text search
- NO EMBEDDINGS: Cannot use semantic/vector search
- Use finbert_score for sentiment filtering (< -0.3 for negative)
- Available metrics: negative_per_1k, uncertainty_per_1k, litigious_per_1k
- PERFORMANCE: Text search is SLOW on millions of sentences
  - Always add ticker filter when possible (via JOIN to sec_filings)
  - Always add date range filter when possible
  - Keep results LIMIT low (10-20 max)
""")

    if 'commodity_positions' in collections:
        rules.append("""
**COMMODITY POSITIONS (CFTC) COLLECTION RULES:**
- Use for CFTC Commitments of Traders data (weekly reports)
- Field: Market_and_Exchange_Names contains commodity type
- Use CONTAINS() for commodity matching (NOT exact ==)
- Traverse from Company via HAS_COMMODITY_POSITION edge
- Shows speculator vs commercial positioning
""")

    return '\n'.join(rules)
