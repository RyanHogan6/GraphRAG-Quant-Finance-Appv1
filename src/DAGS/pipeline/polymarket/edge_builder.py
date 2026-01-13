"""
Polymarket Edge Builder Module
Creates graph edges linking prediction markets to companies
Uses keyword matching for direct mentions, sector relationships, and macro events
"""

from datetime import datetime
from typing import List, Dict

from .config import (
    COMPANY_COL,
    MARKET_COL,
    TRADER_COL,
    POSITION_COL,
    EDGE_DIRECT,
    EDGE_SECTOR,
    EDGE_MACRO,
    EDGE_TRADER_POSITION,
    EDGE_POSITION_MARKET,
    COMPANY_KEYWORDS,
    SECTOR_KEYWORDS,
    MACRO_EVENTS,
    INSERT_BATCH_SIZE
)

# ============================================================================
# DIRECT COMPANY MENTION EDGES
# ============================================================================

def build_direct_mention_edges(db, markets: List[Dict], companies: List[Dict]) -> int:
    """
    Create edges for direct company mentions in market questions/descriptions.

    Uses comprehensive keyword matching to link markets to specific companies.
    Calculates confidence score based on number of keyword matches.

    Args:
        db: ArangoDB database handle
        markets: List of market documents
        companies: List of company documents

    Returns:
        Number of edges created
    """

    print("\n[EDGE BUILDER] Creating direct company mention edges...")
    print("-" * 80)

    edges_coll = db.collection(EDGE_DIRECT)

    # Clear existing edges
    edges_coll.truncate()

    # Build company map for fast lookup
    company_map = {}
    for company in companies:
        ticker = company.get('ticker')
        if ticker:
            company_map[ticker] = company

    print(f"  Company map: {len(company_map)} companies loaded")

    edges_created = 0
    batch = []

    for idx, market in enumerate(markets):
        # Extract text to search
        question = (market.get('question') or '').lower()
        description = (market.get('description') or '').lower()
        combined_text = question + ' ' + description

        # Check each company's keywords
        for ticker, keywords in COMPANY_KEYWORDS.items():
            if ticker not in company_map:
                continue

            # Find matched keywords
            matched_keywords = [kw for kw in keywords if kw.lower() in combined_text]

            if matched_keywords:
                # Calculate confidence score (more matches = higher confidence)
                confidence = min(len(matched_keywords) * 0.25, 1.0)

                edge = {
                    '_from': market['_id'],
                    '_to': company_map[ticker]['_id'],
                    'match_type': 'keyword',
                    'matched_keywords': matched_keywords,
                    'confidence': confidence,
                    'market_volume_24h': float(market.get('volume_24h', 0)),
                    'created_at': datetime.now().isoformat(),
                    'source': 'polymarket'
                }

                batch.append(edge)
                edges_created += 1

                # Insert in batches
                if len(batch) >= INSERT_BATCH_SIZE:
                    edges_coll.insert_many(batch)
                    batch = []

        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(markets)} markets...", end='\r')

    # Insert remaining batch
    if batch:
        edges_coll.insert_many(batch)

    print(f"\n  [OK] Created {edges_created:,} direct mention edges")

    return edges_created


# ============================================================================
# SECTOR-LEVEL EDGES
# ============================================================================

def build_sector_edges(db, markets: List[Dict], companies: List[Dict]) -> int:
    """
    Create edges for sector-level market impacts.

    Links markets mentioning sector keywords to all companies in that sector.
    Examples: "tech stocks", "defense sector", "banking crisis"

    Args:
        db: ArangoDB database handle
        markets: List of market documents
        companies: List of company documents

    Returns:
        Number of edges created
    """

    print("\n[EDGE BUILDER] Creating sector-level edges...")
    print("-" * 80)

    edges_coll = db.collection(EDGE_SECTOR)

    # Clear existing edges
    edges_coll.truncate()

    # Get company sector field name (check first company)
    company_sector_field = None
    if companies:
        sample = companies[0]
        for field in ['sector', 'gics_sector', 'Sector']:
            if field in sample:
                company_sector_field = field
                break

    if not company_sector_field:
        print("  [WARN] No sector field found in companies - skipping sector edges")
        return 0

    print(f"  Using sector field: {company_sector_field}")

    edges_created = 0
    batch = []

    for market in markets:
        # Extract text to search
        combined_text = ((market.get('question') or '') + ' ' + (market.get('description') or '')).lower()

        # Check each sector's keywords
        for sector, keywords in SECTOR_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in combined_text]

            if matched:
                # Find all companies in this sector
                sector_query = f"""
                FOR c IN {COMPANY_COL}
                  FILTER CONTAINS(LOWER(c.{company_sector_field}), '{sector}')
                  LIMIT 100
                  RETURN c._id
                """

                try:
                    sector_companies = list(db.aql.execute(sector_query))

                    for company_id in sector_companies:
                        edge = {
                            '_from': market['_id'],
                            '_to': company_id,
                            'match_type': 'sector',
                            'sector': sector,
                            'matched_keywords': matched,
                            'confidence': 0.4,  # Lower confidence than direct mentions
                            'market_volume_24h': float(market.get('volume_24h', 0)),
                            'created_at': datetime.now().isoformat(),
                            'source': 'polymarket'
                        }

                        batch.append(edge)
                        edges_created += 1

                        # Insert in batches
                        if len(batch) >= INSERT_BATCH_SIZE:
                            edges_coll.insert_many(batch)
                            batch = []

                except Exception as e:
                    print(f"  [WARN] Error querying sector '{sector}': {e}")
                    continue

    # Insert remaining batch
    if batch:
        edges_coll.insert_many(batch)

    print(f"  [OK] Created {edges_created:,} sector-based edges")

    return edges_created


# ============================================================================
# MACRO EVENT EDGES
# ============================================================================

def build_macro_edges(db, markets: List[Dict], companies: List[Dict]) -> int:
    """
    Create edges for macro economic events affecting multiple companies.

    Links markets about macro events (Fed rates, inflation, recession) to
    companies in affected sectors.

    Args:
        db: ArangoDB database handle
        markets: List of market documents
        companies: List of company documents

    Returns:
        Number of edges created
    """

    print("\n[EDGE BUILDER] Creating macro event edges...")
    print("-" * 80)

    edges_coll = db.collection(EDGE_MACRO)

    # Clear existing edges
    edges_coll.truncate()

    # Get company sector field name
    company_sector_field = None
    if companies:
        sample = companies[0]
        for field in ['sector', 'gics_sector', 'Sector']:
            if field in sample:
                company_sector_field = field
                break

    if not company_sector_field:
        print("  [WARN] No sector field found - skipping macro edges")
        return 0

    print(f"  Using sector field: {company_sector_field}")

    edges_created = 0
    batch = []

    for market in markets:
        # Extract text to search
        combined_text = ((market.get('question') or '') + ' ' + (market.get('description') or '')).lower()

        # Check each macro event
        for event_name, event_config in MACRO_EVENTS.items():
            # Check if any keywords match
            if any(kw in combined_text for kw in event_config['keywords']):
                # Find companies in affected sectors
                macro_query = f"""
                FOR c IN {COMPANY_COL}
                  FILTER c.{company_sector_field} IN @sectors
                  LIMIT 200
                  RETURN c._id
                """

                try:
                    affected = list(db.aql.execute(macro_query, bind_vars={'sectors': event_config['sectors']}))

                    for company_id in affected:
                        edge = {
                            '_from': market['_id'],
                            '_to': company_id,
                            'match_type': 'macro_event',
                            'event_type': event_name,
                            'confidence': 0.3,  # Lowest confidence (broad impact)
                            'market_volume_24h': float(market.get('volume_24h', 0)),
                            'created_at': datetime.now().isoformat(),
                            'source': 'polymarket'
                        }

                        batch.append(edge)
                        edges_created += 1

                        # Insert in batches
                        if len(batch) >= INSERT_BATCH_SIZE:
                            edges_coll.insert_many(batch)
                            batch = []

                except Exception as e:
                    print(f"  [WARN] Error querying macro event '{event_name}': {e}")
                    continue

    # Insert remaining batch
    if batch:
        edges_coll.insert_many(batch)

    print(f"  [OK] Created {edges_created:,} macro event edges")

    return edges_created


# ============================================================================
# TRADER POSITION EDGES
# ============================================================================

def build_trader_position_edges(db) -> int:
    """
    Create edges linking traders to their positions.

    Edge: Trader -> Position (trader_has_position)

    Args:
        db: ArangoDB database handle

    Returns:
        Number of edges created
    """

    print("\n[EDGE BUILDER] Creating trader -> position edges...")
    print("-" * 80)

    edges_coll = db.collection(EDGE_TRADER_POSITION)

    # Clear existing edges
    edges_coll.truncate()

    # Query all positions and create edges
    query = f"""
    FOR p IN {POSITION_COL}
        LET trader_id = CONCAT('{TRADER_COL}/', p.trader_key)
        LET position_id = p._id
        RETURN {{
            _from: trader_id,
            _to: position_id,
            position_size: p.size,
            average_price: p.average_price,
            realized_profit: p.realized_profit,
            created_at: DATE_ISO8601(DATE_NOW())
        }}
    """

    try:
        edges = list(db.aql.execute(query))

        if edges:
            # Insert in batches
            batch = []
            edges_created = 0

            for edge in edges:
                batch.append(edge)
                edges_created += 1

                if len(batch) >= INSERT_BATCH_SIZE:
                    edges_coll.insert_many(batch)
                    batch = []

            # Insert remaining
            if batch:
                edges_coll.insert_many(batch)

            print(f"  [OK] Created {edges_created:,} trader->position edges")
        else:
            print(f"  [WARN] No positions found to create edges")
            edges_created = 0

    except Exception as e:
        print(f"  [ERROR] Failed to create trader->position edges: {e}")
        edges_created = 0

    return edges_created


def build_position_market_edges(db) -> int:
    """
    Create edges linking positions to their markets.

    Edge: Position -> Market (position_in_market)

    Args:
        db: ArangoDB database handle

    Returns:
        Number of edges created
    """

    print("\n[EDGE BUILDER] Creating position -> market edges...")
    print("-" * 80)

    edges_coll = db.collection(EDGE_POSITION_MARKET)

    # Clear existing edges
    edges_coll.truncate()

    # Query all positions and link to markets via market_key
    query = f"""
    FOR p IN {POSITION_COL}
        LET market_id = CONCAT('{MARKET_COL}/', p.market_key)
        LET position_id = p._id
        RETURN {{
            _from: position_id,
            _to: market_id,
            outcome_index: p.outcome_index,
            position_size: p.size,
            current_price: p.current_price,
            created_at: DATE_ISO8601(DATE_NOW())
        }}
    """

    try:
        edges = list(db.aql.execute(query))

        if edges:
            # Insert in batches
            batch = []
            edges_created = 0

            for edge in edges:
                batch.append(edge)
                edges_created += 1

                if len(batch) >= INSERT_BATCH_SIZE:
                    edges_coll.insert_many(batch)
                    batch = []

            # Insert remaining
            if batch:
                edges_coll.insert_many(batch)

            print(f"  [OK] Created {edges_created:,} position->market edges")
        else:
            print(f"  [WARN] No positions found to create edges")
            edges_created = 0

    except Exception as e:
        print(f"  [ERROR] Failed to create position->market edges: {e}")
        edges_created = 0

    return edges_created


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def build_all_edges(db) -> dict:
    """
    Convenience function to build ALL edges (market->company AND trader->position->market).

    Loads data from database and creates:
    - Market->Company edges:
      - Direct mention edges
      - Sector-level edges
      - Macro event edges
    - Trader->Position->Market edges:
      - Trader -> Position edges
      - Position -> Market edges

    Args:
        db: ArangoDB database handle

    Returns:
        Dict with edge creation statistics
    """

    print("\n" + "="*80)
    print("EDGE CREATION")
    print("="*80)

    # Load markets from database
    print("\n  Loading markets from database...")
    market_query = f"""
    FOR m IN {MARKET_COL}
    RETURN {{
      _id: m._id,
      _key: m._key,
      question: m.question,
      description: m.description,
      volume_24h: m.volume_24h
    }}
    """
    markets = list(db.aql.execute(market_query))
    print(f"  [OK] Loaded {len(markets):,} markets")

    # Load companies from database
    print("\n  Loading companies from database...")
    company_query = f"""
    FOR c IN {COMPANY_COL}
    RETURN c
    """
    companies = list(db.aql.execute(company_query))
    print(f"  [OK] Loaded {len(companies):,} companies")

    # Build market->company edges
    direct = build_direct_mention_edges(db, markets, companies)
    sector = build_sector_edges(db, markets, companies)
    macro = build_macro_edges(db, markets, companies)

    # Build trader->position->market edges
    trader_position = build_trader_position_edges(db)
    position_market = build_position_market_edges(db)

    market_company_total = direct + sector + macro
    trader_total = trader_position + position_market
    grand_total = market_company_total + trader_total

    print("\n" + "="*80)
    print("[OK] EDGE CREATION COMPLETE")
    print("="*80)
    print("\nMarket -> Company Edges:")
    print(f"  Direct mentions: {direct:,}")
    print(f"  Sector edges: {sector:,}")
    print(f"  Macro edges: {macro:,}")
    print(f"  Subtotal: {market_company_total:,}")
    print("\nTrader -> Position -> Market Edges:")
    print(f"  Trader -> Position: {trader_position:,}")
    print(f"  Position -> Market: {position_market:,}")
    print(f"  Subtotal: {trader_total:,}")
    print(f"\nGRAND TOTAL: {grand_total:,}")
    print("="*80 + "\n")

    return {
        'direct': direct,
        'sector': sector,
        'macro': macro,
        'trader_position': trader_position,
        'position_market': position_market,
        'market_company_total': market_company_total,
        'trader_total': trader_total,
        'total': grand_total
    }


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    from arango import ArangoClient
    from .config import DB_NAME, USERNAME, PASSWORD, ARANGO_HOST

    print("Testing edge builder...")

    try:
        client = ArangoClient(hosts=ARANGO_HOST)
        db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)

        stats = build_all_edges(db)

        print(f"\n[OK] Edge building test completed")
        print(f"  Total edges created: {stats['total']:,}")

    except Exception as e:
        print(f"\n[X] Edge building test failed: {e}")
