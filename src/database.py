"""
database.py - ArangoDB connection and query execution  
Updated: 2026-01-06 with Kalshi support
"""
import streamlit as st
from arango import ArangoClient
from config import *
import re
import pandas as pd
import config as cfg


# ==================== CORE CONNECTION ====================


@st.cache_resource
def get_arango_connection():
    """Establish cached connection to ArangoDB"""
    client = ArangoClient(hosts=cfg.ARANGO_URL)
    db = client.db(cfg.DB_NAME, username=cfg.USERNAME, password=cfg.PASSWORD)
    return db


# ==================== COLLECTION STATS ====================


def get_collection_stats(db, collection_name):
    """Get statistics for a collection"""
    try:
        collection = db.collection(collection_name)
        return {
            "count": collection.count(),
            "name": collection_name
        }
    except:
        return {"count": 0, "name": collection_name}


def get_collections_info(db):
    """Get all collections with their document counts"""
    collections = [
        "Company", 
        "MarketData", 
        "Award", 
        "EconomicData",
        "commodity_positions",
        "prediction_markets_polymarket",
        "prediction_markets_kalshi",  # KALSHI ADDED
        "sec_filings",
        "sec_sections",
        "sec_sentences"
    ]
    stats = []
    for col in collections:
        stats.append(get_collection_stats(db, col))
    return stats


# ==================== BROWSE COLLECTIONS ====================


def browse_collection(db, collection_name, limit=50, filters=None):
    """Browse documents in a collection with optional filters"""
    try:
        if filters and filters.get('field') and filters.get('value'):
            aql = f"""
            FOR doc IN {collection_name}
                FILTER doc.{filters['field']} == @value
                LIMIT @limit
                RETURN doc
            """
            bind_vars = {"value": filters['value'], "limit": limit}
        else:
            aql = f"""
            FOR doc IN {collection_name}
                LIMIT @limit
                RETURN doc
            """
            bind_vars = {"limit": limit}
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        return results
    except Exception as e:
        st.error(f"Browse error: {str(e)}")
        return []


def get_sample_document(db, collection_name):
    """Get one sample document to show schema"""
    try:
        aql = f"FOR doc IN {collection_name} LIMIT 1 RETURN doc"
        cursor = db.aql.execute(aql)
        results = list(cursor)
        return results[0] if results else None
    except:
        return None


# ==================== DATAFRAME FORMATTING ====================


def format_dataframe(df):
    """Format dataframe with proper number formatting and styling"""
    df_display = df.copy()
    
    for col in df_display.columns:
        if df_display[col].dtype in ['float64', 'float32']:
            # Price/amount fields (2 decimals)
            if any(x in col.lower() for x in ['price', 'close', 'open', 'high', 'low', 'amount', 'cap']):
                df_display[col] = df_display[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            # Rate/margin (percentage)
            elif any(x in col.lower() for x in ['rate', 'margin', 'ratio', 'prob']):
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            # Other floats
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        elif df_display[col].dtype in ['int64', 'int32']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "")
    
    return df_display


# ==================== QUERY OPTIMIZATION ====================


def simplify_query_for_speed(question):
    """Convert complex query to fast single-collection query"""
    
    ticker_match = re.search(r'\b([A-Z]{2,5})\b', question)
    ticker = ticker_match.group(1) if ticker_match else None
    
    # Split complex queries into simpler ones
    if ticker and any(word in question.lower() for word in ['award', 'contract']) and \
       any(word in question.lower() for word in ['price', 'stock', 'market']):
        return f"""
Instead of a complex join, answer in two parts:

1. Awards for {ticker}:
FOR award IN Award
  FILTER award.ticker == "{ticker}"
  SORT award.award_amount_float DESC
  LIMIT 10
  RETURN award

2. Latest market data for {ticker}:
FOR market IN MarketData
  FILTER market.ticker == "{ticker}"
  SORT market.date DESC
  LIMIT 1
  RETURN market
        """
    
    return None


# ==================== PERFORMANCE INDEXES ====================


def setup_performance_indexes():
    """Add indexes for fast queries - run once on setup"""
    db = get_arango_connection()
    if not db:
        return
    
    try:
        if db.has_collection('Company'):
            # Company ticker lookup (most common)
            db.collection('Company').add_hash_index(fields=['ticker'], unique=True)
            print("✅ Added ticker index")
        # MarketData indexes
        if db.has_collection('MarketData'):
            market_col = db.collection('MarketData')
            market_col.add_persistent_index(fields=['ticker'], unique=False)
            market_col.add_persistent_index(fields=['date'], unique=False)
            market_col.add_persistent_index(fields=['ticker', 'date'], unique=False)
            print("✅ MarketData indexes created")
        
        # Award indexes
        if db.has_collection('Award'):
            award_col = db.collection('Award')
            award_col.add_persistent_index(fields=['ticker'], unique=False)
            award_col.add_persistent_index(fields=['start_date'], unique=False)
            award_col.add_persistent_index(fields=['award_amount_float'], unique=False)
            print("✅ Award indexes created")
        
        # Commodity positions indexes
        if db.has_collection('commodity_positions'):
            comm_col = db.collection('commodity_positions')
            comm_col.add_persistent_index(fields=['Market_and_Exchange_Names'], unique=False)
            comm_col.add_persistent_index(fields=['as_of_date'], unique=False)
            print("✅ commodity_positions indexes created")
        
        # Polymarket indexes
        if db.has_collection('prediction_markets_polymarket'):
            pm_col = db.collection('prediction_markets_polymarket')
            pm_col.add_persistent_index(fields=['volume_24h'], unique=False)
            pm_col.add_persistent_index(fields=['closed'], unique=False)
            print("✅ prediction_markets_polymarket indexes created")
        
        # Kalshi indexes (ADDED)
        if db.has_collection('prediction_markets_kalshi'):
            kalshi_col = db.collection('prediction_markets_kalshi')
            kalshi_col.add_persistent_index(fields=['volume_24h'], unique=False)
            kalshi_col.add_persistent_index(fields=['status'], unique=False)
            kalshi_col.add_persistent_index(fields=['close_time'], unique=False)
            print("✅ prediction_markets_kalshi indexes created")
        
        # SEC filings indexes
        if db.has_collection('sec_filings'):
            sec_col = db.collection('sec_filings')
            sec_col.add_persistent_index(fields=['ticker'], unique=False)
            sec_col.add_persistent_index(fields=['filing_date'], unique=False)
            sec_col.add_persistent_index(fields=['type'], unique=False)
            print("✅ sec_filings indexes created")

        if db.has_collection('sec_sentences'):
            sent_col = db.collection('sec_sentences')
            
            # Sentiment score index (for FILTER finbert_score < X)
            sent_col.add_skiplist_index(
                fields=['finbert_score'],
                unique=False
            )
            
            # Fulltext index for CONTAINS() searches
            sent_col.add_fulltext_index(
                fields=['text'],
                min_length=3
            )
            
            # Section ID for traversals
            sent_col.add_persistent_index(
                fields=['section_id'],
                unique=False
            )
            
            print("✅ sec_sentences indexes created (HUGE performance boost!)")
        
        # SEC SECTIONS - For filtering by section type
        if db.has_collection('sec_sections'):
            sect_col = db.collection('sec_sections')
            
            sent_col.add_persistent_index(
                fields=['section_type'],
                unique=False
            )
            
            sent_col.add_persistent_index(
                fields=['filing_id'],
                unique=False
            )
            
            print("✅ sec_sections indexes created")
        
        # Edge collections (automatically indexed on _from and _to)
        edge_collections = [
            'HAS_MARKETDATA', 'HAS_AWARD', 'HAS_COMMODITY_POSITION',
            'market_mentions_company_polymarket',
            'market_mentions_company_kalshi',  # KALSHI ADDED
            'market_related_to_sector_kalshi',  # KALSHI ADDED
            'HAS_FILING'
        ]
        for edge_name in edge_collections:
            if db.has_collection(edge_name):
                edge_col = db.collection(edge_name)
                print(f"✅ {edge_name} edge indexes verified")
        
        st.success("Performance indexes configured!")
        
    except Exception as e:
        st.error(f"Index creation error: {str(e)}")


# ==================== AQL QUERY FIXES ====================
def fix_aql_query(query):
    """Fix common LLM mistakes in AQL queries"""
    
    # Fatal error: COSINE_SIMILARITY on SEC content
    if 'COSINE_SIMILARITY' in query and ('sec_sentences' in query or 'sec_sections' in query):
        print("❌ ERROR: SEC content has NO embeddings!")
        print("   Use CONTAINS(LOWER(doc.text), 'keyword') + finbert_score filters")
        return None
    
    # Fatal errors: .content field
    if 'doc.content' in query or 'filing.content' in query:
        print("❌ ERROR: Query uses .content field which doesn't exist!")
        print("   Text is stored in sec_sentences.text")
        return None
    
    # INTO keyword doesn't exist
    if ' INTO ' in query.upper():
        print("❌ ERROR: INTO keyword not supported in AQL")
        print("   Use: LET variable = (subquery)")
        return None
    
    # Collection name fixes ONLY (no field name changes needed!)
    replacements = {
        'SEC_Filings': 'sec_filings',
        'SEC_Sections': 'sec_sections',
        'SEC_Sentences': 'sec_sentences',
        'awards': 'Award',
        'Awards': 'Award',
        'companies': 'Company',
        'Companies': 'Company',
        'market_data': 'MarketData',
        'fred_data': 'EconomicData',
        'FREDData': 'EconomicData',
        'shares_outstanding': 'sharesOutstanding',
        'market_cap': 'marketCap',
        'employees': 'fullTimeEmployees',
        'full_time_employees': 'fullTimeEmployees',
        'sp500_member': 'sp500_member',  # This one is correct
    }
    
    fixed = query
    for wrong, correct in replacements.items():
        fixed = fixed.replace(wrong, correct)
    
    if fixed != query:
        print("🔧 Auto-corrected collection names")
    
    return fixed




# ==================== QUERY EXECUTION ====================


def execute_custom_aql(db, aql_query, bind_vars=None):
    """Execute custom AQL query with auto-correction"""
    try:
        # Auto-fix LLM mistakes
        fixed_query = fix_aql_query(aql_query)
        
        if fixed_query is None:
            return [], "Query contains unfixable errors (see console output)"
        
        cursor = db.aql.execute(fixed_query, bind_vars=bind_vars or {})
        results = list(cursor)
        return results, None
    except Exception as e:
        error_msg = str(e)
        print(f"❌ AQL Error: {error_msg}")
        print(f"Query: {aql_query}")
        return [], error_msg


# ==================== STOCK OVERVIEW ====================


def get_stock_overview(db, ticker):
    """Get comprehensive overview for a stock ticker"""
    overview = {}
    
    # Get company info
    aql_company = """
    FOR company IN Company
        FILTER company.ticker == @ticker
        LIMIT 1
        RETURN company
    """
    cursor = db.aql.execute(aql_company, bind_vars={"ticker": ticker})
    company = list(cursor)
    if company:
        overview['company'] = company[0]
    
    # Get latest market data
    aql_latest = """
    FOR doc IN MarketData
        FILTER doc.ticker == @ticker
        SORT doc.date DESC
        LIMIT 1
        RETURN doc
    """
    cursor = db.aql.execute(aql_latest, bind_vars={"ticker": ticker})
    latest = list(cursor)
    if latest:
        overview['latest'] = latest[0]
    
    # Get historical data (last 90 days)
    aql_history = """
    FOR doc IN MarketData
        FILTER doc.ticker == @ticker
        SORT doc.date DESC
        LIMIT 90
        RETURN doc
    """
    cursor = db.aql.execute(aql_history, bind_vars={"ticker": ticker})
    overview['history'] = list(cursor)
    
    # Get awards
    aql_awards = """
    FOR doc IN Award
        FILTER doc.ticker == @ticker
        SORT doc.start_date DESC
        LIMIT 10
        RETURN doc
    """
    try:
        cursor = db.aql.execute(aql_awards, bind_vars={"ticker": ticker})
        overview['awards'] = list(cursor)
    except:
        overview['awards'] = []
    
    # Get prediction markets (Polymarket + Kalshi)
    aql_polymarkets = """
    FOR edge IN market_mentions_company_polymarket
        FILTER CONTAINS(edge._to, @ticker)
        
        LET market = FIRST(
            FOR m IN prediction_markets_polymarket
                FILTER m._id == edge._from
                RETURN m
        )
        
        FILTER market != null
        FILTER market.closed == false
        SORT market.volume_24h DESC
        LIMIT 5
        
        RETURN {
            source: "Polymarket",
            question: market.question,
            yes_prob: market.yes_probability,
            volume_24h: market.volume_24h,
            confidence: edge.confidence
        }
    """
    
    aql_kalshi = """
    FOR edge IN market_mentions_company_kalshi
        FILTER CONTAINS(edge._to, @ticker)
        
        LET market = FIRST(
            FOR m IN prediction_markets_kalshi
                FILTER m._id == edge._from
                RETURN m
        )
        
        FILTER market != null
        FILTER market.status == "active"
        SORT market.volume_24h DESC
        LIMIT 5
        
        RETURN {
            source: "Kalshi",
            question: market.title,
            yes_prob: market.yes_price,
            volume_24h: market.volume_24h,
            confidence: edge.confidence
        }
    """
    
    try:
        cursor_poly = db.aql.execute(aql_polymarkets, bind_vars={"ticker": ticker})
        cursor_kalshi = db.aql.execute(aql_kalshi, bind_vars={"ticker": ticker})
        
        # Combine both sources
        all_markets = list(cursor_poly) + list(cursor_kalshi)
        all_markets.sort(key=lambda x: x['volume_24h'], reverse=True)
        overview['prediction_markets'] = all_markets[:10]
    except Exception as e:
        print(f"Prediction markets error: {e}")
        overview['prediction_markets'] = []
    
    # Get commodity positions
    aql_commodities = """
    FOR edge IN HAS_COMMODITY_POSITION
        FILTER CONTAINS(edge._from, @ticker)
        LIMIT 5
        RETURN {
            commodity: edge.commodity_name,
            date: edge.as_of_date,
            net_position: edge.net_noncommercial_position
        }
    """
    try:
        cursor = db.aql.execute(aql_commodities, bind_vars={"ticker": ticker})
        overview['commodities'] = list(cursor)
    except:
        overview['commodities'] = []
    
    return overview


def render_stock_overview(db, ticker):
    """Render comprehensive stock overview dashboard"""
    st.subheader(f"📈 {ticker} Stock Overview")
    
    overview = get_stock_overview(db, ticker)
    
    # Company header
    if overview.get('company'):
        company = overview['company']
        st.markdown(f"### {company.get('company', ticker)}")
        st.caption(f"{company.get('sector', 'N/A')} • {company.get('industry', 'N/A')}")
    
    if not overview.get('latest'):
        st.warning(f"No market data found for {ticker}")
        return
    
    latest = overview['latest']
    history = overview['history']
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Latest Close",
            f"${latest.get('close', 0):,.2f}",
            delta=f"{((latest.get('close', 0) - latest.get('open', 0)) / latest.get('open', 1) * 100):.2f}%" if latest.get('open') else None
        )
    
    with col2:
        st.metric("Volume", f"{latest.get('volume', 0):,.0f}")
    
    with col3:
        company_info = overview.get('company', {})
        if company_info.get('marketCap'):
            market_cap_b = company_info['marketCap'] / 1e9
            st.metric("Market Cap", f"${market_cap_b:.2f}B")
        else:
            st.metric("Market Cap", "N/A")
    
    with col4:
        if latest.get('grossMargins'):
            st.metric("Gross Margin", f"{latest['grossMargins']:.2%}")
        else:
            st.metric("Gross Margin", "N/A")
    
    with col5:
        st.metric("Date", latest.get('date', 'N/A'))
    
    st.divider()
    
    # Price history
    if history and len(history) > 1:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 📊 Price History (Last 90 Days)")
            history_df = pd.DataFrame(history)
            history_df = history_df.sort_values('date')
            chart_data = history_df[['date', 'close', 'open', 'high', 'low']].set_index('date')
            st.line_chart(chart_data[['close']], use_container_width=True)
        
        with col_right:
            st.markdown("### 📋 Recent Prices")
            recent_df = history_df[['date', 'close', 'volume']].head(10).copy()
            recent_df['close'] = recent_df['close'].apply(lambda x: f"${x:,.2f}")
            recent_df['volume'] = recent_df['volume'].apply(lambda x: f"{x:,.0f}")
            recent_df.columns = ['Date', 'Close', 'Volume']
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
    
    # Awards
    if overview.get('awards') and len(overview.get('awards')) > 0:
        st.divider()
        st.markdown("### 🏛️ Government Awards")
        awards_df = pd.DataFrame(overview['awards'])
        awards_display = awards_df[['recipient_name', 'award_amount_float', 'awarding_agency', 'start_date']].copy()
        awards_display.columns = ['Recipient', 'Amount', 'Agency', 'Start Date']
        awards_display['Amount'] = pd.to_numeric(awards_display['Amount'], errors='coerce')
        awards_display['Amount'] = awards_display['Amount'].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) and x > 0 else "N/A"
        )
        st.dataframe(awards_display, use_container_width=True, hide_index=True)
    
    # Prediction markets (WITH SOURCE LABELS)
    if overview.get('prediction_markets') and len(overview['prediction_markets']) > 0:
        st.divider()
        st.markdown("### 🔮 Prediction Markets")
        for market in overview['prediction_markets'][:5]:
            source_emoji = "🟣" if market.get('source') == "Polymarket" else "🟢"
            st.markdown(f"{source_emoji} **{market['question']}** ({market.get('source', 'Unknown')})")
            st.caption(f"Yes: {market['yes_prob']:.1%} • Volume: ${market['volume_24h']:,.0f}")
    
    # Commodities
    if overview.get('commodities') and len(overview['commodities']) > 0:
        st.divider()
        st.markdown("### 📦 Commodity Positions")
        comm_df = pd.DataFrame(overview['commodities'])
        st.dataframe(comm_df, use_container_width=True, hide_index=True)


# ==================== TICKER CONFUSION FIX ====================


def fix_ticker_confusion(plan, original_question):
    """Fix when LLM confuses ticker with recipient_name - with context awareness"""
    bind_vars = plan.get("bind_vars", {})
    aql = plan.get("aql_query", "")
    
    # Semantic query detection
    semantic_keywords = ['related to', 'about', 'involving', 'containing', 'with', 'regarding', 'similar to']
    is_semantic_query = any(keyword in original_question.lower() for keyword in semantic_keywords)
    
    # Special cases that are NOT tickers
    not_tickers = {
        'AI': 'artificial intelligence',
        'IT': 'information technology',
        'ML': 'machine learning',
        'AR': 'augmented reality',
        'VR': 'virtual reality',
        'US': 'United States',
        'UK': 'United Kingdom',
        'EU': 'European Union'
    }
    
    # Case 1: recipient_name used instead of ticker
    if "recipient_name" in bind_vars:
        potential_ticker = bind_vars["recipient_name"]
        
        if potential_ticker in not_tickers:
            st.info(f"ℹ️ '{potential_ticker}' = {not_tickers[potential_ticker]} (not a ticker)")
            return plan
        
        if is_semantic_query:
            st.info(f"ℹ️ Semantic query detected - not treating '{potential_ticker}' as ticker")
            return plan
        
        if potential_ticker and potential_ticker.isupper() and 2 <= len(potential_ticker) <= 5:
            st.warning(f"🔧 Auto-fix: '{potential_ticker}' is a ticker, not recipient_name")
            fixed_aql = aql.replace("doc.recipient_name == @recipient_name", "doc.ticker == @ticker")
            fixed_aql = fixed_aql.replace("award.recipient_name == @recipient_name", "award.ticker == @ticker")
            plan["aql_query"] = fixed_aql
            bind_vars["ticker"] = potential_ticker
            del bind_vars["recipient_name"]
            plan["bind_vars"] = bind_vars
            plan["explanation"] = f"Corrected: Using ticker field for {potential_ticker}"
    
    # Case 2: LIKE pattern detection
    if "pattern" in bind_vars:
        pattern = bind_vars["pattern"]
        core_value = pattern.strip('%')
        
        if core_value in not_tickers and is_semantic_query:
            plan["requires_embedding"] = True
            plan["embedding_text"] = core_value
            st.info(f"🔧 Converting '{core_value}' to semantic search")
    
    return plan
