import streamlit as st
from arango import ArangoClient
from config import *
import streamlit as st
from arango import ArangoClient
import re
import pandas as pd
import config as cfg  

# Core Functions
@st.cache_resource
def get_arango_connection():
    """Establish cached connection to ArangoDB"""
    client = ArangoClient(hosts=cfg.ARANGO_URL)
    cert_path = st.secrets['arangodb']['creds']
    db = client.db(cfg.DB_NAME, username=cfg.USERNAME, password=cfg.PASSWORD, verify=cert_path)
    return db



# ==================== DATABASE BROWSER FUNCTIONS ====================

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
    collections = ["Company", "MarketData", "Award", "CommodityPosition", "FREDData", "sec_filings"]
    stats = []
    for col in collections:
        stats.append(get_collection_stats(db, col))
    return stats

def browse_collection(db, collection_name, limit=50, filters=None):
    """Browse documents in a collection with optional filters"""
    try:
        if filters and filters.get('field') and filters.get('value'):
            # Simple filter by field
            aql = f"""
            FOR doc IN {collection_name}
                FILTER doc.{filters['field']} == @value
                LIMIT @limit
                RETURN doc
            """
            bind_vars = {"value": filters['value'], "limit": limit}
        else:
            # No filter - return recent documents
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
    
def format_dataframe(df):
    """Format dataframe with proper number formatting and styling"""
    df_display = df.copy()
    
    # Format numeric columns
    for col in df_display.columns:
        if df_display[col].dtype in ['float64', 'float32']:
            # Check if it's a price/amount field (format with 2 decimals)
            if any(x in col.lower() for x in ['price', 'close', 'open', 'high', 'low', 'amount', 'cap']):
                df_display[col] = df_display[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            # Check if it's a rate/margin (format as percentage)
            elif any(x in col.lower() for x in ['rate', 'margin', 'ratio']):
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            # Other floats (2 decimals with commas)
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        elif df_display[col].dtype in ['int64', 'int32']:
            # Format integers with commas
            df_display[col] = df_display[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "")
    
    return df_display



def simplify_query_for_speed(question):
    """Convert complex query to fast single-collection query"""
    
    # Extract ticker if present
    ticker_match = re.search(r'\b([A-Z]{2,5})\b', question)
    ticker = ticker_match.group(1) if ticker_match else None
    
    # If asking about awards + market data, split into two queries
    if ticker and any(word in question.lower() for word in ['award', 'contract']) and any(word in question.lower() for word in ['price', 'stock', 'market']):
        return f"""
Instead of a complex join, I'll answer this in two parts:

1. Awards for {ticker}:
FOR award IN Award
  FILTER award.ticker == @ticker
  SORT award.award_amount DESC
  LIMIT 10
  RETURN award

2. Latest market data for {ticker}:
FOR market IN MarketData
  FILTER market.ticker == @ticker
  SORT market.date DESC
  LIMIT 1
  RETURN market
        """
    
    return None




def setup_performance_indexes():
    """Add indexes for fast graph traversals - run this once"""
    db = get_arango_connection()
    if not db:
        return
    
    try:
        # Index on MarketData for ticker lookups
        if db.has_collection('MarketData'):
            market_col = db.collection('MarketData')
            # Persistent index on ticker (for fast filtering)
            market_col.add_persistent_index(fields=['ticker'], unique=False)
            # Persistent index on date (for time-series queries)
            market_col.add_persistent_index(fields=['date'], unique=False)
            # Compound index for ticker + date (fastest for combined queries)
            market_col.add_persistent_index(fields=['ticker', 'date'], unique=False)
            print("✅ MarketData indexes created")
        
        # Index on Award for ticker lookups
        if db.has_collection('Award'):
            award_col = db.collection('Award')
            award_col.add_persistent_index(fields=['ticker'], unique=False)
            award_col.add_persistent_index(fields=['start_date'], unique=False)
            award_col.add_persistent_index(fields=['award_amount'], unique=False)
            print("✅ Award indexes created")
        
        # Edge collections automatically have _from and _to indexes
        # but verify they exist
        if db.has_collection('HAS_MARKETDATA'):
            edge_col = db.collection('HAS_MARKETDATA')
            print(f"✅ HAS_MARKETDATA edge indexes: {edge_col.indexes()}")
        
        if db.has_collection('HAS_AWARD'):
            edge_col = db.collection('HAS_AWARD')
            print(f"✅ HAS_AWARD edge indexes: {edge_col.indexes()}")
            
        st.success("Performance indexes configured!")
        
    except Exception as e:
        st.error(f"Index creation error: {str(e)}")

def fix_aql_query(query):
    """Fix common LLM mistakes in AQL queries"""
    
    # Check for fatal errors first
    if 'doc.content' in query or 'filing.content' in query or 'sec.content' in query:
        print("❌ ERROR: Query uses .content field which doesn't exist!")
        print("   Text is stored in sec_sentences.text, not in sec_filings")
        print("   To find Risk Factors, use: sec.section_type == 'Item 1A Risk Factors'")
        return None  # Return None to signal invalid query
    
    if 'LIKE' in query:
        print("⚠️  Warning: ArangoDB doesn't use SQL LIKE syntax")
        print("   Use CONTAINS() or regex instead")
    
    # Normal replacements
    replacements = {
        'SEC_Filings': 'sec_filings',
        'SEC_Sections': 'sec_sections',
        'SEC_Sentences': 'sec_sentences',
        '.filing_type': '.type',
    }
    
    fixed = query
    for wrong, correct in replacements.items():
        fixed = fixed.replace(wrong, correct)
    
    if fixed != query:
        print("🔧 Auto-corrected AQL query")
    
    return fixed


def execute_custom_aql(db, aql_query, bind_vars=None):
    """Execute custom AQL query with auto-correction"""
    try:
        # Auto-fix LLM mistakes
        fixed_query = fix_aql_query(aql_query)
        
        # If query is unfixable, return error
        if fixed_query is None:
            return [], "Query contains unfixable errors (see console output)"
        
        cursor = db.aql.execute(fixed_query, bind_vars=bind_vars or {})
        results = list(cursor)
        return results, None
    except Exception as e:
        print(f"❌ AQL Error: {e}")
        print(f"Query: {aql_query}")
        return [], str(e)


def get_stock_overview(db, ticker):
    """Get comprehensive overview for a stock ticker"""
    overview = {}
    
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
    
    # Get awards for this ticker
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
    
    return overview
def render_stock_overview(db, ticker):
    """Render comprehensive stock overview dashboard"""
    st.subheader(f"📈 {ticker} Stock Overview")
    
    overview = get_stock_overview(db, ticker)
    
    if not overview.get('latest'):
        st.warning(f"No market data found for {ticker}")
        return
    
    latest = overview['latest']
    history = overview['history']
    
    # Key metrics row
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
        if latest.get('marketCap'):
            market_cap_b = latest['marketCap'] / 1e9
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
    
    # Price history table and chart
    if history and len(history) > 1:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 📊 Price History (Last 90 Days)")
            
            # Create price chart data
            history_df = pd.DataFrame(history)
            history_df = history_df.sort_values('date')
            
            # Simple line chart
            chart_data = history_df[['date', 'close', 'open', 'high', 'low']].set_index('date')
            st.line_chart(chart_data[['close']], use_container_width=True)
            
            # Statistics
            st.markdown("### 📈 Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.metric("90-Day High", f"${history_df['high'].max():,.2f}")
                st.metric("90-Day Low", f"${history_df['low'].min():,.2f}")
            
            with stats_col2:
                avg_volume = history_df['volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
                avg_close = history_df['close'].mean()
                st.metric("Avg Close", f"${avg_close:,.2f}")
            
            with stats_col3:
                price_change = ((history_df.iloc[0]['close'] - history_df.iloc[-1]['close']) / history_df.iloc[-1]['close']) * 100
                st.metric("90-Day Change", f"{price_change:.2f}%")
                volatility = history_df['close'].std()
                st.metric("Volatility (σ)", f"${volatility:.2f}")
        
        with col_right:
            st.markdown("### 📋 Recent Prices")
            recent_df = history_df[['date', 'close', 'volume']].head(10).copy()
            recent_df['close'] = recent_df['close'].apply(lambda x: f"${x:,.2f}")
            recent_df['volume'] = recent_df['volume'].apply(lambda x: f"{x:,.0f}")
            recent_df.columns = ['Date', 'Close', 'Volume']
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
    
    # Awards section - FIXED
    if overview.get('awards') and len(overview.get('awards')) > 0:
        st.divider()
        st.markdown("### 🏛️ Government Awards")
        
        awards_df = pd.DataFrame(overview['awards'])
        
        # Select and rename columns
        awards_display = awards_df[['award_id', 'award_amount', 'awarding_agency', 'start_date', 'description']].copy()
        awards_display.columns = ['Award ID', 'Amount', 'Agency', 'Start Date', 'Description']
        
        # FIXED: Convert amount to numeric first, then format
        awards_display['Amount'] = pd.to_numeric(awards_display['Amount'], errors='coerce')
        awards_display['Amount'] = awards_display['Amount'].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) and x > 0 else "N/A"
        )
        
        # Truncate descriptions
        awards_display['Description'] = awards_display['Description'].apply(
            lambda x: str(x)[:100] + "..." if pd.notna(x) and len(str(x)) > 100 else str(x)
        )
        
        st.dataframe(awards_display, use_container_width=True, hide_index=True)
        
        # Award summary
        total_awards = len(overview['awards'])
        total_amount = pd.to_numeric(awards_df['award_amount'], errors='coerce').sum()
        st.info(f"📊 Total: {total_awards} awards worth ${total_amount:,.2f}")


def fix_ticker_confusion(plan, original_question):
    """Fix when LLM confuses ticker with recipient_name - with context awareness"""
    bind_vars = plan.get("bind_vars", {})
    aql = plan.get("aql_query", "")
    
    # Check if this is likely a SEMANTIC search query (don't "fix" these!)
    semantic_keywords = ['related to', 'about', 'involving', 'containing', 'with', 'regarding', 'concerning']
    is_semantic_query = any(keyword in original_question.lower() for keyword in semantic_keywords)
    
    # Special cases that are NOT tickers
    not_tickers = {
        'AI': 'artificial intelligence (use semantic search)',
        'IT': 'information technology (use semantic search)',
        'ML': 'machine learning (use semantic search)',
        'AR': 'augmented reality (use semantic search)',
        'VR': 'virtual reality (use semantic search)',
        'US': 'United States (not a ticker)',
        'UK': 'United Kingdom (not a ticker)'
    }
    
    # Case 1: recipient_name == value
    if "recipient_name" in bind_vars:
        potential_ticker = bind_vars["recipient_name"]
        
        # Check if it's a special case
        if potential_ticker in not_tickers:
            st.info(f"ℹ️ '{potential_ticker}' detected as {not_tickers[potential_ticker]}")
            return plan  # Don't fix
        
        # Check if semantic query
        if is_semantic_query:
            st.info(f"ℹ️ Semantic query detected - not treating '{potential_ticker}' as ticker")
            return plan  # Don't fix
        
        if potential_ticker and potential_ticker.isupper() and 2 <= len(potential_ticker) <= 5:
            st.warning(f"🔧 Auto-fix: '{potential_ticker}' is a ticker")
            
            fixed_aql = aql.replace("doc.recipient_name == @recipient_name", "doc.ticker == @ticker")
            fixed_aql = fixed_aql.replace("award.recipient_name == @recipient_name", "award.ticker == @ticker")
            
            plan["aql_query"] = fixed_aql
            bind_vars["ticker"] = potential_ticker
            del bind_vars["recipient_name"]
            plan["bind_vars"] = bind_vars
            plan["explanation"] = f"Corrected: Using ticker field for {potential_ticker}"
    
    # Case 2: pattern with LIKE
    if "pattern" in bind_vars:
        pattern = bind_vars["pattern"]
        core_value = pattern.strip('%')
        
        # Check special cases
        if core_value in not_tickers:
            st.info(f"ℹ️ '{core_value}' is {not_tickers[core_value]} - not fixing")
            return plan
        
        # Check semantic query
        if is_semantic_query:
            st.info(f"ℹ️ Semantic query detected - keeping LIKE pattern")
            return plan
        
        if core_value and core_value.isupper() and 2 <= len(core_value) <= 5:
            st.warning(f"🔧 Auto-fix: '{core_value}' is a ticker")
            
            fixed_aql = aql.replace("doc.recipient_name LIKE @pattern", "doc.ticker == @ticker")
            fixed_aql = fixed_aql.replace("award.recipient_name LIKE @pattern", "award.ticker == @ticker")
            
            plan["aql_query"] = fixed_aql
            bind_vars["ticker"] = core_value
            del bind_vars["pattern"]
            plan["bind_vars"] = bind_vars
    
    # Case 3: ticker in bind_vars but wrong query type (your current issue!)
    if "ticker" in bind_vars:
        ticker_value = bind_vars["ticker"]
        
        # If ticker is "AI" but query is semantic, fix the query type
        if ticker_value in not_tickers and is_semantic_query:
            st.warning(f"🔧 Converting to semantic search: '{ticker_value}' = {not_tickers[ticker_value]}")
            
            # This needs semantic search, not ticker lookup!
            plan["requires_embedding"] = True
            plan["embedding_text"] = original_question.replace("awards", "").replace("related to", "").strip()
            plan["aql_query"] = """
FOR doc IN Award
  FILTER doc.description_embedding != null
  LET similarity = COSINE_SIMILARITY(doc.description_embedding, @query_vector)
  FILTER similarity >= 0.70
  SORT similarity DESC
  LIMIT 10
  RETURN {
    award_id: doc.award_id,
    recipient_name: doc.recipient_name,
    ticker: doc.ticker,
    description: doc.description,
    award_amount: doc.award_amount,
    start_date: doc.start_date,
    similarity_score: similarity
  }
            """
            # Remove ticker, will be replaced with query_vector
            del bind_vars["ticker"]
            plan["bind_vars"] = bind_vars
            plan["intent"] = "semantic_search"
            plan["explanation"] = f"Converted to semantic search for '{ticker_value}' concepts"
    
    return plan