import streamlit as st
from arango import ArangoClient
import torch
import re
import openai
from datetime import datetime
from dotenv import load_dotenv
import os 
import pandas as pd 
import json

# Load environment variables
load_dotenv() 

# Configuration
ARANGO_URL = "http://localhost:8529"
GRAPH_NAME = "QUANT_v1_FinanceGraph"
DB_NAME = "QUANT_v1"
USERNAME = "root"
PASSWORD = os.getenv('PASSWORD')
COMPANY_COL = "Company"
MARKETDATA_COL = "MarketData"
EDGE_MARKETDATA_COL = "HAS_MARKETDATA"
AWARD_COL = "Award"
EDGE_AWARD_COL = "HAS_AWARD"
FRED_COL = "FREDData"
COMMODITY_COL = "CommodityPosition"
SEC_COL = "SEC_Filings"

openai.api_key = os.getenv('OPENAI_API_KEY')

# Schema description for LLM
SCHEMA_DESCRIPTION = """
Database: QUANT_v1 (ArangoDB Multi-Model Graph)

COLLECTIONS:
1. Company
   - ticker (string): Stock ticker symbol
   - name (string): Company name
   - sector (string): Business sector

2. MarketData
   - ticker (string): Stock ticker
   - date (string): Format YYYY-MM-DD
   - close, open, high, low (float): Price data
   - volume (int): Trading volume
   - marketCap (float): Market capitalization
   - grossMargins, ebitdaMargins (float): Financial ratios
   
3. Award (government contracts)
   - ticker (string): Recipient company ticker
   - award_id (string): Unique award identifier
   - award_amount (float): Contract value in USD
   - recipient_name (string): Company receiving award
   - awarding_agency (string): Government agency
   - start_date, end_date (string): Contract period (YYYY-MM-DD)
   - description (string): Award description
   - description_embedding (array): Semantic vector (1536 dimensions)

4. CommodityPosition (CFTC data)
   - As_of_Date_in_Form_YYYY-MM-DD (string): Report date ⚠️ USE BACKTICKS: doc.`As_of_Date_in_Form_YYYY-MM-DD`
   - Market_and_Exchange_Names (string): Commodity name
   - Open_Interest_All (int): Total open interest

    IMPORTANT: Fields with hyphens MUST be wrapped in backticks!
    Wrong: doc.As_of_Date_in_Form_YYYY-MM-DD
    Correct: doc.`As_of_Date_in_Form_YYYY-MM-DD`
   
5. FREDData (macroeconomic indicators)
   - Unnamed_0 (string): Date in YYYY-MM-DD format
   - S&P_500_Index (float): S&P 500 value
   - Civilian_Unemployment_Rate (float): Unemployment %
   - Federal_Funds_Rate (float): Fed funds rate %

6. SEC_Filings
   - ticker (string): Company ticker
   - filing_type (string): 10-K, 10-Q, etc.
   - filing_date (string): Date filed
   - content (string): Filing text

EDGES (Relationships):
- HAS_MARKETDATA: Company -> MarketData (links company to price history)
- HAS_AWARD: Company -> Award (links company to government contracts)

QUERY CAPABILITIES:
- Exact match: Filter by ticker, date, specific values
- Semantic search: Available for Award.description_embedding using COSINE_SIMILARITY
- Graph traversal: Use OUTBOUND/INBOUND with edge collections
- Time-series: Filter by date ranges
- Aggregations: SUM, AVG, COUNT, MIN, MAX
"""


# Add this to FEW_SHOT_EXAMPLES
OPTIMIZED_TRAVERSAL_EXAMPLES = """
EXAMPLE: Cross-Collection with Traversal (OPTIMIZED)
Question: "Which companies received awards over $5M and what were their stock prices?"
Strategy: Start with Award (smaller dataset), filter early, limit results, then join to MarketData

BAD APPROACH (Will timeout):
FOR company IN Company
  FOR award IN OUTBOUND company HAS_AWARD
    FILTER award.award_amount > 5000000
    FOR market IN OUTBOUND company HAS_MARKETDATA
      RETURN {company, award, market}

GOOD APPROACH (Fast):
FOR award IN Award
  FILTER award.award_amount > 5000000
  FILTER award.ticker != null
  LIMIT 10
  FOR market IN MarketData
    FILTER market.ticker == award.ticker
    SORT market.date DESC
    LIMIT 1
    RETURN {
      ticker: award.ticker,
      recipient: award.recipient_name,
      award_amount: award.award_amount,
      latest_close: market.close,
      market_date: market.date
    }

Key optimizations:
1. Start with Award (pre-filtered dataset)
2. Filter award_amount BEFORE traversal
3. LIMIT to 10 awards early
4. Use direct ticker match instead of graph traversal (faster)
5. LIMIT market data to latest record only

EXAMPLE: Time-Window Traversal (OPTIMIZED)
Question: "Show companies that got awards in 2024 and their stock performance after"
Strategy: Filter by date range first, use indexed fields

OPTIMIZED QUERY:
FOR award IN Award
  FILTER award.start_date >= "2024-01-01" AND award.start_date <= "2024-12-31"
  FILTER award.award_amount > 1000000
  FILTER award.ticker != null
  SORT award.award_amount DESC
  LIMIT 5
  LET market_data = (
    FOR market IN MarketData
      FILTER market.ticker == award.ticker
      FILTER market.date >= award.start_date
      FILTER market.date <= DATE_ADD(award.start_date, 30, 'day')
      SORT market.date ASC
      RETURN {date: market.date, close: market.close}
  )
  RETURN {
    ticker: award.ticker,
    award_date: award.start_date,
    award_amount: award.award_amount,
    price_movement: market_data
  }

Bind Variables: {}
"""

SEMANTIC_EXAMPLES = """
EXAMPLE: Semantic Search (NOT a ticker lookup!)
Question: "Show me awards related to AI" or "Find AI contracts"
Intent: semantic_search (NOT ticker lookup for "AI" symbol!)
Strategy: Generate embedding for "artificial intelligence machine learning", search description_embedding

AQL:
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

Bind Variables: {"query_vector": [embedding from "artificial intelligence machine learning"]}
Requires Embedding: true
Embedding Text: "artificial intelligence machine learning AI deep learning neural networks"

🚨 DO NOT confuse "AI" with ticker symbol! "Awards related to AI" = semantic search, NOT ticker lookup!

TRIGGER WORDS for semantic search (use embeddings, NOT ticker):
- "related to", "about", "involving", "containing"
- "cybersecurity", "AI", "machine learning", "renewable energy", "quantum computing"
"""

# Add to your FEW_SHOT_EXAMPLES


# Update your planning prompt to include these patterns

# Few-shot examples for the LLM
FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Question: "What was Apple's closing price on 2024-01-15?"
Intent: Single value lookup
Query Strategy: Exact match on MarketData
AQL:
FOR doc IN MarketData
  FILTER doc.ticker == @ticker AND doc.date == @date
  RETURN {date: doc.date, close: doc.close, ticker: doc.ticker}
Bind Variables: {"ticker": "AAPL", "date": "2024-01-15"}

EXAMPLE 2:
Question: "Show me government awards for defense contractors in 2024"
Intent: Multi-result retrieval with filtering
Query Strategy: Filter Award collection by agency and date
AQL:
FOR doc IN Award
  FILTER (doc.awarding_agency LIKE "%Defense%" OR doc.awarding_agency LIKE "%DoD%")
  FILTER doc.start_date >= @start_date AND doc.start_date <= @end_date
  SORT doc.award_amount DESC
  LIMIT 20
  RETURN {
    award_id: doc.award_id,
    recipient: doc.recipient_name,
    ticker: doc.ticker,
    amount: doc.award_amount,
    agency: doc.awarding_agency,
    start_date: doc.start_date,
    description: doc.description
  }
Bind Variables: {"start_date": "2024-01-01", "end_date": "2024-12-31"}

EXAMPLE 3:
Question: "What was the unemployment rate on 2024-03-01?"
Intent: Single value lookup from macroeconomic data
Query Strategy: Exact match on FREDData
AQL:
FOR doc IN FREDData
  FILTER doc.Unnamed_0 == @date
  RETURN {
    date: doc.Unnamed_0,
    unemployment_rate: doc.Civilian_Unemployment_Rate,
    fed_funds_rate: doc.Federal_Funds_Rate,
    sp500: doc["S&P_500_Index"]
  }
Bind Variables: {"date": "2024-03-01"}

EXAMPLE 4:
Question: "Find awards with descriptions similar to 'cybersecurity defense systems'"
Intent: Semantic similarity search
Query Strategy: Vector search using embeddings
AQL:
FOR doc IN Award
  FILTER doc.description_embedding != null
  SORT COSINE_SIMILARITY(doc.description_embedding, @query_vector) DESC
  LIMIT 10
  RETURN {
    award_id: doc.award_id,
    recipient: doc.recipient_name,
    ticker: doc.ticker,
    description: doc.description,
    amount: doc.award_amount,
    start_date: doc.start_date
  }
Bind Variables: {"query_vector": [0.123, 0.456, ...]} (embedding generated from query text)

EXAMPLE 5:
Question: "Show me Tesla's market data for the last week of January 2024"
Intent: Time-series data retrieval
Query Strategy: Filter by ticker and date range
AQL:
FOR doc IN MarketData
  FILTER doc.ticker == @ticker
  FILTER doc.date >= @start_date AND doc.date <= @end_date
  SORT doc.date ASC
  RETURN {
    date: doc.date,
    ticker: doc.ticker,
    open: doc.open,
    close: doc.close,
    high: doc.high,
    low: doc.low,
    volume: doc.volume
  }
Bind Variables: {"ticker": "TSLA", "start_date": "2024-01-24", "end_date": "2024-01-31"}

EXAMPLE 6:
Question: "What are the top 5 largest government awards?"
Intent: Aggregation and ranking
Query Strategy: Sort by amount and limit results
AQL:
FOR doc IN Award
  FILTER doc.award_amount != null
  SORT doc.award_amount DESC
  LIMIT 5
  RETURN {
    award_id: doc.award_id,
    recipient: doc.recipient_name,
    ticker: doc.ticker,
    amount: doc.award_amount,
    agency: doc.awarding_agency,
    start_date: doc.start_date
  }
Bind Variables: {}

EXAMPLE 7:
Question: "Show commodity positions for wheat on 2024-06-15"
Intent: Single commodity lookup
Query Strategy: Filter CommodityPosition by name and date
AQL:
FOR doc IN CommodityPosition
  FILTER doc["As_of_Date_in_Form_YYYY-MM-DD"] == @date
  FILTER doc.Market_and_Exchange_Names LIKE @commodity
  RETURN doc
Bind Variables: {"date": "2024-06-15", "commodity": "%WHEAT%"}

# Add this to FEW_SHOT_EXAMPLES:

EXAMPLE *: Commodity Position Query (Special Field Names)
Question: "Show me commodity open interest for corn in June 2024"
Intent: lookup
Collections: ["CommodityPosition"]
AQL:
FOR doc IN CommodityPosition
  FILTER doc.Market_and_Exchange_Names LIKE @commodity
  FILTER doc.`As_of_Date_in_Form_YYYY-MM-DD` >= @start_date 
  FILTER doc.`As_of_Date_in_Form_YYYY-MM-DD` <= @end_date
  SORT doc.`As_of_Date_in_Form_YYYY-MM-DD` ASC
  LIMIT 20
  RETURN {
    date: doc.`As_of_Date_in_Form_YYYY-MM-DD`,
    commodity: doc.Market_and_Exchange_Names,
    open_interest: doc.Open_Interest_All
  }
Bind Variables: {
  "commodity": "%CORN%",
  "start_date": "2024-06-01",
  "end_date": "2024-06-30"
}

"""

# Session state initialization
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Core Functions
@st.cache_resource
def get_arango_connection():
    """Establish cached connection to ArangoDB"""
    client = ArangoClient(hosts=ARANGO_URL)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    return db


def get_query_embedding(text):
    """Generate embedding vector for semantic search"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding generation error: {str(e)}")
        return None

def plan_query_with_llm(question, intent_hint=None):
    """Generate AQL query plan from natural language"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Add intent hint to prompt
    hint_text = ""
    if intent_hint:
        if intent_hint.get("type") == "ticker":
            hint_text = f"\n\n🎯 CONFIRMED: This is a TICKER query for '{intent_hint.get('value')}'. Use doc.ticker == @ticker"
        elif intent_hint.get("type") == "concept":
            hint_text = f"\n\n🎯 CONFIRMED: This is a CONCEPT/SEMANTIC query about '{intent_hint.get('value')}'. Use semantic search with embeddings."
    
    planning_prompt = f"""You are a database query planner for ArangoDB.

{SCHEMA_DESCRIPTION}

{FEW_SHOT_EXAMPLES}

USER QUESTION: "{question}"{hint_text}
Current Date: {current_date}

Generate a JSON response with:
- "intent": classification
- "collections": array
- "requires_embedding": boolean (true for concept queries)
- "embedding_text": text (if semantic)
- "aql_query": AQL query
- "bind_vars": object
- "explanation": strategy explanation

Return ONLY valid JSON.

Response:"""

    # Rest of your existing function...
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=1200,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Query planning error: {str(e)}")
        return None


def quick_intent_check(question):
    """Quick LLM call to determine if ticker or semantic query"""
    
    check_prompt = f"""Question: "{question}"

Is this asking about a TICKER SYMBOL or a CONCEPT?

TICKER: Question mentions a specific stock ticker (2-5 uppercase letters like AAPL, CMI, TSLA)
CONCEPT: Question asks about a topic/theme (AI, cybersecurity, renewable energy, etc.)

Examples:
- "CMI awards" → TICKER (CMI is Cummins stock ticker)
- "awards related to AI" → CONCEPT (AI = artificial intelligence topic)
- "TSLA in 2024" → TICKER
- "renewable energy contracts" → CONCEPT

Return JSON: {{"type": "ticker", "value": "CMI"}} or {{"type": "concept", "value": "artificial intelligence"}}
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": check_prompt}],
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"type": "unknown"}


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

# Add this button to your sidebar or database browser
if st.sidebar.button("⚡ Setup Performance Indexes"):
    setup_performance_indexes()



def validate_aql_syntax(aql_query):
    """Basic syntax validation before execution"""
    errors = []
    
    # Check for common typos
    if "compan." in aql_query and "company" in aql_query:
        errors.append("Typo detected: 'compan.' should be 'company.'")
        aql_query = aql_query.replace("compan.", "company.")
    
    # Check for undeclared variables
    # Find all variable declarations (FOR var IN ...)
    declared_vars = set(re.findall(r'FOR\s+(\w+)\s+IN', aql_query, re.IGNORECASE))
    
    # Find all variable uses (var.field)
    used_vars = set(re.findall(r'(\w+)\.', aql_query))
    
    # Check if any used vars aren't declared
    undeclared = used_vars - declared_vars - {'doc', 'data'}  # doc and data are common
    if undeclared:
        errors.append(f"Undeclared variables: {undeclared}")
    
    # Check bind variables
    bind_params = set(re.findall(r'@(\w+)', aql_query))
    
    return aql_query, errors, bind_params

# Use in execute_planned_query:
def execute_planned_query(plan):
    if not plan or 'aql_query' not in plan:
        return []
    
    db = get_arango_connection()
    if not db:
        return []
    
    try:
        aql_query = plan.get("aql_query", "")
        bind_vars = plan.get("bind_vars", {})
        
        # Validate and fix syntax
        aql_query, syntax_errors, required_bind_vars = validate_aql_syntax(aql_query)
        
        if syntax_errors:
            st.warning(f"⚠️ Query issues detected and auto-fixed:")
            for error in syntax_errors:
                st.caption(f"  - {error}")
        
        # Update the plan with fixed query
        plan["aql_query"] = aql_query
        
        # Check if all required bind variables are provided
        missing_vars = required_bind_vars - set(bind_vars.keys())
        
        # Handle embeddings
        if "@query_vector" in required_bind_vars:
            if plan.get("requires_embedding") and plan.get("embedding_text"):
                embedding = get_query_embedding(plan["embedding_text"])
                if embedding:
                    bind_vars["query_vector"] = embedding
                    missing_vars.discard("query_vector")
                else:
                    st.error("Failed to generate embedding for semantic search")
                    return []
            else:
                st.error("Query requires @query_vector but no embedding_text provided")
                return []
        
        # Remove requires_embedding from bind_vars if it was added (shouldn't happen)
        if "requires_embedding" in bind_vars:
            del bind_vars["requires_embedding"]
        
        if missing_vars:
            st.error(f"❌ Missing bind variables: {missing_vars}")
            with st.expander("🐛 Debug"):
                st.write(f"Required: {required_bind_vars}")
                st.write(f"Provided: {set(bind_vars.keys())}")
            return []
        
        # Execute
        cursor = db.aql.execute(
            aql_query, 
            bind_vars=bind_vars,
            ttl=30,
            batch_size=1000,
            optimizer_rules=["+all"]
        )
        
        results = list(cursor)
        return results
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"Query execution error: {error_msg}")
        
        with st.expander("🐛 Debug Query"):
            st.code(plan.get("aql_query", ""), language="sql")
            st.json(plan.get("bind_vars", {}))
        
        return []





def format_results_for_llm(results, plan):
    """
    Convert query results into structured context for LLM
    Include metadata about query strategy and data sources
    """
    if not results:
        return "No results found. The database contains no matching records for this query."
    
    context_parts = []
    context_parts.append(f"QUERY INTENT: {plan.get('intent', 'Unknown')}")
    context_parts.append(f"DATA SOURCES: {', '.join(plan.get('collections', []))}")
    context_parts.append(f"RESULT COUNT: {len(results)}")
    context_parts.append(f"QUERY STRATEGY: {plan.get('explanation', 'N/A')}")
    context_parts.append("\nRETRIEVED DATA:\n")
    
    for i, doc in enumerate(results[:30]):  # Limit to prevent token overflow
        # Clean up internal fields and embeddings
        clean_doc = {k: v for k, v in doc.items() 
                     if not k.startswith('_') and k != 'description_embedding'}
        
        # Format as readable key-value pairs
        doc_str = " | ".join([f"{k}: {v}" for k, v in clean_doc.items()])
        context_parts.append(f"[{i+1}] {doc_str}")
    
    if len(results) > 30:
        context_parts.append(f"\n... and {len(results) - 30} more results (truncated to save tokens)")
    
    return "\n".join(context_parts)


def create_analysis_prompt(question, formatted_context, plan):
    """
    Create prompt for final LLM analysis with domain expertise
    """
    prompt = f"""You are a quantitative financial analyst providing insights from a multi-source graph database containing market data, government contracts, macroeconomic indicators, and commodity positions.

DATABASE QUERY EXECUTED:
Intent: {plan.get('intent', 'Unknown')}
Strategy: {plan.get('explanation', 'Data retrieved from graph database')}

RETRIEVED DATA:
{formatted_context}

USER QUESTION: {question}

ANALYSIS INSTRUCTIONS:
1. Answer the question directly and concisely using ONLY the provided data
2. For financial data: highlight trends, anomalies, patterns, or notable values
3. For government awards: focus on amounts, agencies, recipients, timing
4. For macroeconomic data: provide context and interpretation
5. For commodity data: explain positions and market indicators
6. If multiple results exist, format as a Markdown table with relevant columns
7. Cite specific data points using [1], [2], etc. corresponding to result numbers
8. Provide quantitative summary when applicable (totals, averages, ranges, changes)
9. If data is incomplete or missing, explicitly state what's absent
10. For semantic searches: explain why results are relevant
11. Keep response concise and data-focused (avoid unnecessary preamble)

FORMAT GUIDELINES:
- Single values: Direct answer with citation
- Multiple items: Markdown table with key columns
- Time-series: Show trends and notable changes
- Comparisons: Highlight differences and similarities

ANSWER:"""
    
    return prompt


def get_llm_analysis(prompt):
    """Get analysis from OpenAI"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {str(e)}"


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
    collections = ["Company", "MarketData", "Award", "CommodityPosition", "FREDData", "SEC_Filings"]
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


def execute_custom_aql(db, aql_query, bind_vars=None):
    """Execute custom AQL query"""
    try:
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars or {})
        results = list(cursor)
        return results, None
    except Exception as e:
        return [], str(e)
# ==================== ENHANCED DATABASE BROWSER FUNCTIONS ====================
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





def render_database_browser_tab():
    """Render the database browser interface"""
    st.header("🗄️ Database Browser")
    
    db = get_arango_connection()
    if not db:
        st.error("Cannot connect to database")
        return
    
    # Layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Collections")
        
        collections_info = get_collections_info(db)
        
        for col_info in collections_info:
            if st.button(
                f"📁 {col_info['name']}\n({col_info['count']:,} docs)",
                key=f"browse_btn_{col_info['name']}",
                use_container_width=True
            ):
                st.session_state.selected_collection = col_info['name']
                st.session_state.show_custom_aql = False
                st.session_state.show_stock_overview = False
                st.rerun()
        
        st.divider()
        
        # Stock overview button
        if st.button("📈 Stock Overview", key="btn_stock_overview", use_container_width=True):
            st.session_state.show_stock_overview = True
            st.session_state.show_custom_aql = False
            st.session_state.selected_collection = None
            st.rerun()
        
        # Custom AQL button
        if st.button("⚡ Custom AQL", key="btn_custom_aql", use_container_width=True):
            st.session_state.show_custom_aql = True
            st.session_state.show_stock_overview = False
            st.session_state.selected_collection = None
            st.rerun()
    
    with col2:
        # Stock Overview Mode
        if st.session_state.get('show_stock_overview', False):
            st.subheader("📈 Stock Overview")
            
            ticker_input = st.text_input(
                "Enter Ticker Symbol:",
                value="AAPL",
                key="stock_ticker_input"
            ).upper()
            
            if st.button("Load Stock Data", type="primary", key="btn_load_stock"):
                render_stock_overview(db, ticker_input)
        
        # Custom AQL Query Interface
        elif st.session_state.get('show_custom_aql', False):
            st.subheader("⚡ Custom AQL Query")
            
            aql_query = st.text_area(
                "Enter AQL Query:",
                value="FOR doc IN MarketData\n  FILTER doc.ticker == 'AAPL'\n  LIMIT 10\n  RETURN doc",
                height=150,
                key="custom_aql_input"
            )
            
            col_a, col_b = st.columns([1, 4])
            with col_a:
                if st.button("Execute", type="primary", key="btn_execute_aql"):
                    results, error = execute_custom_aql(db, aql_query)
                    if error:
                        st.error(f"Query error: {error}")
                    else:
                        st.success(f"✅ Retrieved {len(results)} documents")
                        if results:
                            df = pd.DataFrame(results)
                            cols = [c for c in df.columns if not c.startswith('_') and c != 'description_embedding']
                            if cols:
                                df_display = format_dataframe(df[cols])
                                # FIXED: Only show actual rows
                                st.dataframe(
                                    df_display, 
                                    use_container_width=True, 
                                    hide_index=True,
                                    height=min(len(df_display) * 35 + 38, 600)  # Dynamic height
                                )
            
            with col_b:
                if st.button("Back to Browser", key="btn_back_to_browser"):
                    st.session_state.show_custom_aql = False
                    st.rerun()
        
        # Collection Browser
        elif st.session_state.get('selected_collection'):
            collection_name = st.session_state.selected_collection
            
            st.subheader(f"📊 {collection_name} Collection")
            
            sample_doc = get_sample_document(db, collection_name)
            
            if sample_doc:
                with st.expander("📋 Schema (Sample Document Fields)", expanded=False):
                    fields = [k for k in sample_doc.keys() if not k.startswith('_')]
                    st.code(", ".join(fields))
            
            # Filters
            st.markdown("### Filters")
            col_a, col_b, col_c = st.columns([2, 2, 1])
            
            with col_a:
                filter_field = st.selectbox(
                    "Filter by field:",
                    ["None"] + ([k for k in sample_doc.keys() if not k.startswith('_')] if sample_doc else []),
                    key=f"filter_field_{collection_name}"
                )
            
            with col_b:
                filter_value = st.text_input(
                    "Value:", 
                    "", 
                    key=f"filter_value_{collection_name}"
                ) if filter_field != "None" else None
            
            with col_c:
                limit = st.number_input(
                    "Limit:", 
                    min_value=10, 
                    max_value=500, 
                    value=50, 
                    step=10,
                    key=f"limit_{collection_name}"
                )
            
            if st.button("🔍 Load Data", type="primary", key=f"btn_load_{collection_name}"):
                filters = {"field": filter_field, "value": filter_value} if filter_field != "None" and filter_value else None
                
                with st.spinner("Loading data..."):
                    results = browse_collection(db, collection_name, limit=limit, filters=filters)
                
                if results:
                    st.success(f"✅ Loaded {len(results)} documents")
                    
                    df = pd.DataFrame(results)
                    cols_to_show = [c for c in df.columns if not c.startswith('_') and c != 'description_embedding']
                    
                    if cols_to_show:
                        # Format and display
                        df_display = format_dataframe(df[cols_to_show])
                        
                        # FIXED: Dynamic height based on actual rows
                        row_height = 35  # pixels per row
                        header_height = 38  # header height
                        max_height = 600  # maximum table height
                        calculated_height = min(len(df_display) * row_height + header_height, max_height)
                        
                        st.dataframe(
                            df_display, 
                            use_container_width=True,
                            hide_index=True,
                            height=calculated_height
                        )
                        
                        # Download button
                        csv = df[cols_to_show].to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"{collection_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key=f"download_{collection_name}"
                        )
                        
                        # Summary stats for numeric columns
                        numeric_cols = df[cols_to_show].select_dtypes(include=['float64', 'int64']).columns
                        if len(numeric_cols) > 0:
                            with st.expander("📊 Summary Statistics", expanded=False):
                                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    
                    # Sample JSON
                    with st.expander("🔍 View Sample Document (JSON)", expanded=False):
                        st.json(results[0])
                else:
                    st.warning("No documents found")
        
        else:
            # Welcome screen
            st.info("👈 Select a collection or feature from the sidebar")
            
            st.markdown("### 📊 Database Overview")
            collections_info = get_collections_info(db)
            
            overview_data = {
                "Collection": [c['name'] for c in collections_info],
                "Documents": [f"{c['count']:,}" for c in collections_info]
            }
            st.table(pd.DataFrame(overview_data))

# ==================== STREAMLIT APP ====================

# ==================== STREAMLIT APP (PLACE AT END OF FILE) ====================

st.set_page_config(page_title="GraphRAG Finance Platform", page_icon="📊", layout="wide")

# Initialize ALL session state variables
if 'selected_collection' not in st.session_state:
    st.session_state.selected_collection = None
if 'show_custom_aql' not in st.session_state:
    st.session_state.show_custom_aql = False
if 'show_stock_overview' not in st.session_state:
    st.session_state.show_stock_overview = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

st.title("🔍 GraphRAG Quantitative Finance Platform")
st.markdown("*Multi-source financial intelligence with LLM-powered query generation*")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This platform queries:
    - 📈 Market data
    - 🏛️ Government contracts
    - 📊 Macro indicators
    - 🌾 Commodity positions
    - 📄 SEC filings
    """)
    
    st.divider()
    
    st.header("💡 Examples")
    st.markdown("""
    - "What was Tesla's closing price on 2024-06-15?"
    - "Show me the largest defense contracts"
    - "Find awards related to AI"
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_conv"):
        st.session_state.conversation_history = []
        st.rerun()
    
    st.divider()
    st.caption(f"📍 {DB_NAME}")
    st.caption(f"🔌 {ARANGO_URL}")

# Create tabs
tab1, tab2 = st.tabs(["🤖 AI Query Interface", "🗄️ Database Browser"])

# ==================== TAB 1: AI QUERY ====================
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        user_question = st.text_input(
            "Ask a question about financial data:",
            placeholder="e.g., What was Microsoft's closing price on 2024-05-15?",
            key="question_input"
        )

    with col2:
        search_button = st.button("🔎 Search", type="primary", use_container_width=True, key="search_btn")

    # Conversation history
    if st.session_state.conversation_history:
        with st.expander("💬 Conversation History", expanded=False):
            for i, msg in enumerate(st.session_state.conversation_history[-6:]):
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content'][:150]}...")
                elif msg["role"] == "assistant":
                    st.markdown(f"**Assistant:** {msg['content'][:150]}...")

    # Query execution
    # In your Tab 1 query execution section, after planning:

# In Tab 1, replace your entire query execution block with this:

if (search_button or user_question) and user_question:
    
    # Step 1: Quick intent check
    with st.spinner("🧠 Understanding query type..."):
        intent = quick_intent_check(user_question)
        st.info(f"🎯 Detected: {intent.get('type', 'unknown').upper()} query")
    
    # Step 2: Generate query with intent hint
    with st.spinner("⚙️ Planning query..."):
        # Add intent to the planning prompt
        query_plan = plan_query_with_llm(user_question, intent_hint=intent)
        
        if not query_plan:
            st.error("❌ Could not generate query plan.")
            st.stop()
    
    # Step 3: Show plan
    with st.expander("🔍 Query Plan & Strategy", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Intent", query_plan.get("intent", "Unknown"))
            st.metric("Collections", ", ".join(query_plan.get("collections", [])))
        with col_b:
            st.metric("Semantic Search", "Yes" if query_plan.get("requires_embedding") else "No")
            st.caption(f"**Strategy:** {query_plan.get('explanation', 'N/A')}")
        
        st.code(query_plan.get("aql_query", "No query"), language="sql")
        if query_plan.get("bind_vars"):
            st.json(query_plan.get("bind_vars"))
    
    # Step 4: Execute
    with st.spinner("⚡ Executing query..."):
        results = execute_planned_query(query_plan)

    # Rest of your existing code for displaying results...

        if results:
            st.success(f"✅ Retrieved {len(results)} results")
        else:
            st.warning("⚠️ No results found")
        
        # Step 3: Analysis
        if results:
            with st.spinner("🤖 Analyzing results..."):
                formatted_context = format_results_for_llm(results, query_plan)
                analysis_prompt = create_analysis_prompt(user_question, formatted_context, query_plan)
                answer = get_llm_analysis(analysis_prompt)
            
            st.markdown("### 📊 Analysis")
            st.markdown(answer)
            
            # Raw data
            with st.expander("📋 View Raw Data", expanded=False):
                try:
                    df = pd.DataFrame(results)
                    cols_to_show = [col for col in df.columns if not col.startswith('_') and col != 'description_embedding']
                    if cols_to_show:
                        st.dataframe(df[cols_to_show], use_container_width=True)
                    else:
                        st.json(results[:10])
                except Exception as e:
                    st.json(results[:10])
                    st.caption(f"Could not format as table: {str(e)}")
            
            # Debug
            with st.expander("🔧 Debug: LLM Context", expanded=False):
                st.text(formatted_context[:3000])
                if len(formatted_context) > 3000:
                    st.caption("(Truncated for display)")
        
        else:
            st.info("💡 Try rephrasing your question or check if the data exists in the database.")
            st.markdown("**Suggestions:**")
            st.markdown("- Verify ticker symbols (e.g., AAPL for Apple)")
            st.markdown("- Check date formats (YYYY-MM-DD)")
            st.markdown("- Ensure the collection contains relevant data")

# ==================== TAB 2: DATABASE BROWSER ====================
with tab2:
    render_database_browser_tab()

# Footer
st.divider()
st.caption("🚀 Powered by ArangoDB, OpenAI GPT-4, and text-embedding-3-small | GraphRAG Architecture")
