
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

6. sec_filings
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