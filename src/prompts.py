"""
prompts.py - Schema descriptions and few-shot examples for AQL query generation
Last updated: 2026-01-06 with Kalshi support
"""

# =============================================================================
# CRITICAL AQL RULES (Condensed version - only essential syntax rules)
# =============================================================================

CRITICAL_AQL_RULES = """
⚠️ CRITICAL AQL SYNTAX RULES ⚠️

1. DATE FUNCTIONS:
   ✅ DATE_SUBTRACT(DATE_NOW(), 30, "day")
   ❌ DATE_SUB() - Does not exist in AQL!

2. ORDER OF OPERATIONS:
   FOR → FILTER → SORT → LIMIT → RETURN
   (SORT/LIMIT must come BEFORE RETURN)

   ⚠️ For NESTED FOR loops:
   FOR outer IN collection
     FILTER ...
     LIMIT ...    ← LIMIT for outer loop goes HERE
     FOR inner IN collection2
       FILTER ...
       RETURN ...  ← NOT after this!

3. NO UNION/JOIN - Use Nested FOR Loops or MERGE:
   ❌ UNION ALL - Does not exist in AQL!
   ❌ JOIN - Use nested FOR loops instead
   ✅ Multiple FOR loops to combine data
   ✅ MERGE() to combine objects

   Example (combining SEC + Company data):
   FOR filing IN sec_filings
     FILTER filing.ticker IN ["RTX", "LMT", "BA"]
     LIMIT 50
     FOR company IN Company
       FILTER company.ticker == filing.ticker
       RETURN MERGE(filing, {marketCap: company.marketCap, employees: company.fullTimeEmployees})

   ⚠️ CRITICAL: In nested FOR loops, LIMIT must come BEFORE the inner FOR loop!

4. COLLECTION NAMES (case-sensitive):
   ✅ Award, Company, MarketData, EconomicData
   ✅ sec_filings, sec_sections, sec_sentences
   ✅ commodity_positions, prediction_markets_polymarket, prediction_markets_kalshi
   ✅ polymarket_traders, polymarket_positions
   ❌ awards, companies, market_data

5. CRITICAL FIELD NAMES:
   Award: award_amount_float (for math), start_date, description_embedding (ONLY collection with embeddings!)
   Company: sharesOutstanding, marketCap, fullTimeEmployees (camelCase!)
   MarketData: sma_20, sma_50 (snake_case), targetMeanPrice (camelCase)
   EconomicData: sandp_500_index, federal_funds_rate
   SEC: finbert_score, avg_negative, avg_uncertainty (NO embeddings!)
   Polymarket: question, description, yes_probability, volume_24h, closed (NO embeddings!)
   Polymarket Traders: total_volume, total_profit, is_whale, activity_level (NO embeddings!)
   Polymarket Positions: market_question, size, average_price, realized_profit, unrealizedProfit
   Kalshi: title, yes_price, volume, status (NO embeddings!)

6. SEMANTIC SEARCH - CRITICAL RULES:
   ✅ Award ONLY: HAS description_embedding - use COSINE_SIMILARITY(doc.description_embedding, @query_vector)
   ❌ ALL OTHER COLLECTIONS: NO embeddings - use CONTAINS(LOWER(field), 'keyword')

   Examples:
   - Award semantic: LET sim = COSINE_SIMILARITY(doc.description_embedding, @query_vector) FILTER sim >= 0.7
   - Polymarket text: FILTER CONTAINS(LOWER(doc.question), 'football') OR CONTAINS(LOWER(doc.description), 'super bowl')
   - SEC text: FILTER CONTAINS(LOWER(doc.text), 'cybersecurity')

   ❌ NEVER use embeddings on: SEC, Polymarket, Kalshi, Company, MarketData, EconomicData

7. ALWAYS ADD LIMIT:
   Every query must have LIMIT to prevent timeout.
"""

# =============================================================================
# FULL SCHEMA DESCRIPTION (Used for reference, not injected into prompts)
# =============================================================================

SCHEMA_DESCRIPTION = """
Database: QUANT_v3 (ArangoDB Multi-Model Graph)

⚠️ CRITICAL AQL SYNTAX - READ FIRST ⚠️

AQL DATE FUNCTIONS (MySQL/SQL syntax does NOT work):
- DATE_SUBTRACT(date, amount, unit) ✅ CORRECT - Subtract time
  Example: DATE_SUBTRACT(DATE_NOW(), 30, "day")
- DATE_ADD(date, amount, unit) ✅ CORRECT - Add time
- DATE_NOW() ✅ CORRECT - Current timestamp
- NEVER use: DATE_SUB() ❌ WRONG - Does not exist in AQL!
- NEVER use: DATEADD() ❌ WRONG - Does not exist in AQL!
- NEVER use: DATEDIFF() ❌ WRONG - Does not exist in AQL!

AQL ROUNDING (no decimal places parameter):
- ROUND(value) - Rounds to integer only
- To round to 2 decimals: FLOOR(value * 100) / 100 ✅
- NEVER use: ROUND(value, 2) ❌ WRONG


AQL ORDER OF OPERATIONS: FOR → FILTER → SORT → LIMIT → RETURN ⚠️
   - SORT must come BEFORE RETURN
   - LIMIT must come BEFORE RETURN
   - NEVER put SORT/LIMIT after RETURN

⚠️ CRITICAL FIELD NAMING RULES (EXACT NAMES FROM DATABASE):

Award:
- award_amount_float (for math), start_date, recipient_name, awarding_agency
- description_embedding (for semantic search), contract_year

Company:
- sharesOutstanding (camelCase, NOT shares_outstanding)
- marketCap (camelCase, NOT market_cap)
- fullTimeEmployees (camelCase, NOT employees)
- sp500_member (snake_case - correct!)
- ticker, company, sector, industry, country, cik, website

MarketData:
- Technical: sma_20, sma_50, ema_12, macd_signal, macd_histogram (underscores)
- Flags: golden_cross, death_cross, above_sma20 (underscores)
- Fundamentals: targetMeanPrice, forwardEps, trailingPE (camelCase)

EconomicData:
- sandp_500_index, federal_funds_rate, unemployment_rate (underscores)
- 10y_2y_treasury_spread, yield_curve_inverted

SEC:
- sec_filings: avg_finbert, avg_uncertainty, avg_negative
- sec_sentences: finbert_score, negative_per_1k, uncertainty_per_1k
- ❌ NO embeddings on SEC - use CONTAINS(LOWER(text), keyword)

Polymarket:
- yes_probability, no_probability, volume_24h, market_slug

Commodity:
- Market_and_Exchange_Names (Capital letters!), as_of_date

⚠️ SEMANTIC SEARCH RULES:
- Award descriptions: HAS embeddings - use COSINE_SIMILARITY ✅
- SEC content: NO embeddings - use CONTAINS() text filters ❌
- Never use COSINE_SIMILARITY on sec_sentences or sec_sections

⚠️ COLLECTION NAMES:
- Use "Award" (capital A, singular)
- Use "sec_filings" (with underscores)
- Use "sec_sections" (with underscores)
- Use "sec_sentences" (with underscores)
- commodity_positions (NOT "commodity_position" or "CommodityPositions")
- prediction_markets_polymarket (NOT "polymarket" or "prediction_market")
- prediction_markets_kalshi (NOT "kalshi" or "kalshi_markets")

DOCUMENT COLLECTIONS:

1. Company
   - ticker (string): Stock ticker symbol (e.g., "AAPL", "DG")
   
   ⚠️ WARNING: This collection ONLY contains ticker. No name, sector, or other fields.
   Collection name: "Company" (capital C, singular)

2. MarketData (daily OHLCV + 40+ technical/fundamental indicators)
   - ticker (string): Stock ticker
   - date (string): Format YYYY-MM-DD (e.g., "2016-01-05")
   - open, high, low, close (float): Price data
   - volume (int): Trading volume
   
   Technical Indicators:
   - sma_5, sma_10, sma_20, sma_50, sma_200 (float): Simple moving averages
   - ema_12, ema_26 (float): Exponential moving averages
   - macd, macd_signal, macd_histogram (float): MACD indicator
   - obv (float): On-balance volume
   - dist_from_sma20, dist_from_sma50, dist_from_sma200 (float): Distance from SMAs
   - golden_cross, death_cross (int): 1 if occurred, else 0
   - above_sma20, above_sma50, above_sma200 (int): 1 if above SMA, else 0
   
   Fundamental Data:
   - targetMeanPrice, targetHighPrice, targetLowPrice, targetMedianPrice (float): Analyst targets
   - recommendationKey (string): "buy", "hold", "sell"
   - numberOfAnalystOpinions (int): Number of analysts
   - forwardEps, trailingEps (float): Earnings per share
   - earningsGrowth, revenueGrowth (float): Growth rates
   - returnOnEquity, returnOnAssets (float): Profitability metrics
   - grossMargins, ebitdaMargins, operatingMargins, profitMargins (float): Margin metrics
   - trailingPE, forwardPE (float): Price-to-earnings ratios
   - priceToBook, priceToSalesTrailing12Months (float): Valuation ratios
   - debtToEquity (float): Leverage ratio
   - totalDebt, totalCash (float): Balance sheet items
   - currentRatio, quickRatio (float): Liquidity ratios
   - freeCashflow, operatingCashflow (float): Cash flow metrics
   - dividendRate, dividendYield (float): Dividend data
   - beta (float): Volatility vs market
   - fiftyTwoWeekHigh, fiftyTwoWeekLow (float): 52-week range
   
   Time Features:
   - year, month, quarter, day_of_week, day_of_month (int): Date components

3. Award (government contracts from USASpending.gov)
   - ticker (string): Recipient company ticker
   - recipient_name (string): Company name (e.g., "3M COMPANY")
   - matched_sp500_name (string): Standardized company name
   - start_date (string): Contract start YYYY-MM-DD
   - award_amount (string): Contract value as string (for display)
   - award_amount_float (float): Contract value as number (USE THIS for filtering/sorting)
   - awarding_agency (string): Government agency (e.g., "Department of Defense")
   - description (string): Award description (full text)
   - description_embedding (array): Semantic vector (1536 dimensions) for similarity search
   - contract_year (string): Year as string (e.g., "2017")
   - source_file (string): Origin file (e.g., "contracts_2017.csv")
   - ingested_at (string): Timestamp of data ingestion

4. EconomicData (macroeconomic indicators from FRED)
   ⚠️ CRITICAL: Field names use underscores, not camelCase!
   - date (string): Date YYYY-MM-DD
   - ingested_at (string): Timestamp
   
   Stock Indices:
   - sandp_500_index (float): S&P 500 value (NOT "sp500"!)
   - nasdaq_composite (float): NASDAQ value
   - dow_jones_industrial_average (float): DJIA value
   - vix_volatility_index (float): VIX value
   
   Interest Rates:
   - federal_funds_rate (float): Fed funds rate % (NOT "fed_funds_rate"!)
   - 2_year_treasury_yield (float): 2Y Treasury %
   - 10_year_treasury_yield (float): 10Y Treasury %
   - 30_year_treasury_yield (float): 30Y Treasury %
   - 10y_2y_treasury_spread (float): Yield curve spread
   
   Inflation:
   - consumer_price_index_cpi (float): CPI value
   - core_cpi_ex_food_and_energy (float): Core CPI
   - core_pce_feds_preferred (float): Fed's preferred inflation gauge
   
   Labor Market:
   - unemployment_rate (float): Unemployment %
   - nonfarm_payrolls (int): Payroll count
   - initial_jobless_claims (int): Weekly jobless claims
   
   Economic Activity:
   - real_gdp (float): Real GDP
   - industrial_production (float): Industrial production index
   - retail_sales (float): Retail sales
   - consumer_sentiment (float): Consumer sentiment index
   
   Commodities & Other:
   - crude_oil_price_wti (float): Oil price (WTI)
   - m2_money_supply (float): M2 money supply
   - housing_starts (int): New housing starts
   - case_shiller_home_price_index (float): Home price index
   
   Derived Fields:
   - yield_curve_slope (float): 10Y - 2Y spread
   - yield_curve_inverted (int): 1 if inverted, else 0

5. commodity_positions (CFTC Commitments of Traders data)
   - ticker (string): Company ticker
   - as_of_date (string): Report date YYYY-MM-DD
   - Market_and_Exchange_Names (string): Commodity name (e.g., "CRUDE OIL, LIGHT SWEET - CHICAGO MERCANTILE EXCHANGE")
   - net_noncommercial_position (int): Net speculator position (long - short)
   - net_commercial_position (int): Net hedger position
   - open_interest (int): Total open interest
   - commodity_count (int): Number of commodity types tracked for this ticker
   - total_position_size (int): Total position size across all commodities
   
   ⚠️ Use Case: Track commodity exposure for energy, agriculture, mining companies

6. prediction_markets_polymarket (Polymarket prediction market data)
   - _key (string): Unique market ID (condition_id)
   - condition_id (string): Market condition ID
   - question (string): Market question (e.g., "Will Apple release new iPhone in Q1 2024?")
   - description (string): Detailed market description
   - market_slug (string): URL-friendly slug
   - end_date (string): Market close date
   - volume (float): Total trading volume ($)
   - volume_24h (float): 24-hour trading volume ($)
   - liquidity (float): Current liquidity ($)
   - closed (bool): Market status (true/false)
   - category (string): Market category
   - outcomes (array): Possible outcomes ["Yes", "No"]
   - outcome_prices (array): Current prices [yes_price, no_price]
   - yes_probability (float): Probability of "Yes" outcome (0-1)
   - no_probability (float): Probability of "No" outcome (0-1)
   - fetched_at (string): Data fetch timestamp
   
   ⚠️ Use Case: Forward-looking sentiment, event probabilities, crowd predictions

7. prediction_markets_kalshi (Kalshi prediction market data)
   - _key (string): Unique market ID
   - market_ticker (string): Kalshi market ticker (e.g., "INXD-24JAN19")
   - title (string): Market question (e.g., "Will the Nasdaq be above 18,000 on January 19?")
   - category (string): Market category
   - status (string): Market status ("active", "closed", "settled")
   - close_time (string): Market close timestamp
   - expiration_time (string): Market expiration timestamp
   - volume (float): Total trading volume ($)
   - volume_24h (float): 24-hour trading volume ($)
   - open_interest (float): Current open interest ($)
   - yes_price (float): Current "Yes" price (0-1, equivalent to probability)
   - no_price (float): Current "No" price (0-1)
   - last_price (float): Most recent trade price
   - previous_yes_price (float): Previous yes price for comparison
   - previous_no_price (float): Previous no price
   - strike_date (string): Event date/strike date
   - floor_strike (float): Floor strike price (for ranged markets)
   - cap_strike (float): Cap strike price (for ranged markets)
   - result (string): Market result (if settled)
   - fetched_at (string): Data fetch timestamp
   
   ⚠️ Key Differences from Polymarket:
   - Uses "status" instead of "closed" (values: "active", "closed", "settled")
   - Uses "title" instead of "question"
   - Uses "yes_price" instead of "yes_probability" (both 0-1 scale)
   - Has strike prices for binary markets
   - More structured for financial/index markets
   
   ⚠️ Use Case: Financial markets, economic indicators, index levels

8. polymarket_traders (Polymarket trader/whale data from Data API v1)
   - _key (string): Unique trader key (hash of address)
   - address (string): Wallet address
   - total_volume (float): Lifetime trading volume ($)
   - total_trades (int): Total number of trades
   - total_profit (float): Total realized profit/loss ($)
   - is_whale (bool): True if total_volume >= $50,000
   - fetched_at (string): Data fetch timestamp

   Engineered Features:
   - volume_rank (int): Rank by trading volume
   - avg_position_size (float): Average position size ($)
   - activity_level (string): "casual" | "regular" | "active" | "whale"
   - profit_ratio (float): Profit per dollar traded
   - is_profitable (bool): True if total_profit > 0

   ⚠️ Use Case: Whale tracking, smart money analysis, trader behavior

9. polymarket_positions (Current positions held by traders)
   - _key (string): Unique position key (trader_address + market_id + outcome)
   - position_id (string): Position identifier
   - trader_address (string): Wallet address of trader
   - trader_key (string): FK to polymarket_traders._key
   - market_condition_id (string): Market condition ID
   - market_key (string): FK to prediction_markets_polymarket._key
   - market_question (string): Market question text
   - outcome_index (int): 0 for "No", 1 for "Yes"
   - size (float): Position size (number of shares)
   - average_price (float): Average entry price (0-1)
   - realized_profit (float): Realized P&L ($)
   - unrealizedProfit (float): Unrealized P&L ($)
   - current_value (float): Current position value ($)
   - current_price (float): Current market price (0-1)
   - redeemable (bool): Can be redeemed
   - fetched_at (string): Data fetch timestamp

   ⚠️ Use Case: Track what whales are betting on, position analysis, portfolio exposure

10. sec_filings (SEC document metadata)
   - ticker (string): Company ticker
   - type (string): Filing type ("10-K", "10-Q", "8-K", etc.)
   - accession (string): SEC accession number (unique ID)
   - file_name (string): Source file name
   - filing_date (string): Date filed YYYY-MM-DD
   - fiscal_year (int): Fiscal year
   
   Sentiment Metrics (aggregated from sentences):
   - avg_finbert (float): Average FinBERT sentiment (-1 to +1)
   - avg_uncertainty (float): Uncertainty score per 1000 words
   - avg_positive (float): Positive words per 1000
   - avg_negative (float): Negative words per 1000
   - sentence_count (int): Total sentences in filing
   
   ⚠️ NO CONTENT FIELD: Full text is NOT stored here. Use sec_sections or sec_sentences.

9. sec_sections (sections within filings)
   - filing_id (string): Parent filing ID (format: "sec_filings/{ticker}_{type}_{accession}_{filename}")
   - section_type (string): Section type (e.g., "Full Document", "Risk Factors", "MD&A")
   - start_char (int): Start position in original document
   - length (int): Length in characters
   
   ⚠️ NO EMBEDDING FIELD: Cannot do semantic search on sections.
   ⚠️ NO CONTENT FIELD: Text is NOT stored. Use sec_sentences for actual content.

10. sec_sentences (individual sentences - most granular level)
    - section_id (string): Parent section ID (format: "sec_sections/{ticker}_{type}_{accession}_{filename}_sec{N}")
    - text (string): Sentence text (THIS is where content lives)
    - n_tokens (int): Token count
    
    Sentiment Metrics:
    - finbert_score (float): FinBERT sentiment score (-1 to +1)
    - finbert_probs (object): Probabilities {positive, negative, neutral}
    - negative_per_1k (float): Negative words per 1000
    - positive_per_1k (float): Positive words per 1000
    - uncertainty_per_1k (float): Uncertainty words per 1000
    - litigious_per_1k (float): Legal language per 1000
    
    ⚠️ NO EMBEDDING FIELD: Cannot do vector similarity search.
    ⚠️ For semantic search, use finbert_score filtering instead of cosine similarity.

EDGE COLLECTIONS (Graph Relationships):

1. HAS_MARKETDATA: Company -> MarketData
   - date (string): Market date
   
   Usage: FOR market IN OUTBOUND company HAS_MARKETDATA

2. HAS_AWARD: Company -> Award
   - award_amount (float): Contract value
   
   Usage: FOR award IN OUTBOUND company HAS_AWARD

3. HAS_COMMODITY_POSITION: Company -> commodity_positions
   - commodity_name (string): Commodity type
   - as_of_date (string): CFTC report date
   
   Usage: FOR position IN OUTBOUND company HAS_COMMODITY_POSITION

4. market_mentions_company_polymarket: prediction_markets_polymarket -> Company
   - match_type (string): "keyword" (direct mention)
   - matched_keywords (array): Keywords that matched (e.g., ["tesla", "elon musk"])
   - confidence (float): Match confidence score (0-1)
   - market_volume_24h (float): Market trading volume
   
   Usage: FOR market IN INBOUND company market_mentions_company_polymarket

5. market_related_to_sector_polymarket: prediction_markets_polymarket -> Company
   - match_type (string): "sector" (sector-level relationship)
   - sector (string): Affected sector (e.g., "technology", "defense")
   - matched_keywords (array): Sector keywords matched
   - confidence (float): Lower confidence (typically 0.4)
   
   Usage: FOR market IN INBOUND company market_related_to_sector_polymarket
   
   ⚠️ WARNING: Contains 919k edges (noisy). Use sparingly or prefer market_mentions_company_polymarket.

6. market_affects_company_polymarket: prediction_markets_polymarket -> Company
   - match_type (string): "macro_event" (macroeconomic relationship)
   - event_type (string): Event category (e.g., "fed_rate", "inflation", "recession")
   - confidence (float): Low confidence (typically 0.3)
   
   Usage: FOR market IN INBOUND company market_affects_company_polymarket
   
   ⚠️ WARNING: Contains 119k edges (very noisy). Use only for macro analysis.

7. market_mentions_company_kalshi: prediction_markets_kalshi -> Company
   - match_type (string): "keyword" (direct mention)
   - matched_keywords (array): Keywords that matched
   - confidence (float): Match confidence score (0-1)
   - market_volume_24h (float): Market trading volume
   - source (string): Data source ("kalshi")
   - created_at (string): Edge creation timestamp
   
   Usage: FOR market IN INBOUND company market_mentions_company_kalshi

8. market_related_to_sector_kalshi: prediction_markets_kalshi -> Company
   - match_type (string): "sector" (sector-level relationship)
   - sector (string): Affected sector
   - matched_keywords (array): Sector keywords matched
   - confidence (float): Confidence score
   - market_volume_24h (float): Market trading volume
   
   Usage: FOR market IN INBOUND company market_related_to_sector_kalshi

9. trader_has_position: polymarket_traders -> polymarket_positions
   - position_size (float): Position size (shares)
   - avg_price (float): Average entry price
   - realized_profit (float): Realized P&L
   - created_at (string): Edge creation timestamp

   Usage: FOR position IN OUTBOUND trader trader_has_position

   ⚠️ Use Case: Find all positions for a specific whale trader

10. position_in_market: polymarket_positions -> prediction_markets_polymarket
    - position_size (float): Position size (shares)
    - outcome_index (int): 0="No", 1="Yes"
    - current_price (float): Current market price
    - created_at (string): Edge creation timestamp

    Usage: FOR market IN OUTBOUND position position_in_market

    ⚠️ Use Case: See which markets a position belongs to

11. HAS_FILING: Company -> sec_filings
   - filing_date (string): Date filed
   - filing_type (string): Filing type
   - links companies to SEC filings
   
   Usage: FOR filing IN OUTBOUND company HAS_FILING

10. has_section: sec_filings -> sec_sections
    
    Usage: FOR section IN OUTBOUND filing has_section

11. has_sentence: sec_sections -> sec_sentences
    
    Usage: FOR sentence IN OUTBOUND section has_sentence

GRAPHS:
- QUANT_v3_FinanceGraph: Company-centric financial data graph

## GRAPH STRUCTURE - ALWAYS PREFER EDGES OVER TICKER FILTERING

Edge Definitions:
1. Company → Award (via HAS_AWARD edge)
2. Company → sec_filings (via HAS_FILING edge)
3. Company → MarketData (via HAS_MARKETDATA edge)
4. sec_filings → sec_sections (via has_section edge)
5. sec_sections → sec_sentences (via has_sentence edge)

## CRITICAL: When to Use Graph Traversals
- If query involves Company + Award → USE: FOR award IN OUTBOUND company HAS_AWARD
- If query involves Company + MarketData → USE: FOR m IN OUTBOUND company HAS_MARKETDATA
- If query involves Company + Filings → USE: FOR filing IN OUTBOUND company HAS_FILING
- For multi-hop traversal → USE nested OUTBOUND (e.g., Company → Filing → Section → Sentence)

## Graph Traversal Examples:
```aql
// Companies with awards (PREFERRED)
FOR company IN Company
  FOR award IN OUTBOUND company HAS_AWARD
    RETURN {company, award}

// Companies with market data (PREFERRED)
FOR company IN Company
  FOR m IN OUTBOUND company HAS_MARKETDATA
    FILTER m.date >= "2025-01-01"
    RETURN {company, market: m}

// Deep traversal: Company → Filing → Section → Sentence
FOR company IN Company
  FILTER company.ticker == @ticker
  FOR filing IN OUTBOUND company HAS_FILING
    FOR section IN OUTBOUND filing has_section
      FOR sentence IN OUTBOUND section has_sentence
        FILTER sentence.finbertscore < -0.3
        RETURN {company, filing, section, sentence}

"""

# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

FEW_SHOT_EXAMPLES = """
EXAMPLE 1 - Market Data Lookup:
Question: "What was Apple's closing price on 2016-01-05?"
Intent: single_value_lookup
Collections: ["MarketData"]
AQL:
FOR doc IN MarketData
  FILTER doc.ticker == @ticker AND doc.date == @date
  RETURN {date: doc.date, close: doc.close, volume: doc.volume}
Bind Variables: {"ticker": "AAPL", "date": "2016-01-05"}
Requires Embedding: false

---

EXAMPLE 2 - Award Lookup (CORRECT field name!):
Question: "Show me the top 5 largest government awards"
Intent: ranking
Collections: ["Award"]
AQL:
FOR doc IN Award
  FILTER doc.award_amount_float != null
  SORT doc.award_amount_float DESC
  LIMIT 5
  RETURN {
    recipient: doc.recipient_name,
    ticker: doc.ticker,
    amount: doc.award_amount_float,
    agency: doc.awarding_agency,
    start_date: doc.start_date,
    description: SUBSTRING(doc.description, 0, 200)
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 3 - Economic Data (CORRECT field names!):
Question: "What was the unemployment rate and S&P 500 on 2016-01-04?"
Intent: single_value_lookup
Collections: ["EconomicData"]
AQL:
FOR doc IN EconomicData
  FILTER doc.date == @date
  RETURN {
    date: doc.date,
    sandp_500: doc.sandp_500_index,
    unemployment: doc.unemployment_rate,
    fed_rate: doc.federal_funds_rate,
    vix: doc.vix_volatility_index,
    yield_curve: doc.yield_curve_slope
  }
Bind Variables: {"date": "2016-01-04"}
Requires Embedding: false

---

EXAMPLE 4 - Award Semantic Search:
Question: "Find awards related to artificial intelligence"
Intent: semantic_search
Collections: ["Award"]
AQL:
FOR doc IN Award
  FILTER doc.description_embedding != null
  LET similarity = COSINE_SIMILARITY(doc.description_embedding, @query_vector)
  FILTER similarity >= 0.70
  SORT similarity DESC
  LIMIT 10
  RETURN {
    recipient: doc.recipient_name,
    ticker: doc.ticker,
    description: SUBSTRING(doc.description, 0, 300),
    amount: doc.award_amount_float,
    start_date: doc.start_date,
    similarity: similarity
  }
Bind Variables: {"query_vector": [0.123, ...]}
Requires Embedding: true
Embedding Text: "artificial intelligence AI machine learning deep learning neural networks"

---

EXAMPLE 5 - SEC Filing Sentiment:
Question: "Show me Apple's most negative 10-K filings"
Intent: sentiment_analysis
Collections: ["sec_filings"]
AQL:
FOR doc IN sec_filings
  FILTER doc.ticker == @ticker
  FILTER doc.type == "10-K"
  FILTER doc.avg_finbert != null
  SORT doc.avg_finbert ASC
  LIMIT 5
  RETURN {
    ticker: doc.ticker,
    filing_date: doc.filing_date,
    fiscal_year: doc.fiscal_year,
    sentiment: doc.avg_finbert,
    negative_score: doc.avg_negative,
    uncertainty: doc.avg_uncertainty
  }
Bind Variables: {"ticker": "AAPL"}
Requires Embedding: false

---

EXAMPLE 6 - SEC Sentence Search (NO embeddings, use text filter):
Question: "Find SEC sentences mentioning supply chain risk"
Intent: text_search
Collections: ["sec_sentences"]
AQL:
FOR doc IN sec_sentences
  FILTER CONTAINS(LOWER(doc.text), "supply chain")
  FILTER CONTAINS(LOWER(doc.text), "risk")
  FILTER doc.finbert_score < -0.3
  SORT doc.finbert_score ASC
  LIMIT 10
  RETURN {
    text: SUBSTRING(doc.text, 0, 500),
    sentiment: doc.finbert_score,
    section_id: doc.section_id,
    negative_words: doc.negative_per_1k
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 7 - Graph Traversal (Company -> Awards):
Question: "Show me awards for ticker DG"
Intent: graph_traversal
Collections: ["Company", "Award"]
Edges: ["HAS_AWARD"]
AQL:
FOR company IN Company
  FILTER company.ticker == @ticker
  FOR award IN OUTBOUND company HAS_AWARD
    SORT award.start_date DESC
    LIMIT 10
    RETURN {
      ticker: company.ticker,
      recipient: award.recipient_name,
      amount: award.award_amount_float,
      agency: award.awarding_agency,
      start_date: award.start_date,
      description: SUBSTRING(award.description, 0, 200)
    }
Bind Variables: {"ticker": "DG"}
Requires Embedding: false

---

EXAMPLE 8 - Market Data with Indicators:
Question: "Show me stocks with price above 20-day SMA on 2016-01-05"
Intent: technical_screening
Collections: ["MarketData"]
AQL:
FOR doc IN MarketData
  FILTER doc.date == @date
  FILTER doc.sma_20 != null
  FILTER doc.above_sma20 == 1
  SORT doc.close DESC
  LIMIT 20
  RETURN {
    ticker: doc.ticker,
    close: doc.close,
    sma_20: doc.sma_20,
    dist_from_sma20: doc.dist_from_sma20,
    volume: doc.volume
  }
Bind Variables: {"date": "2016-01-05"}
Requires Embedding: false

---

EXAMPLE 9 - Date Range Query:
Question: "Show me Tesla's stock prices for January 2016"
Intent: time_series
Collections: ["MarketData"]
AQL:
FOR doc IN MarketData
  FILTER doc.ticker == @ticker
  FILTER doc.date >= @start_date AND doc.date < @end_date
  SORT doc.date ASC
  LIMIT 50
  RETURN {
    date: doc.date,
    ticker: doc.ticker,
    open: doc.open,
    close: doc.close,
    volume: doc.volume
  }
Bind Variables: {"ticker": "TSLA", "start_date": "2016-01-01", "end_date": "2016-02-01"}
Requires Embedding: false

---

EXAMPLE 10 - Aggregation:
Question: "What's the total value of defense awards in 2017?"
Intent: aggregation
Collections: ["Award"]
AQL:
FOR doc IN Award
  FILTER doc.contract_year == "2017"
  FILTER doc.awarding_agency LIKE "%Defense%" OR doc.awarding_agency LIKE "%DoD%"
  COLLECT AGGREGATE total = SUM(doc.award_amount_float), count = COUNT(1)
  RETURN {total_amount: total, award_count: count}
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 11 - Commodity Positions:
Question: "Show me companies with crude oil exposure"
Intent: commodity_exposure
Collections: ["commodity_positions"]
AQL:
FOR doc IN commodity_positions
  FILTER CONTAINS(LOWER(doc.Market_and_Exchange_Names), "crude oil")
  FILTER doc.net_noncommercial_position != 0
  SORT ABS(doc.net_noncommercial_position) DESC
  LIMIT 20
  RETURN {
    ticker: doc.ticker,
    commodity: doc.Market_and_Exchange_Names,
    report_date: doc.as_of_date,
    net_position: doc.net_noncommercial_position,
    open_interest: doc.open_interest
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 12 - Polymarket Prediction Markets:
Question: "Find Polymarket predictions about Tesla"
Intent: prediction_market_search
Collections: ["prediction_markets_polymarket"]
AQL:
FOR market IN prediction_markets_polymarket
  FILTER CONTAINS(LOWER(market.question), "tesla") OR CONTAINS(LOWER(market.question), "musk")
  FILTER market.volume_24h > 1000
  FILTER market.closed == false
  SORT market.volume_24h DESC
  LIMIT 10
  RETURN {
    question: market.question,
    yes_prob: market.yes_probability,
    volume_24h: market.volume_24h,
    liquidity: market.liquidity,
    end_date: market.end_date,
    description: SUBSTRING(market.description, 0, 200)
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 12b - Kalshi Prediction Markets:
Question: "Find Kalshi markets about Nasdaq"
Intent: prediction_market_search
Collections: ["prediction_markets_kalshi"]
AQL:
FOR market IN prediction_markets_kalshi
  FILTER CONTAINS(LOWER(market.title), "nasdaq") OR 
         CONTAINS(LOWER(market.title), "inxd")
  FILTER market.status == "active"
  FILTER market.volume_24h > 1000
  SORT market.volume_24h DESC
  LIMIT 10
  RETURN {
    title: market.title,
    yes_price: market.yes_price,
    volume_24h: market.volume_24h,
    open_interest: market.open_interest,
    close_time: market.close_time
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 13 - Graph: Company -> Polymarket:
Question: "What are Polymarket prediction markets saying about Apple?"
Intent: graph_traversal
Collections: ["Company", "prediction_markets_polymarket"]
Edges: ["market_mentions_company_polymarket"]
AQL:
FOR company IN Company
  FILTER company.ticker == @ticker
  FOR market IN INBOUND company market_mentions_company_polymarket
    FILTER market.volume_24h > 5000
    FILTER market.closed == false
    SORT market.volume_24h DESC
    LIMIT 10
    RETURN {
      ticker: company.ticker,
      question: market.question,
      yes_prob: market.yes_probability,
      volume_24h: market.volume_24h,
      matched_keywords: market.matched_keywords,
      confidence: market.confidence
    }
Bind Variables: {"ticker": "AAPL"}
Requires Embedding: false

---

EXAMPLE 13b - Graph: Company -> Kalshi:
Question: "What Kalshi markets mention Microsoft?"
Intent: graph_traversal
Collections: ["Company", "prediction_markets_kalshi"]
Edges: ["market_mentions_company_kalshi"]
AQL:
FOR edge IN market_mentions_company_kalshi
  FILTER CONTAINS(edge._to, "MSFT")
  
  LET market = FIRST(
    FOR m IN prediction_markets_kalshi
      FILTER m._id == edge._from
      RETURN m
  )
  
  FILTER market != null
  FILTER market.status == "active"
  FILTER market.volume_24h > 5000
  
  SORT market.volume_24h DESC
  LIMIT 10
  
  RETURN {
    title: market.title,
    yes_price: market.yes_price,
    volume_24h: market.volume_24h,
    matched_keywords: edge.matched_keywords,
    confidence: edge.confidence
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 14 - Combined: Market + Awards + Commodities:
Question: "Show me defense contractors with commodity exposure"
Intent: multi_source_analysis
Collections: ["Company", "Award", "commodity_positions"]
Edges: ["HAS_AWARD", "HAS_COMMODITY_POSITION"]
AQL:
FOR company IN Company
  LET awards = (
    FOR award IN OUTBOUND company HAS_AWARD
      FILTER award.awarding_agency LIKE "%Defense%"
      RETURN award
  )
  
  LET commodities = (
    FOR position IN OUTBOUND company HAS_COMMODITY_POSITION
      RETURN position
  )
  
  FILTER LENGTH(awards) > 0 AND LENGTH(commodities) > 0
  
  RETURN {
    ticker: company.ticker,
    award_count: LENGTH(awards),
    total_awards: SUM(awards[*].award_amount_float),
    commodities: commodities[*].Market_and_Exchange_Names,
    net_positions: commodities[*].net_noncommercial_position
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 15 - Polymarket Sentiment Score:
Question: "Calculate bullish sentiment for tech stocks from Polymarket"
Intent: sentiment_aggregation
Collections: ["Company", "prediction_markets_polymarket"]
Edges: ["market_mentions_company_polymarket"]
AQL:
FOR company IN Company
  FILTER company.ticker IN ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
  
  LET markets = (
    FOR market IN INBOUND company market_mentions_company_polymarket
      FILTER market.volume_24h > 10000
      FILTER market.closed == false
      RETURN market
  )
  
  FILTER LENGTH(markets) > 0
  
  LET avg_bullish = AVG(markets[*].yes_probability)
  LET total_volume = SUM(markets[*].volume_24h)
  
  SORT avg_bullish DESC
  
  RETURN {
    ticker: company.ticker,
    market_count: LENGTH(markets),
    avg_yes_prob: ROUND(avg_bullish * 100, 1),
    total_volume_24h: total_volume,
    sentiment: avg_bullish > 0.6 ? "🚀 BULLISH" : 
               avg_bullish < 0.4 ? "📉 BEARISH" : "⚖️ NEUTRAL"
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE 16 - Combined Polymarket + Kalshi Sentiment:
Question: "Compare Polymarket and Kalshi sentiment for Apple"
Intent: multi_source_comparison
Collections: ["Company", "prediction_markets_polymarket", "prediction_markets_kalshi"]
Edges: ["market_mentions_company_polymarket", "market_mentions_company_kalshi"]
AQL:
LET polymarket_data = (
  FOR edge IN market_mentions_company_polymarket
    FILTER CONTAINS(edge._to, "AAPL")
    LET market = DOCUMENT(edge._from)
    FILTER market != null AND market.closed == false
    RETURN market.yes_probability
)

LET kalshi_data = (
  FOR edge IN market_mentions_company_kalshi
    FILTER CONTAINS(edge._to, "AAPL")
    LET market = DOCUMENT(edge._from)
    FILTER market != null AND market.status == "active"
    RETURN market.yes_price
)

RETURN {
  ticker: "AAPL",
  polymarket: {
    market_count: LENGTH(polymarket_data),
    avg_sentiment: AVG(polymarket_data)
  },
  kalshi: {
    market_count: LENGTH(kalshi_data),
    avg_sentiment: AVG(kalshi_data)
  },
  combined_sentiment: (AVG(polymarket_data) + AVG(kalshi_data)) / 2
}
Bind Variables: {}
Requires Embedding: false

--- EXAMPLE: SEC Text Search with Date + Company Data ---
Question: Show me cybersecurity risks in 2022 with company details
Intent: sec_text_search_with_date_and_company
Collections: sec_filings, sec_sections, sec_sentences, Company
AQL:
FOR filing IN sec_filings
  FILTER filing.filing_date >= '2022-01-01'
  AND filing.filing_date <= '2022-12-31'
  
  LET company = FIRST(
    FOR c IN Company
    FILTER c.ticker == filing.ticker
    RETURN c
  )
  
  FILTER company != null
  
  LET risks = (
    FOR section IN sec_sections
    FILTER section.filing_id == filing._id
      FOR sentence IN sec_sentences
      FILTER sentence.section_id == section._id
      AND sentence.finbert_score < -0.3
      AND CONTAINS(LOWER(sentence.text), 'cybersecurity')
      LIMIT 3
      RETURN {
        text: SUBSTRING(sentence.text, 0, 300),
        sentiment: sentence.finbert_score
      }
  )
  
  FILTER LENGTH(risks) > 0
  LIMIT 10
  
  RETURN {
    ticker: filing.ticker,
    company: company.company,
    cik: company.cik,
    marketCap: company.marketCap,
    sharesOutstanding: company.sharesOutstanding,
    sector: company.sector,
    filing_date: filing.filing_date,
    risks: risks
  }

Bind Variables: {}
Requires Embedding: false

💡 Strategy:
- Start with sec_filings (date index makes this fast)
- Filter by date FIRST (reduces from 7,495 to ~250 filings)
- Join to Company via ticker
- Navigate filing → sections → sentences using ID fields
- Use subquery for risks to keep result clean
- NO @ticker bind variable (we're searching all companies)
---

--- EXAMPLE: SEC Keyword Search (Simple) ---
Question: What are the biggest risks mentioned in filings?
Intent: sec_keyword_search
Collections: sec_sentences
AQL:
FOR sentence IN sec_sentences
  FILTER sentence.finbert_score < -0.3
  AND (
    CONTAINS(LOWER(sentence.text), 'risk')
    OR CONTAINS(LOWER(sentence.text), 'threat')
  )
  SORT sentence.finbert_score ASC
  LIMIT 20
  RETURN {
    text: SUBSTRING(sentence.text, 0, 400),
    sentiment: sentence.finbert_score,
    negative_per_1k: sentence.negative_per_1k
  }

Bind Variables: {}
Requires Embedding: false

💡 Strategy:
- Simple sentence-level search (no joins needed)
- Use text filters + sentiment
- Sort by most negative
---

--- EXAMPLE: SEC Company-Specific Search ---
Question: Show Apple's cybersecurity risks in their latest 10-K
Intent: sec_company_specific
Collections: sec_filings, sec_sections, sec_sentences, Company
AQL:
FOR filing IN sec_filings
  FILTER filing.ticker == @ticker
  AND filing.type == '10-K'
  SORT filing.filing_date DESC
  LIMIT 1
  
  FOR section IN sec_sections
  FILTER section.filing_id == filing._id
    FOR sentence IN sec_sentences
    FILTER sentence.section_id == section._id
    AND sentence.finbert_score < -0.3
    AND CONTAINS(LOWER(sentence.text), 'cybersecurity')
    SORT sentence.finbert_score ASC
    LIMIT 10
    RETURN {
      text: SUBSTRING(sentence.text, 0, 400),
      sentiment: sentence.finbert_score,
      filing_date: filing.filing_date
    }

Bind Variables: {"ticker": "AAPL"}
Requires Embedding: false

💡 Strategy:
- When specific ticker mentioned, use @ticker bind variable
- Get latest filing first, then drill down
- More efficient than scanning all sentences

---

EXAMPLE - Price Change Over Period (CORRECT ORDER)
Question: "What are the top 15 companies that have seen their stock price increase by more than 2% in the last 180 days?"
Intent: price_screening
Collections: Company, MarketData
Edges: HAS_MARKETDATA

AQL:
LET six_months_ago = DATE_SUBTRACT(DATE_NOW(), 180, "day")

FOR company IN Company
  LIMIT 500
  
  LET market_data = (
    FOR m IN OUTBOUND company HAS_MARKETDATA
      FILTER m.date >= six_months_ago
      SORT m.date ASC
      RETURN m
  )
  
  FILTER LENGTH(market_data) >= 2
  
  LET oldest_price = FIRST(market_data).close
  LET newest_price = LAST(market_data).close
  LET price_change_pct = (newest_price - oldest_price) / oldest_price * 100
  
  FILTER price_change_pct > 2
  
  SORT price_change_pct DESC
  LIMIT 15
  RETURN {
    ticker: company.ticker,
    start_price: oldest_price,
    end_price: newest_price,
    change_pct: FLOOR(price_change_pct * 100) / 100,
    days: LENGTH(market_data)
  }

Bind Variables: {}
Requires_Embedding: false

Strategy:
✅ CORRECT ORDER: FOR → FILTER → SORT → LIMIT → RETURN
❌ WRONG: Don't put SORT after RETURN
- Compare first vs last price over 180 days (not daily close > open)
- Use DATE_SUBTRACT(DATE_NOW(), 180, "day")
- SORT DESC before RETURN to get top performers

---

EXAMPLE: Prediction Markets - Top Markets by Volume
Question: "What are the top 10 most active Polymarket markets?"
Intent: ranking
Collections: ["prediction_markets_polymarket"]
AQL:
FOR market IN prediction_markets_polymarket
  FILTER market.volume_24h > 0
  FILTER market.closed == false
  SORT market.volume_24h DESC
  LIMIT 10
  RETURN {
    question: market.question,
    volume_24h: market.volume_24h,
    yes_probability: market.yes_probability,
    liquidity: market.liquidity,
    end_date: market.end_date,
    category: market.category
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE: Prediction Markets - Markets Mentioning Company
Question: "Show me prediction markets about Tesla"
Intent: company_related_search
Collections: ["prediction_markets_polymarket", "Company"]
AQL:
FOR company IN Company
  FILTER company.ticker == @ticker
  FOR market IN INBOUND company market_mentions_company_polymarket
    FILTER market.closed == false
    SORT market.volume_24h DESC
    LIMIT 20
    RETURN {
      question: market.question,
      yes_probability: market.yes_probability,
      volume_24h: market.volume_24h,
      confidence: market.confidence,
      matched_keywords: market.matched_keywords
    }
Bind Variables: {"ticker": "TSLA"}
Requires Embedding: false

Strategy:
✅ Use graph edges (market_mentions_company_polymarket) not text search
✅ Filter by ticker on Company collection first
✅ INBOUND traversal to find markets pointing to company

---

EXAMPLE: Whale Traders - Top Traders by Profit
Question: "Who are the most profitable whale traders on Polymarket?"
Intent: ranking
Collections: ["polymarket_traders"]
AQL:
FOR trader IN polymarket_traders
  FILTER trader.is_whale == true
  FILTER trader.total_profit > 0
  SORT trader.total_profit DESC
  LIMIT 10
  RETURN {
    address: trader.address,
    total_volume: trader.total_volume,
    total_profit: trader.total_profit,
    total_trades: trader.total_trades,
    profit_ratio: trader.profit_ratio,
    activity_level: trader.activity_level
  }
Bind Variables: {}
Requires Embedding: false

---

EXAMPLE: Whale Positions - What Markets Are Whales Betting On
Question: "What markets are whale traders betting Yes on?"
Intent: graph_traversal
Collections: ["polymarket_traders", "polymarket_positions", "prediction_markets_polymarket"]
AQL:
FOR trader IN polymarket_traders
  FILTER trader.is_whale == true
  FOR position IN OUTBOUND trader trader_has_position
    FILTER position.outcome_index == 1
    FILTER position.size > 100
    FOR market IN OUTBOUND position position_in_market
      FILTER market.closed == false
      SORT position.size DESC
      LIMIT 20
      RETURN DISTINCT {
        market_question: market.question,
        yes_probability: market.yes_probability,
        position_size: position.size,
        trader_address: trader.address,
        trader_volume: trader.total_volume,
        current_price: position.current_price
      }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ CORRECT: Graph traversal trader -> position -> market
✅ Filter outcome_index == 1 for "Yes" bets
✅ OUTBOUND for trader->position, OUTBOUND for position->market
✅ Use DISTINCT to avoid duplicate markets

---

⚠️ FIELD NAME CHEAT SHEET (Common Mistakes):

WRONG → CORRECT
- sp500 → sandp_500_index
- fed_funds_rate → federal_funds_rate
- award_amount (for math) → award_amount_float
- sec_filings.content → sec_sentences.text
- sec_sections.embedding → (DOESN'T EXIST, use finbert_score filter)
- polymarket → prediction_markets_polymarket
- kalshi → prediction_markets_kalshi
- commodity_position → commodity_positions

⚠️ PREDICTION MARKET FIELD DIFFERENCES:

POLYMARKET:
- Collection: prediction_markets_polymarket
- Question field: .question
- Probability field: .yes_probability
- Status field: .closed (boolean)
- Volume field: .volume_24h

KALSHI:
- Collection: prediction_markets_kalshi
- Question field: .title
- Probability field: .yes_price
- Status field: .status (string: "active", "closed", "settled")
- Volume field: .volume_24h
- Additional: .open_interest, .strike_date, .market_ticker

EDGES:
- Polymarket edges: market_mentions_company_polymarket
- Kalshi edges: market_mentions_company_kalshi
- Both have same edge structure (matched_keywords, confidence)

⚠️ SEMANTIC SEARCH RULES:
- Award descriptions: ✅ Has description_embedding (use COSINE_SIMILARITY)
- SEC content: ❌ NO embeddings (use CONTAINS() text filters + finbert_score instead)
- Polymarket: ❌ NO embeddings (use CONTAINS() on question/description)
- Kalshi: ❌ NO embeddings (use CONTAINS() on title)
- Trigger words for semantic: "related to", "about", "similar to", "involving"
- Concepts that need semantic: "AI", "cybersecurity", "renewable energy", "blockchain"

⚠️ PREDICTION MARKET EDGE USAGE:
- market_mentions_company_polymarket: ✅ Clean (15k edges, use THIS)
- market_related_to_sector_polymarket: ⚠️ Noisy (919k edges, use cautiously)
- market_affects_company_polymarket: ⚠️ Very noisy (119k edges, avoid unless doing macro)
- market_mentions_company_kalshi: ✅ Clean (use for Kalshi)
- market_related_to_sector_kalshi: ⚠️ Use cautiously

⚠️ ALWAYS ADD LIMIT:
- Default: 20
- Semantic: 10
- Time-series: 50
- Prediction markets: 10
- Never omit LIMIT or query may timeout
"""


# """
# prompts.py - Schema descriptions and few-shot examples for AQL query generation
# Last updated: 2026-01-04
# """

# # =============================================================================
# # SCHEMA DESCRIPTION (Based on actual ArangoDB collections)
# # =============================================================================

# SCHEMA_DESCRIPTION = """
# Database: QUANT_v2 (ArangoDB Multi-Model Graph)
# 🚨 CRITICAL: Collection names are CASE-SENSITIVE and SINGULAR:
# - Company (NOT "companies" or "company")
# - MarketData (NOT "marketdata" or "market_data")
# - Award (NOT "awards")
# - EconomicData (NOT "economicdata" or "economic_data")
# - sec_filings (NOT "sec_filing" or "secFilings")
# - sec_sections (NOT "sec_section")
# - sec_sentences (NOT "sec_sentence")
# DOCUMENT COLLECTIONS:

# 1. Company
#    - ticker (string): Stock ticker symbol (e.g., "AAPL", "DG")
#     WARNING: This collection ONLY contains ticker. No name, sector, or other fields.
#     Collection name: "Company" (capital C, singular)
   
# 2. MarketData (daily OHLCV + 40+ technical/fundamental indicators)
#    - ticker (string): Stock ticker
#    - date (string): Format YYYY-MM-DD (e.g., "2016-01-05")
#    - open, high, low, close (float): Price data
#    - volume (int): Trading volume
   
#    Technical Indicators:
#    - sma_5, sma_10, sma_20, sma_50, sma_200 (float): Simple moving averages
#    - ema_12, ema_26 (float): Exponential moving averages
#    - macd, macd_signal, macd_histogram (float): MACD indicator
#    - obv (float): On-balance volume
#    - dist_from_sma20, dist_from_sma50, dist_from_sma200 (float): Distance from SMAs
#    - golden_cross, death_cross (int): 1 if occurred, else 0
#    - above_sma20, above_sma50, above_sma200 (int): 1 if above SMA, else 0
   
#    Fundamental Data:
#    - targetMeanPrice, targetHighPrice, targetLowPrice, targetMedianPrice (float): Analyst targets
#    - recommendationKey (string): "buy", "hold", "sell"
#    - numberOfAnalystOpinions (int): Number of analysts
#    - forwardEps, trailingEps (float): Earnings per share
#    - earningsGrowth, revenueGrowth (float): Growth rates
#    - returnOnEquity, returnOnAssets (float): Profitability metrics
#    - grossMargins, ebitdaMargins, operatingMargins, profitMargins (float): Margin metrics
#    - trailingPE, forwardPE (float): Price-to-earnings ratios
#    - priceToBook, priceToSalesTrailing12Months (float): Valuation ratios
#    - debtToEquity (float): Leverage ratio
#    - totalDebt, totalCash (float): Balance sheet items
#    - currentRatio, quickRatio (float): Liquidity ratios
#    - freeCashflow, operatingCashflow (float): Cash flow metrics
#    - dividendRate, dividendYield (float): Dividend data
#    - beta (float): Volatility vs market
#    - fiftyTwoWeekHigh, fiftyTwoWeekLow (float): 52-week range
   
#    Time Features:
#    - year, month, quarter, day_of_week, day_of_month (int): Date components

# 3. Award (government contracts)
#    - ticker (string): Recipient company ticker
#    - recipient_name (string): Company name (e.g., "3M COMPANY")
#    - matched_sp500_name (string): Standardized company name
#    - start_date (string): Contract start YYYY-MM-DD
#    - award_amount (string): Contract value as string (for display)
#    - award_amount_float (float): Contract value as number (USE THIS for filtering/sorting)
#    - awarding_agency (string): Government agency (e.g., "Department of Defense")
#    - description (string): Award description (full text)
#    - description_embedding (array): Semantic vector (1536 dimensions) for similarity search
#    - contract_year (string): Year as string (e.g., "2017")
#    - source_file (string): Origin file (e.g., "contracts_2017.csv")
#    - ingested_at (string): Timestamp of data ingestion

# 4. EconomicData (macroeconomic indicators)
#     CRITICAL: Field names use underscores, not camelCase!
   
#    - date (string): Date YYYY-MM-DD
#    - ingested_at (string): Timestamp
   
#    Stock Indices:
#    - sandp_500_index (float): S&P 500 value (NOT "sp500"!)
#    - nasdaq_composite (float): NASDAQ value
#    - dow_jones_industrial_average (float): DJIA value
#    - vix_volatility_index (float): VIX value
   
#    Interest Rates:
#    - federal_funds_rate (float): Fed funds rate % (NOT "fed_funds_rate"!)
#    - 2_year_treasury_yield (float): 2Y Treasury %
#    - 10_year_treasury_yield (float): 10Y Treasury %
#    - 30_year_treasury_yield (float): 30Y Treasury %
#    - 10y_2y_treasury_spread (float): Yield curve spread
   
#    Inflation:
#    - consumer_price_index_cpi (float): CPI value
#    - core_cpi_ex_food_and_energy (float): Core CPI
#    - core_pce_feds_preferred (float): Fed's preferred inflation gauge
   
#    Labor Market:
#    - unemployment_rate (float): Unemployment %
#    - nonfarm_payrolls (int): Payroll count
#    - initial_jobless_claims (int): Weekly jobless claims
   
#    Economic Activity:
#    - real_gdp (float): Real GDP
#    - industrial_production (float): Industrial production index
#    - retail_sales (float): Retail sales
#    - consumer_sentiment (float): Consumer sentiment index
   
#    Commodities & Other:
#    - crude_oil_price_wti (float): Oil price (WTI)
#    - m2_money_supply (float): M2 money supply
#    - housing_starts (int): New housing starts
#    - case_shiller_home_price_index (float): Home price index
   
#    Derived Fields:
#    - yield_curve_slope (float): 10Y - 2Y spread
#    - yield_curve_inverted (int): 1 if inverted, else 0

# 5. sec_filings (SEC document metadata)
#    - ticker (string): Company ticker
#    - type (string): Filing type ("10-K", "10-Q", "8-K", etc.)
#    - accession (string): SEC accession number (unique ID)
#    - file_name (string): Source file name
#    - filing_date (string): Date filed YYYY-MM-DD
#    - fiscal_year (int): Fiscal year
   
#    Sentiment Metrics (aggregated from sentences):
#    - avg_finbert (float): Average FinBERT sentiment (-1 to +1)
#    - avg_uncertainty (float): Uncertainty score per 1000 words
#    - avg_positive (float): Positive words per 1000
#    - avg_negative (float): Negative words per 1000
#    - sentence_count (int): Total sentences in filing
   
#     NO CONTENT FIELD: Full text is NOT stored here. Use sec_sections or sec_sentences.

# 6. sec_sections (sections within filings)
#    - filing_id (string): Parent filing ID (format: "sec_filings/{ticker}_{type}_{accession}_{filename}")
#    - section_type (string): Section type (e.g., "Full Document", "Risk Factors", "MD&A")
#    - start_char (int): Start position in original document
#    - length (int): Length in characters
   
#     NO EMBEDDING FIELD: Cannot do semantic search on sections.
#     NO CONTENT FIELD: Text is NOT stored. Use sec_sentences for actual content.

# 7. sec_sentences (individual sentences - most granular level)
#    - section_id (string): Parent section ID (format: "sec_sections/{ticker}_{type}_{accession}_{filename}_sec{N}")
#    - text (string): Sentence text (THIS is where content lives)
#    - n_tokens (int): Token count
   
#    Sentiment Metrics:
#    - finbert_score (float): FinBERT sentiment score (-1 to +1)
#    - finbert_probs (object): Probabilities {positive, negative, neutral}
#    - negative_per_1k (float): Negative words per 1000
#    - positive_per_1k (float): Positive words per 1000
#    - uncertainty_per_1k (float): Uncertainty words per 1000
#    - litigious_per_1k (float): Legal language per 1000
   
#     NO EMBEDDING FIELD: Cannot do vector similarity search.
#     For semantic search, use finbert_score filtering instead of cosine similarity.

# EDGE COLLECTIONS (Graph Relationships):

# 1. HAS_MARKETDATA: Company -> MarketData
#    - date (string): Market date
#    Usage: FOR market IN OUTBOUND company HAS_MARKETDATA

# 2. HAS_AWARD: Company -> Award
#    - award_amount (float): Contract value
#    Usage: FOR award IN OUTBOUND company HAS_AWARD

# 3. HAS_FILING: Company -> sec_filings
#    - filing_date (string): Date filed
#    - filing_type (string): Filing type
#    Usage: FOR filing IN OUTBOUND company HAS_FILING

# 4. has_section: sec_filings -> sec_sections
#    Usage: FOR section IN OUTBOUND filing has_section

# 5. has_sentence: sec_sections -> sec_sentences
#    Usage: FOR sentence IN OUTBOUND section has_sentence

# GRAPHS:
# - QUANT_v2_FinanceGraph: Company + MarketData + Award (financial data)
# - sec_graph: sec_filings + sec_sections + sec_sentences (SEC document hierarchy)

#  CRITICAL LIMITATIONS:
# 1. NO SEMANTIC SEARCH on SEC data (no embeddings in sec_sections or sec_sentences)
# 2. Company collection is MINIMAL (only ticker, no name/sector/industry)
# 3. SEC content is ONLY in sec_sentences.text (not in filings or sections)
# 4. EconomicData field names use snake_case with full names (sandp_500_index, not sp500)
# 5. Award amounts: Use award_amount_float for math, award_amount for display

#  IMPORTANT RULES:
# 1. All dates are strings in YYYY-MM-DD format
# 2. For Award filtering, use award_amount_float (not award_amount)
# 3. For EconomicData, use full field names: sandp_500_index, federal_funds_rate
# 4. For SEC content, query sec_sentences.text (NOT sec_filings.content)
# 5. For SEC sentiment search, filter by finbert_score (no vector similarity)
# 6. Always add LIMIT to prevent timeout (default 20)
# """


# # =============================================================================
# # FEW-SHOT EXAMPLES
# # =============================================================================

# FEW_SHOT_EXAMPLES = """
# EXAMPLE 1 - Market Data Lookup:
# Question: "What was Apple's closing price on 2016-01-05?"
# Intent: single_value_lookup
# Collections: ["MarketData"]
# AQL:
# FOR doc IN MarketData
#   FILTER doc.ticker == @ticker AND doc.date == @date
#   RETURN {date: doc.date, close: doc.close, volume: doc.volume}
# Bind Variables: {"ticker": "AAPL", "date": "2016-01-05"}
# Requires Embedding: false

# ---

# EXAMPLE 2 - Award Lookup (CORRECT field name!):
# Question: "Show me the top 5 largest government awards"
# Intent: ranking
# Collections: ["Award"]
# AQL:
# FOR doc IN Award
#   FILTER doc.award_amount_float != null
#   SORT doc.award_amount_float DESC
#   LIMIT 5
#   RETURN {
#     recipient: doc.recipient_name,
#     ticker: doc.ticker,
#     amount: doc.award_amount_float,
#     agency: doc.awarding_agency,
#     start_date: doc.start_date,
#     description: SUBSTRING(doc.description, 0, 200)
#   }
# Bind Variables: {}
# Requires Embedding: false

# ---

# EXAMPLE 3 - Economic Data (CORRECT field names!):
# Question: "What was the unemployment rate and S&P 500 on 2016-01-04?"
# Intent: single_value_lookup
# Collections: ["EconomicData"]
# AQL:
# FOR doc IN EconomicData
#   FILTER doc.date == @date
#   RETURN {
#     date: doc.date,
#     sandp_500: doc.sandp_500_index,
#     unemployment: doc.unemployment_rate,
#     fed_rate: doc.federal_funds_rate,
#     vix: doc.vix_volatility_index,
#     yield_curve: doc.yield_curve_slope
#   }
# Bind Variables: {"date": "2016-01-04"}
# Requires Embedding: false

# ---

# EXAMPLE 4 - Award Semantic Search:
# Question: "Find awards related to artificial intelligence"
# Intent: semantic_search
# Collections: ["Award"]
# AQL:
# FOR doc IN Award
#   FILTER doc.description_embedding != null
#   LET similarity = COSINE_SIMILARITY(doc.description_embedding, @query_vector)
#   FILTER similarity >= 0.70
#   SORT similarity DESC
#   LIMIT 10
#   RETURN {
#     recipient: doc.recipient_name,
#     ticker: doc.ticker,
#     description: SUBSTRING(doc.description, 0, 300),
#     amount: doc.award_amount_float,
#     start_date: doc.start_date,
#     similarity: similarity
#   }
# Bind Variables: {"query_vector": [0.123, ...]}
# Requires Embedding: true
# Embedding Text: "artificial intelligence AI machine learning deep learning neural networks"

# ---

# EXAMPLE 5 - SEC Filing Sentiment:
# Question: "Show me Apple's most negative 10-K filings"
# Intent: sentiment_analysis
# Collections: ["sec_filings"]
# AQL:
# FOR doc IN sec_filings
#   FILTER doc.ticker == @ticker
#   FILTER doc.type == "10-K"
#   FILTER doc.avg_finbert != null
#   SORT doc.avg_finbert ASC
#   LIMIT 5
#   RETURN {
#     ticker: doc.ticker,
#     filing_date: doc.filing_date,
#     fiscal_year: doc.fiscal_year,
#     sentiment: doc.avg_finbert,
#     negative_score: doc.avg_negative,
#     uncertainty: doc.avg_uncertainty
#   }
# Bind Variables: {"ticker": "AAPL"}
# Requires Embedding: false

# ---

# EXAMPLE 6 - SEC Sentence Search (NO embeddings, use text filter):
# Question: "Find SEC sentences mentioning supply chain risk"
# Intent: text_search
# Collections: ["sec_sentences"]
# AQL:
# FOR doc IN sec_sentences
#   FILTER CONTAINS(LOWER(doc.text), "supply chain")
#   FILTER CONTAINS(LOWER(doc.text), "risk")
#   FILTER doc.finbert_score < -0.3
#   SORT doc.finbert_score ASC
#   LIMIT 10
#   RETURN {
#     text: SUBSTRING(doc.text, 0, 500),
#     sentiment: doc.finbert_score,
#     section_id: doc.section_id,
#     negative_words: doc.negative_per_1k
#   }
# Bind Variables: {}
# Requires Embedding: false

# ---

# EXAMPLE 7 - Graph Traversal (Company -> Awards):
# Question: "Show me awards for ticker DG"
# Intent: graph_traversal
# Collections: ["Company", "Award"]
# Edges: ["HAS_AWARD"]
# AQL:
# FOR company IN Company
#   FILTER company.ticker == @ticker
#   FOR award IN OUTBOUND company HAS_AWARD
#     SORT award.start_date DESC
#     LIMIT 10
#     RETURN {
#       ticker: company.ticker,
#       recipient: award.recipient_name,
#       amount: award.award_amount_float,
#       agency: award.awarding_agency,
#       start_date: award.start_date,
#       description: SUBSTRING(award.description, 0, 200)
#     }
# Bind Variables: {"ticker": "DG"}
# Requires Embedding: false

# ---

# EXAMPLE 8 - Market Data with Indicators:
# Question: "Show me stocks with price above 20-day SMA on 2016-01-05"
# Intent: technical_screening
# Collections: ["MarketData"]
# AQL:
# FOR doc IN MarketData
#   FILTER doc.date == @date
#   FILTER doc.sma_20 != null
#   FILTER doc.above_sma20 == 1
#   SORT doc.close DESC
#   LIMIT 20
#   RETURN {
#     ticker: doc.ticker,
#     close: doc.close,
#     sma_20: doc.sma_20,
#     dist_from_sma20: doc.dist_from_sma20,
#     volume: doc.volume
#   }
# Bind Variables: {"date": "2016-01-05"}
# Requires Embedding: false

# ---

# EXAMPLE 9 - Date Range Query:
# Question: "Show me Tesla's stock prices for January 2016"
# Intent: time_series
# Collections: ["MarketData"]
# AQL:
# FOR doc IN MarketData
#   FILTER doc.ticker == @ticker
#   FILTER doc.date >= @start_date AND doc.date < @end_date
#   SORT doc.date ASC
#   LIMIT 50
#   RETURN {
#     date: doc.date,
#     ticker: doc.ticker,
#     open: doc.open,
#     close: doc.close,
#     volume: doc.volume
#   }
# Bind Variables: {"ticker": "TSLA", "start_date": "2016-01-01", "end_date": "2016-02-01"}
# Requires Embedding: false

# ---

# EXAMPLE 10 - Aggregation:
# Question: "What's the total value of defense awards in 2017?"
# Intent: aggregation
# Collections: ["Award"]
# AQL:
# FOR doc IN Award
#   FILTER doc.contract_year == "2017"
#   FILTER doc.awarding_agency LIKE "%Defense%" OR doc.awarding_agency LIKE "%DoD%"
#   COLLECT AGGREGATE total = SUM(doc.award_amount_float), count = COUNT(1)
#   RETURN {total_amount: total, award_count: count}
# Bind Variables: {}
# Requires Embedding: false

# ---

#  FIELD NAME CHEAT SHEET (Common Mistakes):
# WRONG → CORRECT
# - sp500 → sandp_500_index
# - fed_funds_rate → federal_funds_rate
# - award_amount (for math) → award_amount_float
# - sec_filings.content → sec_sentences.text
# - sec_sections.embedding → (DOESN'T EXIST, use finbert_score filter)

#  SEMANTIC SEARCH RULES:
# - Award descriptions:  Has description_embedding (use COSINE_SIMILARITY)
# - SEC content: NO embeddings (use CONTAINS() text filters + finbert_score instead)
# - Trigger words for semantic: "related to", "about", "similar to", "involving"
# - Concepts that need semantic: "AI", "cybersecurity", "renewable energy", "blockchain"

#  ALWAYS ADD LIMIT:
# - Default: 20, Semantic: 10, Time-series: 50
# - Never omit LIMIT or query may timeout
# """
