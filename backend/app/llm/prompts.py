"""
prompts.py - Schema descriptions and few-shot examples for AQL query generation
Last updated: 2026-01-06 with Kalshi support
"""

# =============================================================================
# CRITICAL AQL RULES (Condensed version - only essential syntax rules)
# =============================================================================

CRITICAL_AQL_RULES = """
⚠️ CRITICAL AQL SYNTAX RULES ⚠️

0. DISAMBIGUATE "MARKETS" (CRITICAL!):
   ✅ "prediction markets" / "betting markets" / "polymarket" / "kalshi" → prediction_markets_polymarket
   ✅ "stock prices" / "OHLCV" / "trading data" / "closing price" → MarketData

   Examples:
   - "prediction markets about Tesla" → Use prediction_markets_polymarket + graph edges
   - "Tesla stock price" → Use MarketData collection
   - "markets that whales are betting on" → Use prediction_markets_polymarket (context: betting/whales)
   - "markets with high volume last week" → AMBIGUOUS - default to MarketData unless "prediction" mentioned

   ⚠️ When user says "markets" without context:
   - If question mentions: whales, betting, prediction, probability, polymarket, kalshi → prediction_markets_polymarket
   - If question mentions: stock, price, OHLCV, technical indicators → MarketData
   - Default: Ask for clarification or prefer prediction_markets_polymarket in whale/trader context

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
   ✅ commodity_positions, futures_prices, options_flow
   ✅ prediction_markets_polymarket, prediction_markets_kalshi
   ✅ polymarket_traders, polymarket_positions
   ✅ eia_crude_inventory, eia_natgas_storage, eia_natgas_production, eia_lng_exports
   ❌ awards, companies, market_data, futures, options

5. CRITICAL FIELD NAMES:
   Award: award_amount_float (for math), start_date, description_embedding (for semantic search)
   Company: sharesOutstanding, marketCap, fullTimeEmployees (camelCase!)
   MarketData: sma_20, sma_50 (snake_case), targetMeanPrice (camelCase)
   EconomicData: sandp_500_index, federal_funds_rate
   SEC: finbert_score, avg_negative, avg_uncertainty (NO embeddings!)
   Polymarket: question, description, yes_probability, volume_24h, closed, question_embedding (for semantic search!)
   Polymarket Traders: total_volume, total_profit, is_whale, activity_level (NO embeddings!)
   Polymarket Positions: market_question, size, average_price, realized_profit, unrealizedProfit
   Kalshi: title, yes_price, volume, status (NO embeddings!)
   Futures: commodity, contract_symbol, sma_20, rsi_14, volatility_30d, daily_return
   Options: call_volume, put_volume, put_call_volume_ratio, iv_rank, call_volume_unusual, potential_call_sweep, unusual_total_activity
   EIA Crude: crude_stocks, crude_stocks_change, cushing_stocks, refinery_utilization
   EIA NatGas Storage: total_stocks, stocks_change, stocks_vs_5yr_pct
   Commodity Positions: Market_and_Exchange_Names (Capital M!), net_noncommercial_position

6. SEMANTIC SEARCH - CRITICAL RULES:
   ✅ Award: HAS description_embedding - use COSINE_SIMILARITY(doc.description_embedding, @query_vector)
   ✅ Polymarket: HAS question_embedding - use COSINE_SIMILARITY(doc.question_embedding, @query_vector)
   ❌ OTHER COLLECTIONS: NO embeddings - use CONTAINS(LOWER(field), 'keyword')
   ❌ NO embeddings: SEC, Kalshi, Futures, Options, EIA, Commodity Positions

   🚨 PERFORMANCE - ALWAYS PRE-FILTER BEFORE COSINE_SIMILARITY:
   ✅ CORRECT (Fast):
   FOR doc IN Award
     FILTER doc.award_amount_float > 500000    ← Pre-filter first!
     FILTER doc.description_embedding != null
     LIMIT 3000                                 ← Limit docs before similarity calc!
     LET sim = COSINE_SIMILARITY(doc.description_embedding, @query_vector)
     FILTER sim >= 0.72
     LIMIT 15

   ❌ WRONG (Times out after 60 seconds):
   FOR doc IN Award
     LET sim = COSINE_SIMILARITY(doc.description_embedding, @query_vector)  ← Scans ALL docs!
     FILTER sim >= 0.70

   ⚠️ Similarity thresholds:
   - Award: >= 0.72 (not 0.70 - higher = faster + better matches)
   - Polymarket: >= 0.68 (not 0.65 - higher = faster)

   ❌ NEVER use embeddings on: SEC, Kalshi, Company, MarketData, EconomicData

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
- futures_prices (NOT "futures" or "cme_futures")
- options_flow (NOT "options" or "options_activity")
- eia_crude_inventory (NOT "crude_inventory" or "eia_crude")
- eia_natgas_storage (NOT "natgas_storage" or "eia_gas")
- eia_natgas_production (NOT "natgas_production")
- eia_lng_exports (NOT "lng_exports")
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

6. futures_prices (CME commodity futures prices)
   - _key (string): Unique ID (format: "{commodity}_{date}")
   - commodity (string): Commodity type (e.g., "CRUDE_OIL", "NATURAL_GAS", "GOLD", "CORN")
   - date (string): Trading date YYYY-MM-DD
   - open, high, low, close (float): OHLCV price data
   - volume (int): Trading volume (contracts)
   - contract_symbol (string): Futures contract symbol (e.g., "CL=F", "NG=F", "GC=F")
   - unit (string): Price unit (e.g., "USD/barrel", "USD/MMBtu", "USD/oz")

   Technical Indicators:
   - sma_20, sma_50 (float): Simple moving averages
   - rsi_14 (float): Relative Strength Index (0-100)
   - volatility_30d (float): 30-day volatility
   - dist_from_52w_high, dist_from_52w_low (float): Distance from 52-week extremes
   - macd, macd_signal (float): MACD indicator

   Momentum & Returns:
   - daily_return, weekly_return, monthly_return (float): Price returns
   - above_sma20, above_sma50 (int): 1 if above SMA, else 0

   ⚠️ Use Case: Commodity price tracking, correlation with CFTC positions, macro trend analysis
   ⚠️ Links to: CFTC positions, EIA inventory data, EconomicData

7. options_flow (Daily options activity for insider trading detection)
   - _key (string): Unique ID (format: "{ticker}_{date}")
   - ticker (string): Stock ticker
   - date (string): Trading date YYYY-MM-DD
   - stock_price (float): Underlying stock price

   Volume Metrics:
   - call_volume, put_volume, total_volume (int): Options volume
   - call_open_interest, put_open_interest, total_open_interest (int): Open interest
   - put_call_volume_ratio, put_call_oi_ratio (float): Put/call ratios

   Implied Volatility:
   - call_iv_avg, put_iv_avg (float): Average IV for near-the-money options
   - iv_rank (float): IV percentile vs 52-week range (0-1)

   Premium Flow:
   - call_premium, put_premium (float): Total premium ($) traded
   - call_contracts, put_contracts (int): Number of contracts

   Unusual Activity Detection (requires 20+ days of history):
   - call_volume_unusual, put_volume_unusual (float): Volume vs 20-day average
   - unusual_total_activity (int): 1 if total volume > 2x average, else 0
   - unusual_call_activity (int): 1 if call volume unusually high
   - unusual_put_activity (int): 1 if put volume unusually high

   Sentiment Signals:
   - bullish_signal (int): 1 if high call volume + low P/C ratio
   - bearish_signal (int): 1 if high put volume + high P/C ratio
   - potential_call_sweep (int): 1 if extreme call buying (>3x avg, P/C < 0.5)
   - potential_put_sweep (int): 1 if extreme put buying (>3x avg, P/C > 2.0)

   ⚠️ Use Case: Detect unusual options activity before contract announcements, insider trading signals
   ⚠️ Links to: Company, MarketData, Award (for pre-announcement activity), sec_filings (for pre-filing activity)

8. eia_crude_inventory (EIA Crude Oil Inventory - Weekly)
   - date (string): Report week ending YYYY-MM-DD
   - crude_stocks (float): Crude oil stocks (million barrels)
   - crude_stocks_change (float): Weekly change (million barrels)
   - cushing_stocks (float): Cushing, OK storage (key delivery point)
   - gasoline_stocks (float): Gasoline stocks (million barrels)
   - distillate_stocks (float): Distillate (diesel) stocks
   - refinery_utilization (float): Refinery utilization rate (%)

   ⚠️ Use Case: Fundamental analysis for crude oil futures, supply/demand analysis
   ⚠️ Links to: futures_prices (CRUDE_OIL)

9. eia_natgas_storage (EIA Natural Gas Storage - Weekly)
   - date (string): Report week ending YYYY-MM-DD
   - total_stocks (float): Natural gas in storage (Bcf - billion cubic feet)
   - stocks_change (float): Weekly injection/withdrawal (Bcf)
   - stocks_vs_5yr_avg (float): Deviation from 5-year average (Bcf)
   - stocks_vs_5yr_pct (float): % vs 5-year average

   ⚠️ Use Case: Natural gas supply analysis, seasonal storage patterns
   ⚠️ Links to: futures_prices (NATURAL_GAS)

10. eia_natgas_production (EIA Natural Gas Production - Monthly)
    - date (string): Month YYYY-MM-DD
    - dry_production (float): Dry natural gas production (Bcf)
    - marketed_production (float): Gross marketed production (Bcf)

    ⚠️ Use Case: Long-term production trends, supply forecasting

11. eia_lng_exports (EIA LNG Exports - Monthly)
    - date (string): Month YYYY-MM-DD
    - lng_exports (float): LNG exports (Bcf)
    - lng_export_terminals (int): Number of active export terminals

    ⚠️ Use Case: Global LNG demand tracking, export capacity analysis

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
   - question_embedding (array[1536]): Semantic embedding of question+description for similarity search
   - fetched_at (string): Data fetch timestamp

   ⚠️ Use Case: Forward-looking sentiment, event probabilities, crowd predictions
   ✅ HAS EMBEDDINGS: Use COSINE_SIMILARITY(doc.question_embedding, @query_vector) for semantic search

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
   - yes_probability (float): Current "Yes" probability (0-1)
   - no_probability (float): Current "No" probability (0-1)
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

10. sec_filings (SEC document metadata - 36,175 total filings)
   - ticker (string): Company ticker
   - type (string): Filing type - 12 types available:
     • "10-K" (4,960 filings) - Annual report with audited financials
     • "10-Q" (5,019 filings) - Quarterly report
     • "8-K" (5,025 filings) - Material events (M&A, earnings, leadership changes)
     • "4" (6,120 filings) - Insider ownership changes (buy/sell transactions) ⚠️ HAS TRADES FIELD
     • "5" (3,593 filings) - Annual insider ownership report ⚠️ HAS TRADES FIELD
     • "6-K" (134 filings) - Foreign issuer current report
     • "S-1" (317 filings) - IPO registration statement
     • "SC 13D" (1,685 filings) - Beneficial ownership >5% with intent to influence
     • "SC 13G" (5,801 filings) - Passive beneficial ownership >5%
     • "13F-HR" (756 filings) - Institutional investor holdings (quarterly)
     • "DEF 14A" (2,469 filings) - Proxy statement (shareholder meetings)
     • "424B4" (301 filings) - Prospectus filed pursuant to Rule 424(b)(4)

   - accession (string): SEC accession number (unique ID)
   - file_name (string): Source file name
   - filing_date (string): Date filed YYYY-MM-DD
   - fiscal_year (int): Fiscal year

   ⚠️ FORM 4/5 ONLY - Insider Transaction Data (CRITICAL!):
   - trades (array of objects): Structured insider transaction data
     • type (string): "non-derivative" (stock) or "derivative" (options)
     • code (string): Transaction code
       - "P" = Purchase (INSIDER BUYING - BULLISH SIGNAL)
       - "S" = Sale (INSIDER SELLING - BEARISH SIGNAL)
       - "F" = Tax withholding (automatic, NOT informed trade)
       - "M" = Exercise of options
       - "A" = Grant/Award (compensation)
     • shares (int): Number of shares (negative = sold, positive = bought)
     • price (float): Transaction price per share
     • post_shares (float): Total shares held AFTER transaction
     • is_informed (bool): true = informed trade (P/S), false = automatic (F/M/A)

   ⚠️ INSIDER BUYING DETECTION:
   FILTER filing.type == "4"
   FILTER filing.trades[? ANY.code == "P"]  ← Purchases only
   FILTER filing.trades[? ANY.is_informed == true]  ← Exclude tax withholding

   ⚠️ INSIDER SELLING DETECTION:
   FILTER filing.type == "4"
   FILTER filing.trades[? ANY.code == "S"]  ← Sales only
   FILTER filing.trades[? ANY.is_informed == true]

   Sentiment Metrics (10-K, 10-Q, 8-K, etc. - aggregated from sentences):
   - avg_finbert (float): Average FinBERT sentiment (-1 to +1)
   - avg_uncertainty (float): Uncertainty score per 1000 words
   - avg_positive (float): Positive words per 1000
   - avg_negative (float): Negative words per 1000
   - sentence_count (int): Total sentences in filing

   ⚠️ NO CONTENT FIELD: Full text is NOT stored here. Use sec_sections or sec_sentences.
   ⚠️ INSIDER TRADING SIGNALS: Form 4/5 have `trades` field with buy/sell data, SC 13D/G show large institutional positions
   ⚠️ IPO/OFFERING DATA: S-1 and 424B4 for new offerings

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

12. POSITION_ON_COMMODITY: commodity_positions -> futures_prices
    - commodity_name (string): Commodity type
    - as_of_date (string): CFTC report date

    Usage: FOR price IN OUTBOUND position POSITION_ON_COMMODITY

    ⚠️ Use Case: Link CFTC trader positions to actual futures prices

13. INVENTORY_AFFECTS_PRICE: eia_crude_inventory -> futures_prices
    - inventory_date (string): EIA report date
    - commodity_type (string): "CRUDE_OIL"

    Usage: FOR price IN OUTBOUND inventory INVENTORY_AFFECTS_PRICE

    ⚠️ Use Case: Analyze crude oil price response to inventory changes

14. STORAGE_AFFECTS_PRICE: eia_natgas_storage -> futures_prices
    - storage_date (string): EIA report date
    - commodity_type (string): "NATURAL_GAS"

    Usage: FOR price IN OUTBOUND storage STORAGE_AFFECTS_PRICE

    ⚠️ Use Case: Analyze natural gas price response to storage levels

15. MACRO_IMPACTS_COMMODITY: EconomicData -> futures_prices
    - economic_date (string): Date
    - commodity_type (string): Affected commodity

    Usage: FOR price IN OUTBOUND econ MACRO_IMPACTS_COMMODITY

    ⚠️ Use Case: Macro correlation analysis (inflation → commodities, dollar → gold)

16. HAS_OPTIONS_ACTIVITY: MarketData -> options_flow
    - date (string): Trading date

    Usage: FOR options IN OUTBOUND market HAS_OPTIONS_ACTIVITY

    ⚠️ Use Case: Link stock price movement to options activity

17. COMPANY_HAS_OPTIONS: Company -> options_flow
    - ticker (string): Stock ticker

    Usage: FOR options IN OUTBOUND company COMPANY_HAS_OPTIONS

    ⚠️ Use Case: Query all options activity for a company

18. OPTIONS_BEFORE_AWARD: options_flow -> Award
    - days_before (int): Days between options activity and award announcement
    - unusual_activity (bool): True if unusual volume detected
    - activity_type (string): "call_sweep" | "put_sweep" | "high_volume"

    Usage: FOR award IN OUTBOUND options OPTIONS_BEFORE_AWARD

    ⚠️ Use Case: INSIDER TRADING DETECTION - Find unusual options activity 1-90 days before contract announcements
    ⚠️ Only created for UNUSUAL activity (volume > 2x average or call/put sweeps)

19. OPTIONS_BEFORE_FILING: options_flow -> sec_filings
    - days_before (int): Days between options activity and filing
    - filing_type (string): Filing type ("8-K", "10-Q", etc.)
    - unusual_activity (bool): True if unusual volume detected

    Usage: FOR filing IN OUTBOUND options OPTIONS_BEFORE_FILING

    ⚠️ Use Case: INSIDER TRADING DETECTION - Find unusual options activity 1-30 days before SEC filings (especially 8-K)
    ⚠️ Only created for UNUSUAL activity before significant filings

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

EXAMPLE 3a - Options Flow (CRITICAL: Use options_flow collection!):
Question: "Show me stocks with unusual call volume"
Intent: options_screening
Collections: ["options_flow"]
AQL:
FOR opt IN options_flow
  FILTER opt.unusual_call_activity == 1 OR opt.call_volume_unusual > 2
  FILTER opt.date >= DATE_SUBTRACT(DATE_NOW(), 7, "day")
  SORT opt.call_volume DESC
  LIMIT 20
  RETURN {
    ticker: opt.ticker,
    date: opt.date,
    call_volume: opt.call_volume,
    unusual_ratio: opt.call_volume_unusual,
    put_call_ratio: opt.put_call_volume_ratio,
    iv: opt.iv_avg,
    stock_price: opt.stock_price
  }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Use options_flow collection (NOT MarketData!)
✅ Filter by unusual_call_activity flag or call_volume_unusual ratio
✅ Options data is only available from 2026-01-24 onwards (Day 1 baseline)
⚠️ Unusual activity flags require 20+ days of data to be reliable
💡 Keywords: "options", "calls", "puts", "sweeps", "unusual activity" → use options_flow

---

EXAMPLE 3b - Futures Prices (CRITICAL: Use UPPERCASE commodity names!):
Question: "Show me recent crude oil futures prices"
Intent: commodity_price_lookup
Collections: ["futures_prices"]
AQL:
FOR doc IN futures_prices
  FILTER doc.commodity == "CRUDE_OIL"
  SORT doc.date DESC
  LIMIT 30
  RETURN {
    date: doc.date,
    commodity: doc.commodity,
    close: doc.close,
    high: doc.high,
    low: doc.low,
    volume: doc.volume,
    rsi: doc.rsi_14,
    volatility: doc.daily_range_pct
  }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ ALWAYS use UPPERCASE commodity names: "CRUDE_OIL", "NATURAL_GAS", "GOLD", "SILVER", "COPPER"
✅ Use futures_prices collection (NOT MarketData with fake tickers!)
✅ Available commodities: CRUDE_OIL, NATURAL_GAS, GOLD, SILVER, COPPER, CORN, WHEAT, SOYBEANS
💡 Keywords: "futures", "commodities", "crude oil", "natural gas", "gold" → use futures_prices

---

EXAMPLE 3c - EIA Energy Data (CRITICAL: Use correct EIA collections!):
Question: "Show me crude oil inventory levels"
Intent: energy_fundamental_analysis
Collections: ["eia_crude_inventory"]
AQL:
FOR doc IN eia_crude_inventory
  FILTER doc.product-name == "Crude Oil"
  SORT doc.report_date DESC
  LIMIT 20
  RETURN {
    date: doc.report_date,
    value: doc.value,
    change: doc.change_from_previous,
    pct_change: doc.pct_change,
    units: doc.units,
    series: doc.series-description
  }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Use eia_crude_inventory for crude oil inventory (NOT MarketData!)
✅ Use eia_natgas_storage for natural gas storage (NOT MarketData with ticker='NATGAS'!)
✅ Use eia_natgas_production for natural gas production
✅ Field names have hyphens: product-name, series-description, area-name
💡 Keywords: "inventory", "storage", "production", "EIA", "crude stocks" → use EIA collections

---

EXAMPLE 4 - Award Semantic Search (OPTIMIZED FOR PERFORMANCE):
Question: "Find awards related to artificial intelligence"
Intent: semantic_search
Collections: ["Award"]
AQL:
FOR doc IN Award
  FILTER doc.award_amount_float > 500000
  FILTER doc.description_embedding != null
  LIMIT 3000
  LET similarity = COSINE_SIMILARITY(doc.description_embedding, @query_vector)
  FILTER similarity >= 0.72
  SORT similarity DESC
  LIMIT 15
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

⚠️ CRITICAL PERFORMANCE OPTIMIZATION:
✅ ALWAYS pre-filter BEFORE COSINE_SIMILARITY to reduce documents scanned
✅ Add LIMIT 3000 after filter, BEFORE similarity calculation (prevents timeout)
✅ Use award_amount_float > 500000 to focus on substantial contracts
✅ Increase similarity threshold to 0.72 (not 0.70) to get better matches
✅ Final LIMIT 15 (not 20) to return focused results

❌ NEVER DO THIS (causes timeout):
FOR doc IN Award
  LET sim = COSINE_SIMILARITY(doc.description_embedding, @query_vector)  ← Scans ALL awards!
  FILTER sim >= 0.70

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

EXAMPLE 7 - Awards by Company Name (CRITICAL: Use CONTAINS pattern!):
Question: "Show me Lockheed Martin's government contracts"
Intent: award_lookup_by_company_name
Collections: ["Award"]
AQL:
FOR doc IN Award
  FILTER CONTAINS(LOWER(doc.recipient_name), 'lockheed')
  SORT doc.award_amount_float DESC
  LIMIT 20
  RETURN {
    recipient: doc.recipient_name,
    amount: doc.award_amount_float,
    agency: doc.awarding_agency,
    start_date: doc.start_date,
    description: SUBSTRING(doc.description, 0, 200)
  }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Use CONTAINS(LOWER(...), 'keyword') for flexible matching
✅ Never use exact == match on recipient_name (names have variations!)
✅ Common variations: "LOCKHEED MARTIN CORP", "LOCKHEED MARTIN CORPORATION", "LOCKHEED MARTIN", etc.
💡 This works better than graph traversal when you don't know the ticker

---

EXAMPLE 7b - Graph Traversal (Company -> Awards):
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

EXAMPLE 9 - Date Range Query (CRITICAL: Month/Year Parsing):
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

⚠️ CRITICAL: "January 2016" → start_date: "2016-01-01", end_date: "2016-02-01" (first day of NEXT month)
⚠️ CRITICAL: "October 2020" → start_date: "2020-10-01", end_date: "2020-11-01"
⚠️ CRITICAL: Always use < (less than) for end_date, not <= (to exclude next month)

⚠️ CRITICAL FIELD NAMES - AVOID CONFUSION:
- MarketData collection: use doc.volume (NOT volume_24h!)
- prediction_markets_polymarket: use market.volume_24h
- prediction_markets_kalshi: use market.volume_24h

---

EXAMPLE 9b - Time Series Summary (For date ranges > 7 days):
Question: "Tesla stock performance during October 2020"
Intent: time_series_summary
Collections: ["MarketData"]
AQL:
LET data = (
  FOR doc IN MarketData
    FILTER doc.ticker == @ticker
    FILTER doc.date >= @start_date AND doc.date < @end_date
    SORT doc.date ASC
    RETURN doc
)

LET first = FIRST(data)
LET last = LAST(data)

RETURN {
  ticker: @ticker,
  period: CONCAT(@start_date, " to ", @end_date),
  trading_days: LENGTH(data),
  opening_price: first.open,
  closing_price: last.close,
  period_high: MAX(data[*].high),
  period_low: MIN(data[*].low),
  price_change: last.close - first.open,
  percent_change: ROUND(((last.close - first.open) / first.open) * 100 * 100) / 100,
  avg_daily_volume: ROUND(AVG(data[*].volume))
}
Bind Variables: {"ticker": "TSLA", "start_date": "2020-10-01", "end_date": "2020-11-01"}
Requires Embedding: false

💡 Use this summary pattern when:
- Date range > 7 days
- User asks for "performance", "how did X do", "summary"
- User doesn't explicitly ask for daily/granular data

💡 Use Example 9 (daily data) when:
- Date range <= 7 days
- User asks for "daily prices", "show me each day", "detailed data"

---

EXAMPLE 9c - Multi-Ticker Comparison (Separate summaries per ticker):
Question: "Compare AAPL's stock trends with MSFT and GOOGL over the past year"
Intent: multi_ticker_comparison
Collections: ["MarketData"]
AQL:
FOR ticker IN ["AAPL", "MSFT", "GOOGL"]
  LET data = (
    FOR doc IN MarketData
      FILTER doc.ticker == ticker
      FILTER doc.date >= @start_date AND doc.date < @end_date
      SORT doc.date ASC
      RETURN doc
  )
  
  LET first = FIRST(data)
  LET last = LAST(data)
  
  RETURN {
    ticker: ticker,
    period: CONCAT(@start_date, " to ", @end_date),
    trading_days: LENGTH(data),
    opening_price: first.open,
    closing_price: last.close,
    period_high: MAX(data[*].high),
    period_low: MIN(data[*].low),
    price_change: last.close - first.open,
    percent_change: ROUND(((last.close - first.open) / first.open) * 100 * 100) / 100,
    avg_daily_volume: ROUND(AVG(data[*].volume))
  }
Bind Variables: {"start_date": "2025-01-01", "end_date": "2026-01-01"}
Requires Embedding: false

💡 This pattern returns 3 separate rows (one per ticker) for easy comparison
💡 Each ticker gets its own summary statistics
💡 Frontend can display as separate cards or comparison table
💡 Use when user says "compare", "versus", "vs", or lists multiple tickers

⚠️ DO NOT interleave data from multiple tickers in a single result set
⚠️ DO NOT use UNION - use FOR loop over ticker array instead

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

EXAMPLE 11 - Companies with Commodity Exposure (Use COMPANY_TRADES_COMMODITY Edge):
Question: "Show me companies with crude oil exposure"
Intent: commodity_exposure
Collections: ["Company", "futures_prices"]
AQL:
FOR company IN Company
  FOR futures IN OUTBOUND company COMPANY_TRADES_COMMODITY
    FILTER CONTAINS(LOWER(futures.commodity_type), "crude") OR futures.commodity_type == "CRUDE_OIL"
    SORT futures.date DESC
    LIMIT 1
    RETURN DISTINCT {
      ticker: company.ticker,
      company_name: company.company,
      commodity: futures.commodity_type,
      latest_price: futures.close,
      latest_date: futures.date,
      sector: company.sector
    }
Bind Variables: {}
Requires Embedding: false
Why This Works: Uses COMPANY_TRADES_COMMODITY edge to find companies linked to commodity futures.
Companies with exposure: XOM, CVX, COP (crude oil), FCX, NEM (copper/gold), ADM (agriculture)

---

EXAMPLE 11b - Mining Companies with Multiple Commodity Exposure:
Question: "Find mining companies with exposure to gold and copper"
Intent: multi_commodity_exposure
Collections: ["Company", "futures_prices"]
AQL:
FOR company IN Company
  FILTER company.sector == "Materials" OR company.industry IN ["Metals & Mining", "Gold"]
  LET commodities = (
    FOR futures IN OUTBOUND company COMPANY_TRADES_COMMODITY
      FILTER futures.commodity_type IN ["GOLD", "COPPER"]
      RETURN DISTINCT futures.commodity_type
  )
  FILTER LENGTH(commodities) >= 2
  RETURN {
    ticker: company.ticker,
    company: company.company,
    sector: company.sector,
    industry: company.industry,
    commodities: commodities,
    market_cap: company.marketCap
  }
Bind Variables: {}
Requires Embedding: false
Why This Works: Filters for Materials sector, checks COMPANY_TRADES_COMMODITY edges for GOLD and COPPER.
Known mining companies: FCX (Freeport-McMoRan - copper/gold), NEM (Newmont - gold), GOLD (Barrick - gold)

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

EXAMPLE 12c - Polymarket Semantic Search (OPTIMIZED FOR PERFORMANCE):
Question: "Find prediction markets about artificial intelligence and technology"
Intent: semantic_prediction_market_search
Collections: ["prediction_markets_polymarket"]
AQL:
FOR market IN prediction_markets_polymarket
  FILTER market.closed == false
  FILTER market.question_embedding != null
  LIMIT 2000
  LET similarity = COSINE_SIMILARITY(market.question_embedding, @query_vector)
  FILTER similarity >= 0.68
  SORT similarity DESC
  LIMIT 12
  RETURN {
    question: market.question,
    yes_probability: market.yes_probability,
    volume_24h: market.volume_24h,
    liquidity: market.liquidity,
    similarity: similarity,
    category: market.category
  }
Bind Variables: {"query_vector": [0.123, ...]}
Requires Embedding: true
Embedding Text: "artificial intelligence AI technology machine learning GPT robots automation software"

⚠️ CRITICAL PERFORMANCE OPTIMIZATION:
✅ Pre-filter closed markets FIRST (reduces docs from ~thousands to hundreds)
✅ Add LIMIT 2000 BEFORE COSINE_SIMILARITY (prevents timeout on large collections)
✅ Increased threshold to 0.68 (from 0.65) for better quality matches
✅ Final LIMIT 12 to return focused results

💡 Strategy:
- Use semantic search when user asks about concepts/topics (not specific keywords)
- Filter out closed markets unless user explicitly wants historical data
- Returns similarity score for transparency

⚠️ When to use semantic vs keyword:
- Semantic: "markets about AI", "predictions related to climate change", "markets similar to..."
- Keyword: "markets mentioning Tesla", "Trump election markets" (specific entity names)

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

---

EXAMPLE 16b - Comprehensive Company Workup (WITH SEC FILINGS + TOP SENTENCES):
Question: "Show me a complete analysis of AAPL" OR "Tell me about Tesla" OR "Show me NVDA"
Intent: company_comprehensive_workup
Collections: ["Company", "MarketData", "sec_filings", "sec_sections", "sec_sentences", "Award", "options_flow"]
Edges: ["HAS_MARKETDATA", "HAS_FILING", "has_section", "has_sentence", "HAS_AWARD", "COMPANY_HAS_OPTIONS"]
AQL:
FOR company IN Company
  FILTER company.ticker == @ticker
  LIMIT 1

  LET market_data = (
    FOR m IN OUTBOUND company HAS_MARKETDATA
      SORT m.date DESC
      LIMIT 365
      RETURN m
  )

  LET sec_filings = (
    FOR filing IN OUTBOUND company HAS_FILING
      SORT filing.filing_date DESC
      LIMIT 20
      LET top_sentences = (
        FOR section IN OUTBOUND filing has_section
          FOR sentence IN OUTBOUND section has_sentence
            FILTER sentence.finbertscore != null
            SORT ABS(sentence.finbertscore) DESC
            LIMIT 5
            RETURN {
              text: sentence.text,
              score: sentence.finbertscore
            }
      )
      RETURN MERGE(filing, { top_sentences: top_sentences })
  )

  LET awards = (
    FOR award IN OUTBOUND company HAS_AWARD
      SORT award.start_date DESC
      LIMIT 20
      RETURN award
  )

  LET options_flow = (
    FOR opt IN OUTBOUND company COMPANY_HAS_OPTIONS
      SORT opt.date DESC
      LIMIT 20
      RETURN opt
  )

  RETURN MERGE(company, {
    MarketData: market_data,
    sec_filings: sec_filings,
    Award: awards,
    options_flow: options_flow
  })
Bind Variables: {"ticker": "AAPL"}
Requires_Embedding: false

Strategy:
✅ CRITICAL: Use LET subqueries to fetch related data (NOT nested FOR loops in RETURN)
✅ For SEC filings: Fetch top_sentences using nested traversal (filing -> section -> sentence)
✅ Sort sentences by ABS(finbertscore) to get most extreme sentiment
✅ Use MERGE() to combine company data with nested collections
✅ This query structure matches what CompanyWorkup component expects
⚠️ This query enables full company overview with clickable SEC filings that show sentiment excerpts

---

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

EXAMPLE: DISAMBIGUATION - "Markets" with Company Context
Question: "Show me Tesla-related markets that whales are betting on"
Intent: graph_traversal_prediction_markets (NOT stock market data!)
Collections: ["Company", "prediction_markets_polymarket", "polymarket_traders", "polymarket_positions"]
AQL:
FOR company IN Company
  FILTER company.ticker == @ticker
  FOR market IN INBOUND company market_mentions_company_polymarket
    FILTER market.closed == false
    FOR position IN INBOUND market position_in_market
      FOR trader IN INBOUND position trader_has_position
        FILTER trader.is_whale == true
        SORT position.size DESC
        LIMIT 15
        RETURN DISTINCT {
          market_question: market.question,
          yes_probability: market.yes_probability,
          volume_24h: market.volume_24h,
          whale_position_size: position.size,
          outcome: position.outcome_index == 1 ? "Yes" : "No",
          trader_volume: trader.total_volume
        }
Bind Variables: {"ticker": "TSLA"}
Requires Embedding: false

Strategy:
⚠️ CRITICAL: "markets" in whale/betting context = prediction_markets_polymarket (NOT MarketData!)
✅ Graph path: Company -> prediction markets -> positions -> traders
✅ INBOUND traversal: market -> position -> trader (backwards through edges)
✅ Filter is_whale to get only whale traders
❌ WRONG: Using MarketData collection for this query (that's stock OHLCV data)

---

EXAMPLE 17 - Futures Prices with CFTC Positions:
Question: "Show me crude oil futures prices when speculators had large long positions"
Intent: commodity_correlation_analysis
Collections: ["commodity_positions", "futures_prices"]
Edges: ["POSITION_ON_COMMODITY"]
AQL:
FOR position IN commodity_positions
  FILTER CONTAINS(LOWER(position.Market_and_Exchange_Names), "crude oil")
  FILTER position.net_noncommercial_position > 100000
  FOR price IN OUTBOUND position POSITION_ON_COMMODITY
    FILTER price.commodity == "CRUDE_OIL"
    SORT price.date DESC
    LIMIT 20
    RETURN {
      date: price.date,
      crude_price: price.close,
      net_spec_position: position.net_noncommercial_position,
      open_interest: position.open_interest,
      price_change_pct: price.daily_return,
      rsi: price.rsi_14
    }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Use graph edge POSITION_ON_COMMODITY to link CFTC data to prices
✅ Filter positions first (smaller dataset), then traverse to prices
✅ Combines trader positioning with actual price movement
⚠️ CFTC data is weekly, futures prices are daily

---

EXAMPLE 18 - Crude Oil Inventory Impact on Prices:
Question: "How do crude oil prices respond to large inventory builds?"
Intent: fundamental_price_analysis
Collections: ["eia_crude_inventory", "futures_prices"]
Edges: ["INVENTORY_AFFECTS_PRICE"]
AQL:
FOR inventory IN eia_crude_inventory
  FILTER inventory.crude_stocks_change > 5
  FOR price IN OUTBOUND inventory INVENTORY_AFFECTS_PRICE
    FILTER price.commodity == "CRUDE_OIL"
    SORT inventory.date DESC
    LIMIT 15
    RETURN {
      report_date: inventory.date,
      inventory_build: inventory.crude_stocks_change,
      crude_price: price.close,
      price_change_pct: price.weekly_return,
      total_stocks: inventory.crude_stocks,
      cushing_stocks: inventory.cushing_stocks
    }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Filter for significant inventory builds (>5 million barrels)
✅ Traverse to corresponding futures prices
✅ Track price response to supply changes
💡 Typical pattern: Large builds → price decline (bearish for oil)

---

EXAMPLE 19 - Unusual Options Activity (Insider Trading Signals):
Question: "Show me stocks with unusual call buying yesterday"
Intent: options_screening
Collections: ["options_flow"]
AQL:
FOR opt IN options_flow
  FILTER opt.date >= DATE_SUBTRACT(DATE_NOW(), 2, "day")
  FILTER opt.potential_call_sweep == 1 OR opt.unusual_call_activity == 1
  FILTER opt.total_volume > 1000
  SORT opt.call_volume_unusual DESC
  LIMIT 20
  RETURN {
    ticker: opt.ticker,
    date: opt.date,
    stock_price: opt.stock_price,
    call_volume: opt.call_volume,
    unusual_ratio: opt.call_volume_unusual,
    put_call_ratio: opt.put_call_volume_ratio,
    call_premium: opt.call_premium,
    iv_rank: opt.iv_rank,
    signal: opt.potential_call_sweep == 1 ? "🚨 CALL SWEEP" : "⚠️ HIGH CALL VOLUME"
  }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Filter by unusual activity flags (pre-calculated in pipeline)
✅ potential_call_sweep = extreme buying (>3x average, P/C < 0.5)
✅ unusual_call_activity = high call volume (>2x average)
⚠️ Requires 20+ days of data for baseline averages
💡 Call sweeps before earnings/contracts may indicate insider knowledge

---

EXAMPLE 20 - Options Activity Before Contract Announcements (Insider Trading):
Question: "Find unusual options activity before defense contract awards"
Intent: insider_trading_detection
Collections: ["Company", "options_flow", "Award"]
Edges: ["COMPANY_HAS_OPTIONS", "OPTIONS_BEFORE_AWARD"]
AQL:
FOR company IN Company
  FILTER company.ticker IN ["LMT", "RTX", "NOC", "BA", "GD"]
  FOR options IN OUTBOUND company COMPANY_HAS_OPTIONS
    FOR award IN OUTBOUND options OPTIONS_BEFORE_AWARD
      FILTER award.award_amount_float > 10000000
      FILTER award.awarding_agency LIKE "%Defense%"
      SORT award.start_date DESC
      LIMIT 15
      RETURN {
        ticker: company.ticker,
        options_date: options.date,
        award_date: award.start_date,
        days_before: DATE_DIFF(award.start_date, options.date, "day"),
        award_amount: award.award_amount_float,
        call_volume: options.call_volume,
        unusual_ratio: options.call_volume_unusual,
        put_call_ratio: options.put_call_volume_ratio,
        activity_type: options.potential_call_sweep == 1 ? "Call Sweep" : "High Volume",
        award_description: SUBSTRING(award.description, 0, 200)
      }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Uses OPTIONS_BEFORE_AWARD edge (only created for unusual activity)
✅ Filters defense contractors and large awards (>$10M)
✅ Shows days_before to identify timing patterns
⚠️ Edge only created for activity 1-90 days before award announcement
💡 Pattern: Unusual call buying 30-60 days before major contract = potential insider tip

---

EXAMPLE 21 - Options Activity Before SEC Filings:
Question: "Show me unusual options activity before 8-K filings"
Intent: insider_trading_detection
Collections: ["options_flow", "sec_filings"]
Edges: ["OPTIONS_BEFORE_FILING"]
AQL:
FOR options IN options_flow
  FILTER options.potential_call_sweep == 1 OR options.potential_put_sweep == 1
  FOR filing IN OUTBOUND options OPTIONS_BEFORE_FILING
    FILTER filing.type == "8-K"
    SORT filing.filing_date DESC
    LIMIT 20
    RETURN {
      ticker: options.ticker,
      options_date: options.date,
      filing_date: filing.filing_date,
      days_before: DATE_DIFF(filing.filing_date, options.date, "day"),
      filing_type: filing.type,
      sentiment: filing.avg_finbert,
      call_volume: options.call_volume,
      put_volume: options.put_volume,
      unusual_call: options.call_volume_unusual,
      unusual_put: options.put_volume_unusual,
      signal: options.potential_call_sweep == 1 ? "🟢 Call Sweep" : "🔴 Put Sweep"
    }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Uses OPTIONS_BEFORE_FILING edge (only unusual activity)
✅ Focuses on 8-K filings (material events, M&A, earnings)
✅ Call sweeps before positive news, put sweeps before negative
⚠️ Edge only created for activity 1-30 days before filing
💡 Call sweep + positive 8-K sentiment = potential insider trading

---

EXAMPLE 22 - Natural Gas Storage vs Prices:
Question: "Show me natural gas prices when storage was below 5-year average"
Intent: commodity_fundamental_analysis
Collections: ["eia_natgas_storage", "futures_prices"]
Edges: ["STORAGE_AFFECTS_PRICE"]
AQL:
FOR storage IN eia_natgas_storage
  FILTER storage.stocks_vs_5yr_pct < -10
  FOR price IN OUTBOUND storage STORAGE_AFFECTS_PRICE
    FILTER price.commodity == "NATURAL_GAS"
    SORT storage.date DESC
    LIMIT 20
    RETURN {
      date: storage.date,
      natgas_price: price.close,
      total_stocks: storage.total_stocks,
      vs_5yr_avg: storage.stocks_vs_5yr_pct,
      weekly_change: storage.stocks_change,
      price_change: price.weekly_return
    }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Filter for low storage (<-10% vs 5-year average)
✅ Link to natural gas futures prices
💡 Pattern: Low storage → higher prices (supply tightness)
💡 Seasonal: Injections (summer), withdrawals (winter)

---

EXAMPLE 23 - Multi-Commodity Price Comparison:
Question: "Compare crude oil and gold prices over the last 90 days"
Intent: multi_commodity_analysis
Collections: ["futures_prices"]
AQL:
LET start_date = DATE_SUBTRACT(DATE_NOW(), 90, "day")

FOR commodity IN ["CRUDE_OIL", "GOLD"]
  LET prices = (
    FOR price IN futures_prices
      FILTER price.commodity == commodity
      FILTER price.date >= start_date
      SORT price.date ASC
      RETURN price
  )

  LET first = FIRST(prices)
  LET last = LAST(prices)

  RETURN {
    commodity: commodity,
    current_price: last.close,
    starting_price: first.close,
    change_pct: FLOOR(((last.close - first.close) / first.close) * 100 * 100) / 100,
    high_90d: MAX(prices[*].high),
    low_90d: MIN(prices[*].low),
    avg_volume: ROUND(AVG(prices[*].volume)),
    volatility: FIRST(prices).volatility_30d,
    rsi: last.rsi_14
  }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Loop through commodity array for separate summaries
✅ Calculate 90-day performance for each
✅ Includes technical indicators (RSI, volatility)
💡 Use for flight-to-safety analysis (gold up when oil down)

---

EXAMPLE 24 - Company Options + Stock Price + Awards (Multi-Source):
Question: "Show me RTX's options activity around recent contract awards"
Intent: multi_source_insider_analysis
Collections: ["Company", "options_flow", "MarketData", "Award"]
Edges: ["COMPANY_HAS_OPTIONS", "OPTIONS_BEFORE_AWARD", "HAS_MARKETDATA"]
AQL:
FOR company IN Company
  FILTER company.ticker == @ticker

  FOR options IN OUTBOUND company COMPANY_HAS_OPTIONS
    FILTER options.date >= DATE_SUBTRACT(DATE_NOW(), 180, "day")
    FILTER options.unusual_total_activity == 1

    LET market = FIRST(
      FOR m IN OUTBOUND company HAS_MARKETDATA
        FILTER m.date == options.date
        RETURN m
    )

    LET awards = (
      FOR award IN OUTBOUND options OPTIONS_BEFORE_AWARD
        FILTER award.award_amount_float > 5000000
        RETURN award
    )

    FILTER LENGTH(awards) > 0

    SORT options.date DESC
    LIMIT 10

    RETURN {
      date: options.date,
      stock_price: market.close,
      stock_change: market.close - market.open,
      options_volume: options.total_volume,
      unusual_ratio: options.call_volume_unusual,
      put_call: options.put_call_volume_ratio,
      awards_announced: LENGTH(awards),
      total_award_value: SUM(awards[*].award_amount_float),
      days_until_awards: awards[0].start_date
    }
Bind Variables: {"ticker": "RTX"}
Requires Embedding: false

Strategy:
✅ Combines options, stock prices, and awards for single ticker
✅ Filters for unusual activity only
✅ Uses subquery for awards (may be multiple per options date)
✅ Shows complete picture: options spike → award announcement
💡 Pattern detection: Unusual buying 30-90 days before multi-million dollar contracts

---

EXAMPLE 25 - Insider Buying/Selling (Form 4 Filings):
Question: "Show me recent insider buying for tech companies"
Intent: insider_trading_analysis
Collections: ["sec_filings", "Company"]
Edges: ["HAS_FILING"]
AQL:
FOR filing IN sec_filings
  FILTER filing.type == "4"
  FILTER filing.filing_date >= DATE_SUBTRACT(DATE_NOW(), 90, "day")
  FILTER filing.trades != null

  LET purchases = filing.trades[* FILTER CURRENT.code == "P" AND CURRENT.is_informed == true]
  FILTER LENGTH(purchases) > 0

  FOR company IN Company
    FILTER company.ticker == filing.ticker
    FILTER company.sector == "Technology"

    LET total_shares_bought = SUM(purchases[*].shares)
    LET avg_price = AVG(purchases[*].price)
    LET total_value = SUM(purchases[*].shares * purchases[*].price)

    SORT filing.filing_date DESC
    LIMIT 20

    RETURN {
      ticker: filing.ticker,
      company_name: company.company,
      filing_date: filing.filing_date,
      shares_bought: total_shares_bought,
      avg_price: avg_price,
      total_value: total_value,
      num_transactions: LENGTH(purchases),
      accession: filing.accession
    }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Form 4 = Insider ownership changes with structured `trades` data
✅ Filter for code == "P" (Purchase) to find insider BUYING
✅ Filter is_informed == true to exclude automatic tax withholdings (code "F")
✅ Calculate total shares, average price, total dollar value
💡 Correlation opportunity: Check if unusual options activity preceded this Form 4 filing
⚠️ Form 4 filed within 2 business days of transaction
⚠️ Large insider purchases (>$100k) are strong bullish signals

---

EXAMPLE 26 - Institutional Holdings (13F-HR Filings):
Question: "What are the most recent 13F filings showing large tech holdings?"
Intent: institutional_holdings_analysis
Collections: ["sec_filings", "sec_sentences"]
Edges: ["has_section", "has_sentence"]
AQL:
FOR filing IN sec_filings
  FILTER filing.type == "13F-HR"
  FILTER filing.filing_date >= DATE_SUBTRACT(DATE_NOW(), 120, "day")

  FOR section IN 1..1 OUTBOUND filing has_section
    FOR sentence IN 1..1 OUTBOUND section has_sentence
      FILTER CONTAINS(LOWER(sentence.text), "apple")
        OR CONTAINS(LOWER(sentence.text), "microsoft")
        OR CONTAINS(LOWER(sentence.text), "nvidia")

      COLLECT
        ticker = filing.ticker,
        filing_date = filing.filing_date,
        accession = filing.accession
      INTO sentences = sentence.text

      SORT filing_date DESC
      LIMIT 10

      RETURN {
        institution_ticker: ticker,
        filing_date: filing_date,
        accession: accession,
        mentions: LENGTH(sentences),
        sample_text: sentences[0]
      }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ 13F-HR = Quarterly institutional holdings reports (hedge funds, asset managers)
✅ Filed by institutions managing >$100M in assets
✅ Shows what "smart money" is holding (Buffett's Berkshire, Bridgewater, etc.)
💡 Cross-reference: Compare 13F positions with prediction market sentiment
⚠️ 13F filed 45 days after quarter end (Q4 2025 filings due mid-Feb 2026)

---

EXAMPLE 27 - Insider Buying with Options Correlation:
Question: "Find stocks where insiders bought shares and there was unusual call buying beforehand"
Intent: advanced_insider_trading_detection
Collections: ["sec_filings", "options_flow", "Company"]
Edges: ["COMPANY_HAS_OPTIONS"]
AQL:
FOR filing IN sec_filings
  FILTER filing.type == "4"
  FILTER filing.filing_date >= DATE_SUBTRACT(DATE_NOW(), 90, "day")
  FILTER filing.trades != null

  LET purchases = filing.trades[* FILTER CURRENT.code == "P" AND CURRENT.is_informed == true]
  FILTER LENGTH(purchases) > 0

  FOR company IN Company
    FILTER company.ticker == filing.ticker

    FOR options IN OUTBOUND company COMPANY_HAS_OPTIONS
      FILTER options.date >= DATE_SUBTRACT(filing.filing_date, 30, "day")
      FILTER options.date < filing.filing_date
      FILTER options.unusual_call_activity == 1

      LET total_shares_bought = SUM(purchases[*].shares)
      LET total_purchase_value = SUM(purchases[*].shares * purchases[*].price)
      LET days_before_filing = DATE_DIFF(options.date, filing.filing_date, "day")

      SORT filing.filing_date DESC
      LIMIT 10

      RETURN {
        ticker: filing.ticker,
        company_name: company.company,
        sector: company.sector,
        insider_filing_date: filing.filing_date,
        shares_bought: total_shares_bought,
        purchase_value: total_purchase_value,
        options_date: options.date,
        days_before_insider: ABS(days_before_filing),
        call_volume: options.call_volume,
        call_unusual_ratio: options.call_volume_unusual,
        put_call_ratio: options.put_call_volume_ratio,
        potential_call_sweep: options.potential_call_sweep,
        signal_strength: "STRONG"
      }
Bind Variables: {}
Requires Embedding: false

Strategy:
✅ Find Form 4 with insider purchases (code == "P")
✅ Look back 30 days for unusual call buying BEFORE insider filing
✅ Unusual call activity + subsequent insider buying = potential insider knowledge
💡 STRONG SIGNAL: Insider buys shares after unusual call buying (likely same person/connected)
💡 Timeframe: Options activity typically 1-30 days before Form 4 filing
⚠️ This pattern suggests informed trading (bullish)
⚠️ Cross-reference with 8-K filings for material events announced after

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
- futures → futures_prices
- options → options_flow
- eia_crude → eia_crude_inventory
- eia_natgas → eia_natgas_storage (for storage data)
- eia_gas_production → eia_natgas_production

⚠️ CRITICAL: COMMODITY NAMES (futures_prices.commodity field)
ALWAYS use UPPERCASE with UNDERSCORES:
- "crude oil" → "CRUDE_OIL" (UPPERCASE!)
- "natural gas" → "NATURAL_GAS" (UPPERCASE!)
- "gold" → "GOLD"
- "silver" → "SILVER"
- "copper" → "COPPER"
- "corn" → "CORN"
- "wheat" → "WHEAT"
- "soybeans" → "SOYBEANS"

⚠️ CRITICAL: OPTIONS vs MARKET DATA
When user asks about OPTIONS activity, use options_flow collection:
- "unusual call volume" → options_flow (NOT MarketData!)
- "put/call ratio" → options_flow.put_call_volume_ratio
- "option sweeps" → options_flow.potential_call_sweep
- "implied volatility" → options_flow.iv_avg
- "options activity" → options_flow (NOT MarketData!)

⚠️ CRITICAL: EIA DATA COLLECTIONS
When user asks about energy inventory/storage:
- "crude oil inventory" → eia_crude_inventory (NOT MarketData!)
- "natural gas storage" → eia_natgas_storage (NOT MarketData!)
- "natural gas production" → eia_natgas_production
- "LNG exports" → eia_lng_exports
NEVER use MarketData with ticker='NATGAS' or ticker='CRUDE' - these don't exist!

⚠️ CRITICAL: RECIPIENT NAME VARIATIONS (Award collection)
Common company name variations in Award.recipient_name:
- "Lockheed Martin" search → Use CONTAINS(LOWER(doc.recipient_name), 'lockheed')
- Never use exact match like == "LOCKHEED MARTIN CORPORATION" (may not match!)
- Recipient names have variations: "LOCKHEED MARTIN CORP", "LOCKHEED MARTIN CORPORATION", etc.

⚠️ SEC FORM TYPES (Use Case Guide):

sec_filings.type values and when to use them:
- "10-K" → Annual reports, full financials, risk factors, strategy (most comprehensive)
- "10-Q" → Quarterly updates, interim financials
- "8-K" → Material events (M&A, earnings releases, CEO changes, lawsuits)
- "4" → Insider buy/sell transactions (detect insider sentiment, use with options_flow for correlation)
- "5" → Annual insider holdings summary
- "SC 13D" → Activist investor positions >5% (intent to influence company)
- "SC 13G" → Passive institutional positions >5% (no activist intent)
- "13F-HR" → Hedge fund/institutional holdings (quarterly snapshots, e.g., Buffett's Berkshire)
- "S-1" → IPO filings (pre-IPO financials, risk factors, use case)
- "6-K" → Foreign company reports (non-US headquarters)
- "DEF 14A" → Proxy statements (executive comp, shareholder proposals)
- "424B4" → Final prospectus for offerings

⚠️ INSIDER TRADING DETECTION STRATEGY:
1. Form 4 trades field → Parse structured buy/sell data (code "P" = buy, "S" = sell)
2. Filter is_informed == true → Exclude automatic tax withholdings
3. Correlate with options_flow → Unusual call buying 1-30 days before insider buying = STRONG SIGNAL
4. Correlate with 8-K filings → Material events announced shortly after insider activity
5. SC 13D filings → Activist campaigns, often followed by stock price movement
6. Calculate purchase value: SUM(trades[*].shares * trades[*].price)

⚠️ FORM 4/5 TRADES FIELD SYNTAX:
- Get all purchases: filing.trades[* FILTER CURRENT.code == "P"]
- Get all sales: filing.trades[* FILTER CURRENT.code == "S"]
- Get informed trades only: filing.trades[* FILTER CURRENT.is_informed == true]
- Total shares bought: SUM(filing.trades[*].shares)
- Average price: AVG(filing.trades[*].price)
- Total value: SUM(filing.trades[*].shares * filing.trades[*].price)

⚠️ NEW COLLECTIONS FIELD NAMES:

futures_prices:
- commodity (string): "CRUDE_OIL", "NATURAL_GAS", "GOLD", "CORN", etc.
- contract_symbol (string): "CL=F", "NG=F", "GC=F" (Yahoo Finance symbols)
- volume (int): Trading volume in contracts (NOT volume_24h)
- Technical: sma_20, rsi_14, volatility_30d, macd

options_flow:
- call_volume, put_volume, total_volume (int): Options volume
- put_call_volume_ratio, put_call_oi_ratio (float): Ratios
- call_iv_avg, put_iv_avg (float): Implied volatility
- iv_rank (float): IV percentile (0-1)
- call_volume_unusual, put_volume_unusual (float): vs 20-day average
- potential_call_sweep, potential_put_sweep (int): 1 or 0
- unusual_total_activity (int): 1 if volume > 2x average

EIA Collections:
- eia_crude_inventory: crude_stocks, crude_stocks_change, cushing_stocks, refinery_utilization
- eia_natgas_storage: total_stocks, stocks_change, stocks_vs_5yr_avg, stocks_vs_5yr_pct
- eia_natgas_production: dry_production, marketed_production
- eia_lng_exports: lng_exports, lng_export_terminals

⚠️ TERMINOLOGY DISAMBIGUATION (CRITICAL!):

"MARKETS" CAN MEAN TWO DIFFERENT THINGS:
1. MarketData = Stock market data (OHLCV, technical indicators, fundamentals)
   - Questions: "stock price", "closing price", "trading volume" (stock context), "SMA", "MACD"

2. prediction_markets_polymarket = Prediction/betting markets (Polymarket, Kalshi)
   - Questions: "prediction markets", "betting", "whales", "probability", "polymarket", "kalshi"

CONTEXT CLUES:
- "Tesla stock price" → MarketData
- "Tesla prediction markets" → prediction_markets_polymarket
- "markets that whales bet on" → prediction_markets_polymarket (whale = prediction market trader)
- "markets with high trading volume" → AMBIGUOUS! Check for context:
  - If question mentions whales/betting → prediction_markets_polymarket
  - If question mentions stock/price → MarketData

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
- Polymarket questions: ✅ Has question_embedding (use COSINE_SIMILARITY)
- SEC content: ❌ NO embeddings (use CONTAINS() text filters + finbert_score instead)
- Kalshi: ❌ NO embeddings (use CONTAINS() on title)
- Futures prices: ❌ NO embeddings (use CONTAINS() on commodity field or exact match)
- Options flow: ❌ NO embeddings (filter by ticker, date, or unusual activity flags)
- EIA data: ❌ NO embeddings (use date/value filters)
- Commodity positions: ❌ NO embeddings (use CONTAINS() on Market_and_Exchange_Names)
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
