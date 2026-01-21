
import { Node, Edge } from '@xyflow/react'

export interface SchemaNode {
    name: string
    collection: string
    description: string
    keyFields: string[] // Renaming to 'fields' might break things, so we'll keep 'keyFields' but fill it with all fields
    // New: Graph connections with explicit direction and edge collection
    connections: {
        target: string // key in GRAPH_SCHEMA
        edge: string // edge collection name
        direction: 'INBOUND' | 'OUTBOUND'
        type: 'direct' | 'multi_hop'
        hops?: string[] // For multi-hop (e.g. SEC)
    }[]
    exampleQuery: string
}

export const GRAPH_SCHEMA: Record<string, SchemaNode> = {
    company: {
        name: 'Companies',
        collection: 'Company',
        description: 'S&P 500 Companies',
        keyFields: [
            "cik", "city", "company", "country", "fullTimeEmployees", "industry", "lastUpdated",
            "marketCap", "recordCount", "sector", "sharesOutstanding", "sp500_member", "ticker", "website"
        ],
        connections: [
            { target: 'marketdata', edge: 'HAS_MARKETDATA', direction: 'OUTBOUND', type: 'direct' },
            { target: 'awards', edge: 'HAS_AWARD', direction: 'OUTBOUND', type: 'direct' },
            { target: 'sec', edge: 'HAS_FILING', direction: 'OUTBOUND', type: 'direct' },
            { target: 'predictionmarkets', edge: 'market_mentions_company_polymarket', direction: 'INBOUND', type: 'direct' },
            { target: 'kalshi', edge: 'market_mentions_company_kalshi', direction: 'INBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR c IN Company FILTER c.ticker == "AAPL" RETURN c'
    },
    marketdata: {
        name: 'Market Data',
        collection: 'MarketData',
        description: 'Daily OHLCV & Indicators',
        keyFields: [
            "above_sma20", "above_sma200", "above_sma50", "beta", "calculated_div_yield", "close", "currentRatio",
            "daily_return", "date", "day_of_month", "day_of_week", "death_cross", "debtToEquity", "dist_from_sma20",
            "dist_from_sma200", "dist_from_sma50", "dividendRate", "dividendYield", "earningsGrowth", "earningsQuarterlyGrowth",
            "ebitdaMargins", "ema_12", "ema_26", "enterpriseToEbitda", "enterpriseToRevenue", "fiftyDayAverage",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "fiveYearAvgDividendYield", "forwardEps", "forwardPE", "freeCashflow",
            "golden_cross", "grossMargins", "high", "low", "macd", "macd_histogram", "macd_signal", "month",
            "numberOfAnalystOpinions", "obv", "open", "operatingCashflow", "operatingMargins", "payoutRatio", "priceToBook",
            "priceToSalesTrailing12Months", "profitMargins", "quarter", "quickRatio", "recommendationKey", "returnOnAssets",
            "returnOnEquity", "revenueGrowth", "revenuePerShare", "sma_10", "sma_20", "sma_200", "sma_5",
            "sma_50", "targetHighPrice", "targetLowPrice", "targetMeanPrice", "targetMedianPrice", "ticker",
            "totalCash", "totalDebt", "tr", "trailingEps", "trailingPE", "twoHundredDayAverage", "volume", "year"
        ],
        connections: [
            { target: 'company', edge: 'HAS_MARKETDATA', direction: 'INBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR m IN MarketData FILTER m.ticker == "AAPL" RETURN m'
    },
    awards: {
        name: 'Gov Contracts',
        collection: 'Award',
        description: 'Federal Contract Awards',
        keyFields: [
            "award_amount", "award_amount_float", "awarding_agency", "contract_year", "description",
            "description_embedding", "ingested_at", "matched_sp500_name", "recipient_name", "source_file",
            "start_date", "ticker"
        ],
        connections: [
            { target: 'company', edge: 'HAS_AWARD', direction: 'INBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR a IN Award FILTER a.recipient_name == "LOCKHEED" RETURN a'
    },
    economicdata: {
        name: 'Economic Data',
        collection: 'EconomicData',
        description: 'FRED Economic Indicators',
        keyFields: [
            "10_year_treasury_yield", "10y_2y_treasury_spread", "2_year_treasury_yield", "30_year_mortgage_rate",
            "30_year_treasury_yield", "case_shiller_home_price_index", "consumer_price_index_cpi", "consumer_sentiment",
            "core_cpi_ex_food_and_energy", "core_pce_feds_preferred", "crude_oil_price_wti", "date", "dow_jones_industrial_average",
            "fed_balance_sheet_size", "federal_funds_rate", "housing_starts", "industrial_production", "ingested_at",
            "initial_jobless_claims", "m2_money_supply", "nasdaq_composite", "nonfarm_payrolls", "real_gdp",
            "retail_sales", "sandp_500_index", "unemployment_rate", "vix_volatility_index", "yield_curve_inverted", "yield_curve_slope"
        ],
        connections: [],
        exampleQuery: 'FOR e IN EconomicData FILTER e.date > "2023-01-01" RETURN e'
    },
    sec: {
        name: 'SEC Filings',
        collection: 'sec_filings',
        description: '10-K/10-Q Filings',
        keyFields: [
            "accession", "avg_finbert", "avg_negative", "avg_positive", "avg_uncertainty", "file_name",
            "filing_date", "fiscal_year", "sentence_count", "ticker", "type"
        ],
        connections: [
            { target: 'sec_sections', edge: 'has_section', direction: 'OUTBOUND', type: 'direct' },
            { target: 'sec_sentences', edge: 'has_section', direction: 'OUTBOUND', type: 'multi_hop' },
            { target: 'company', edge: 'HAS_FILING', direction: 'INBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR f IN sec_filings FILTER f.ticker == "TSLA" RETURN f'
    },
    sec_sections: {
        name: 'SEC Sections',
        collection: 'sec_sections',
        description: 'Filing Sections',
        keyFields: ["filing_id", "length", "section_type", "start_char"],
        connections: [
            { target: 'sec', edge: 'has_section', direction: 'INBOUND', type: 'direct' },
            { target: 'sec_sentences', edge: 'has_sentence', direction: 'OUTBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR s IN sec_sections LIMIT 5 RETURN s'
    },
    sec_sentences: {
        name: 'SEC Sentences',
        collection: 'sec_sentences',
        description: 'Sentiment & FinBERT Scores',
        keyFields: [
            "finbert_probs", "finbert_score", "litigious_per_1k", "n_tokens", "negative_per_1k", "positive_per_1k",
            "section_id", "text", "uncertainty_per_1k"
        ],
        connections: [
            { target: 'sec_sections', edge: 'has_sentence', direction: 'INBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR s IN sec_sentences FILTER s.finbert_score < -0.5 RETURN s'
    },
    predictionmarkets: {
        name: 'Polymarket',
        collection: 'prediction_markets_polymarket',
        description: 'Polymarket Events',
        keyFields: [
            "category", "closed", "condition_id", "description", "end_date", "fetched_at", "liquidity",
            "market_slug", "no_probability", "outcome_prices", "outcomes", "question", "question_embedding",
            "volume", "volume_24h", "yes_probability"
        ],
        connections: [
            { target: 'company', edge: 'market_mentions_company_polymarket', direction: 'OUTBOUND', type: 'direct' },
            { target: 'polymarket_positions', edge: 'position_in_market', direction: 'INBOUND', type: 'direct' },
            { target: 'polymarket_price_history', edge: 'market_has_price_history', direction: 'OUTBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR m IN prediction_markets_polymarket FILTER m.volume_24h > 1000 RETURN m'
    },
    kalshi: {
        name: 'Kalshi',
        collection: 'prediction_markets_kalshi',
        description: 'Kalshi Events',
        keyFields: [
            "category", "close_time", "fetched_at", "market_ticker", "no_probability", "open_interest",
            "status", "title", "updated_at", "volume", "volume_24h", "yes_probability"
        ],
        connections: [
            { target: 'company', edge: 'market_mentions_company_kalshi', direction: 'OUTBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR k IN prediction_markets_kalshi RETURN k'
    },
    cftc: {
        name: 'CFTC Positions',
        collection: 'commodity_positions',
        description: 'Futures Positioning',
        keyFields: [
            "%_of_OI_Commercial_Long_All", "%_of_OI_Commercial_Long_Old", "%_of_OI_Commercial_Long_Other",
            "%_of_OI_Commercial_Short_All", "%_of_OI_Commercial_Short_Old", "%_of_OI_Commercial_Short_Other",
            "%_of_OI_Noncommercial_Long_All", "%_of_OI_Noncommercial_Long_Old", "%_of_OI_Noncommercial_Long_Other",
            "%_of_OI_Noncommercial_Short_All", "%_of_OI_Noncommercial_Short_Old", "%_of_OI_Noncommercial_Short_Other",
            "%_of_OI_Noncommercial_Spreading_All", "%_of_OI_Noncommercial_Spreading_Old", "%_of_OI_Noncommercial_Spreading_Other",
            "%_of_OI_Nonreportable_Long_All", "%_of_OI_Nonreportable_Long_Old", "%_of_OI_Nonreportable_Long_Other",
            "%_of_OI_Nonreportable_Short_All", "%_of_OI_Nonreportable_Short_Old", "%_of_OI_Nonreportable_Short_Other",
            "%_of_OI_Total_Reportable_Long_All", "%_of_OI_Total_Reportable_Long_Old", "%_of_OI_Total_Reportable_Long_Other",
            "%_of_OI_Total_Reportable_Short_All", "%_of_OI_Total_Reportable_Short_Old", "%_of_OI_Total_Reportable_Short_Other",
            "%_of_Open_Interest_OIOld", "%_of_Open_Interest_OI_All", "%_of_Open_Interest_OI_Other", "As_of_Date_in_Form_YYMMDD",
            "CFTC_Commodity_Code_Quotes", "CFTC_Contract_Market_Code", "CFTC_Contract_Market_Code_Quotes",
            "CFTC_Market_Code_in_Initials", "CFTC_Market_Code_in_Initials_Quotes", "CFTC_Region_Code",
            "Change_in_Commercial_Long_All", "Change_in_Commercial_Short_All", "Change_in_Noncommercial_Long_All",
            "Change_in_Noncommercial_Short_All", "Change_in_Noncommercial_Spreading_All", "Change_in_Nonreportable_Long_All",
            "Change_in_Nonreportable_Short_All", "Change_in_Open_Interest_All", "Change_in_Total_Reportable_Long_All",
            "Change_in_Total_Reportable_Short_All", "Commercial_Positions_Long_All", "Commercial_Positions_Long_Old",
            "Commercial_Positions_Long_Other", "Commercial_Positions_Short_All", "Commercial_Positions_Short_Old",
            "Commercial_Positions_Short_Other", "Concentration_Gross_LT_=4_TDR_Long_Old", "Concentration_Gross_LT_=4_TDR_Long_Other",
            "Concentration_Gross_LT_=4_TDR_ShortOther", "Concentration_Gross_LT_=4_TDR_Short_All", "Concentration_Gross_LT_=4_TDR_Short_Old",
            "Concentration_Gross_LT_=8_TDR_Long_All", "Concentration_Gross_LT_=8_TDR_Long_Old", "Concentration_Gross_LT_=8_TDR_Long_Other",
            "Concentration_Gross_LT_=8_TDR_ShortOther", "Concentration_Gross_LT_=8_TDR_Short_All", "Concentration_Gross_LT_=8_TDR_Short_Old",
            "Concentration_Gross_LT_=_4_TDR_Long_All", "Concentration_Net_LT_=4_TDR_Long_All", "Concentration_Net_LT_=4_TDR_Long_Old",
            "Concentration_Net_LT_=4_TDR_Long_Other", "Concentration_Net_LT_=4_TDR_Short_All", "Concentration_Net_LT_=4_TDR_Short_Old",
            "Concentration_Net_LT_=4_TDR_Short_Other", "Concentration_Net_LT_=8_TDR_Long_All", "Concentration_Net_LT_=8_TDR_Long_Old",
            "Concentration_Net_LT_=8_TDR_Long_Other", "Concentration_Net_LT_=8_TDR_Short_All", "Concentration_Net_LT_=8_TDR_Short_Old",
            "Concentration_Net_LT_=8_TDR_Short_Other", "Contract_Units", "Market_and_Exchange_Names", "Noncommercial_Positions_Long_All",
            "Noncommercial_Positions_Long_Old", "Noncommercial_Positions_Long_Other", "Noncommercial_Positions_Short_All",
            "Noncommercial_Positions_Short_Old", "Noncommercial_Positions_Short_Other", "Noncommercial_Positions_Spreading_All",
            "Noncommercial_Positions_Spreading_Old", "Noncommercial_Positions_Spreading_Other", "Nonreportable_Positions_Long_All",
            "Nonreportable_Positions_Long_Old", "Nonreportable_Positions_Long_Other", "Nonreportable_Positions_Short_All",
            "Nonreportable_Positions_Short_Old", "Nonreportable_Positions_Short_Other", "Open_Interest_All", "Open_Interest_Old",
            "Open_Interest_Other", "Total_Reportable_Positions_Long_All", "Total_Reportable_Positions_Long_Old",
            "Total_Reportable_Positions_Long_Other", "Total_Reportable_Positions_Short_All", "Total_Reportable_Positions_Short_Old",
            "Total_Reportable_Positions_Short_Other", "Traders_Commercial_Long_All", "Traders_Commercial_Long_Old",
            "Traders_Commercial_Long_Other", "Traders_Commercial_Short_All", "Traders_Commercial_Short_Old",
            "Traders_Commercial_Short_Other", "Traders_Noncommercial_Long_All", "Traders_Noncommercial_Long_Old",
            "Traders_Noncommercial_Long_Other", "Traders_Noncommercial_Short_All", "Traders_Noncommercial_Short_Old",
            "Traders_Noncommercial_Short_Other", "Traders_Noncommercial_Spreading_All", "Traders_Noncommercial_Spreading_Old",
            "Traders_Noncommercial_Spreading_Other", "Traders_Total_All", "Traders_Total_Old", "Traders_Total_Other",
            "Traders_Total_Reportable_Long_All", "Traders_Total_Reportable_Long_Old", "Traders_Total_Reportable_Long_Other",
            "Traders_Total_Reportable_Short_All", "Traders_Total_Reportable_Short_Old", "Traders_Total_Reportable_Short_Other",
            "as_of_date", "commodity_code", "data_source", "source_file"
        ],
        connections: [],
        exampleQuery: 'FOR p IN commodity_positions RETURN p'
    },
    polymarket_traders: {
        name: 'Polymarket Traders',
        collection: 'polymarket_traders',
        description: 'Whales & Profit Makers',
        keyFields: [
            "activity_level", "address", "avg_position_size", "fetched_at", "is_profitable", "is_whale",
            "profit_ratio", "total_profit", "total_trades", "total_volume", "updated_at", "volume_rank"
        ],
        connections: [
            { target: 'polymarket_positions', edge: 'trader_has_position', direction: 'OUTBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR t IN polymarket_traders FILTER t.is_whale == true RETURN t'
    },
    polymarket_positions: {
        name: 'Poly Positions',
        collection: 'polymarket_positions',
        description: 'Trader Positions',
        keyFields: [
            "average_price", "fetched_at", "market_condition_id", "market_key", "market_question",
            "outcome_index", "position_id", "realized_profit", "size", "trader_address", "trader_key",
            "unrealized_profit", "updated_at"
        ],
        connections: [
            { target: 'predictionmarkets', edge: 'position_in_market', direction: 'OUTBOUND', type: 'direct' },
            { target: 'polymarket_traders', edge: 'trader_has_position', direction: 'INBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR p IN polymarket_positions SORT p.size DESC LIMIT 5 RETURN p'
    },
    polymarket_price_history: {
        name: 'Poly Price History',
        collection: 'polymarket_price_history',
        description: 'Price Action',
        keyFields: [
            "condition_id", "datetime", "liquidity", "market_id", "no_price", "timestamp", "volume",
            "volume_24h", "yes_price"
        ],
        connections: [
            { target: 'predictionmarkets', edge: 'market_has_price_history', direction: 'INBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR h IN polymarket_price_history LIMIT 5 RETURN h'
    }
}

// Helper to check if connection is valid
export const isValidConnection = (source: string, target: string): boolean => {
    const schema = GRAPH_SCHEMA[source.toLowerCase()]
    if (!schema) return false
    return schema.connections.some(c => c.target === target.toLowerCase())
}
