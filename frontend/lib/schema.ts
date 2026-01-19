import { Node, Edge } from '@xyflow/react'

// Define the strict schema for valid connections
export interface SchemaNode {
    name: string
    collection: string
    description: string
    keyFields: string[]
    validConnections: string[] // List of collections this can connect TO
    exampleQuery: string
}

export const GRAPH_SCHEMA: Record<string, SchemaNode> = {
    company: {
        name: 'Companies',
        collection: 'Company',
        description: 'S&P 500 Companies',
        keyFields: ['ticker', 'name', 'sector', 'industry', 'marketCap'],
        validConnections: ['marketdata', 'awards', 'sec', 'predictionmarkets', 'cftc'],
        exampleQuery: 'FOR c IN Company FILTER c.ticker == "AAPL" RETURN c'
    },
    marketdata: {
        name: 'Market Data',
        collection: 'MarketData',
        description: 'Daily OHLCV & Indicators',
        keyFields: ['date', 'close', 'volume', 'rsi_14', 'sma_50'],
        validConnections: [], // Leaf node typically
        exampleQuery: 'FOR m IN MarketData FILTER m.ticker == "AAPL" RETURN m'
    },
    awards: {
        name: 'Gov Contracts',
        collection: 'Award',
        description: 'Federal Contract Awards',
        keyFields: ['recipient_name', 'award_amount_float', 'agency', 'description'],
        validConnections: ['company'], // Can connect back to company
        exampleQuery: 'FOR a IN Award FILTER a.recipient_name == "LOCKHEED" RETURN a'
    },
    sec: {
        name: 'SEC Sentences',
        collection: 'sec_sentences',
        description: '10-K/10-Q Sentiment',
        keyFields: ['sentence', 'sentiment_score', 'sentiment_label', 'filing_date'],
        validConnections: [],
        exampleQuery: 'FOR s IN sec_sentences FILTER s.ticker == "TSLA" SORT s.filing_date DESC RETURN s'
    },
    predictionmarkets: {
        name: 'Prediction Markets',
        collection: 'prediction_markets_polymarket',
        description: 'Polymarket/Kalshi Events',
        keyFields: ['question', 'volume_24h', 'yes_prob', 'no_prob'],
        validConnections: ['company'],
        exampleQuery: 'FOR m IN prediction_markets_polymarket FILTER m.volume_24h > 1000 RETURN m'
    },
    cftc: {
        name: 'CFTC Positions',
        collection: 'commodity_positions',
        description: 'Futures Positioning',
        keyFields: ['commodity', 'long_positions', 'short_positions'],
        validConnections: [],
        exampleQuery: 'FOR p IN commodity_positions RETURN p'
    },
    fred: {
        name: 'Federal Reserve',
        collection: 'EconomicData',
        description: 'Macro Indicators',
        keyFields: ['series_id', 'value', 'date'],
        validConnections: [],
        exampleQuery: 'FOR e IN EconomicData RETURN e'
    }
}

// Helper to check if connection is valid
export const isValidConnection = (source: string, target: string): boolean => {
    const schema = GRAPH_SCHEMA[source.toLowerCase()]
    if (!schema) return false
    return schema.validConnections.includes(target.toLowerCase())
}
