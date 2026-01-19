import { Node, Edge } from '@xyflow/react'

export interface SchemaNode {
    name: string
    collection: string
    description: string
    keyFields: string[]
    // New: Graph connections with explicit direction and edge collection
    connections: {
        target: string // key in GRAPH_SCHEMA
        edge: string // edge collection name
        direction: 'INBOUND' | 'OUTBOUND'
        type: 'direct' | 'multi_hop'
        hops?: string[] // For multi-hop (e.g. SEC) - list of collections/edges? simplified for MVP
    }[]
    exampleQuery: string
}

export const GRAPH_SCHEMA: Record<string, SchemaNode> = {
    company: {
        name: 'Companies',
        collection: 'Company',
        description: 'S&P 500 Companies',
        keyFields: ['ticker', 'name', 'sector', 'industry', 'marketCap'],
        connections: [
            { target: 'marketdata', edge: 'HAS_MARKETDATA', direction: 'OUTBOUND', type: 'direct' },
            { target: 'awards', edge: 'HAS_AWARD', direction: 'OUTBOUND', type: 'direct' },
            { target: 'sec', edge: 'HAS_FILING', direction: 'OUTBOUND', type: 'direct' }, // To filings
            { target: 'predictionmarkets', edge: 'market_mentions_company_polymarket', direction: 'INBOUND', type: 'direct' },
            { target: 'kalshi', edge: 'market_mentions_company_kalshi', direction: 'INBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR c IN Company FILTER c.ticker == "AAPL" RETURN c'
    },
    marketdata: {
        name: 'Market Data',
        collection: 'MarketData',
        description: 'Daily OHLCV & Indicators',
        keyFields: ['date', 'close', 'volume', 'rsi_14', 'sma_50'],
        connections: [],
        exampleQuery: 'FOR m IN MarketData FILTER m.ticker == "AAPL" RETURN m'
    },
    awards: {
        name: 'Gov Contracts',
        collection: 'Award',
        description: 'Federal Contract Awards',
        keyFields: ['recipient_name', 'award_amount_float', 'agency', 'description'],
        connections: [],
        exampleQuery: 'FOR a IN Award FILTER a.recipient_name == "LOCKHEED" RETURN a'
    },
    sec: {
        name: 'SEC Filings',
        collection: 'sec_filings',
        description: '10-K/10-Q Filings',
        keyFields: ['filing_type', 'filing_date', 'form_type'],
        connections: [
            { target: 'sec_sentences', edge: 'has_section', direction: 'OUTBOUND', type: 'multi_hop' } # Simplified representation
        ],
        exampleQuery: 'FOR f IN sec_filings FILTER f.ticker == "TSLA" RETURN f'
    },
    sec_sentences: {
        name: 'SEC Sentences',
        collection: 'sec_sentences',
        description: 'Sentiment & FinBERT Scores',
        keyFields: ['sentence', 'sentiment_score', 'sentiment_label', 'filing_date'],
        connections: [],
        exampleQuery: 'FOR s IN sec_sentences FILTER s.ticker == "TSLA" RETURN s'
    },
    predictionmarkets: {
        name: 'Polymarket',
        collection: 'prediction_markets_polymarket',
        description: 'Polymarket Events',
        keyFields: ['question', 'volume_24h', 'yes_prob', 'no_prob'],
        connections: [
            { target: 'company', edge: 'market_mentions_company_polymarket', direction: 'OUTBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR m IN prediction_markets_polymarket FILTER m.volume_24h > 1000 RETURN m'
    },
    kalshi: {
        name: 'Kalshi',
        collection: 'prediction_markets_kalshi',
        description: 'Kalshi Events',
        keyFields: ['title', 'yes_bid', 'no_bid', 'status'],
        connections: [
            { target: 'company', edge: 'market_mentions_company_kalshi', direction: 'OUTBOUND', type: 'direct' }
        ],
        exampleQuery: 'FOR k IN prediction_markets_kalshi RETURN k'
    },
    cftc: {
        name: 'CFTC Positions',
        collection: 'commodity_positions',
        description: 'Futures Positioning',
        keyFields: ['commodity', 'long_positions', 'short_positions'],
        connections: [],
        exampleQuery: 'FOR p IN commodity_positions RETURN p'
    }
}

// Helper to check if connection is valid
export const isValidConnection = (source: string, target: string): boolean => {
    const schema = GRAPH_SCHEMA[source.toLowerCase()]
    if (!schema) return false
    return schema.connections.some(c => c.target === target.toLowerCase())
}
