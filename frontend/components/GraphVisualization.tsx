'use client'

import { useCallback, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  MarkerType,
  NodeProps,
  Handle,
  Position,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

// Collection metadata with real schema info
const collectionData = {
  company: {
    name: 'Company',
    count: '612',
    description: 'Core entity connecting all data sources',
    keyFields: ['ticker', 'company', 'sector', 'industry', 'marketCap', 'sharesOutstanding'],
    edges: ['HAS_MARKETDATA', 'HAS_AWARD', 'HAS_FILING', 'COMPANY_HAS_OPTIONS', 'COMPANY_TRADES_COMMODITY', 'market_mentions_company'],
    exampleQuery: 'FOR c IN Company FILTER c.ticker == "AAPL" RETURN c',
    sampleData: {
      ticker: 'AAPL',
      company: 'Apple Inc.',
      sector: 'Technology',
      marketCap: 3450000000000,
    },
    highlight: 'Central hub connecting all data sources - S&P 500 companies',
  },
  marketdata: {
    name: 'MarketData',
    count: '2M+',
    description: 'Daily OHLCV + 40+ technical & fundamental indicators',
    keyFields: ['ticker', 'date', 'close', 'volume', 'sma_50', 'rsi_14', 'trailingPE', 'forwardEps'],
    edges: ['HAS_MARKETDATA (FROM Company)', 'HAS_OPTIONS_ACTIVITY → options_flow'],
    exampleQuery: 'FOR m IN MarketData FILTER m.ticker == "AAPL" SORT m.date DESC LIMIT 30 RETURN m',
    sampleData: {
      ticker: 'AAPL',
      date: '2026-01-24',
      close: 185.43,
      volume: 58234571,
      sma_50: 182.15,
      rsi_14: 62.3,
    },
    highlight: '40+ technical and fundamental indicators per day',
  },
  awards: {
    name: 'Award',
    count: '500K+',
    description: 'Federal contracts with AI embeddings for semantic search',
    keyFields: ['award_amount_float', 'description', 'start_date', 'recipient_name', 'description_embedding'],
    edges: ['HAS_AWARD (FROM Company)', 'OPTIONS_BEFORE_AWARD (FROM options_flow)'],
    exampleQuery: 'FOR a IN Award LET sim = COSINE_SIMILARITY(a.description_embedding, @query_vector) FILTER sim >= 0.7 SORT sim DESC LIMIT 10 RETURN a',
    sampleData: {
      recipient_name: 'LOCKHEED MARTIN',
      award_amount_float: 450000000,
      description: 'F-35 Joint Strike Fighter...',
      start_date: '2025-06-15',
    },
    highlight: 'Semantic search enabled with 1536-dim embeddings',
  },
  sec: {
    name: 'SEC Filings',
    count: '7.5K+',
    description: '12 form types: 10-K, 10-Q, 8-K, Form 4/5, SC 13D/G, 13F, S-1, 6-K, DEF 14A, 424B4',
    keyFields: ['ticker', 'type', 'filing_date', 'avg_finbert', 'trades'],
    edges: ['HAS_FILING (FROM Company)', 'has_section → sec_sections', 'has_exhibit → sec_exhibits', 'has_xbrl_data → sec_xbrl_data', 'OPTIONS_BEFORE_FILING (FROM options_flow)'],
    exampleQuery: 'FOR f IN sec_filings FILTER f.ticker == "TSLA" AND f.avg_finbert < -0.5 RETURN f',
    sampleData: {
      ticker: 'TSLA',
      type: '10-K',
      filing_date: '2025-02-08',
      avg_finbert: -0.32,
      sentence_count: 1847,
    },
    highlight: 'Hierarchical: Filings → Sections → Sentences; Exhibits; XBRL',
  },
  sec_sections: {
    name: 'SEC Sections',
    count: '50K+',
    description: 'Filing section structure (MD&A, Risk, etc.)',
    keyFields: ['section_type', 'filing_key', 'order'],
    edges: ['has_section (FROM sec_filings)', 'has_sentence → sec_sentences'],
    exampleQuery: 'FOR s IN sec_sections FILTER s.section_type == "Item 1A" RETURN s',
    sampleData: { section_type: 'Item 1A', filing_key: '...', order: 5 },
    highlight: 'Bridge between filings and sentence-level sentiment',
  },
  sec_sentences: {
    name: 'SEC Sentences',
    count: '4M+',
    description: 'Sentence-level text with FinBERT sentiment scores',
    keyFields: ['text', 'finbert_score', 'section_key'],
    edges: ['has_sentence (FROM sec_sections)'],
    exampleQuery: 'FOR s IN sec_sentences FILTER s.finbert_score < -0.8 RETURN s',
    sampleData: { text: 'Revenue declined...', finbert_score: -0.65 },
    highlight: 'Doc2Vec embeddings + FinBERT for semantic search',
  },
  sec_exhibits: {
    name: 'SEC Exhibits',
    count: '10K+',
    description: 'Exhibit documents attached to filings',
    keyFields: ['exhibit_type', 'filing_date', 'description'],
    edges: ['has_exhibit (FROM sec_filings)'],
    exampleQuery: 'FOR e IN sec_exhibits FILTER e.exhibit_type == "EX-10" RETURN e',
    sampleData: { exhibit_type: 'EX-10', filing_date: '2025-01-15' },
    highlight: 'Exhibit-level search and attachments',
  },
  sec_xbrl_data: {
    name: 'SEC XBRL Data',
    count: '5K+',
    description: 'Structured financial data (balance sheet, income, cash flow)',
    keyFields: ['concept', 'value', 'filing_date', 'unit'],
    edges: ['has_xbrl_data (FROM sec_filings)'],
    exampleQuery: 'FOR x IN sec_xbrl_data FILTER x.concept == "Revenue" RETURN x',
    sampleData: { concept: 'Revenue', value: 394328000000, unit: 'USD' },
    highlight: 'Structured fundamentals for company workups',
  },
  options: {
    name: 'Options Flow',
    count: '612+ daily',
    description: 'Daily options activity for insider trading detection',
    keyFields: ['ticker', 'date', 'call_volume', 'put_volume', 'put_call_ratio', 'unusual_call_activity', 'potential_sweep'],
    edges: ['COMPANY_HAS_OPTIONS (FROM Company)', 'HAS_OPTIONS_ACTIVITY (FROM MarketData)', 'OPTIONS_BEFORE_AWARD → Award', 'OPTIONS_BEFORE_FILING → sec_filings'],
    exampleQuery: 'FOR o IN options_flow FILTER o.unusual_call_activity == true RETURN o',
    sampleData: {
      ticker: 'LMT',
      date: '2026-01-24',
      call_volume: 12450,
      put_volume: 3200,
      put_call_ratio: 0.26,
      unusual_call_activity: true,
    },
    highlight: 'Unusual activity detection + insider trading signals (Day 20+)',
  },
  futures: {
    name: 'Futures Prices',
    count: '64K+',
    description: 'CME commodity futures: crude oil, natural gas, gold, copper, corn, wheat, etc.',
    keyFields: ['commodity_type', 'date', 'close', 'volume', 'sma_20', 'rsi_14', 'volatility_30d'],
    edges: ['COMPANY_TRADES_COMMODITY (FROM Company)', 'POSITION_ON_COMMODITY (FROM commodity_positions)', 'INVENTORY_AFFECTS_PRICE (FROM EIA)', 'MACRO_IMPACTS_COMMODITY (FROM FRED)'],
    exampleQuery: 'FOR f IN futures_prices FILTER f.commodity_type == "CRUDE_OIL" SORT f.date DESC LIMIT 30 RETURN f',
    sampleData: {
      commodity_type: 'CRUDE_OIL',
      date: '2026-01-24',
      close: 78.45,
      volume: 245000,
      rsi_14: 58.2,
    },
    highlight: '18 commodities with technical indicators + EIA/CFTC links',
  },
  eia: {
    name: 'EIA Energy Data',
    count: '1K+',
    description: 'EIA: crude inventory, natgas storage, natgas production, LNG exports (4 collections)',
    keyFields: ['date', 'crude_stocks', 'stocks_change', 'natgas_storage', 'lng_exports'],
    edges: ['INVENTORY_AFFECTS_PRICE → futures_prices', 'STORAGE_AFFECTS_PRICE → futures_prices'],
    exampleQuery: 'FOR e IN eia_crude_inventory FILTER e.crude_stocks_change > 5 RETURN e',
    sampleData: {
      date: '2026-01-17',
      crude_stocks: 425000000,
      crude_stocks_change: 7500000,
      cushing_stocks: 25000000,
    },
    highlight: '4 collections: crude, natgas storage, production, LNG',
  },
  predictionmarkets: {
    name: 'Prediction Markets',
    count: '18K+',
    description: 'Polymarket & Kalshi prediction markets with trader positions & price history',
    keyFields: ['question', 'yes_probability', 'volume_24h', 'liquidity', 'question_embedding'],
    edges: ['market_mentions_company → Company', 'position_in_market (FROM polymarket_positions)', 'price_history → polymarket_price_history'],
    exampleQuery: 'FOR m IN prediction_markets_polymarket FILTER m.volume_24h > 100000 SORT m.volume_24h DESC RETURN m',
    sampleData: {
      question: 'Will Trump win 2024?',
      yes_probability: 0.58,
      volume_24h: 2500000,
      liquidity: 1200000,
    },
    highlight: 'Polymarket + Kalshi; whale tracking + price history',
  },
  polymarket_positions: {
    name: 'Polymarket Positions',
    count: '2K+',
    description: 'Current trader positions in prediction markets',
    keyFields: ['market', 'outcome', 'shares', 'avg_price', 'trader_address'],
    edges: ['trader_has_position (FROM polymarket_traders)', 'position_in_market → prediction_markets'],
    exampleQuery: 'FOR p IN polymarket_positions FILTER p.shares > 10000 RETURN p',
    sampleData: { market: '...', outcome: 'Yes', shares: 5000, avg_price: 0.62 },
    highlight: 'Links whales to specific markets',
  },
  polymarket_price_history: {
    name: 'Polymarket Price History',
    count: '50K+',
    description: 'Historical yes/no prices per market over time',
    keyFields: ['market', 'date', 'yes_price', 'no_price', 'volume'],
    edges: ['price_history (FROM prediction_markets)'],
    exampleQuery: 'FOR h IN polymarket_price_history FILTER h.market == @id SORT h.date DESC LIMIT 30 RETURN h',
    sampleData: { date: '2026-01-24', yes_price: 0.58, no_price: 0.42 },
    highlight: 'Time series for market sentiment',
  },
  traders: {
    name: 'Polymarket Traders',
    count: '500+',
    description: 'Whale traders and profit makers on Polymarket',
    keyFields: ['address', 'total_volume', 'total_profit', 'is_whale', 'is_profitable', 'volume_rank'],
    edges: ['trader_has_position → polymarket_positions'],
    exampleQuery: 'FOR t IN polymarket_traders FILTER t.is_whale == true AND t.is_profitable == true RETURN t',
    sampleData: {
      address: '0x1234...abcd',
      total_volume: 5000000,
      total_profit: 450000,
      is_whale: true,
      volume_rank: 15,
    },
    highlight: 'Track smart money and whale positioning',
  },
  fred: {
    name: 'Economic Data',
    count: '8.9K+',
    description: 'Federal Reserve Economic Data (FRED) - macroeconomic indicators',
    keyFields: ['date', 'federal_funds_rate', 'unemployment_rate', 'cpi', 'gdp', '10_year_treasury'],
    edges: ['MACRO_IMPACTS_COMMODITY → futures_prices'],
    exampleQuery: 'FOR e IN EconomicData FILTER e.date > "2024-01-01" RETURN e',
    sampleData: {
      date: '2026-01-01',
      federal_funds_rate: 4.5,
      unemployment_rate: 3.7,
      cpi: 3.2,
    },
    highlight: 'Macro indicators driving market conditions',
  },
  cftc: {
    name: 'CFTC Positions',
    count: '5K+',
    description: 'Commodity Futures Trading Commission - institutional positioning data',
    keyFields: ['Market_and_Exchange_Names', 'as_of_date', 'Noncommercial_Positions_Long', 'Commercial_Positions_Short'],
    edges: ['HAS_COMMODITY_POSITION (FROM Company)', 'POSITION_ON_COMMODITY → futures_prices'],
    exampleQuery: 'FOR c IN commodity_positions FILTER CONTAINS(c.Market_and_Exchange_Names, "CRUDE") RETURN c',
    sampleData: {
      Market_and_Exchange_Names: 'CRUDE OIL, LIGHT SWEET',
      as_of_date: '2026-01-14',
      Noncommercial_Positions_Long_All: 450000,
      Commercial_Positions_Short_All: 620000,
    },
    highlight: 'Institutional sentiment and positioning (weekly reports)',
  },
  web: {
    name: 'Web Search',
    count: 'Real-time',
    description: 'Perplexity API for current events & news context',
    keyFields: ['summary', 'sources', 'citations'],
    edges: ['Augments all queries with real-time data'],
    exampleQuery: 'Parallel execution: DB query + Web search → Hybrid synthesis',
    sampleData: {
      summary: 'Recent news context...',
      sources: ['https://example.com'],
      citations: [{ number: 1, url: 'https://example.com' }],
    },
    highlight: 'Always runs in parallel with database queries',
  },
}

// Custom node component with dramatic styling
function CustomNode({ data }: NodeProps) {
  const label = data.label as string
  const count = data.count as string
  const isCenter = data.isCenter as boolean

  if (isCenter) {
    // Center hub - Extra dramatic
    return (
      <div
        className="px-8 py-5 rounded-xl border-4 shadow-2xl cursor-pointer transition-all hover:scale-110 hover:rotate-1 relative overflow-hidden"
        style={{
          background: 'linear-gradient(135deg, #D4AF37 0%, #F4D03F 50%, #D4AF37 100%)',
          color: '#1a1a1a',
          borderColor: '#FFD700',
          minWidth: '200px',
          boxShadow: '0 0 40px rgba(212, 175, 55, 0.6), 0 0 80px rgba(212, 175, 55, 0.3), inset 0 0 20px rgba(255, 255, 255, 0.1)',
        }}
      >
        <Handle type="target" position={Position.Top} style={{ background: '#FFD700', width: '12px', height: '12px', boxShadow: '0 0 10px rgba(255, 215, 0, 0.8)' }} />
        <div className="text-center relative z-10">
          <div className="font-black text-xl mb-1 tracking-wide drop-shadow-sm">{label}</div>
          <div className="text-sm font-bold opacity-80">{count}</div>
        </div>
        <Handle type="source" position={Position.Bottom} style={{ background: '#FFD700', width: '12px', height: '12px', boxShadow: '0 0 10px rgba(255, 215, 0, 0.8)' }} />
        {/* Animated glow pulse */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse" />
      </div>
    )
  }

  // Regular nodes - Enhanced with glow
  return (
    <div
      className="px-5 py-3 rounded-lg border-2 shadow-lg cursor-pointer transition-all hover:scale-110 hover:shadow-2xl relative"
      style={{
        background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
        color: '#D4AF37',
        borderColor: 'rgba(212, 175, 55, 0.5)',
        minWidth: '160px',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5), 0 0 20px rgba(212, 175, 55, 0.15)',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#D4AF37', width: '10px', height: '10px', boxShadow: '0 0 8px rgba(212, 175, 55, 0.6)' }} />
      <div className="text-center relative z-10">
        <div className="font-bold text-base mb-1 drop-shadow-sm">{label}</div>
        <div className="text-xs opacity-70 font-medium">{count}</div>
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: '#D4AF37', width: '10px', height: '10px', boxShadow: '0 0 8px rgba(212, 175, 55, 0.6)' }} />
      {/* Subtle hover glow */}
      <div className="absolute inset-0 rounded-lg transition-opacity opacity-0 hover:opacity-100" style={{
        background: 'radial-gradient(circle at center, rgba(212, 175, 55, 0.1) 0%, transparent 70%)'
      }} />
    </div>
  )
}

const nodeTypes = {
  custom: CustomNode,
}

export default function GraphVisualization() {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  // Define nodes (radial layout around Company hub)
  const initialNodes: Node[] = [
    // Center hub
    {
      id: 'company',
      type: 'custom',
      position: { x: 500, y: 400 },
      data: { label: 'Company', count: '612 S&P 500', isCenter: true },
    },
    // Top tier - Real-time data
    {
      id: 'web',
      type: 'custom',
      position: { x: 500, y: 50 },
      data: { label: 'Web Search', count: 'Real-time', isCenter: false },
    },
    // Second tier - Core financial data (left to right)
    {
      id: 'marketdata',
      type: 'custom',
      position: { x: 100, y: 200 },
      data: { label: 'Market Data', count: '2M+ OHLCV', isCenter: false },
    },
    {
      id: 'options',
      type: 'custom',
      position: { x: 300, y: 150 },
      data: { label: 'Options Flow', count: '612 daily', isCenter: false },
    },
    {
      id: 'awards',
      type: 'custom',
      position: { x: 700, y: 150 },
      data: { label: 'Gov Contracts', count: '500K awards', isCenter: false },
    },
    {
      id: 'sec',
      type: 'custom',
      position: { x: 900, y: 200 },
      data: { label: 'SEC Filings', count: '7.5K filings', isCenter: false },
    },
    // SEC hierarchy (all 21 doc collections represented)
    {
      id: 'sec_sections',
      type: 'custom',
      position: { x: 1050, y: 140 },
      data: { label: 'SEC Sections', count: '50K+', isCenter: false },
    },
    {
      id: 'sec_sentences',
      type: 'custom',
      position: { x: 1050, y: 240 },
      data: { label: 'SEC Sentences', count: '4M+', isCenter: false },
    },
    {
      id: 'sec_exhibits',
      type: 'custom',
      position: { x: 1050, y: 340 },
      data: { label: 'SEC Exhibits', count: '10K+', isCenter: false },
    },
    {
      id: 'sec_xbrl_data',
      type: 'custom',
      position: { x: 1050, y: 440 },
      data: { label: 'SEC XBRL', count: '5K+', isCenter: false },
    },
    // Third tier - Markets & Commodities
    {
      id: 'futures',
      type: 'custom',
      position: { x: 100, y: 600 },
      data: { label: 'Futures Prices', count: '64K records', isCenter: false },
    },
    {
      id: 'eia',
      type: 'custom',
      position: { x: 300, y: 650 },
      data: { label: 'EIA Energy', count: 'Weekly reports', isCenter: false },
    },
    {
      id: 'cftc',
      type: 'custom',
      position: { x: 500, y: 700 },
      data: { label: 'CFTC Positions', count: '5K reports', isCenter: false },
    },
    {
      id: 'predictionmarkets',
      type: 'custom',
      position: { x: 700, y: 650 },
      data: { label: 'Prediction Markets', count: '18K markets', isCenter: false },
    },
    {
      id: 'traders',
      type: 'custom',
      position: { x: 900, y: 600 },
      data: { label: 'Whale Traders', count: '500+ tracked', isCenter: false },
    },
    {
      id: 'polymarket_positions',
      type: 'custom',
      position: { x: 780, y: 540 },
      data: { label: 'Poly Positions', count: '2K+', isCenter: false },
    },
    {
      id: 'polymarket_price_history',
      type: 'custom',
      position: { x: 820, y: 720 },
      data: { label: 'Poly Price Hist', count: '50K+', isCenter: false },
    },
    // Bottom tier - Macro
    {
      id: 'fred',
      type: 'custom',
      position: { x: 500, y: 900 },
      data: { label: 'Economic Data', count: 'FRED 8.9K', isCenter: false },
    },
  ]

  // Define edges with labels (all actual ArangoDB graph edges)
  const initialEdges: Edge[] = [
    // Web search enrichment (dashed - not in DB)
    {
      id: 'web-company',
      source: 'web',
      target: 'company',
      label: 'enriches queries',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 3, strokeDasharray: '5 5', filter: 'drop-shadow(0 0 4px rgba(212, 175, 55, 0.6))' },
      labelStyle: { fill: '#FFD700', fontSize: 10, fontWeight: 700 },
      labelBgStyle: { fill: '#000000', fillOpacity: 0.9 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#FFD700', width: 20, height: 20 },
    },
    // Company → Core Data
    {
      id: 'company-marketdata',
      source: 'company',
      target: 'marketdata',
      label: 'HAS_MARKETDATA',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 3, filter: 'drop-shadow(0 0 4px rgba(212, 175, 55, 0.6))' },
      labelStyle: { fill: '#FFD700', fontSize: 10, fontWeight: 700 },
      labelBgStyle: { fill: '#000000', fillOpacity: 0.9 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#FFD700', width: 20, height: 20 },
    },
    {
      id: 'company-awards',
      source: 'company',
      target: 'awards',
      label: 'HAS_AWARD',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 2 },
      labelStyle: { fill: '#D4AF37', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#D4AF37' },
    },
    {
      id: 'company-sec',
      source: 'company',
      target: 'sec',
      label: 'HAS_FILING',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 2 },
      labelStyle: { fill: '#D4AF37', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#D4AF37' },
    },
    {
      id: 'company-options',
      source: 'company',
      target: 'options',
      label: 'COMPANY_HAS_OPTIONS',
      animated: true,
      style: { stroke: '#10B981', strokeWidth: 2 },
      labelStyle: { fill: '#10B981', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#10B981' },
    },
    // Company → Commodities (CRITICAL NEW EDGE)
    {
      id: 'company-futures',
      source: 'company',
      target: 'futures',
      label: 'COMPANY_TRADES_COMMODITY',
      animated: true,
      style: { stroke: '#F59E0B', strokeWidth: 3 },
      labelStyle: { fill: '#F59E0B', fontSize: 9, fontWeight: 700 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.9 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#F59E0B' },
    },
    {
      id: 'company-cftc',
      source: 'company',
      target: 'cftc',
      label: 'HAS_COMMODITY_POSITION',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 2 },
      labelStyle: { fill: '#D4AF37', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#D4AF37' },
    },
    // MarketData → Options
    {
      id: 'marketdata-options',
      source: 'marketdata',
      target: 'options',
      label: 'HAS_OPTIONS_ACTIVITY',
      animated: true,
      style: { stroke: '#10B981', strokeWidth: 2 },
      labelStyle: { fill: '#10B981', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#10B981' },
    },
    // Options → Insider Trading
    {
      id: 'options-awards',
      source: 'options',
      target: 'awards',
      label: 'OPTIONS_BEFORE_AWARD',
      animated: true,
      style: { stroke: '#EF4444', strokeWidth: 2 },
      labelStyle: { fill: '#EF4444', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#EF4444' },
    },
    {
      id: 'options-sec',
      source: 'options',
      target: 'sec',
      label: 'OPTIONS_BEFORE_FILING',
      animated: true,
      style: { stroke: '#EF4444', strokeWidth: 2 },
      labelStyle: { fill: '#EF4444', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#EF4444' },
    },
    // Commodities ecosystem
    {
      id: 'cftc-futures',
      source: 'cftc',
      target: 'futures',
      label: 'POSITION_ON_COMMODITY',
      animated: true,
      style: { stroke: '#F59E0B', strokeWidth: 2 },
      labelStyle: { fill: '#F59E0B', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#F59E0B' },
    },
    {
      id: 'eia-futures',
      source: 'eia',
      target: 'futures',
      label: 'INVENTORY_AFFECTS_PRICE',
      animated: true,
      style: { stroke: '#F59E0B', strokeWidth: 2 },
      labelStyle: { fill: '#F59E0B', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#F59E0B' },
    },
    {
      id: 'eia-futures-storage',
      source: 'eia',
      target: 'futures',
      label: 'STORAGE_AFFECTS_PRICE',
      animated: true,
      style: { stroke: '#F59E0B', strokeWidth: 1.5, strokeDasharray: '4 4' },
      labelStyle: { fill: '#F59E0B', fontSize: 8, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#F59E0B' },
    },
    {
      id: 'sec-sec_sections',
      source: 'sec',
      target: 'sec_sections',
      label: 'has_section',
      animated: true,
      style: { stroke: '#A78BFA', strokeWidth: 2 },
      labelStyle: { fill: '#A78BFA', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#A78BFA' },
    },
    {
      id: 'sec_sections-sec_sentences',
      source: 'sec_sections',
      target: 'sec_sentences',
      label: 'has_sentence',
      animated: true,
      style: { stroke: '#A78BFA', strokeWidth: 2 },
      labelStyle: { fill: '#A78BFA', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#A78BFA' },
    },
    {
      id: 'sec-sec_exhibits',
      source: 'sec',
      target: 'sec_exhibits',
      label: 'has_exhibit',
      animated: true,
      style: { stroke: '#A78BFA', strokeWidth: 2 },
      labelStyle: { fill: '#A78BFA', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#A78BFA' },
    },
    {
      id: 'sec-sec_xbrl_data',
      source: 'sec',
      target: 'sec_xbrl_data',
      label: 'has_xbrl_data',
      animated: true,
      style: { stroke: '#A78BFA', strokeWidth: 2 },
      labelStyle: { fill: '#A78BFA', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#A78BFA' },
    },
    {
      id: 'traders-polymarket_positions',
      source: 'traders',
      target: 'polymarket_positions',
      label: 'trader_has_position',
      animated: true,
      style: { stroke: '#8B5CF6', strokeWidth: 2 },
      labelStyle: { fill: '#8B5CF6', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#8B5CF6' },
    },
    {
      id: 'polymarket_positions-predictionmarkets',
      source: 'polymarket_positions',
      target: 'predictionmarkets',
      label: 'position_in_market',
      animated: true,
      style: { stroke: '#8B5CF6', strokeWidth: 2 },
      labelStyle: { fill: '#8B5CF6', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#8B5CF6' },
    },
    {
      id: 'predictionmarkets-polymarket_price_history',
      source: 'predictionmarkets',
      target: 'polymarket_price_history',
      label: 'price_history',
      animated: true,
      style: { stroke: '#8B5CF6', strokeWidth: 2 },
      labelStyle: { fill: '#8B5CF6', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#8B5CF6' },
    },
    {
      id: 'fred-futures',
      source: 'fred',
      target: 'futures',
      label: 'MACRO_IMPACTS_COMMODITY',
      animated: true,
      style: { stroke: '#F59E0B', strokeWidth: 2 },
      labelStyle: { fill: '#F59E0B', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#F59E0B' },
    },
    // Prediction Markets
    {
      id: 'predictionmarkets-company',
      source: 'predictionmarkets',
      target: 'company',
      label: 'market_mentions_company',
      animated: true,
      style: { stroke: '#8B5CF6', strokeWidth: 2 },
      labelStyle: { fill: '#8B5CF6', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#8B5CF6' },
    },
    {
      id: 'traders-predictionmarkets',
      source: 'traders',
      target: 'predictionmarkets',
      label: 'trader_has_position',
      animated: true,
      style: { stroke: '#8B5CF6', strokeWidth: 2 },
      labelStyle: { fill: '#8B5CF6', fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#8B5CF6' },
    },
  ]

  const onNodeClick = useCallback((_event: any, node: Node) => {
    setSelectedNode(node.id)
  }, [])

  const onNodesChange = useCallback(() => { }, [])
  const onEdgesChange = useCallback(() => { }, [])

  const nodeInfo = selectedNode ? collectionData[selectedNode as keyof typeof collectionData] : null

  return (
    <div className="relative">
      <div className="h-[720px] w-full bg-dark-900 rounded-lg border border-gold/20">
        <ReactFlow
          nodes={initialNodes}
          edges={initialEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag={false}
          panOnScroll={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          attributionPosition="bottom-left"
          defaultEdgeOptions={{
            type: 'smoothstep',
          }}
        >
          <Background color="#D4AF37" gap={20} size={1.5} style={{ opacity: 0.15 }} />
        </ReactFlow>
      </div>

      {/* Node Details Modal */}
      {nodeInfo && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-2 md:p-4"
          onClick={() => setSelectedNode(null)}
        >
          <div
            className="bg-dark-800 border border-gold/30 rounded-lg max-w-full md:max-w-2xl lg:max-w-3xl w-full max-h-[80vh] md:max-h-[90vh] overflow-y-auto m-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-3 md:p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-gold">{nodeInfo.name}</h3>
                  <p className="text-gray-400 text-sm mt-1">{nodeInfo.count} records</p>
                </div>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="text-gray-500 hover:text-gold transition-colors text-2xl"
                >
                  ×
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-semibold text-gold mb-2">Description</h4>
                  <p className="text-gray-300 text-sm">{nodeInfo.description}</p>
                </div>

                {nodeInfo.highlight && (
                  <div className="bg-gold/10 border border-gold/30 rounded-lg p-3">
                    <p className="text-gold text-sm font-semibold">⚡ {nodeInfo.highlight}</p>
                  </div>
                )}

                <div>
                  <h4 className="text-sm font-semibold text-gold mb-2">Key Fields</h4>
                  <div className="flex flex-wrap gap-2">
                    {nodeInfo.keyFields.map((field) => (
                      <code key={field} className="text-xs bg-dark-700 text-gray-300 px-2 py-1 rounded border border-gold/20">
                        {field}
                      </code>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-gold mb-2">Graph Edges</h4>
                  <div className="space-y-1">
                    {nodeInfo.edges.map((edge) => (
                      <div key={edge} className="text-xs text-gray-400 font-mono bg-dark-700 px-3 py-1 rounded">
                        {edge}
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-gold mb-2">Sample Data</h4>
                  <div className="bg-dark-700 rounded-lg p-3 overflow-x-auto">
                    <pre className="text-xs text-gray-300 font-mono">
                      {JSON.stringify(nodeInfo.sampleData, null, 2)}
                    </pre>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-gold mb-2">Example Query</h4>
                  <div className="bg-dark-700 rounded-lg p-3 overflow-x-auto">
                    <code className="text-xs text-green-400 font-mono whitespace-pre-wrap">
                      {nodeInfo.exampleQuery}
                    </code>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hint text */}
      {!selectedNode && (
        <div className="mt-4 text-center">
          <p className="text-gray-500 text-sm">
            💡 Click any node to view collection details, schema, and sample queries
          </p>
        </div>
      )}
    </div>
  )
}
