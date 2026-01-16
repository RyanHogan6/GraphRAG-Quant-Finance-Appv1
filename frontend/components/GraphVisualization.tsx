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
    count: '852',
    description: 'Core entity connecting all data sources',
    keyFields: ['ticker', 'name', 'sector', 'industry', 'marketCap', 'sharesOutstanding'],
    edges: ['HAS_MARKETDATA', 'HAS_AWARD', 'HAS_FILING', 'market_mentions_company_polymarket'],
    exampleQuery: 'FOR c IN Company FILTER c.ticker == "AAPL" RETURN c',
    sampleData: {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      sector: 'Technology',
      marketCap: 3450000000000,
    },
    highlight: 'Central hub connecting all data sources',
  },
  marketdata: {
    name: 'MarketData',
    count: '2M+',
    description: 'Daily OHLCV + 40+ technical & fundamental indicators',
    keyFields: ['ticker', 'date', 'close', 'volume', 'sma_50', 'rsi_14', 'trailingPE', 'forwardEps'],
    edges: ['Connected FROM Company via HAS_MARKETDATA'],
    exampleQuery: 'FOR m IN MarketData FILTER m.ticker == "AAPL" SORT m.date DESC LIMIT 30 RETURN m',
    sampleData: {
      ticker: 'AAPL',
      date: '2026-01-13',
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
    edges: ['Connected FROM Company via HAS_AWARD'],
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
    count: '15K+',
    description: 'Filings → Sections → Sentences with FinBERT sentiment',
    keyFields: ['ticker', 'filing_type', 'filing_date', 'sentiment_score', 'risk_mentions'],
    edges: ['Connected FROM Company via HAS_FILING', 'has_section → sec_sections', 'has_sentence → sec_sentences'],
    exampleQuery: 'FOR f IN sec_filings FILTER f.ticker == "TSLA" AND f.sentiment_score < -0.5 RETURN f',
    sampleData: {
      ticker: 'TSLA',
      filing_type: '10-K',
      filing_date: '2025-02-08',
      sentiment_score: -0.32,
      risk_mentions: 47,
    },
    highlight: 'Hierarchical: Filings → Sections → Sentences',
  },
  polymarket: {
    name: 'Polymarket',
    count: '10K+',
    description: 'Prediction market data with trader positions',
    keyFields: ['market_slug', 'question', 'outcome_yes', 'outcome_no', 'volume', 'liquidity'],
    edges: ['market_mentions_company_polymarket → Company', 'market_related_to_sector_polymarket → Company'],
    exampleQuery: 'FOR m IN prediction_markets_polymarket FILTER m.question =~ "Trump" SORT m.volume DESC LIMIT 10 RETURN m',
    sampleData: {
      question: 'Will Trump win 2024?',
      outcome_yes: 0.58,
      outcome_no: 0.42,
      volume: 15234000,
      liquidity: 2500000,
    },
    highlight: 'Connected to companies via semantic edges',
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

// Custom node component with click handler
function CustomNode({ data }: NodeProps) {
  const label = data.label as string
  const count = data.count as string
  const isCenter = data.isCenter as boolean

  return (
    <div
      className="px-6 py-4 rounded-lg border-2 shadow-lg cursor-pointer transition-all hover:scale-105 hover:shadow-xl"
      style={{
        background: isCenter ? '#D4AF37' : '#2a2a2a',
        color: isCenter ? '#1a1a1a' : '#D4AF37',
        borderColor: isCenter ? '#D4AF37' : 'rgba(212, 175, 55, 0.4)',
        minWidth: isCenter ? '180px' : '160px',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#D4AF37' }} />
      <div className="text-center">
        <div className="font-bold text-base mb-1">{label}</div>
        <div className="text-xs opacity-70">{count}</div>
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: '#D4AF37' }} />
    </div>
  )
}

const nodeTypes = {
  custom: CustomNode,
}

export default function GraphVisualization() {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  // Define nodes
  const initialNodes: Node[] = [
    {
      id: 'company',
      type: 'custom',
      position: { x: 500, y: 350 },
      data: { label: 'Company', count: '852 companies', isCenter: true },
    },
    {
      id: 'marketdata',
      type: 'custom',
      position: { x: 200, y: 150 },
      data: { label: 'Market Data', count: '2M+ records', isCenter: false },
    },
    {
      id: 'awards',
      type: 'custom',
      position: { x: 800, y: 150 },
      data: { label: 'Gov Contracts', count: '500K+ awards', isCenter: false },
    },
    {
      id: 'sec',
      type: 'custom',
      position: { x: 100, y: 550 },
      data: { label: 'SEC Filings', count: '15K+ filings', isCenter: false },
    },
    {
      id: 'predictionmarkets',
      type: 'custom',
      position: { x: 700, y: 550 },
      data: { label: 'Prediction Markets', count: '18K+ markets', isCenter: false },
    },
    {
      id: 'web',
      type: 'custom',
      position: { x: 500, y: 50 },
      data: { label: 'Web Search', count: 'Real-time', isCenter: false },
    },
    {
      id: 'fred',
      type: 'custom',
      position: { x: 300, y: 550 },
      data: { label: 'FRED Data', count: 'Economic indicators', isCenter: false },
    },
    {
      id: 'cftc',
      type: 'custom',
      position: { x: 900, y: 350 },
      data: { label: 'CFTC Positions', count: 'Futures data', isCenter: false },
    },
  ]

  // Define edges with labels
  const initialEdges: Edge[] = [
    {
      id: 'company-marketdata',
      source: 'company',
      target: 'marketdata',
      label: 'daily prices & indicators',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 2 },
      labelStyle: { fill: '#D4AF37', fontSize: 10, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#D4AF37',
      },
    },
    {
      id: 'company-awards',
      source: 'company',
      target: 'awards',
      label: 'government contracts',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 2 },
      labelStyle: { fill: '#D4AF37', fontSize: 10, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#D4AF37',
      },
    },
    {
      id: 'company-sec',
      source: 'company',
      target: 'sec',
      label: 'regulatory filings',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 2 },
      labelStyle: { fill: '#D4AF37', fontSize: 10, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#D4AF37',
      },
    },
    {
      id: 'polymarket-company',
      source: 'polymarket',
      target: 'company',
      label: 'mentions company',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 2 },
      labelStyle: { fill: '#D4AF37', fontSize: 10, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#D4AF37',
      },
    },
    {
      id: 'web-company',
      source: 'web',
      target: 'company',
      label: 'real-time context',
      animated: true,
      style: { stroke: '#D4AF37', strokeWidth: 2, strokeDasharray: '5 5' },
      labelStyle: { fill: '#D4AF37', fontSize: 10, fontWeight: 600 },
      labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#D4AF37',
      },
    },
  ]

  const onNodeClick = useCallback((_event: any, node: Node) => {
    setSelectedNode(node.id)
  }, [])

  const onNodesChange = useCallback(() => {}, [])
  const onEdgesChange = useCallback(() => {}, [])

  const nodeInfo = selectedNode ? collectionData[selectedNode as keyof typeof collectionData] : null

  return (
    <div className="relative">
      <div className="h-[600px] w-full bg-dark-900 rounded-lg border border-gold/20">
        <ReactFlow
          nodes={initialNodes}
          edges={initialEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          fitView
          attributionPosition="bottom-left"
          defaultEdgeOptions={{
            type: 'smoothstep',
          }}
        >
          <Background color="#D4AF37" gap={16} size={1} />
          <Controls />
        </ReactFlow>
      </div>

      {/* Tooltip Modal */}
      {nodeInfo && (
        <div className="mt-6 bg-dark-800 border border-gold/30 rounded-lg p-6 animate-in fade-in duration-300">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-2xl font-bold text-gold">{nodeInfo.name}</h3>
              <p className="text-gray-400 text-sm mt-1">{nodeInfo.count} records</p>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-500 hover:text-gold transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
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
      )}

      {/* Hint text */}
      {!selectedNode && (
        <div className="mt-4 text-center">
          <p className="text-gray-500 text-sm">
            💡 Click any node to see collection details, schema, and sample queries
          </p>
        </div>
      )}
    </div>
  )
}
