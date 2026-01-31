'use client'

import { useState, useCallback, useMemo, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GRAPH_SCHEMA } from '@/lib/schema'

interface GraphNode {
  id: string
  collectionKey: string | null // null for uninitialized starter node
  label: string
  x: number
  y: number
  isDragging?: boolean
}

interface GraphEdge {
  id: string
  from: string
  to: string
  label: string
  direction: 'OUTBOUND' | 'INBOUND'
}

interface GraphExplorerProps {
  onQueryChange: (aql: string, description: string) => void
}

const COLLECTION_OPTIONS = [
  { key: 'company', label: 'Companies', icon: '🏢', color: '#fbbf24' },
  { key: 'marketdata', label: 'Stock Prices', icon: '📈', color: '#10b981' },
  { key: 'options', label: 'Options Flow', icon: '⚡', color: '#8b5cf6' },
  { key: 'futures', label: 'Commodities', icon: '🛢️', color: '#f59e0b' },
  { key: 'awards', label: 'Gov Contracts', icon: '🎖️', color: '#fbbf24' },
  { key: 'sec', label: 'SEC Filings', icon: '📄', color: '#3b82f6' },
  { key: 'predictionmarkets', label: 'Polymarket', icon: '🎲', color: '#ec4899' },
]

// Edge descriptions for tooltips
const EDGE_DESCRIPTIONS: Record<string, string> = {
  'HAS_MARKETDATA': 'Links company to daily stock prices, volume, and 40+ technical indicators',
  'HAS_AWARD': 'Connects company to federal contract awards they received',
  'HAS_FILING': 'Links company to their SEC filings (10-K, 10-Q, 8-K, Form 4, etc.)',
  'COMPANY_HAS_OPTIONS': 'Connects company to daily options flow data for insider trading detection',
  'COMPANY_TRADES_COMMODITY': 'Links company to commodity futures they produce/trade (energy, metals, agriculture)',
  'HAS_COMMODITY_POSITION': 'CFTC positioning data showing who holds futures positions',
  'market_mentions_company_polymarket': 'Prediction market explicitly mentions this company',
  'market_mentions_company_kalshi': 'Kalshi market explicitly mentions this company',
  'market_related_to_sector_polymarket': 'Prediction market related to company sector/industry',
  'market_related_to_sector_kalshi': 'Kalshi market related to company sector/industry',
  'HAS_OPTIONS_ACTIVITY': 'Links stock price data to same-day options activity',
  'OPTIONS_BEFORE_AWARD': 'Unusual options activity detected before contract announcement',
  'OPTIONS_BEFORE_FILING': 'Unusual options activity detected before SEC filing',
  'POSITION_ON_COMMODITY': 'CFTC position data linked to commodity futures prices',
  'INVENTORY_AFFECTS_PRICE': 'EIA crude oil inventory levels correlated with futures prices',
  'STORAGE_AFFECTS_PRICE': 'EIA natural gas storage levels correlated with futures prices',
  'MACRO_IMPACTS_COMMODITY': 'FRED economic indicators affecting commodity prices',
  'has_section': 'Filing contains multiple sections (Risk Factors, MD&A, etc.)',
  'has_sentence': 'Section broken into sentences with FinBERT sentiment scores',
  'has_exhibit': 'Filing includes exhibits (contracts, financial statements)',
  'has_xbrl_data': 'Filing has XBRL structured financial data',
  'trader_has_position': 'Whale/trader holds position in prediction market',
  'position_in_market': 'Position links to specific market',
}

export default function GraphExplorer({ onQueryChange }: GraphExplorerProps) {
  const [nodes, setNodes] = useState<GraphNode[]>([
    { id: 'starter', collectionKey: null, label: 'Start', x: 500, y: 250 }
  ])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [expandingNode, setExpandingNode] = useState<string | null>(null)
  const [selectedFields, setSelectedFields] = useState<Record<string, string[]>>({})
  const [showAQL, setShowAQL] = useState(false)
  const [draggedNode, setDraggedNode] = useState<string | null>(null)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [tickerSearch, setTickerSearch] = useState('')

  // Pan and zoom state
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState({ x: 0, y: 0 })
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)

  // Report generation state
  const [isExecuting, setIsExecuting] = useState(false)
  const [reportData, setReportData] = useState<any>(null)
  const [showReport, setShowReport] = useState(false)

  // Company ticker selection
  const [showTickerSelector, setShowTickerSelector] = useState(false)
  const [selectedTickers, setSelectedTickers] = useState<string[]>([])
  const [tickerInput, setTickerInput] = useState('')

  const svgRef = useRef<SVGSVGElement>(null)

  // Get node color
  const getNodeColor = (collectionKey: string | null) => {
    if (!collectionKey) return '#6b7280'
    const option = COLLECTION_OPTIONS.find(o => o.key === collectionKey)
    return option?.color || '#6b7280'
  }

  // Initialize starter node with collection
  const handleInitializeNode = useCallback((nodeId: string, collectionKey: string) => {
    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    setNodes(prev => prev.map(n =>
      n.id === nodeId
        ? { ...n, collectionKey, label: schema.name }
        : n
    ))
    setExpandingNode(null)
    setSelectedNode(nodeId)

    // Auto-select all fields for this node
    setSelectedFields(prev => ({
      ...prev,
      [nodeId]: schema.keyFields
    }))
  }, [])

  // Add new node from connection (auto-connects)
  const handleAddNode = useCallback((fromNodeId: string, collectionKey: string) => {
    // Check if already exists
    if (nodes.some(n => n.collectionKey === collectionKey)) return

    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    const fromNode = nodes.find(n => n.id === fromNodeId)
    if (!fromNode) return

    // Position new node offset from parent
    const newNode: GraphNode = {
      id: collectionKey,
      collectionKey,
      label: schema.name,
      x: fromNode.x + 200 + Math.random() * 100,
      y: fromNode.y + (Math.random() - 0.5) * 150
    }

    // Find edge info
    const fromSchema = GRAPH_SCHEMA[fromNode.collectionKey!]
    const connection = fromSchema?.connections.find(c => c.target === collectionKey)

    if (connection) {
      const newEdge: GraphEdge = {
        id: `edge-${fromNodeId}-${collectionKey}`,
        from: fromNodeId,
        to: collectionKey,
        label: connection.edge,
        direction: connection.direction
      }

      setNodes(prev => [...prev, newNode])
      setEdges(prev => [...prev, newEdge])
      setExpandingNode(null)
      setSelectedNode(newNode.id)

      // Auto-select all fields for this node
      setSelectedFields(prev => ({
        ...prev,
        [newNode.id]: schema.keyFields
      }))
    }
  }, [nodes])

  // Connect to existing node
  const handleConnectExisting = useCallback((fromNodeId: string, toNodeId: string) => {
    const fromNode = nodes.find(n => n.id === fromNodeId)
    const toNode = nodes.find(n => n.id === toNodeId)
    if (!fromNode || !toNode) return

    // Check if edge exists
    const exists = edges.some(e =>
      (e.from === fromNodeId && e.to === toNodeId) ||
      (e.from === toNodeId && e.to === fromNodeId)
    )
    if (exists) return

    const fromSchema = GRAPH_SCHEMA[fromNode.collectionKey!]
    const connection = fromSchema?.connections.find(c => c.target === toNode.collectionKey)

    if (connection) {
      const newEdge: GraphEdge = {
        id: `edge-${fromNodeId}-${toNodeId}`,
        from: fromNodeId,
        to: toNodeId,
        label: connection.edge,
        direction: connection.direction
      }

      setEdges(prev => [...prev, newEdge])
      setExpandingNode(null)
    }
  }, [nodes, edges])

  // Get connections for node
  const getAvailableConnections = useCallback((node: GraphNode) => {
    if (!node.collectionKey) return []

    const schema = GRAPH_SCHEMA[node.collectionKey]
    if (!schema) return []

    // Get existing nodes that can be connected
    const existingOptions = nodes
      .filter(n => n.id !== node.id && n.collectionKey)
      .map(n => {
        const canConnect = schema.connections.some(c => c.target === n.collectionKey)
        const alreadyConnected = edges.some(e =>
          (e.from === node.id && e.to === n.id) ||
          (e.from === n.id && e.to === node.id)
        )

        if (!canConnect || alreadyConnected) return null

        return {
          target: n.collectionKey!,
          targetNodeId: n.id,
          label: `← Connect to ${n.label}`,
          isExisting: true
        }
      })
      .filter((c): c is NonNullable<typeof c> => c !== null)

    // Get new node options
    const newOptions = schema.connections
      .map(conn => {
        const targetSchema = GRAPH_SCHEMA[conn.target]
        if (!targetSchema) return null

        const exists = nodes.some(n => n.collectionKey === conn.target)
        if (exists) return null

        return {
          target: conn.target,
          targetNodeId: undefined,
          label: `${conn.direction === 'OUTBOUND' ? '→' : '←'} ${targetSchema.name}`,
          isExisting: false
        }
      })
      .filter((c): c is NonNullable<typeof c> => c !== null)

    return [...newOptions, ...existingOptions]
  }, [nodes, edges])

  // Mouse down on node - start drag
  const handleMouseDown = useCallback((nodeId: string, e: React.MouseEvent) => {
    if (!svgRef.current) return

    const node = nodes.find(n => n.id === nodeId)
    if (!node) return

    const svgRect = svgRef.current.getBoundingClientRect()
    const svgX = (e.clientX - svgRect.left) * (1000 / svgRect.width)
    const svgY = (e.clientY - svgRect.top) * (500 / svgRect.height)

    setDraggedNode(nodeId)
    setDragOffset({ x: svgX - node.x, y: svgY - node.y })
  }, [nodes])

  // Pan move
  const handlePanMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      setPanOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      })
    }
  }, [isPanning, panStart])

  // Mouse move - handle both panning and node dragging
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      handlePanMove(e)
    } else if (draggedNode && svgRef.current) {
      const svgRect = svgRef.current.getBoundingClientRect()
      const svgX = (e.clientX - svgRect.left) * (1000 / svgRect.width)
      const svgY = (e.clientY - svgRect.top) * (500 / svgRect.height)

      // Constrain to boundaries (with padding for node radius)
      const padding = 40
      const constrainedX = Math.max(padding, Math.min(1000 - padding, svgX - dragOffset.x))
      const constrainedY = Math.max(padding, Math.min(500 - padding, svgY - dragOffset.y))

      setNodes(prev => prev.map(n =>
        n.id === draggedNode
          ? { ...n, x: constrainedX, y: constrainedY, isDragging: true }
          : n
      ))
    }
  }, [draggedNode, dragOffset, isPanning, handlePanMove])

  // Mouse up - stop drag
  const handleMouseUp = useCallback(() => {
    if (draggedNode) {
      setNodes(prev => prev.map(n =>
        n.id === draggedNode ? { ...n, isDragging: false } : n
      ))
      setDraggedNode(null)
    }
    if (isPanning) {
      setIsPanning(false)
    }
  }, [draggedNode, isPanning])

  // Pan start - middle mouse button
  const handlePanStart = useCallback((e: React.MouseEvent) => {
    if (e.button === 1) { // Middle mouse button
      e.preventDefault()
      setIsPanning(true)
      setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y })
    }
  }, [panOffset])

  // Zoom with scroll wheel
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setZoom(prev => Math.max(0.1, Math.min(3, prev * delta)))
  }, [])

  // Generate English description
  const englishDescription = useMemo(() => {
    const realNodes = nodes.filter(n => n.collectionKey)
    if (realNodes.length === 0) return 'No query yet'
    if (realNodes.length === 1) return `Get all ${realNodes[0].label}`

    return `Get ${realNodes[0].label} with ${realNodes.slice(1).map(n => n.label).join(', ')}`
  }, [nodes])

  // Generate AQL
  const aqlQuery = useMemo(() => {
    const realNodes = nodes.filter(n => n.collectionKey)
    if (realNodes.length === 0) return ''

    const root = realNodes[0]
    const rootSchema = GRAPH_SCHEMA[root.collectionKey!]
    const collection = rootSchema?.collection || root.collectionKey

    let query = `FOR doc IN ${collection}\n`

    // Add ticker filter if company node has selected tickers
    if (root.collectionKey === 'company' && selectedTickers.length > 0) {
      const tickerList = selectedTickers.map(t => `"${t}"`).join(', ')
      query += `  FILTER doc.ticker IN [${tickerList}]\n`
    }

    query += `  SORT doc.date DESC\n  LIMIT 100\n  RETURN doc`
    return query
  }, [nodes, selectedTickers])

  // Handle ticker search
  const handleTickerSearch = useCallback(() => {
    if (!tickerSearch.trim()) return

    const starterNode = nodes.find(n => n.collectionKey === null)
    if (starterNode) {
      handleInitializeNode(starterNode.id, 'company')
      // Store ticker for AQL generation
      setSelectedFields(prev => ({
        ...prev,
        '__ticker__': [tickerSearch.trim().toUpperCase()]
      }))
    }
  }, [tickerSearch, nodes, handleInitializeNode])

  // Execute query and get raw results
  const handleExecuteQuery = useCallback(async () => {
    if (!aqlQuery || !edges.length) return

    setIsExecuting(true)
    setShowReport(true)

    try {
      // Use backend API URL
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

      // Execute AQL query directly
      const response = await fetch(`${API_BASE_URL}/api/query/execute-aql`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          aql_query: aqlQuery,
          bind_vars: {}
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setReportData({
        results: data.results || [],
        count: data.count || 0,
        execution_time: data.execution_time || 0,
        aql_query: aqlQuery,
        description: englishDescription
      })

      // Also notify parent component
      onQueryChange(aqlQuery, englishDescription)

    } catch (error) {
      console.error('Execute query error:', error)
      alert(`Failed to execute query: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsExecuting(false)
    }
  }, [aqlQuery, englishDescription, edges.length, nodes, onQueryChange])

  return (
    <div className="bg-dark-900/50 p-4 rounded-lg border border-gold/10 h-full flex flex-col">
      {/* Header - just description */}
      <div className="mb-3">
        <div className="text-sm text-gray-300">
          {englishDescription}
        </div>
      </div>

      {/* Graph canvas - 80% height */}
      <div className="flex-1 bg-dark-800/30 rounded-lg border border-green-500/10 relative overflow-hidden">
        {/* Checkered grid background */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />

        {/* AQL Display Overlay */}
        {showAQL && aqlQuery && (
          <div className="absolute top-4 left-4 right-4 bg-dark-900/95 border border-green-500/30 rounded-lg p-3 backdrop-blur-sm z-20">
            <div className="flex items-start justify-between gap-2 mb-2">
              <div className="text-xs font-semibold text-green-400">Generated AQL Query</div>
              <button
                onClick={() => setShowAQL(false)}
                className="text-gray-500 hover:text-white"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <pre className="text-xs text-green-300 font-mono overflow-x-auto max-h-40 overflow-y-auto">
              {aqlQuery}
            </pre>
          </div>
        )}
        <svg
          ref={svgRef}
          className="w-full h-full relative z-10"
          viewBox="0 0 1000 500"
          onMouseDown={handlePanStart}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          style={{ cursor: isPanning ? 'grabbing' : 'default' }}
        >
          <defs>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
                <marker id="arrow-out" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L8,3 z" fill="#10b981" />
                </marker>
                <marker id="arrow-in" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L8,3 z" fill="#3b82f6" />
                </marker>
              </defs>

              {/* Pan/Zoom transform group */}
              <g transform={`translate(${panOffset.x},${panOffset.y}) scale(${zoom})`}>
                {/* Edges */}
                {edges.map(edge => {
                const from = nodes.find(n => n.id === edge.from)
                const to = nodes.find(n => n.id === edge.to)
                if (!from || !to) return null

                const color = edge.direction === 'OUTBOUND' ? '#10b981' : '#3b82f6'
                const marker = edge.direction === 'OUTBOUND' ? 'url(#arrow-out)' : 'url(#arrow-in)'

                const angle = Math.atan2(to.y - from.y, to.x - from.x)
                const fromX = from.x + 35 * Math.cos(angle)
                const fromY = from.y + 35 * Math.sin(angle)
                const toX = to.x - 35 * Math.cos(angle)
                const toY = to.y - 35 * Math.sin(angle)

                const tooltip = EDGE_DESCRIPTIONS[edge.label] || `${edge.direction} edge via ${edge.label}`

                return (
                  <g key={edge.id} style={{ cursor: 'help' }}>
                    <title>{tooltip}</title>
                    <line
                      x1={fromX}
                      y1={fromY}
                      x2={toX}
                      y2={toY}
                      stroke={color}
                      strokeWidth="2"
                      opacity="0.5"
                      markerEnd={marker}
                    />
                    <text
                      x={(fromX + toX) / 2}
                      y={(fromY + toY) / 2 - 5}
                      fill={color}
                      fontSize="9"
                      opacity="0.7"
                      textAnchor="middle"
                    >
                      {edge.label}
                    </text>
                  </g>
                )
              })}

              {/* Nodes */}
              {nodes.map(node => {
                const color = getNodeColor(node.collectionKey)
                const isSelected = selectedNode === node.id
                const isExpanding = expandingNode === node.id
                const connections = getAvailableConnections(node)

                return (
                  <g key={node.id}>
                    {/* Node circle */}
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={35}
                      fill="rgba(17, 24, 39, 0.95)"
                      stroke={color}
                      strokeWidth={isSelected ? 3 : 2}
                      filter={isSelected ? 'url(#glow)' : undefined}
                      style={{ cursor: node.isDragging ? 'grabbing' : 'grab' }}
                      onClick={() => {
                        setSelectedNode(node.id)
                        // Show ticker selector if clicking on company node
                        if (node.collectionKey === 'company') {
                          setShowTickerSelector(true)
                        }
                      }}
                      onMouseDown={(e) => handleMouseDown(node.id, e as any)}
                    />

                    {/* Label */}
                    <text
                      x={node.x}
                      y={node.y}
                      fill={color}
                      fontSize={11}
                      fontWeight="600"
                      textAnchor="middle"
                      dominantBaseline="middle"
                      style={{ pointerEvents: 'none' }}
                    >
                      {node.label}
                    </text>

                    {/* Expand button - only show if node has connections or is uninitialized */}
                    {(!node.collectionKey || connections.length > 0) && (
                      <g
                        onClick={(e) => {
                          e.stopPropagation()
                          setExpandingNode(isExpanding ? null : node.id)
                        }}
                        style={{ cursor: 'pointer' }}
                      >
                        <circle
                          cx={node.x + 26}
                          cy={node.y - 26}
                          r={10}
                          fill={isExpanding ? '#10b981' : '#374151'}
                          stroke="#6b7280"
                          strokeWidth="1.5"
                        />
                        <text
                          x={node.x + 26}
                          y={node.y - 21}
                          fill="white"
                          fontSize="16"
                          fontWeight="bold"
                          textAnchor="middle"
                          style={{ pointerEvents: 'none' }}
                        >
                          {isExpanding ? '−' : '+'}
                        </text>
                      </g>
                    )}

                    {/* Connection menu */}
                    {isExpanding && (
                      <g>
                        {/* Menu for starter node (choose collection) */}
                        {!node.collectionKey && (
                          <>
                            <rect
                              x={node.x + 45}
                              y={node.y - 60}
                              width="180"
                              height={Math.min(COLLECTION_OPTIONS.length * 28 + 15, 220)}
                              rx="4"
                              fill="rgba(31, 41, 55, 0.98)"
                              stroke="#10b981"
                              strokeWidth="1.5"
                            />
                            <text x={node.x + 55} y={node.y - 42} fill="#9ca3af" fontSize="10">
                              Choose Collection:
                            </text>
                            {COLLECTION_OPTIONS.map((opt, i) => (
                              <g key={opt.key}>
                                <rect
                                  x={node.x + 50}
                                  y={node.y - 25 + i * 28}
                                  width="170"
                                  height="24"
                                  rx="3"
                                  fill="rgba(55, 65, 81, 0.5)"
                                  style={{ cursor: 'pointer' }}
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleInitializeNode(node.id, opt.key)
                                  }}
                                  className="hover:fill-[rgba(75,85,99,0.9)]"
                                />
                                <text
                                  x={node.x + 58}
                                  y={node.y - 9 + i * 28}
                                  fill="#e5e7eb"
                                  fontSize="11"
                                  style={{ pointerEvents: 'none' }}
                                >
                                  {opt.icon} {opt.label}
                                </text>
                              </g>
                            ))}
                          </>
                        )}

                        {/* Menu for initialized nodes (add connections) */}
                        {node.collectionKey && connections.length > 0 && (
                          <>
                            <rect
                              x={node.x + 45}
                              y={node.y - 40}
                              width="200"
                              height={Math.min(connections.length * 28 + 15, 240)}
                              rx="4"
                              fill="rgba(31, 41, 55, 0.98)"
                              stroke="#10b981"
                              strokeWidth="1.5"
                            />
                            <text x={node.x + 55} y={node.y - 22} fill="#9ca3af" fontSize="10">
                              Add Connection:
                            </text>
                            {connections.slice(0, 7).map((conn, i) => (
                              <g key={i}>
                                <rect
                                  x={node.x + 50}
                                  y={node.y - 5 + i * 28}
                                  width="190"
                                  height="24"
                                  rx="3"
                                  fill="rgba(55, 65, 81, 0.5)"
                                  style={{ cursor: 'pointer' }}
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    if (conn.isExisting && conn.targetNodeId) {
                                      handleConnectExisting(node.id, conn.targetNodeId)
                                    } else {
                                      handleAddNode(node.id, conn.target)
                                    }
                                  }}
                                  className="hover:fill-[rgba(75,85,99,0.9)]"
                                />
                                <text
                                  x={node.x + 58}
                                  y={node.y + 11 + i * 28}
                                  fill="#e5e7eb"
                                  fontSize="10"
                                  style={{ pointerEvents: 'none' }}
                                >
                                  {conn.label}
                                </text>
                              </g>
                            ))}
                          </>
                        )}
                      </g>
                    )}
                  </g>
                )
              })}
              </g>
            </svg>
      </div>

      {/* Bottom 20% - Query info */}
      <div className="bg-dark-800/50 rounded-lg border border-gray-700 p-3 flex gap-4 min-h-[140px]">
        {/* Query summary - left side */}
        <div className="flex-shrink-0 w-48 space-y-2">
          <h3 className="text-xs font-bold text-green-400 uppercase tracking-wider">Query</h3>
          <div className="space-y-1 text-xs text-gray-400">
            <div>{nodes.filter(n => n.collectionKey).length} collections</div>
            <div>{edges.length} connections</div>
          </div>
          {edges.length > 0 && (
            <button
              onClick={handleExecuteQuery}
              disabled={isExecuting}
              className="w-full px-3 py-1.5 bg-green-500 hover:bg-green-600 rounded text-white text-xs font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isExecuting ? 'Executing...' : 'Execute Query'}
            </button>
          )}
        </div>

        {/* Field selection - right side */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <AnimatePresence>
            {selectedNode && (() => {
              const node = nodes.find(n => n.id === selectedNode)
              if (!node?.collectionKey) return null

              const schema = GRAPH_SCHEMA[node.collectionKey]
              if (!schema) return null

              const fields = selectedFields[node.id] || []

              return (
                <motion.div
                  key={selectedNode}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex-1 flex flex-col min-h-0"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <h3 className="text-xs font-bold text-white">{node.label}</h3>
                      <p className="text-[10px] text-gray-500 truncate">{schema.description}</p>
                    </div>
                    <button
                      onClick={() => {
                        setSelectedFields(prev => ({
                          ...prev,
                          [node.id]: fields.length === schema.keyFields.length ? [] : schema.keyFields
                        }))
                      }}
                      className="text-xs text-green-400 hover:text-green-300"
                    >
                      {fields.length === schema.keyFields.length ? 'Clear' : 'All'}
                    </button>
                  </div>

                  <div className="flex-1 overflow-y-auto min-h-0">
                    <div className="grid grid-cols-3 gap-1">
                      {schema.keyFields.map(field => {
                        const selected = fields.includes(field)
                        return (
                          <button
                            key={field}
                            onClick={() => {
                              setSelectedFields(prev => {
                                const current = prev[node.id] || []
                                return {
                                  ...prev,
                                  [node.id]: selected
                                    ? current.filter(f => f !== field)
                                    : [...current, field]
                                }
                              })
                            }}
                            className={`px-2 py-1 rounded text-left text-[10px] transition-colors truncate ${
                              selected
                                ? 'bg-green-500/20 text-green-300'
                                : 'bg-dark-700 text-gray-400 hover:bg-dark-600'
                            }`}
                            title={field}
                          >
                            {field}
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  {/* Node-specific controls */}
                  <div className="mt-3 pt-3 border-t border-gray-700 space-y-2">
                    {/* Ticker Search - only for company nodes */}
                    {node.collectionKey === 'company' && (
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={tickerSearch}
                          onChange={(e) => setTickerSearch(e.target.value.toUpperCase())}
                          onKeyDown={(e) => e.key === 'Enter' && handleTickerSearch()}
                          placeholder="Filter ticker (e.g. AAPL)"
                          className="flex-1 px-2 py-1 text-xs bg-dark-800 border border-gray-600 rounded text-white placeholder-gray-500 focus:border-gold/50 outline-none"
                        />
                        <button
                          onClick={handleTickerSearch}
                          className="px-2 py-1 text-xs bg-gold/20 hover:bg-gold/30 border border-gold/50 rounded text-gold transition-colors"
                        >
                          Go
                        </button>
                      </div>
                    )}

                    {/* Show AQL */}
                    {edges.length > 0 && (
                      <button
                        onClick={() => setShowAQL(!showAQL)}
                        className="w-full px-3 py-1 text-xs bg-dark-700 hover:bg-dark-600 border border-gray-600 rounded text-white transition-colors"
                      >
                        {showAQL ? 'Hide' : 'Show'} AQL
                      </button>
                    )}
                  </div>
                </motion.div>
              )
            })()}
          </AnimatePresence>
        </div>
      </div>

      {/* Ticker Selector Modal */}
      <AnimatePresence>
        {showTickerSelector && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
            onClick={() => setShowTickerSelector(false)}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="bg-dark-800 rounded-lg border border-gold/30 w-full max-w-md p-6"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-bold text-gold mb-4">Filter by Ticker</h3>

              {/* Ticker Input */}
              <div className="mb-4">
                <label className="text-xs text-gray-400 mb-2 block">Enter ticker symbols (comma-separated)</label>
                <input
                  type="text"
                  value={tickerInput}
                  onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
                  placeholder="AAPL, MSFT, TSLA"
                  className="w-full px-3 py-2 bg-dark-900 border border-gray-600 rounded text-white placeholder-gray-500 focus:border-gold/50 outline-none"
                  autoFocus
                />
              </div>

              {/* Selected Tickers */}
              {selectedTickers.length > 0 && (
                <div className="mb-4">
                  <div className="text-xs text-gray-400 mb-2">Selected:</div>
                  <div className="flex flex-wrap gap-2">
                    {selectedTickers.map(ticker => (
                      <div
                        key={ticker}
                        className="px-2 py-1 bg-gold/20 border border-gold/50 rounded text-xs text-gold flex items-center gap-2"
                      >
                        {ticker}
                        <button
                          onClick={() => setSelectedTickers(prev => prev.filter(t => t !== ticker))}
                          className="text-gold/70 hover:text-gold"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    const tickers = tickerInput.split(',').map(t => t.trim()).filter(Boolean)
                    setSelectedTickers(prev => Array.from(new Set([...prev, ...tickers])))
                    setTickerInput('')
                  }}
                  className="flex-1 px-4 py-2 bg-gold/20 hover:bg-gold/30 border border-gold/50 rounded text-gold transition-colors"
                >
                  Add Tickers
                </button>
                <button
                  onClick={() => {
                    setShowTickerSelector(false)
                    // TODO: Apply ticker filter to AQL query
                  }}
                  className="flex-1 px-4 py-2 bg-green-500 hover:bg-green-600 rounded text-white transition-colors"
                >
                  Apply Filter
                </button>
              </div>

              <button
                onClick={() => {
                  setSelectedTickers([])
                  setTickerInput('')
                }}
                className="w-full mt-2 text-xs text-gray-500 hover:text-gray-300"
              >
                Clear All
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Report Modal */}
      <AnimatePresence>
        {showReport && reportData && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
            onClick={() => setShowReport(false)}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="bg-dark-800 rounded-lg border border-gold/30 w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-gold/20">
                <div>
                  <h2 className="text-lg font-bold text-gold">Query Results</h2>
                  <p className="text-xs text-gray-400 mt-1">{reportData.description}</p>
                </div>
                <button
                  onClick={() => setShowReport(false)}
                  className="p-2 text-gray-400 hover:text-white transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {/* Stats */}
                <div className="flex items-center gap-6 text-sm">
                  <div>
                    <span className="text-gray-500">Results:</span>
                    <span className="ml-2 text-white font-semibold">{reportData.count}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Execution Time:</span>
                    <span className="ml-2 text-green-400 font-mono">{reportData.execution_time?.toFixed(3)}s</span>
                  </div>
                </div>

                {/* Results Table */}
                {reportData.results && reportData.results.length > 0 ? (
                  <div className="bg-dark-900 rounded border border-gray-700 overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-gray-700">
                          {Object.keys(reportData.results[0]).map((key) => (
                            <th key={key} className="text-left p-2 text-gray-400 font-semibold">
                              {key}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {reportData.results.slice(0, 100).map((row: any, idx: number) => (
                          <tr key={idx} className="border-b border-gray-800 hover:bg-dark-800">
                            {Object.keys(row).map((key) => (
                              <td key={key} className="p-2 text-gray-300">
                                {typeof row[key] === 'object'
                                  ? JSON.stringify(row[key])
                                  : String(row[key])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {reportData.results.length > 100 && (
                      <div className="p-2 text-center text-xs text-gray-500 border-t border-gray-700">
                        Showing first 100 of {reportData.count} results
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    No results found
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
