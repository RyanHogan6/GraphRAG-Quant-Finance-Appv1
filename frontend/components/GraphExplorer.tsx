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

  // Pan move
  const handlePanMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      setPanOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      })
    }
  }, [isPanning, panStart])

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

    return `FOR doc IN ${collection}\n  SORT doc.date DESC\n  LIMIT 100\n  RETURN doc`
  }, [nodes])

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

  return (
    <div className="bg-dark-900/50 p-4 rounded-lg border border-gold/10 space-y-3 h-full flex flex-col">
      {/* Header with ticker search, description and AQL toggle */}
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm text-gray-300 flex-shrink">
          {englishDescription}
        </div>

        {/* Ticker Search */}
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={tickerSearch}
            onChange={(e) => setTickerSearch(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === 'Enter' && handleTickerSearch()}
            placeholder="Ticker (e.g. AAPL)"
            className="px-2 py-1 text-xs bg-dark-800 border border-gray-600 rounded text-white placeholder-gray-500 focus:border-gold/50 outline-none w-28"
          />
          <button
            onClick={handleTickerSearch}
            className="px-2 py-1 text-xs bg-gold/20 hover:bg-gold/30 border border-gold/50 rounded text-gold transition-colors"
          >
            Go
          </button>
        </div>

        {edges.length > 0 && (
          <button
            onClick={() => setShowAQL(!showAQL)}
            className="px-3 py-1 text-xs bg-dark-700 hover:bg-dark-600 border border-gray-600 rounded text-white transition-colors flex-shrink-0"
          >
            {showAQL ? 'Hide' : 'Show'} AQL
          </button>
        )}
      </div>

      {/* AQL display */}
      {showAQL && aqlQuery && (
        <div className="p-3 bg-dark-800 border border-gray-700 rounded font-mono text-xs text-green-400">
          {aqlQuery}
        </div>
      )}

      {/* Graph canvas - 80% height */}
      <div className="flex-1 bg-dark-800/30 rounded-lg border border-green-500/10 relative overflow-hidden">
        {/* Checkered grid background */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
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
                      onClick={() => setSelectedNode(node.id)}
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
      <div className="bg-dark-800/50 rounded-lg border border-gray-700 p-3 flex gap-4 h-[20%] min-h-[120px]">
        {/* Query summary - left side */}
        <div className="flex-shrink-0 w-48 space-y-2">
          <h3 className="text-xs font-bold text-green-400 uppercase tracking-wider">Query</h3>
          <div className="space-y-1 text-xs text-gray-400">
            <div>{nodes.filter(n => n.collectionKey).length} collections</div>
            <div>{edges.length} connections</div>
          </div>
          {edges.length > 0 && (
            <button
              onClick={() => {
                onQueryChange(aqlQuery, englishDescription)
              }}
              className="w-full px-3 py-1.5 bg-green-500 hover:bg-green-600 rounded text-white text-xs font-semibold transition-colors"
            >
              Execute Query
            </button>
          )}
        </div>

        {/* Field selection - right side */}
        <div className="flex-1 overflow-hidden">
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
                  className="h-full flex flex-col"
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

                  <div className="flex-1 overflow-y-auto">
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
                </motion.div>
              )
            })()}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
