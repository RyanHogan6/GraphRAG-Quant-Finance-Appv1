'use client'

import { useState, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GRAPH_SCHEMA } from '@/lib/schema'

interface GraphNode {
  id: string
  collectionKey: string
  label: string
  x: number
  y: number
  layer: number
  parentId: string | null
}

interface GraphEdge {
  id: string
  from: string
  to: string
  label: string
  direction: 'OUTBOUND' | 'INBOUND'
  edgeCollection: string
}

interface GraphExplorerProps {
  onQueryChange: (aql: string, description: string) => void
}

const COLLECTION_OPTIONS = [
  { key: 'company', label: 'Companies', icon: '🏢', description: 'S&P 500 companies' },
  { key: 'marketdata', label: 'Stock Prices', icon: '📈', description: 'Daily OHLCV data' },
  { key: 'options', label: 'Options Flow', icon: '⚡', description: 'Options activity', badge: 'NEW' },
  { key: 'futures', label: 'Commodities', icon: '🛢️', description: 'Futures prices', badge: 'RARE' },
  { key: 'awards', label: 'Gov Contracts', icon: '🎖️', description: 'Government awards' },
  { key: 'sec', label: 'SEC Filings', icon: '📄', description: 'SEC documents' },
  { key: 'sec_sentences', label: 'SEC Sentences', icon: '📝', description: 'SEC filing text' },
  { key: 'sec_exhibits', label: 'SEC Exhibits', icon: '📋', description: 'SEC exhibits' },
  { key: 'predictionmarkets', label: 'Polymarket', icon: '🎲', description: 'Prediction markets' },
  { key: 'kalshi', label: 'Kalshi', icon: '📊', description: 'Kalshi markets' },
  { key: 'commodity_positions', label: 'CFTC Positions', icon: '📈', description: 'Commodity positions' },
]

export default function GraphExplorer({ onQueryChange }: GraphExplorerProps) {
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [expandingNode, setExpandingNode] = useState<string | null>(null)
  const [selectedFields, setSelectedFields] = useState<Record<string, string[]>>({})
  const [showStartPicker, setShowStartPicker] = useState(false)

  // Create root node from collection selection
  const handleStartQuery = useCallback((collectionKey: string) => {
    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    const rootNode: GraphNode = {
      id: `root-${collectionKey}`,
      collectionKey,
      label: schema.name,
      x: 200,
      y: 300,
      layer: 0,
      parentId: null
    }

    setNodes([rootNode])
    setEdges([])
    setSelectedNode(rootNode.id)
    setShowStartPicker(false)
  }, [])

  // Get node color
  const getNodeColor = (collectionKey: string) => {
    if (collectionKey === 'company') return '#fbbf24'
    if (collectionKey === 'awards') return '#fbbf24'
    if (collectionKey.startsWith('sec')) return '#3b82f6'
    if (collectionKey === 'marketdata') return '#10b981'
    if (collectionKey === 'options') return '#8b5cf6'
    if (collectionKey === 'futures' || collectionKey.startsWith('eia')) return '#f59e0b'
    if (collectionKey.includes('prediction') || collectionKey === 'kalshi') return '#ec4899'
    if (collectionKey === 'commodity_positions') return '#f97316'
    return '#6b7280'
  }

  // Get available connections for a node
  const getAvailableConnections = useCallback((node: GraphNode) => {
    const schema = GRAPH_SCHEMA[node.collectionKey]
    if (!schema) return []

    const options: Array<{
      target: string
      targetNodeId?: string
      edge: string
      direction: 'OUTBOUND' | 'INBOUND'
      label: string
      isExisting: boolean
    }> = []

    // Add new node options (OUTBOUND and INBOUND)
    schema.connections.forEach(conn => {
      const targetSchema = GRAPH_SCHEMA[conn.target]
      if (!targetSchema) return

      // Check if we already have edges to prevent duplicate connections
      const existingEdge = edges.find(e =>
        e.from === node.id &&
        e.to.includes(conn.target) &&
        e.edgeCollection === conn.edge
      )

      if (!existingEdge) {
        const dirLabel = conn.direction === 'OUTBOUND' ? '→' : '←'
        options.push({
          target: conn.target,
          edge: conn.edge,
          direction: conn.direction,
          label: `${dirLabel} ${targetSchema.name} (new)`,
          isExisting: false
        })
      }
    })

    // Add existing node options (cross-connections)
    nodes.forEach(existingNode => {
      if (existingNode.id === node.id) return

      // Check if there's a schema connection between these collections
      const outboundConn = schema.connections.find(c =>
        c.target === existingNode.collectionKey &&
        c.direction === 'OUTBOUND'
      )

      const inboundConn = schema.connections.find(c =>
        c.target === existingNode.collectionKey &&
        c.direction === 'INBOUND'
      )

      // Check for existing edge
      const hasExistingEdge = edges.some(e =>
        e.from === node.id && e.to === existingNode.id
      )

      if (outboundConn && !hasExistingEdge) {
        options.push({
          target: existingNode.collectionKey,
          targetNodeId: existingNode.id,
          edge: outboundConn.edge,
          direction: 'OUTBOUND',
          label: `→ ${existingNode.label} (existing)`,
          isExisting: true
        })
      }

      if (inboundConn && !hasExistingEdge) {
        options.push({
          target: existingNode.collectionKey,
          targetNodeId: existingNode.id,
          edge: inboundConn.edge,
          direction: 'INBOUND',
          label: `← ${existingNode.label} (existing)`,
          isExisting: true
        })
      }
    })

    return options
  }, [nodes, edges])

  // Add connection
  const handleAddConnection = useCallback((
    fromNodeId: string,
    targetKey: string,
    targetNodeId: string | undefined,
    edgeCollection: string,
    direction: 'OUTBOUND' | 'INBOUND'
  ) => {
    const fromNode = nodes.find(n => n.id === fromNodeId)
    if (!fromNode) return

    let toNode: GraphNode

    if (targetNodeId) {
      // Connect to existing node
      toNode = nodes.find(n => n.id === targetNodeId)!
    } else {
      // Create new node
      const targetSchema = GRAPH_SCHEMA[targetKey]
      if (!targetSchema) return

      // Calculate position - layer to the right
      const targetLayer = fromNode.layer + 1
      const nodesInLayer = nodes.filter(n => n.layer === targetLayer)

      const baseY = nodesInLayer.length > 0
        ? Math.max(...nodesInLayer.map(n => n.y)) + 100
        : fromNode.y

      toNode = {
        id: `${targetKey}-${Date.now()}`,
        collectionKey: targetKey,
        label: targetSchema.name,
        x: fromNode.x + 300,
        y: baseY,
        layer: targetLayer,
        parentId: fromNodeId
      }

      setNodes(prev => [...prev, toNode])
    }

    // Create edge
    const newEdge: GraphEdge = {
      id: `edge-${Date.now()}`,
      from: fromNodeId,
      to: toNode.id,
      label: edgeCollection,
      direction,
      edgeCollection
    }

    setEdges(prev => [...prev, newEdge])
    setExpandingNode(null)
    setSelectedNode(toNode.id)
  }, [nodes])

  // Calculate viewBox
  const viewBox = useMemo(() => {
    if (nodes.length === 0) return '0 0 1200 600'

    const padding = 150
    const xs = nodes.map(n => n.x)
    const ys = nodes.map(n => n.y)

    const minX = Math.min(...xs) - padding
    const maxX = Math.max(...xs) + padding + 200
    const minY = Math.min(...ys) - padding
    const maxY = Math.max(...ys) + padding

    return `${minX} ${minY} ${maxX - minX} ${maxY - minY}`
  }, [nodes])

  // Generate AQL query
  const generateAQL = useCallback(() => {
    if (nodes.length === 0) return ''

    // Find root node (layer 0)
    const rootNode = nodes.find(n => n.layer === 0)
    if (!rootNode) return ''

    // Simple query for now - will enhance later
    const collectionNames = nodes.map(n => {
      const schema = GRAPH_SCHEMA[n.collectionKey]
      return schema?.collection || n.collectionKey
    }).join(', ')

    const rootCollection = GRAPH_SCHEMA[rootNode.collectionKey]?.collection || rootNode.collectionKey

    let aql = `FOR doc IN ${rootCollection}\n`
    aql += `  SORT doc.date DESC\n`
    aql += `  LIMIT 100\n`
    aql += `  RETURN doc`

    return aql
  }, [nodes, edges])

  // Reset
  const handleReset = () => {
    setNodes([])
    setEdges([])
    setSelectedNode(null)
    setExpandingNode(null)
    setSelectedFields({})
    setShowStartPicker(false)
  }

  return (
    <div className="w-full h-full flex bg-dark-900">
      {/* Main container */}
      <div className="flex-1 flex flex-col p-6">
        <div className="flex-1 bg-dark-800/50 rounded-lg border border-green-500/20 overflow-hidden relative">

          {/* Empty state - Start Query button */}
          {nodes.length === 0 && !showStartPicker && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="absolute inset-0 flex items-center justify-center"
            >
              <button
                onClick={() => setShowStartPicker(true)}
                className="px-8 py-4 bg-green-500/20 hover:bg-green-500/30 border-2 border-green-500 rounded-lg text-green-400 font-bold text-lg transition-all hover:scale-105"
              >
                ▶ Start Query
              </button>
            </motion.div>
          )}

          {/* Collection picker modal */}
          {showStartPicker && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute inset-0 bg-dark-900/95 flex items-center justify-center z-50"
            >
              <div className="w-full max-w-2xl p-8">
                <div className="mb-6 text-center">
                  <h2 className="text-2xl font-bold text-white mb-2">Select Starting Collection</h2>
                  <p className="text-gray-400 text-sm">Choose where to begin your query</p>
                </div>

                <div className="grid grid-cols-2 gap-3 max-h-[500px] overflow-y-auto">
                  {COLLECTION_OPTIONS.map(option => (
                    <button
                      key={option.key}
                      onClick={() => handleStartQuery(option.key)}
                      className="p-4 bg-dark-800 hover:bg-dark-700 border border-gray-700 hover:border-green-500/50 rounded-lg text-left transition-all group"
                    >
                      <div className="flex items-start gap-3">
                        <span className="text-2xl">{option.icon}</span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-semibold text-white group-hover:text-green-400">
                              {option.label}
                            </h3>
                            {option.badge && (
                              <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[9px] font-bold rounded">
                                {option.badge}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-gray-500">{option.description}</p>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                <button
                  onClick={() => setShowStartPicker(false)}
                  className="mt-6 w-full py-2 text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          )}

          {/* Graph SVG */}
          {nodes.length > 0 && (
            <svg className="w-full h-full" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
              <defs>
                <filter id="node-glow">
                  <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>

                {/* Arrow markers for different edge types */}
                <marker id="arrow-outbound" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L9,3 z" fill="#10b981" />
                </marker>
                <marker id="arrow-inbound" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L9,3 z" fill="#3b82f6" />
                </marker>
              </defs>

              {/* Draw edges */}
              {edges.map((edge) => {
                const fromNode = nodes.find(n => n.id === edge.from)
                const toNode = nodes.find(n => n.id === edge.to)
                if (!fromNode || !toNode) return null

                const isOutbound = edge.direction === 'OUTBOUND'
                const color = isOutbound ? '#10b981' : '#3b82f6'
                const marker = isOutbound ? 'url(#arrow-outbound)' : 'url(#arrow-inbound)'

                const fromX = fromNode.x + 35
                const toX = toNode.x - 35
                const midX = (fromX + toX) / 2
                const midY = (fromNode.y + toNode.y) / 2

                // Use curved path for cross-layer connections
                const isCrossConnection = Math.abs(fromNode.layer - toNode.layer) > 1 ||
                                        fromNode.layer === toNode.layer

                return (
                  <motion.g key={edge.id}>
                    <motion.path
                      d={isCrossConnection
                        ? `M ${fromX} ${fromNode.y} Q ${midX} ${midY - 50} ${toX} ${toNode.y}`
                        : `M ${fromX} ${fromNode.y} Q ${midX} ${fromNode.y} ${toX} ${toNode.y}`
                      }
                      stroke={color}
                      strokeWidth="2"
                      fill="none"
                      strokeOpacity="0.6"
                      markerEnd={marker}
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5 }}
                    />

                    {/* Edge label */}
                    <text
                      x={midX}
                      y={isCrossConnection ? midY - 55 : fromNode.y - 10}
                      textAnchor="middle"
                      fill={color}
                      fontSize="9"
                      fontWeight="600"
                      opacity="0.8"
                    >
                      {edge.label}
                    </text>
                  </motion.g>
                )
              })}

              {/* Draw nodes */}
              {nodes.map((node) => {
                const color = getNodeColor(node.collectionKey)
                const isSelected = selectedNode === node.id
                const isExpanding = expandingNode === node.id
                const availableConnections = getAvailableConnections(node)
                const canExpand = availableConnections.length > 0

                return (
                  <motion.g
                    key={node.id}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    {/* Node circle */}
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={35}
                      fill="rgba(17, 24, 39, 0.95)"
                      stroke={color}
                      strokeWidth={isSelected ? 3 : 2}
                      filter={isSelected ? 'url(#node-glow)' : undefined}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setSelectedNode(node.id)}
                    />

                    {/* Node label */}
                    <text
                      x={node.x}
                      y={node.y}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill={color}
                      fontSize={11}
                      fontWeight="600"
                      style={{ cursor: 'pointer', pointerEvents: 'none' }}
                    >
                      {node.label}
                    </text>

                    {/* Layer badge */}
                    <circle cx={node.x - 26} cy={node.y - 26} r="9" fill="#374151" opacity="0.9" />
                    <text
                      x={node.x - 26}
                      y={node.y - 22}
                      textAnchor="middle"
                      fill="white"
                      fontSize="9"
                      fontWeight="bold"
                      style={{ pointerEvents: 'none' }}
                    >
                      {node.layer + 1}
                    </text>

                    {/* Expand button */}
                    {canExpand && (
                      <g onClick={(e) => {
                        e.stopPropagation()
                        setExpandingNode(isExpanding ? null : node.id)
                      }}>
                        <circle
                          cx={node.x + 26}
                          cy={node.y - 26}
                          r="11"
                          fill={isExpanding ? '#10b981' : '#374151'}
                          stroke={isExpanding ? '#10b981' : '#6b7280'}
                          strokeWidth="2"
                          style={{ cursor: 'pointer' }}
                        />
                        <text
                          x={node.x + 26}
                          y={node.y - 21}
                          textAnchor="middle"
                          fill="white"
                          fontSize="16"
                          fontWeight="bold"
                          style={{ cursor: 'pointer', pointerEvents: 'none' }}
                        >
                          {isExpanding ? '−' : '+'}
                        </text>
                      </g>
                    )}

                    {/* Connection menu */}
                    {isExpanding && availableConnections.length > 0 && (
                      <motion.g
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                      >
                        <rect
                          x={node.x + 50}
                          y={node.y - 40}
                          width="200"
                          height={Math.min(availableConnections.length * 30 + 20, 300)}
                          rx="6"
                          fill="rgba(31, 41, 55, 0.98)"
                          stroke="#10b981"
                          strokeWidth="2"
                        />
                        <text
                          x={node.x + 60}
                          y={node.y - 20}
                          fill="#9ca3af"
                          fontSize="10"
                          fontWeight="600"
                        >
                          Add Connection:
                        </text>

                        {availableConnections.slice(0, 8).map((conn, i) => {
                          const connY = node.y + i * 30

                          return (
                            <g key={i}>
                              <rect
                                x={node.x + 55}
                                y={connY}
                                width="190"
                                height="26"
                                rx="4"
                                fill="rgba(55, 65, 81, 0.5)"
                                style={{ cursor: 'pointer' }}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleAddConnection(
                                    node.id,
                                    conn.target,
                                    conn.targetNodeId,
                                    conn.edge,
                                    conn.direction
                                  )
                                }}
                                className="hover:fill-[rgba(75,85,99,0.8)]"
                              />
                              <text
                                x={node.x + 65}
                                y={connY + 17}
                                fill="#e5e7eb"
                                fontSize="11"
                                fontWeight="500"
                                style={{ cursor: 'pointer', pointerEvents: 'none' }}
                              >
                                {conn.label}
                              </text>
                            </g>
                          )
                        })}
                      </motion.g>
                    )}
                  </motion.g>
                )
              })}

              {/* Execute panel */}
              {edges.length > 0 && (() => {
                const maxX = Math.max(...nodes.map(n => n.x))
                const avgY = nodes.reduce((sum, n) => sum + n.y, 0) / nodes.length

                return (
                  <motion.g
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    <rect
                      x={maxX + 80}
                      y={avgY - 70}
                      width="160"
                      height="140"
                      rx="8"
                      fill="rgba(34, 197, 94, 0.1)"
                      stroke="#10b981"
                      strokeWidth="2"
                    />
                    <text x={maxX + 160} y={avgY - 40} textAnchor="middle" fill="#10b981" fontSize="13" fontWeight="bold">
                      Query Summary
                    </text>
                    <text x={maxX + 160} y={avgY - 18} textAnchor="middle" fill="#9ca3af" fontSize="11">
                      {nodes.length} collections
                    </text>
                    <text x={maxX + 160} y={avgY} textAnchor="middle" fill="#9ca3af" fontSize="11">
                      {edges.length} connections
                    </text>

                    <rect
                      x={maxX + 100}
                      y={avgY + 25}
                      width="120"
                      height="35"
                      rx="6"
                      fill="#10b981"
                      style={{ cursor: 'pointer' }}
                      onClick={() => {
                        const aql = generateAQL()
                        onQueryChange(aql, `Journey: ${nodes.map(n => n.label).join(' → ')}`)
                      }}
                    />
                    <text
                      x={maxX + 160}
                      y={avgY + 48}
                      textAnchor="middle"
                      fill="white"
                      fontSize="12"
                      fontWeight="bold"
                      style={{ cursor: 'pointer', pointerEvents: 'none' }}
                    >
                      Execute Query
                    </text>
                  </motion.g>
                )
              })()}
            </svg>
          )}

          {/* Instructions */}
          {nodes.length > 0 && edges.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute bottom-6 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-green-500/10 border border-green-500/30 rounded-lg"
            >
              <p className="text-green-400 text-sm">
                Click the <span className="font-bold">+</span> button to add connections
              </p>
            </motion.div>
          )}
        </div>

        {/* Bottom controls */}
        <div className="mt-4 flex gap-3">
          {nodes.length > 0 && (
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-dark-700 hover:bg-dark-600 border border-gray-600 rounded-lg text-sm text-white transition-colors"
            >
              Start Over
            </button>
          )}
          {edges.length > 0 && (
            <div className="flex-1 text-right text-xs text-gray-500">
              <span className="text-green-400">→ OUTBOUND</span> · <span className="text-blue-400">← INBOUND</span>
            </div>
          )}
        </div>
      </div>

      {/* Right panel - Field selection */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ x: 320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 320, opacity: 0 }}
            className="w-80 bg-dark-800 border-l border-green-500/20 p-6 overflow-y-auto"
          >
            {(() => {
              const node = nodes.find(n => n.id === selectedNode)
              if (!node) return null

              const schema = GRAPH_SCHEMA[node.collectionKey]
              if (!schema) return null

              const nodeFields = selectedFields[node.id] || []

              return (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-bold text-white mb-1">{node.label}</h3>
                    <p className="text-xs text-gray-400">{schema.description}</p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-sm font-semibold text-gray-300">Select Fields</p>
                      <button
                        onClick={() => {
                          const allFields = schema.keyFields
                          setSelectedFields(prev => ({
                            ...prev,
                            [node.id]: nodeFields.length === allFields.length ? [] : allFields
                          }))
                        }}
                        className="text-xs text-green-400 hover:text-green-300"
                      >
                        {nodeFields.length === schema.keyFields.length ? 'Deselect All' : 'Select All'}
                      </button>
                    </div>

                    <div className="space-y-2 max-h-[400px] overflow-y-auto">
                      {schema.keyFields.map(fieldName => {
                        const isSelected = nodeFields.includes(fieldName)

                        return (
                          <button
                            key={fieldName}
                            onClick={() => {
                              setSelectedFields(prev => {
                                const current = prev[node.id] || []
                                const updated = isSelected
                                  ? current.filter(f => f !== fieldName)
                                  : [...current, fieldName]
                                return { ...prev, [node.id]: updated }
                              })
                            }}
                            className={`w-full p-2 rounded border text-left transition-all ${
                              isSelected
                                ? 'bg-green-500/20 border-green-500/40 text-green-300'
                                : 'bg-dark-700 border-gray-600 text-gray-400 hover:border-gray-500'
                            }`}
                          >
                            <div className="flex items-start gap-2">
                              <div className={`mt-0.5 w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 ${
                                isSelected ? 'bg-green-500 border-green-500' : 'border-gray-500'
                              }`}>
                                {isSelected && (
                                  <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                  </svg>
                                )}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="text-xs font-mono truncate">{fieldName}</div>
                              </div>
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  <div className="pt-4 border-t border-gray-700">
                    <p className="text-xs text-gray-500">
                      Layer {node.layer + 1} • {nodeFields.length} of {schema.keyFields.length} fields selected
                    </p>
                  </div>
                </div>
              )
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
