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
  { key: 'options', label: 'Options Flow', icon: '⚡', color: '#8b5cf6', badge: 'NEW' },
  { key: 'futures', label: 'Commodities', icon: '🛢️', color: '#f59e0b', badge: 'RARE' },
  { key: 'awards', label: 'Gov Contracts', icon: '🎖️', color: '#fbbf24' },
  { key: 'sec', label: 'SEC Filings', icon: '📄', color: '#3b82f6' },
  { key: 'predictionmarkets', label: 'Polymarket', icon: '🎲', color: '#ec4899' },
  { key: 'kalshi', label: 'Kalshi', icon: '📊', color: '#ec4899' },
]

export default function GraphExplorer({ onQueryChange }: GraphExplorerProps) {
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [expandingNode, setExpandingNode] = useState<string | null>(null)
  const [selectedFields, setSelectedFields] = useState<Record<string, string[]>>({})

  // Get node color
  const getNodeColor = (collectionKey: string) => {
    const option = COLLECTION_OPTIONS.find(o => o.key === collectionKey)
    return option?.color || '#6b7280'
  }

  // Add first node (shows dropdown immediately)
  const handleAddFirstNode = useCallback((collectionKey: string) => {
    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    const newNode: GraphNode = {
      id: collectionKey, // Use collectionKey as ID (ensures uniqueness)
      collectionKey,
      label: schema.name,
      x: 400,
      y: 300
    }

    setNodes([newNode])
    setSelectedNode(newNode.id)
  }, [])

  // Add additional node (only if doesn't exist)
  const handleAddNode = useCallback((collectionKey: string, fromNodeId?: string) => {
    // Check if node already exists
    if (nodes.some(n => n.collectionKey === collectionKey)) {
      console.log('Node already exists:', collectionKey)
      return
    }

    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    // Calculate position - arrange in circle around center
    const angleStep = (2 * Math.PI) / (nodes.length + 1)
    const radius = 250
    const centerX = 400
    const centerY = 300
    const angle = nodes.length * angleStep

    const newNode: GraphNode = {
      id: collectionKey,
      collectionKey,
      label: schema.name,
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle)
    }

    setNodes(prev => [...prev, newNode])
    setSelectedNode(newNode.id)

    // If adding from another node, create edge
    if (fromNodeId) {
      handleAddEdge(fromNodeId, collectionKey)
    }

    setExpandingNode(null)
  }, [nodes])

  // Add edge between two nodes
  const handleAddEdge = useCallback((fromId: string, toId: string) => {
    const fromNode = nodes.find(n => n.id === fromId)
    const toNode = nodes.find(n => n.id === toId)
    if (!fromNode || !toNode) return

    // Check if edge already exists
    const edgeExists = edges.some(e =>
      (e.from === fromId && e.to === toId) ||
      (e.from === toId && e.to === fromId)
    )
    if (edgeExists) return

    // Find edge definition in schema
    const fromSchema = GRAPH_SCHEMA[fromNode.collectionKey]
    const connection = fromSchema?.connections.find(c => c.target === toNode.collectionKey)

    if (connection) {
      const newEdge: GraphEdge = {
        id: `edge-${fromId}-${toId}`,
        from: fromId,
        to: toId,
        label: connection.edge,
        direction: connection.direction
      }

      setEdges(prev => [...prev, newEdge])
    }

    setExpandingNode(null)
  }, [nodes, edges])

  // Get available connections for a node
  const getAvailableConnections = useCallback((node: GraphNode) => {
    const schema = GRAPH_SCHEMA[node.collectionKey]
    if (!schema) return []

    return schema.connections
      .map(conn => {
        const targetSchema = GRAPH_SCHEMA[conn.target]
        if (!targetSchema) return null

        const targetExists = nodes.some(n => n.collectionKey === conn.target)
        const edgeExists = edges.some(e =>
          (e.from === node.id && e.to === conn.target) ||
          (e.from === conn.target && e.to === node.id)
        )

        if (edgeExists) return null // Already connected

        const dirLabel = conn.direction === 'OUTBOUND' ? '→' : '←'

        return {
          target: conn.target,
          label: `${dirLabel} ${targetSchema.name}${targetExists ? ' (connect)' : ''}`,
          exists: targetExists,
          direction: conn.direction,
          edgeLabel: conn.edge
        }
      })
      .filter(Boolean)
  }, [nodes, edges])

  // Handle connection click
  const handleConnectionClick = useCallback((fromNodeId: string, targetKey: string, exists: boolean) => {
    if (exists) {
      // Just add edge to existing node
      handleAddEdge(fromNodeId, targetKey)
    } else {
      // Add new node + edge
      handleAddNode(targetKey, fromNodeId)
    }
  }, [handleAddEdge, handleAddNode])

  // Calculate viewBox
  const viewBox = useMemo(() => {
    if (nodes.length === 0) return '0 0 800 600'

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

    const rootNode = nodes[0]
    const rootCollection = GRAPH_SCHEMA[rootNode.collectionKey]?.collection || rootNode.collectionKey

    let aql = `FOR doc IN ${rootCollection}\n`
    aql += `  SORT doc.date DESC\n`
    aql += `  LIMIT 100\n`
    aql += `  RETURN doc`

    return aql
  }, [nodes])

  // Reset
  const handleReset = () => {
    setNodes([])
    setEdges([])
    setSelectedNode(null)
    setExpandingNode(null)
    setSelectedFields({})
  }

  return (
    <div className="w-full h-full flex bg-dark-900">
      {/* Main container */}
      <div className="flex-1 flex flex-col p-6">
        <div className="flex-1 bg-dark-800/50 rounded-lg border border-green-500/20 overflow-hidden relative">

          {/* Empty state - Dropdown selector */}
          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-96">
                <h2 className="text-xl font-bold text-white mb-4 text-center">Start Your Query</h2>
                <div className="space-y-2">
                  {COLLECTION_OPTIONS.map(option => (
                    <button
                      key={option.key}
                      onClick={() => handleAddFirstNode(option.key)}
                      className="w-full p-3 bg-dark-800 hover:bg-dark-700 border border-gray-700 hover:border-green-500/50 rounded-lg text-left transition-all group flex items-center gap-3"
                    >
                      <span className="text-2xl">{option.icon}</span>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-white group-hover:text-green-400">
                            {option.label}
                          </span>
                          {option.badge && (
                            <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[9px] font-bold rounded">
                              {option.badge}
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
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

                const angle = Math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x)
                const fromX = fromNode.x + 40 * Math.cos(angle)
                const fromY = fromNode.y + 40 * Math.sin(angle)
                const toX = toNode.x - 40 * Math.cos(angle)
                const toY = toNode.y - 40 * Math.sin(angle)

                const midX = (fromX + toX) / 2
                const midY = (fromY + toY) / 2

                return (
                  <motion.g key={edge.id}>
                    <motion.line
                      x1={fromX}
                      y1={fromY}
                      x2={toX}
                      y2={toY}
                      stroke={color}
                      strokeWidth="2"
                      strokeOpacity="0.6"
                      markerEnd={marker}
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5 }}
                    />
                    <text
                      x={midX}
                      y={midY - 8}
                      textAnchor="middle"
                      fill={color}
                      fontSize="10"
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
                      r={40}
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
                      fontSize={12}
                      fontWeight="600"
                      style={{ cursor: 'pointer', pointerEvents: 'none' }}
                    >
                      {node.label}
                    </text>

                    {/* Expand button */}
                    {canExpand && (
                      <g onClick={(e) => {
                        e.stopPropagation()
                        setExpandingNode(isExpanding ? null : node.id)
                      }}>
                        <circle
                          cx={node.x + 28}
                          cy={node.y - 28}
                          r="12"
                          fill={isExpanding ? '#10b981' : '#374151'}
                          stroke={isExpanding ? '#10b981' : '#6b7280'}
                          strokeWidth="2"
                          style={{ cursor: 'pointer' }}
                        />
                        <text
                          x={node.x + 28}
                          y={node.y - 22}
                          textAnchor="middle"
                          fill="white"
                          fontSize="18"
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
                          x={node.x + 55}
                          y={node.y - 50}
                          width="220"
                          height={Math.min(availableConnections.length * 32 + 20, 320)}
                          rx="6"
                          fill="rgba(31, 41, 55, 0.98)"
                          stroke="#10b981"
                          strokeWidth="2"
                        />
                        <text
                          x={node.x + 65}
                          y={node.y - 28}
                          fill="#9ca3af"
                          fontSize="11"
                          fontWeight="600"
                        >
                          Add Connection:
                        </text>

                        {availableConnections.slice(0, 8).map((conn, i) => {
                          const connY = node.y - 8 + i * 32

                          return (
                            <g key={i}>
                              <rect
                                x={node.x + 60}
                                y={connY}
                                width="210"
                                height="28"
                                rx="4"
                                fill="rgba(55, 65, 81, 0.5)"
                                style={{ cursor: 'pointer' }}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleConnectionClick(node.id, conn.target, conn.exists)
                                }}
                                className="hover:fill-[rgba(75,85,99,0.8)]"
                              />
                              <text
                                x={node.x + 70}
                                y={connY + 18}
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
              {edges.length > 0 && (
                <motion.g
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <rect
                    x={50}
                    y={50}
                    width="180"
                    height="140"
                    rx="8"
                    fill="rgba(34, 197, 94, 0.1)"
                    stroke="#10b981"
                    strokeWidth="2"
                  />
                  <text x={140} y={85} textAnchor="middle" fill="#10b981" fontSize="14" fontWeight="bold">
                    Query Summary
                  </text>
                  <text x={140} y={108} textAnchor="middle" fill="#9ca3af" fontSize="11">
                    {nodes.length} collections
                  </text>
                  <text x={140} y={128} textAnchor="middle" fill="#9ca3af" fontSize="11">
                    {edges.length} connections
                  </text>

                  <rect
                    x={70}
                    y={145}
                    width="140"
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
                    x={140}
                    y={167}
                    textAnchor="middle"
                    fill="white"
                    fontSize="12"
                    fontWeight="bold"
                    style={{ cursor: 'pointer', pointerEvents: 'none' }}
                  >
                    Execute Query
                  </text>
                </motion.g>
              )}
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
                Click <span className="font-bold">+</span> to add connections
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
                      {nodeFields.length} of {schema.keyFields.length} fields selected
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
