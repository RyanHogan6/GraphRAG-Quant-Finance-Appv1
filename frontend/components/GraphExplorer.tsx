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
  pathFromRoot: string[] // Track collections in path to prevent loops
}

interface GraphEdge {
  from: string
  to: string
  label: string
}

interface GraphExplorerProps {
  onQueryChange: (aql: string, description: string) => void
}

const STARTING_OPTIONS = [
  { key: 'company', label: 'Companies', icon: '🏢' },
  { key: 'awards', label: 'Gov Contracts', icon: '🎖️' },
  { key: 'sec', label: 'SEC Filings', icon: '📄' },
  { key: 'marketdata', label: 'Stock Prices', icon: '📈' },
  { key: 'options', label: 'Options Flow', icon: '⚡', badge: 'NEW' },
  { key: 'futures', label: 'Commodities', icon: '🛢️', badge: 'RARE' },
]

export default function GraphExplorer({ onQueryChange }: GraphExplorerProps) {
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [expandingNode, setExpandingNode] = useState<string | null>(null) // Show connection menu
  const [showRootSelector, setShowRootSelector] = useState(true)
  const [selectedFields, setSelectedFields] = useState<Record<string, string[]>>({}) // nodeId -> fields

  // Set root type
  const handleSetRootType = useCallback((collectionKey: string) => {
    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    const rootNode: GraphNode = {
      id: 'root',
      collectionKey,
      label: schema.name,
      x: 200,
      y: 300,
      layer: 0,
      parentId: null,
      pathFromRoot: [collectionKey]
    }

    setNodes([rootNode])
    setShowRootSelector(false)
    setSelectedNode('root')
  }, [])

  // Add a single connection from a node
  const handleAddConnection = useCallback((fromNodeId: string, targetKey: string, edgeLabel: string) => {
    const fromNode = nodes.find(n => n.id === fromNodeId)
    if (!fromNode) return

    const targetSchema = GRAPH_SCHEMA[targetKey]
    if (!targetSchema) return

    // Check for circular loop
    if (fromNode.pathFromRoot.includes(targetKey)) {
      console.log('Prevented circular loop:', targetKey)
      return
    }

    // Calculate position - find existing nodes in this layer
    const targetLayer = fromNode.layer + 1
    const nodesInTargetLayer = nodes.filter(n => n.layer === targetLayer)

    const layerX = fromNode.x + 350
    const baseY = nodesInTargetLayer.length > 0
      ? Math.max(...nodesInTargetLayer.map(n => n.y)) + 120
      : fromNode.y

    const newNodeId = `${targetKey}-${Date.now()}`
    const newNode: GraphNode = {
      id: newNodeId,
      collectionKey: targetKey,
      label: targetSchema.name,
      x: layerX,
      y: baseY,
      layer: targetLayer,
      parentId: fromNodeId,
      pathFromRoot: [...fromNode.pathFromRoot, targetKey]
    }

    const newEdge: GraphEdge = {
      from: fromNodeId,
      to: newNodeId,
      label: edgeLabel
    }

    setNodes(prev => [...prev, newNode])
    setEdges(prev => [...prev, newEdge])
    setExpandingNode(null)
    setSelectedNode(newNodeId)
  }, [nodes])

  // Get node color
  const getNodeColor = (collectionKey: string) => {
    if (collectionKey === 'company') return '#fbbf24'
    if (collectionKey === 'awards') return '#fbbf24'
    if (collectionKey.startsWith('sec')) return '#3b82f6'
    if (collectionKey === 'marketdata') return '#10b981'
    if (collectionKey === 'options') return '#8b5cf6'
    if (collectionKey === 'futures' || collectionKey.startsWith('eia')) return '#f59e0b'
    if (collectionKey.includes('prediction') || collectionKey === 'kalshi') return '#ec4899'
    return '#6b7280'
  }

  // Calculate viewBox
  const viewBox = useMemo(() => {
    if (nodes.length === 0) return '0 0 1200 600'

    const padding = 100
    const xs = nodes.map(n => n.x)
    const ys = nodes.map(n => n.y)

    const minX = Math.min(...xs) - padding
    const maxX = Math.max(...xs) + padding + 200
    const minY = Math.min(...ys) - padding
    const maxY = Math.max(...ys) + padding

    return `${minX} ${minY} ${maxX - minX} ${maxY - minY}`
  }, [nodes])

  // Get available connections (excluding loops)
  const getAvailableConnections = useCallback((node: GraphNode) => {
    const schema = GRAPH_SCHEMA[node.collectionKey]
    if (!schema) return []

    return schema.connections.filter(conn =>
      !node.pathFromRoot.includes(conn.target) // Prevent loops
    )
  }, [])

  // Reset
  const handleReset = () => {
    setNodes([])
    setEdges([])
    setSelectedNode(null)
    setExpandingNode(null)
    setShowRootSelector(true)
    setSelectedFields({})
  }

  return (
    <div className="w-full h-full flex bg-dark-900">
      {/* Main container - like LLM interface */}
      <div className="flex-1 flex flex-col p-6">
        <div className="flex-1 bg-dark-800/50 rounded-lg border border-green-500/20 overflow-hidden relative">
          {/* Starting selector */}
          {showRootSelector && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="max-w-2xl w-full p-8">
                <h2 className="text-xl font-bold text-white mb-4 text-center">Select Starting Point</h2>
                <div className="grid grid-cols-3 gap-3">
                  {STARTING_OPTIONS.map(option => (
                    <button
                      key={option.key}
                      onClick={() => handleSetRootType(option.key)}
                      className="p-4 bg-dark-700 hover:bg-dark-600 border border-green-500/20 hover:border-green-500/40
                               rounded-lg transition-all text-center group relative"
                    >
                      {option.badge && (
                        <span className="absolute top-2 right-2 px-1.5 py-0.5 text-[10px] font-bold bg-green-500/20 text-green-400 rounded">
                          {option.badge}
                        </span>
                      )}
                      <div className="text-3xl mb-2">{option.icon}</div>
                      <div className="text-sm font-medium text-white group-hover:text-green-400 transition-colors">
                        {option.label}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Graph SVG */}
          {!showRootSelector && (
            <svg className="w-full h-full" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
              <defs>
                <filter id="node-glow">
                  <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
                <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L9,3 z" fill="#10b981" />
                </marker>
              </defs>

              {/* Draw edges */}
              {edges.map((edge, index) => {
                const fromNode = nodes.find(n => n.id === edge.from)
                const toNode = nodes.find(n => n.id === edge.to)
                if (!fromNode || !toNode) return null

                const fromX = fromNode.x + 45
                const toX = toNode.x - 45
                const midX = (fromX + toX) / 2

                return (
                  <motion.g key={`edge-${edge.from}-${edge.to}`}>
                    <motion.path
                      d={`M ${fromX} ${fromNode.y} Q ${midX} ${fromNode.y} ${toX} ${toNode.y}`}
                      stroke="#10b981"
                      strokeWidth="2"
                      fill="none"
                      strokeOpacity="0.4"
                      markerEnd="url(#arrow-green)"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5 }}
                    />
                  </motion.g>
                )
              })}

              {/* Draw nodes */}
              {nodes.map((node, index) => {
                const color = getNodeColor(node.collectionKey)
                const isSelected = selectedNode === node.id
                const isExpanding = expandingNode === node.id
                const availableConnections = getAvailableConnections(node)
                const canExpand = availableConnections.length > 0 && node.layer < 3

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

                    {/* Layer badge */}
                    <circle cx={node.x - 28} cy={node.y - 28} r="10" fill="#374151" opacity="0.9" />
                    <text
                      x={node.x - 28}
                      y={node.y - 25}
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
                          y={node.y - 24}
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
                          x={node.x + 60}
                          y={node.y - 40}
                          width="180"
                          height={Math.min(availableConnections.length * 32 + 16, 200)}
                          rx="6"
                          fill="rgba(31, 41, 55, 0.98)"
                          stroke="#10b981"
                          strokeWidth="2"
                        />
                        <text
                          x={node.x + 70}
                          y={node.y - 20}
                          fill="#9ca3af"
                          fontSize="10"
                          fontWeight="600"
                        >
                          Add Connection:
                        </text>

                        {availableConnections.slice(0, 5).map((conn, i) => {
                          const targetSchema = GRAPH_SCHEMA[conn.target]
                          const connectionY = node.y + i * 32 - 6

                          return (
                            <g key={i}>
                              <rect
                                x={node.x + 65}
                                y={connectionY}
                                width="170"
                                height="28"
                                rx="4"
                                fill="rgba(55, 65, 81, 0.5)"
                                style={{ cursor: 'pointer' }}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleAddConnection(node.id, conn.target, conn.edge)
                                }}
                                className="hover:fill-[rgba(75,85,99,0.8)]"
                              />
                              <text
                                x={node.x + 75}
                                y={connectionY + 18}
                                fill="#e5e7eb"
                                fontSize="11"
                                fontWeight="500"
                                style={{ cursor: 'pointer', pointerEvents: 'none' }}
                              >
                                {targetSchema?.name || conn.target}
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
              {nodes.length > 0 && (() => {
                const maxX = Math.max(...nodes.map(n => n.x))
                const executeX = maxX + 250

                return (
                  <motion.g
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    <rect
                      x={executeX - 90}
                      y={250}
                      width="180"
                      height="160"
                      rx="8"
                      fill="rgba(34, 197, 94, 0.1)"
                      stroke="#10b981"
                      strokeWidth="2"
                    />
                    <text x={executeX} y={285} textAnchor="middle" fill="#10b981" fontSize="14" fontWeight="bold">
                      Query Summary
                    </text>
                    <text x={executeX} y={310} textAnchor="middle" fill="#9ca3af" fontSize="11">
                      {nodes.length} collections
                    </text>
                    <text x={executeX} y={330} textAnchor="middle" fill="#9ca3af" fontSize="11">
                      {edges.length} connections
                    </text>

                    <rect
                      x={executeX - 60}
                      y={355}
                      width="120"
                      height="36"
                      rx="6"
                      fill="#10b981"
                      style={{ cursor: 'pointer' }}
                      onClick={() => {
                        const aql = `FOR doc IN ${nodes[0].collectionKey}\n  LIMIT 20\n  RETURN doc`
                        onQueryChange(aql, `Journey: ${nodes.map(n => n.label).join(' → ')}`)
                      }}
                    />
                    <text
                      x={executeX}
                      y={378}
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
          {nodes.length === 1 && !expandingNode && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute bottom-6 left-1/2 transform -translate-x-1/2 px-4 py-2 bg-green-500/10 border border-green-500/30 rounded-lg"
            >
              <p className="text-green-400 text-sm">
                Click the <span className="font-bold">+</span> button to add connections
              </p>
            </motion.div>
          )}
        </div>

        {/* Reset button */}
        {!showRootSelector && (
          <div className="mt-4">
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-dark-700 hover:bg-dark-600 border border-gray-600 rounded-lg text-sm text-white transition-colors"
            >
              Start Over
            </button>
          </div>
        )}
      </div>

      {/* Right panel - Field selection */}
      <AnimatePresence>
        {selectedNode && !showRootSelector && (
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
