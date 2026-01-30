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
  isExpanded: boolean
  isRoot: boolean
  depth: number
}

interface GraphEdge {
  from: string
  to: string
  label: string
}

interface GraphExplorerProps {
  onQueryChange: (aql: string, description: string) => void
}

export default function GraphExplorer({ onQueryChange }: GraphExplorerProps) {
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [showStartSelector, setShowStartSelector] = useState(true)

  // Start exploration from a collection
  const handleStartExploration = useCallback((collectionKey: string) => {
    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    const rootNode: GraphNode = {
      id: `root-${collectionKey}`,
      collectionKey,
      label: schema.name,
      x: 0,
      y: 0,
      isExpanded: false,
      isRoot: true,
      depth: 0
    }

    setNodes([rootNode])
    setEdges([])
    setSelectedNode(rootNode.id)
    setShowStartSelector(false)
  }, [])

  // Expand a node to show its connections
  const handleExpandNode = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId)
    if (!node || node.isExpanded) return

    // Depth limit: Don't expand beyond depth 2
    if (node.depth >= 2) {
      console.log('Max depth reached - not expanding')
      return
    }

    const schema = GRAPH_SCHEMA[node.collectionKey]
    if (!schema) return

    // Calculate positions in a radial layout around the node
    const connections = schema.connections
    const angleStep = (Math.PI * 2) / connections.length
    const radius = 250

    const newNodes: GraphNode[] = []
    const newEdges: GraphEdge[] = []

    connections.forEach((conn, index) => {
      const targetSchema = GRAPH_SCHEMA[conn.target]
      if (!targetSchema) return

      // Check if a node with this collectionKey already exists
      const existingNode = nodes.find(n => n.collectionKey === conn.target)

      if (existingNode) {
        // Reuse existing node - just add an edge
        newEdges.push({
          from: nodeId,
          to: existingNode.id,
          label: conn.edge
        })
      } else {
        // Create new node
        const angle = angleStep * index - Math.PI / 2 // Start from top
        const x = node.x + Math.cos(angle) * radius
        const y = node.y + Math.sin(angle) * radius

        const newNodeId = `${conn.target}-${Date.now()}-${index}` // Unique ID based on collection type

        newNodes.push({
          id: newNodeId,
          collectionKey: conn.target,
          label: targetSchema.name,
          x,
          y,
          isExpanded: false,
          isRoot: false,
          depth: node.depth + 1
        })

        newEdges.push({
          from: nodeId,
          to: newNodeId,
          label: conn.edge
        })
      }
    })

    setNodes(prev => [...prev, ...newNodes])
    setEdges(prev => [...prev, ...newEdges])
    setNodes(prev => prev.map(n => n.id === nodeId ? { ...n, isExpanded: true } : n))
  }, [nodes])

  // Get color for collection
  const getNodeColor = useCallback((collectionKey: string) => {
    if (collectionKey === 'company') return '#fbbf24' // amber
    if (collectionKey === 'awards') return '#fbbf24' // gold
    if (collectionKey.startsWith('sec')) return '#3b82f6' // blue
    if (collectionKey === 'marketdata') return '#10b981' // green
    if (collectionKey === 'options') return '#8b5cf6' // purple
    if (collectionKey === 'futures' || collectionKey.startsWith('eia')) return '#f59e0b' // amber
    if (collectionKey.includes('prediction') || collectionKey === 'kalshi') return '#ec4899' // pink
    return '#6b7280' // gray
  }, [])

  // Calculate viewBox to fit all nodes
  const viewBox = useMemo(() => {
    if (nodes.length === 0) return '0 0 1200 800'

    const padding = 150
    const xs = nodes.map(n => n.x)
    const ys = nodes.map(n => n.y)

    const minX = Math.min(...xs) - padding
    const maxX = Math.max(...xs) + padding
    const minY = Math.min(...ys) - padding
    const maxY = Math.max(...ys) + padding

    const width = maxX - minX
    const height = maxY - minY

    return `${minX} ${minY} ${width} ${height}`
  }, [nodes])

  if (showStartSelector) {
    return <StartSelector onSelect={handleStartExploration} />
  }

  return (
    <div className="w-full h-full flex bg-dark-900">
      {/* Main Graph View */}
      <div className="flex-1 relative">
        <svg className="w-full h-full" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
          <defs>
            {/* Glow effect */}
            <filter id="node-glow">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>

            {/* Arrow markers for each color */}
            <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
              <path d="M0,0 L0,6 L9,3 z" fill="#10b981" />
            </marker>
          </defs>

          {/* Draw edges */}
          {edges.map((edge, index) => {
            const fromNode = nodes.find(n => n.id === edge.from)
            const toNode = nodes.find(n => n.id === edge.to)
            if (!fromNode || !toNode) return null

            return (
              <motion.g key={`edge-${index}`}>
                <motion.line
                  x1={fromNode.x}
                  y1={fromNode.y}
                  x2={toNode.x}
                  y2={toNode.y}
                  stroke="#10b981"
                  strokeWidth="2"
                  strokeOpacity="0.3"
                  markerEnd="url(#arrow-green)"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.05 }}
                />
                {/* Edge label */}
                <text
                  x={(fromNode.x + toNode.x) / 2}
                  y={(fromNode.y + toNode.y) / 2}
                  fill="#6b7280"
                  fontSize="10"
                  textAnchor="middle"
                  opacity="0.6"
                >
                  {edge.label}
                </text>
              </motion.g>
            )
          })}

          {/* Draw nodes */}
          {nodes.map((node, index) => {
            const color = getNodeColor(node.collectionKey)
            const isSelected = selectedNode === node.id
            const isAtMaxDepth = node.depth >= 2
            const canExpand = !node.isExpanded && !isAtMaxDepth

            return (
              <motion.g
                key={node.id}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                style={{ cursor: canExpand ? 'pointer' : 'default' }}
                onClick={() => {
                  setSelectedNode(node.id)
                  if (canExpand) {
                    handleExpandNode(node.id)
                  }
                }}
              >
                {/* Node circle */}
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={node.isRoot ? 60 : 45}
                  fill="rgba(17, 24, 39, 0.95)"
                  stroke={color}
                  strokeWidth={isSelected ? 4 : 2}
                  filter={isSelected ? 'url(#node-glow)' : undefined}
                />

                {/* Expand indicator - only show if can expand */}
                {canExpand && (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.isRoot ? 50 : 35}
                    fill="none"
                    stroke={color}
                    strokeWidth="1"
                    strokeDasharray="4"
                    opacity="0.5"
                  />
                )}

                {/* Max depth indicator */}
                {isAtMaxDepth && !node.isExpanded && (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={35}
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth="1"
                    strokeDasharray="2"
                    opacity="0.3"
                  />
                )}

                {/* Node label */}
                <text
                  x={node.x}
                  y={node.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={color}
                  fontSize={node.isRoot ? 14 : 12}
                  fontWeight="bold"
                >
                  {node.label}
                </text>

                {/* Depth indicator badge */}
                {node.depth > 0 && (
                  <circle
                    cx={node.x + 35}
                    cy={node.y - 35}
                    r="12"
                    fill={isAtMaxDepth ? '#ef4444' : '#6b7280'}
                    opacity="0.8"
                  />
                )}
                {node.depth > 0 && (
                  <text
                    x={node.x + 35}
                    y={node.y - 32}
                    textAnchor="middle"
                    fill="white"
                    fontSize="10"
                    fontWeight="bold"
                  >
                    {node.depth}
                  </text>
                )}

                {/* Expand hint */}
                {canExpand && (
                  <text
                    x={node.x}
                    y={node.y + (node.isRoot ? 75 : 60)}
                    textAnchor="middle"
                    fill="#6b7280"
                    fontSize="10"
                  >
                    Click to expand
                  </text>
                )}

                {/* Max depth reached hint */}
                {isAtMaxDepth && !node.isExpanded && (
                  <text
                    x={node.x}
                    y={node.y + 60}
                    textAnchor="middle"
                    fill="#ef4444"
                    fontSize="9"
                    opacity="0.7"
                  >
                    Max depth
                  </text>
                )}
              </motion.g>
            )
          })}
        </svg>

        {/* Controls overlay */}
        <div className="absolute top-4 right-4 flex flex-col gap-2">
          <button
            onClick={() => {
              setNodes([])
              setEdges([])
              setShowStartSelector(true)
              setSelectedNode(null)
            }}
            className="px-4 py-2 bg-dark-800 hover:bg-dark-700 border border-gray-600 rounded-lg text-sm text-white transition-colors"
          >
            Start Over
          </button>
        </div>

        {/* Instructions */}
        {nodes.length === 1 && !nodes[0].isExpanded && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-green-500/10 border border-green-500/30 rounded-lg"
          >
            <p className="text-green-400 text-sm text-center">
              Click the node to see available connections
            </p>
          </motion.div>
        )}
      </div>

      {/* Minimal sidebar for selected node details */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            className="w-80 bg-dark-800 border-l border-green-500/20 p-6 overflow-y-auto"
          >
            {(() => {
              const node = nodes.find(n => n.id === selectedNode)
              if (!node) return null

              const schema = GRAPH_SCHEMA[node.collectionKey]
              if (!schema) return null

              return (
                <div className="space-y-4">
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2">{node.label}</h3>
                    <p className="text-sm text-gray-400">{schema.description}</p>
                  </div>

                  {node.isExpanded && (
                    <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                      <p className="text-xs text-green-400">
                        ✓ Expanded - showing {schema.connections.length} connections
                      </p>
                    </div>
                  )}

                  {!node.isExpanded && node.depth >= 2 && (
                    <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                      <p className="text-xs text-red-400">
                        🛑 Max depth reached (Level {node.depth})
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        Click "Start Over" to explore a different path
                      </p>
                    </div>
                  )}

                  {!node.isExpanded && node.depth < 2 && (
                    <div className="space-y-2">
                      <p className="text-xs text-gray-500 uppercase tracking-wider">Available Connections ({schema.connections.length}):</p>
                      <ul className="space-y-1">
                        {schema.connections.map((conn, i) => {
                          const targetSchema = GRAPH_SCHEMA[conn.target]
                          // Check if this target already exists in graph
                          const alreadyExists = nodes.find(n => n.collectionKey === conn.target)
                          return (
                            <li key={i} className="text-sm text-gray-400 flex items-center gap-2">
                              <span className="text-green-500">→</span>
                              {targetSchema?.name || conn.target}
                              {alreadyExists && (
                                <span className="text-xs text-amber-400">(reuse existing)</span>
                              )}
                            </li>
                          )
                        })}
                      </ul>
                    </div>
                  )}

                  <div className="pt-4 border-t border-gray-700 space-y-2">
                    <p className="text-xs text-gray-500">Collection: <span className="text-gray-300 font-mono">{schema.collection}</span></p>
                    <p className="text-xs text-gray-500">
                      Depth: <span className={node.depth >= 2 ? 'text-red-400' : 'text-gray-300'}>{node.depth}</span>
                      <span className="text-gray-600"> / 2 max</span>
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

// Starting point selector
function StartSelector({ onSelect }: { onSelect: (key: string) => void }) {
  const startingPoints = [
    { key: 'company', label: 'Companies', icon: '🏢', description: 'Start with a specific company' },
    { key: 'awards', label: 'Gov Contracts', icon: '🎖️', description: 'Browse federal contract awards' },
    { key: 'sec', label: 'SEC Filings', icon: '📄', description: 'Search regulatory filings' },
    { key: 'marketdata', label: 'Stock Prices', icon: '📈', description: 'Analyze market data' },
    { key: 'options', label: 'Options Flow', icon: '⚡', description: 'Unusual options activity', badge: 'NEW' },
    { key: 'futures', label: 'Commodities', icon: '🛢️', description: 'Commodity futures prices', badge: 'RARE' },
  ]

  return (
    <div className="w-full h-full flex items-center justify-center bg-dark-900">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-3xl w-full px-8"
      >
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white mb-3">Start Your Data Journey</h2>
          <p className="text-gray-400">Choose a starting point to begin exploring connections</p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {startingPoints.map((point, index) => (
            <motion.button
              key={point.key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onSelect(point.key)}
              className="p-6 bg-dark-800/50 hover:bg-dark-800 border border-green-500/20 hover:border-green-500/40
                       rounded-xl transition-all duration-200 text-left group relative"
            >
              {point.badge && (
                <span className="absolute top-3 right-3 px-2 py-0.5 text-xs font-bold bg-green-500/20 text-green-400 rounded-full border border-green-500/30">
                  {point.badge}
                </span>
              )}
              <div className="text-4xl mb-4">{point.icon}</div>
              <div className="text-lg font-semibold text-white group-hover:text-green-400 transition-colors mb-2">
                {point.label}
              </div>
              <div className="text-sm text-gray-400">{point.description}</div>
            </motion.button>
          ))}
        </div>

        <div className="mt-12 text-center text-xs text-gray-500">
          Click any starting point to see its connections
        </div>
      </motion.div>
    </div>
  )
}
