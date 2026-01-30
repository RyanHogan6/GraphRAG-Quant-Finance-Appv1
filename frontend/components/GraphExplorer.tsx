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
  isExpanded: boolean
  isRoot: boolean
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
  const [showRootSelector, setShowRootSelector] = useState(false)

  // Initialize with unselected root node
  const initializeRoot = useCallback(() => {
    if (nodes.length === 0) {
      const rootNode: GraphNode = {
        id: 'root',
        collectionKey: '',
        label: 'Select Starting Point',
        x: 150,
        y: 400,
        layer: 0,
        isExpanded: false,
        isRoot: true
      }
      setNodes([rootNode])
      setSelectedNode('root')
    }
  }, [nodes.length])

  // Set root type
  const handleSetRootType = useCallback((collectionKey: string) => {
    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    setNodes([{
      id: 'root',
      collectionKey,
      label: schema.name,
      x: 150,
      y: 400,
      layer: 0,
      isExpanded: false,
      isRoot: true
    }])
    setShowRootSelector(false)
    setSelectedNode('root')
  }, [])

  // Expand node - horizontal layers with vertical spread
  const handleExpandNode = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId)
    if (!node || node.isExpanded || node.layer >= 2) return

    const schema = GRAPH_SCHEMA[node.collectionKey]
    if (!schema) return

    const connections = schema.connections
    const layerX = node.x + 400 // Move right 400px
    const baseY = node.y
    const verticalSpacing = 120

    // Calculate vertical positions to fan out
    const startY = baseY - ((connections.length - 1) * verticalSpacing) / 2

    const newNodes: GraphNode[] = []
    const newEdges: GraphEdge[] = []

    connections.forEach((conn, index) => {
      const targetSchema = GRAPH_SCHEMA[conn.target]
      if (!targetSchema) return

      // Check if this collection already exists in this layer
      const existingNode = nodes.find(n =>
        n.collectionKey === conn.target && n.layer === node.layer + 1
      )

      if (existingNode) {
        // Reuse existing node
        newEdges.push({
          from: nodeId,
          to: existingNode.id,
          label: conn.edge
        })
      } else {
        // Create new node
        const y = startY + (index * verticalSpacing)
        const newNodeId = `${conn.target}-layer${node.layer + 1}-${Date.now()}-${index}`

        newNodes.push({
          id: newNodeId,
          collectionKey: conn.target,
          label: targetSchema.name,
          x: layerX,
          y,
          layer: node.layer + 1,
          isExpanded: false,
          isRoot: false
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

  // Initialize on mount
  useMemo(() => initializeRoot(), [initializeRoot])

  // Calculate viewBox
  const viewBox = useMemo(() => {
    if (nodes.length === 0) return '0 0 1200 800'

    const padding = 150
    const xs = nodes.map(n => n.x)
    const ys = nodes.map(n => n.y)

    const minX = Math.min(...xs) - padding
    const maxX = Math.max(...xs) + padding + 200 // Extra for execute panel
    const minY = Math.min(...ys) - padding
    const maxY = Math.max(...ys) + padding

    return `${minX} ${minY} ${maxX - minX} ${maxY - minY}`
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

  // Check if ready to execute
  const canExecute = nodes.length > 1 && nodes[0].collectionKey !== ''

  // Get rightmost node position for execute panel
  const executeX = useMemo(() => {
    if (nodes.length === 0) return 0
    return Math.max(...nodes.map(n => n.x)) + 400
  }, [nodes])

  return (
    <div className="w-full h-full flex bg-dark-900 relative">
      {/* Main Graph View */}
      <div className="flex-1 relative overflow-auto">
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

            return (
              <motion.g key={`edge-${index}`}>
                <motion.path
                  d={`M ${fromNode.x + 60} ${fromNode.y} Q ${(fromNode.x + toNode.x) / 2} ${fromNode.y} ${toNode.x - 60} ${toNode.y}`}
                  stroke="#10b981"
                  strokeWidth="2"
                  fill="none"
                  strokeOpacity="0.4"
                  markerEnd="url(#arrow-green)"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.05 }}
                />
              </motion.g>
            )
          })}

          {/* Draw nodes */}
          {nodes.map((node, index) => {
            const color = node.collectionKey ? getNodeColor(node.collectionKey) : '#6b7280'
            const isSelected = selectedNode === node.id
            const canExpand = !node.isExpanded && node.layer < 2 && node.collectionKey !== ''

            return (
              <motion.g
                key={node.id}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                {/* Node circle */}
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={node.isRoot ? 70 : 50}
                  fill="rgba(17, 24, 39, 0.95)"
                  stroke={color}
                  strokeWidth={isSelected ? 4 : 2}
                  filter={isSelected ? 'url(#node-glow)' : undefined}
                  style={{ cursor: canExpand || node.isRoot ? 'pointer' : 'default' }}
                  onClick={() => {
                    setSelectedNode(node.id)
                    if (node.isRoot && node.collectionKey === '') {
                      setShowRootSelector(!showRootSelector)
                    } else if (canExpand) {
                      handleExpandNode(node.id)
                    }
                  }}
                />

                {/* Expand indicator */}
                {canExpand && (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={40}
                    fill="none"
                    stroke={color}
                    strokeWidth="1"
                    strokeDasharray="4"
                    opacity="0.5"
                  />
                )}

                {/* Node label */}
                <text
                  x={node.x}
                  y={node.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={color}
                  fontSize={node.isRoot ? 16 : 13}
                  fontWeight="bold"
                  style={{ cursor: canExpand || node.isRoot ? 'pointer' : 'default' }}
                  onClick={() => {
                    setSelectedNode(node.id)
                    if (node.isRoot && node.collectionKey === '') {
                      setShowRootSelector(!showRootSelector)
                    } else if (canExpand) {
                      handleExpandNode(node.id)
                    }
                  }}
                >
                  {node.label}
                </text>

                {/* Layer badge */}
                {!node.isRoot && (
                  <>
                    <circle cx={node.x - 40} cy={node.y - 40} r="12" fill="#374151" opacity="0.9" />
                    <text x={node.x - 40} y={node.y - 37} textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">
                      {node.layer}
                    </text>
                  </>
                )}
              </motion.g>
            )
          })}

          {/* Execute Panel */}
          {canExecute && (
            <motion.g
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
            >
              <rect
                x={executeX - 100}
                y={300}
                width="200"
                height="200"
                rx="10"
                fill="rgba(34, 197, 94, 0.1)"
                stroke="#10b981"
                strokeWidth="2"
              />
              <text x={executeX} y={350} textAnchor="middle" fill="#10b981" fontSize="14" fontWeight="bold">
                Query Summary
              </text>
              <text x={executeX} y={380} textAnchor="middle" fill="#9ca3af" fontSize="11">
                {nodes.filter(n => n.collectionKey).length} collections
              </text>
              <text x={executeX} y={400} textAnchor="middle" fill="#9ca3af" fontSize="11">
                {edges.length} connections
              </text>

              {/* Execute button */}
              <rect
                x={executeX - 70}
                y={430}
                width="140"
                height="40"
                rx="8"
                fill="#10b981"
                style={{ cursor: 'pointer' }}
                onClick={() => {
                  // Generate and execute query
                  const aql = `FOR doc IN ${nodes[0].collectionKey}\n  LIMIT 20\n  RETURN doc`
                  onQueryChange(aql, `Journey: ${nodes.map(n => n.label).join(' → ')}`)
                }}
              />
              <text
                x={executeX}
                y={455}
                textAnchor="middle"
                fill="white"
                fontSize="13"
                fontWeight="bold"
                style={{ cursor: 'pointer' }}
                onClick={() => {
                  const aql = `FOR doc IN ${nodes[0].collectionKey}\n  LIMIT 20\n  RETURN doc`
                  onQueryChange(aql, `Journey: ${nodes.map(n => n.label).join(' → ')}`)
                }}
              >
                Execute Query
              </text>
            </motion.g>
          )}
        </svg>

        {/* Root type selector dropdown */}
        <AnimatePresence>
          {showRootSelector && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2
                       bg-dark-800 border-2 border-green-500/50 rounded-lg p-4 shadow-2xl z-50"
              style={{ width: '300px' }}
            >
              <div className="text-sm font-bold text-white mb-3">Select Starting Point:</div>
              <div className="space-y-2">
                {STARTING_OPTIONS.map(option => (
                  <button
                    key={option.key}
                    onClick={() => handleSetRootType(option.key)}
                    className="w-full p-3 bg-dark-700 hover:bg-dark-600 border border-green-500/20 hover:border-green-500/40
                             rounded-lg transition-all text-left flex items-center gap-3"
                  >
                    <span className="text-2xl">{option.icon}</span>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-white">{option.label}</div>
                    </div>
                    {option.badge && (
                      <span className="px-2 py-0.5 text-xs font-bold bg-green-500/20 text-green-400 rounded-full">
                        {option.badge}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Controls - Fixed below navbar */}
        <div className="absolute top-16 right-4 flex flex-col gap-2 z-40">
          <button
            onClick={() => {
              setNodes([])
              setEdges([])
              setSelectedNode(null)
              setShowRootSelector(false)
              initializeRoot()
            }}
            className="px-4 py-2 bg-dark-800 hover:bg-dark-700 border border-gray-600 rounded-lg text-sm text-white transition-colors shadow-lg"
          >
            Reset
          </button>
        </div>

        {/* Instructions */}
        {nodes.length === 1 && nodes[0].collectionKey === '' && !showRootSelector && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-green-500/10 border border-green-500/30 rounded-lg"
          >
            <p className="text-green-400 text-sm text-center">
              Click the root node to select a starting collection
            </p>
          </motion.div>
        )}

        {nodes.length === 1 && nodes[0].collectionKey !== '' && !nodes[0].isExpanded && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-green-500/10 border border-green-500/30 rounded-lg"
          >
            <p className="text-green-400 text-sm text-center">
              Click the node to expand and see connections →
            </p>
          </motion.div>
        )}
      </div>

      {/* Right sidebar - connection info */}
      <AnimatePresence>
        {selectedNode && nodes.find(n => n.id === selectedNode)?.collectionKey && (
          <motion.div
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 300, opacity: 0 }}
            className="w-80 bg-dark-800 border-l border-green-500/20 p-6 overflow-y-auto"
          >
            {(() => {
              const node = nodes.find(n => n.id === selectedNode)
              if (!node || !node.collectionKey) return null

              const schema = GRAPH_SCHEMA[node.collectionKey]
              if (!schema) return null

              return (
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-bold text-white mb-2">{node.label}</h3>
                    <p className="text-xs text-gray-400">{schema.description}</p>
                  </div>

                  {!node.isExpanded && node.layer < 2 && (
                    <div className="space-y-2">
                      <p className="text-xs text-gray-500 uppercase">Available ({schema.connections.length}):</p>
                      <ul className="space-y-1">
                        {schema.connections.map((conn, i) => {
                          const targetSchema = GRAPH_SCHEMA[conn.target]
                          return (
                            <li key={i} className="text-xs text-gray-400 flex items-center gap-2">
                              <span className="text-green-500">→</span>
                              {targetSchema?.name || conn.target}
                            </li>
                          )
                        })}
                      </ul>
                    </div>
                  )}

                  <div className="pt-4 border-t border-gray-700 text-xs text-gray-500">
                    Layer: <span className="text-white">{node.layer}</span> / 2
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
