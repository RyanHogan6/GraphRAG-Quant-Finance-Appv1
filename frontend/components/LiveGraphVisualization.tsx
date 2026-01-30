'use client'

import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { JourneyStep } from '@/lib/journey-types'
import { GRAPH_SCHEMA } from '@/lib/schema'

interface LiveGraphVisualizationProps {
  steps: JourneyStep[]
  currentStepIndex: number
  onNodeClick: (stepIndex: number) => void
}

export default function LiveGraphVisualization({ steps, currentStepIndex, onNodeClick }: LiveGraphVisualizationProps) {
  // Calculate node positions in a vertical layout
  const nodes = useMemo(() => {
    return steps.map((step, index) => {
      const schema = GRAPH_SCHEMA[step.collectionKey]
      const isActive = index === currentStepIndex
      const isCompleted = index < currentStepIndex

      return {
        id: step.id,
        label: schema?.name || step.collectionKey,
        x: 150, // Center X
        y: 80 + index * 100, // Vertical spacing
        isActive,
        isCompleted,
        color: isActive ? '#22c55e' : isCompleted ? '#16a34a' : '#4b5563'
      }
    })
  }, [steps, currentStepIndex])

  // Calculate connections between nodes
  const edges = useMemo(() => {
    const result = []
    for (let i = 0; i < nodes.length - 1; i++) {
      result.push({
        from: nodes[i],
        to: nodes[i + 1],
        color: i < currentStepIndex ? '#22c55e' : '#4b5563'
      })
    }
    return result
  }, [nodes, currentStepIndex])

  if (steps.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center text-gray-500 text-sm p-8">
          <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                  d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
          <div>Your journey path</div>
          <div className="text-xs mt-1">will appear here</div>
        </div>
      </div>
    )
  }

  return (
    <svg className="w-full h-full" viewBox="0 0 300 600">
      <defs>
        {/* Glow effect for active nodes */}
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>

        {/* Arrow marker */}
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L9,3 z" fill="#22c55e" />
        </marker>
      </defs>

      {/* Draw edges (connections) */}
      {edges.map((edge, index) => {
        const y1 = edge.from.y + 30
        const y2 = edge.to.y - 30

        return (
          <motion.line
            key={`edge-${index}`}
            x1={edge.from.x}
            y1={y1}
            x2={edge.to.x}
            y2={y2}
            stroke={edge.color}
            strokeWidth="2"
            strokeDasharray="4"
            markerEnd={edge.color === '#22c55e' ? 'url(#arrowhead)' : undefined}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.6 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          />
        )
      })}

      {/* Draw nodes */}
      {nodes.map((node, index) => {
        // Pre-calculate positions for icons
        const iconX1 = node.x - 8
        const iconY1 = node.y - 6
        const iconX2 = node.x - 2
        const iconY2 = node.y - 2
        const iconX3 = node.x + 4
        const iconY3 = node.y - 10
        const badgeX = node.x + 18
        const badgeY = node.y - 18
        const badgeTextY = node.y - 14
        const labelY = node.y + 45

        return (
          <g key={node.id} onClick={() => onNodeClick(index)} style={{ cursor: 'pointer' }}>
            <motion.circle
              cx={node.x}
              cy={node.y}
              r="28"
              fill="rgba(17, 24, 39, 0.9)"
              stroke={node.color}
              strokeWidth="2"
              filter={node.isActive ? 'url(#glow)' : undefined}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              whileHover={{ scale: 1.1 }}
            />

            {/* Node icon based on collection type */}
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: index * 0.1 + 0.2 }}
            >
              {index === 0 && (
                // Company icon
                <path
                  d="M150 65 L150 55 L160 55 L160 65 L165 65 L165 80 L135 80 L135 65 L140 65 L140 55 L150 55"
                  fill={node.color}
                  stroke="none"
                />
              )}
              {index === 1 && steps[index].collectionKey === 'awards' && (
                // Award icon
                <path
                  d="M150 70 L145 75 L147 81 L150 78 L153 81 L155 75 Z"
                  fill={node.color}
                  stroke="none"
                />
              )}
              {index >= 1 && (
                // Default: Chart icon
                <>
                  <rect x={iconX1} y={iconY1} width="4" height="12" fill={node.color} rx="1" />
                  <rect x={iconX2} y={iconY2} width="4" height="8" fill={node.color} rx="1" />
                  <rect x={iconX3} y={iconY3} width="4" height="16" fill={node.color} rx="1" />
                </>
              )}
            </motion.g>

            {/* Step number badge */}
            <motion.circle
              cx={badgeX}
              cy={badgeY}
              r="10"
              fill={node.isActive ? node.color : 'rgba(75, 85, 99, 0.8)'}
              stroke="rgba(17, 24, 39, 0.9)"
              strokeWidth="2"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: index * 0.1 + 0.3 }}
            />
            <text
              x={badgeX}
              y={badgeTextY}
              textAnchor="middle"
              fill="white"
              fontSize="10"
              fontWeight="bold"
            >
              {index + 1}
            </text>

            {/* Node label */}
            <text
              x={node.x}
              y={labelY}
              textAnchor="middle"
              fill={node.isActive ? node.color : '#9ca3af'}
              fontSize="12"
              fontWeight={node.isActive ? 'bold' : 'normal'}
            >
              {node.label}
            </text>
          </g>
        )
      })}

      {/* Suggested next connections (dimmed) */}
      {steps.length > 0 && currentStepIndex === steps.length - 1 && (() => {
        const lastNode = nodes[nodes.length - 1]
        const nextY = lastNode.y + 100
        const nextTextY = lastNode.y + 105

        return (
          <g opacity="0.3">
            <circle
              cx="150"
              cy={nextY}
              r="24"
              fill="none"
              stroke="#4b5563"
              strokeWidth="2"
              strokeDasharray="4"
            />
            <text
              x="150"
              y={nextTextY}
              textAnchor="middle"
              fill="#6b7280"
              fontSize="12"
            >
              +
            </text>
          </g>
        )
      })()}
    </svg>
  )
}
