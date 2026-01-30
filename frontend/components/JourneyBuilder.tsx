'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { JourneyStep, JourneyState } from '@/lib/journey-types'
import { GRAPH_SCHEMA } from '@/lib/schema'
import JourneyBar from './JourneyBar'
import LiveGraphVisualization from './LiveGraphVisualization'
import ConnectionSuggestions from './ConnectionSuggestions'

interface JourneyBuilderProps {
  onQueryChange: (aql: string, description: string) => void
}

export default function JourneyBuilder({ onQueryChange }: JourneyBuilderProps) {
  const [journeyState, setJourneyState] = useState<JourneyState>({
    steps: [],
    connections: [],
    currentStepIndex: 0
  })

  const [showStartSelector, setShowStartSelector] = useState(true)

  // Generate AQL query from journey
  useEffect(() => {
    if (journeyState.steps.length === 0) {
      onQueryChange('', '')
      return
    }

    const aql = generateAQLFromJourney(journeyState)
    const description = generateDescriptionFromJourney(journeyState)
    onQueryChange(aql, description)
  }, [journeyState, onQueryChange])

  // Start a new journey
  const handleStartJourney = useCallback((collectionKey: string, initialLabel?: string) => {
    const schema = GRAPH_SCHEMA[collectionKey]
    if (!schema) return

    const newStep: JourneyStep = {
      id: `step-${Date.now()}`,
      collectionKey,
      label: initialLabel || schema.name,
      filters: [],
      previewData: {
        count: 0 // Will be populated by preview fetch
      }
    }

    setJourneyState({
      steps: [newStep],
      connections: [],
      currentStepIndex: 0
    })
    setShowStartSelector(false)
  }, [])

  // Add a connection to the journey
  const handleAddConnection = useCallback((targetKey: string, edge: string, direction: 'OUTBOUND' | 'INBOUND') => {
    const targetSchema = GRAPH_SCHEMA[targetKey]
    if (!targetSchema) return

    const currentStep = journeyState.steps[journeyState.currentStepIndex]
    if (!currentStep) return

    const newStep: JourneyStep = {
      id: `step-${Date.now()}`,
      collectionKey: targetKey,
      label: targetSchema.name,
      filters: [],
      previewData: {
        count: 0
      }
    }

    setJourneyState(prev => ({
      ...prev,
      steps: [...prev.steps, newStep],
      connections: [...prev.connections, {
        from: currentStep.id,
        to: newStep.id,
        edge,
        direction
      }],
      currentStepIndex: prev.steps.length // Move to new step
    }))
  }, [journeyState.currentStepIndex, journeyState.steps])

  // Navigate to a specific step
  const handleStepClick = useCallback((index: number) => {
    setJourneyState(prev => ({
      ...prev,
      currentStepIndex: index
    }))
  }, [])

  // Remove a step
  const handleRemoveStep = useCallback((index: number) => {
    if (index === 0) return // Can't remove first step

    setJourneyState(prev => ({
      ...prev,
      steps: prev.steps.filter((_, i) => i !== index),
      connections: prev.connections.filter((conn, i) => i !== index - 1),
      currentStepIndex: Math.min(prev.currentStepIndex, index - 1)
    }))
  }, [])

  // Reset journey
  const handleReset = useCallback(() => {
    setJourneyState({
      steps: [],
      connections: [],
      currentStepIndex: 0
    })
    setShowStartSelector(true)
  }, [])

  return (
    <div className="w-full h-full flex flex-col bg-dark-900">
      {/* Journey Bar at top */}
      <JourneyBar
        steps={journeyState.steps}
        currentStepIndex={journeyState.currentStepIndex}
        onStepClick={handleStepClick}
        onRemoveStep={handleRemoveStep}
      />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Current step builder or start selector */}
        <div className="flex-1 overflow-y-auto p-6">
          <AnimatePresence mode="wait">
            {showStartSelector ? (
              <StartingPointSelector onSelect={handleStartJourney} key="start-selector" />
            ) : journeyState.steps.length > 0 ? (
              <CurrentStepEditor
                step={journeyState.steps[journeyState.currentStepIndex]}
                key={journeyState.steps[journeyState.currentStepIndex]?.id}
              />
            ) : null}
          </AnimatePresence>

          {/* Reset button */}
          {journeyState.steps.length > 0 && (
            <button
              onClick={handleReset}
              className="mt-6 px-4 py-2 text-sm text-gray-400 hover:text-white border border-gray-700
                       hover:border-gray-500 rounded-lg transition-colors"
            >
              Start Over
            </button>
          )}
        </div>

        {/* Right: Split panel */}
        <div className="w-[400px] border-l border-green-500/10 flex flex-col">
          {/* Top: Connection suggestions */}
          <div className="flex-1 border-b border-green-500/10 overflow-hidden">
            {journeyState.steps.length > 0 && journeyState.currentStepIndex === journeyState.steps.length - 1 ? (
              <ConnectionSuggestions
                currentCollectionKey={journeyState.steps[journeyState.currentStepIndex].collectionKey}
                onAddConnection={handleAddConnection}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center p-8">
                <div className="text-center text-gray-500 text-sm">
                  Click the last step to see connection options
                </div>
              </div>
            )}
          </div>

          {/* Bottom: Live graph visualization */}
          <div className="h-[300px] bg-dark-900/50">
            <LiveGraphVisualization
              steps={journeyState.steps}
              currentStepIndex={journeyState.currentStepIndex}
              onNodeClick={handleStepClick}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

// Starting point selector
function StartingPointSelector({ onSelect }: { onSelect: (key: string, label?: string) => void }) {
  const startingPoints = [
    { key: 'company', label: 'Companies', icon: '🏢', description: 'Start with a specific company' },
    { key: 'awards', label: 'Gov Contracts', icon: '🎖️', description: 'Browse federal contract awards' },
    { key: 'sec', label: 'SEC Filings', icon: '📄', description: 'Search regulatory filings' },
    { key: 'marketdata', label: 'Stock Prices', icon: '📈', description: 'Analyze market data' },
    { key: 'options', label: 'Options Flow', icon: '⚡', description: 'Unusual options activity', badge: 'NEW' },
    { key: 'futures', label: 'Commodities', icon: '🛢️', description: 'Commodity futures prices', badge: 'RARE' },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-3xl mx-auto"
    >
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">Start Your Data Journey</h2>
        <p className="text-gray-400">Choose a starting point to begin exploring connections</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {startingPoints.map((point, index) => (
          <motion.button
            key={point.key}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.05 }}
            onClick={() => onSelect(point.key)}
            className="p-6 bg-dark-800/50 hover:bg-dark-800 border border-green-500/20 hover:border-green-500/40
                     rounded-lg transition-all duration-200 text-left group relative"
          >
            {point.badge && (
              <span className="absolute top-3 right-3 px-2 py-0.5 text-xs font-bold bg-green-500/20 text-green-400 rounded-full">
                {point.badge}
              </span>
            )}
            <div className="text-3xl mb-3">{point.icon}</div>
            <div className="text-lg font-semibold text-white group-hover:text-green-400 transition-colors mb-1">
              {point.label}
            </div>
            <div className="text-sm text-gray-400">{point.description}</div>
          </motion.button>
        ))}
      </div>
    </motion.div>
  )
}

// Current step editor (placeholder for now)
function CurrentStepEditor({ step }: { step: JourneyStep }) {
  const schema = GRAPH_SCHEMA[step.collectionKey]

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div>
        <h3 className="text-xl font-bold text-white mb-2">{schema?.name}</h3>
        <p className="text-gray-400 text-sm">{schema?.description}</p>
      </div>

      <div className="p-4 bg-dark-800/30 rounded-lg border border-green-500/10">
        <div className="text-sm text-gray-400 mb-2">Filters</div>
        <div className="text-sm text-gray-500">No filters applied</div>
        {/* TODO: Add filter UI here in Phase 2 */}
      </div>

      <div className="p-4 bg-dark-800/30 rounded-lg border border-green-500/10">
        <div className="text-sm text-gray-400 mb-2">Preview</div>
        <div className="text-sm text-gray-500">Select filters to preview data</div>
        {/* TODO: Add live preview here in Phase 3 */}
      </div>
    </motion.div>
  )
}

// Generate AQL from journey (placeholder - will be enhanced)
function generateAQLFromJourney(journey: JourneyState): string {
  if (journey.steps.length === 0) return ''

  // For now, just query the first collection
  const firstStep = journey.steps[0]
  return `FOR doc IN ${GRAPH_SCHEMA[firstStep.collectionKey]?.collection}\n  LIMIT 20\n  RETURN doc`
}

function generateDescriptionFromJourney(journey: JourneyState): string {
  if (journey.steps.length === 0) return ''

  const stepNames = journey.steps.map(s => GRAPH_SCHEMA[s.collectionKey]?.name).join(' → ')
  return `Journey: ${stepNames}`
}
