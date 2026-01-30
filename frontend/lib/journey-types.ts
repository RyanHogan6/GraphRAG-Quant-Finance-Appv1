/**
 * Types for the Journey Builder (Visual Query Builder Phase 1)
 */

export interface JourneyStep {
  id: string
  collectionKey: string // Key in GRAPH_SCHEMA
  label: string // Display name (e.g., "Lockheed Martin")
  filters: JourneyFilter[]
  previewData?: {
    count: number
    sample?: any[]
    sparkline?: number[]
  }
}

export interface JourneyFilter {
  field: string
  operator: string
  value: string
  displayValue?: string // Human-readable value
}

export interface JourneyConnection {
  from: string // Step ID
  to: string // Step ID
  edge: string // Edge collection name
  direction: 'OUTBOUND' | 'INBOUND'
}

export interface JourneyState {
  steps: JourneyStep[]
  connections: JourneyConnection[]
  currentStepIndex: number
}

export interface AvailableConnection {
  targetKey: string
  targetLabel: string
  edge: string
  direction: 'OUTBOUND' | 'INBOUND'
  description: string
  estimatedCount?: number
  rarity?: 'common' | 'rare' | 'very-rare'
  isNew?: boolean
}
