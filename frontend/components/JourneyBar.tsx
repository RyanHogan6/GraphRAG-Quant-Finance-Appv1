'use client'

import { motion } from 'framer-motion'
import { JourneyStep } from '@/lib/journey-types'
import { GRAPH_SCHEMA } from '@/lib/schema'

interface JourneyBarProps {
  steps: JourneyStep[]
  currentStepIndex: number
  onStepClick: (index: number) => void
  onRemoveStep: (index: number) => void
}

export default function JourneyBar({ steps, currentStepIndex, onStepClick, onRemoveStep }: JourneyBarProps) {
  if (steps.length === 0) {
    return (
      <div className="w-full px-6 py-4 bg-dark-900/50 border-b border-green-500/10">
        <div className="text-center text-gray-500 text-sm">
          Select a starting point below to begin your data journey
        </div>
      </div>
    )
  }

  return (
    <div className="w-full px-6 py-4 bg-dark-900/50 border-b border-green-500/10 overflow-x-auto">
      <div className="flex items-center gap-3 min-w-max">
        {steps.map((step, index) => {
          const isActive = index === currentStepIndex
          const isCompleted = index < currentStepIndex
          const schema = GRAPH_SCHEMA[step.collectionKey]

          return (
            <div key={step.id} className="flex items-center gap-3">
              {/* Journey Step Bubble */}
              <motion.button
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => onStepClick(index)}
                className={`
                  relative group px-6 py-3 rounded-full border-2 transition-all duration-300
                  ${isActive
                    ? 'bg-green-500/20 border-green-500 shadow-[0_0_30px_rgba(34,197,94,0.3)]'
                    : isCompleted
                    ? 'bg-green-500/10 border-green-500/50 hover:border-green-500'
                    : 'bg-dark-800/50 border-gray-600/30 hover:border-gray-500'
                  }
                `}
              >
                {/* Step Number Badge */}
                <div className={`
                  absolute -top-2 -left-2 w-6 h-6 rounded-full border-2 flex items-center justify-center text-xs font-bold
                  ${isActive
                    ? 'bg-green-500 border-green-400 text-white'
                    : isCompleted
                    ? 'bg-green-500/50 border-green-500/50 text-white'
                    : 'bg-dark-700 border-gray-600 text-gray-400'
                  }
                `}>
                  {index + 1}
                </div>

                {/* Step Content */}
                <div className="flex flex-col items-start min-w-[180px]">
                  <div className="text-xs text-gray-400 mb-1">{schema?.name || step.collectionKey}</div>
                  <div className={`text-sm font-medium ${isActive ? 'text-green-400' : 'text-gray-300'}`}>
                    {step.label}
                  </div>
                  {step.previewData && (
                    <div className="text-xs text-gray-500 mt-1">
                      {step.previewData.count.toLocaleString()} records
                    </div>
                  )}
                </div>

                {/* Remove Button (on hover) */}
                {index > 0 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onRemoveStep(index)
                    }}
                    className="absolute -top-2 -right-2 w-6 h-6 bg-red-500/80 hover:bg-red-500 rounded-full
                             opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
                  >
                    <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </motion.button>

              {/* Arrow */}
              {index < steps.length - 1 && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 + 0.05 }}
                  className="flex items-center"
                >
                  <svg className="w-6 h-6 text-green-500/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </motion.div>
              )}
            </div>
          )
        })}

        {/* Add Step Button */}
        {steps.length > 0 && (
          <motion.button
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: steps.length * 0.1 }}
            className="px-4 py-3 rounded-full border-2 border-dashed border-green-500/30
                     hover:border-green-500 hover:bg-green-500/10 transition-all duration-300
                     flex items-center gap-2 text-sm text-gray-400 hover:text-green-400"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Add Connection
          </motion.button>
        )}
      </div>
    </div>
  )
}
