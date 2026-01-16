'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function DisclaimerBanner() {
  const [isVisible, setIsVisible] = useState(true)

  if (!isVisible) return null

  return (
    <div className="bg-gradient-to-r from-yellow-900/30 via-yellow-800/30 to-yellow-900/30 border-b border-yellow-700/40 sticky top-0 z-50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 flex-1">
            <span className="text-2xl">⚠️</span>
            <div className="flex-1">
              <p className="text-sm text-yellow-100 font-medium">
                <strong>Not Financial Advice:</strong> This platform is for informational purposes only.
                Market data may be delayed. Always conduct your own research.{' '}
                <Link href="/disclaimer" className="underline hover:text-yellow-200 transition-colors">
                  Read full disclaimer
                </Link>
              </p>
            </div>
          </div>
          <button
            onClick={() => setIsVisible(false)}
            className="text-yellow-200 hover:text-yellow-100 transition-colors text-xl leading-none"
            aria-label="Close disclaimer"
          >
            ×
          </button>
        </div>
      </div>
    </div>
  )
}
