'use client'

import { useState } from 'react'

const collections = [
  { name: 'Company', count: 612, icon: '🏢' },
  { name: 'MarketData', count: 2000000, icon: '📊' },
  { name: 'Award', count: 100000, icon: '🏛️' },
  { name: 'sec_filings', count: 7495, icon: '📄' },
  { name: 'sec_sentences', count: 4362211, icon: '📝' },
  { name: 'prediction_markets_polymarket', count: 15000, icon: '🎲' },
  { name: 'prediction_markets_kalshi', count: 6000, icon: '🎯' },
  { name: 'EconomicData', count: 9000, icon: '💹' },
  { name: 'options_flow', count: 612, icon: '📈' },
  { name: 'futures_prices', count: 64000, icon: '🌾' },
  { name: 'commodity_positions', count: 5000, icon: '📊' },
  { name: 'eia_crude_inventory', count: 500, icon: '⚡' },
  { name: 'eia_natgas_storage', count: 500, icon: '⚡' },
  { name: 'polymarket_traders', count: 1000, icon: '👤' },
  { name: 'polymarket_positions', count: 5000, icon: '💼' },
]

export default function DatabasePage() {
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gold mb-2">Database Browser</h1>
        <p className="text-gray-500">Explore ArangoDB collections and graph structure</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
          <div className="text-gray-500 text-sm mb-1">Collections</div>
          <div className="text-3xl font-bold text-gold">{collections.length}</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
          <div className="text-gray-500 text-sm mb-1">Total Documents</div>
          <div className="text-3xl font-bold text-gold">
            {(collections.reduce((sum, c) => sum + c.count, 0) / 1000000).toFixed(1)}M
          </div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
          <div className="text-gray-500 text-sm mb-1">Edge Collections</div>
          <div className="text-3xl font-bold text-gold">22</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
          <div className="text-gray-500 text-sm mb-1">Database</div>
          <div className="text-xl font-bold text-gold">QUANT_v3</div>
        </div>
      </div>

      {/* Collections Grid */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-gold mb-4">Document Collections</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {collections.map((collection) => (
            <button
              key={collection.name}
              onClick={() => setSelectedCollection(collection.name)}
              className={`bg-dark-800 border rounded-lg p-5 text-left transition-all ${
                selectedCollection === collection.name
                  ? 'border-gold/60 glow-gold'
                  : 'border-gold/20 hover:border-gold/40'
              }`}
            >
              <div className="text-3xl mb-2">{collection.icon}</div>
              <div className="text-gold font-semibold mb-1">{collection.name}</div>
              <div className="text-xs text-gray-500">
                {collection.count.toLocaleString()} docs
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Collection Details */}
      {selectedCollection && (
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gold">
              {selectedCollection}
            </h2>
            <button
              onClick={() => setSelectedCollection(null)}
              className="text-gray-500 hover:text-gold transition-colors"
            >
              ✕
            </button>
          </div>

          <div className="bg-dark-700 border border-gold/20 rounded-lg p-4 mb-4">
            <div className="text-sm text-gray-400 mb-2">Sample documents will appear here once backend is connected</div>
            <div className="text-xs text-gray-600">Backend integration coming soon...</div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-dark-700 border border-gold/20 rounded p-3">
              <div className="text-xs text-gray-500 mb-1">Documents</div>
              <div className="text-lg font-semibold text-gold">
                {collections.find(c => c.name === selectedCollection)?.count.toLocaleString()}
              </div>
            </div>
            <div className="bg-dark-700 border border-gold/20 rounded p-3">
              <div className="text-xs text-gray-500 mb-1">Avg Size</div>
              <div className="text-lg font-semibold text-gold">2.4 KB</div>
            </div>
            <div className="bg-dark-700 border border-gold/20 rounded p-3">
              <div className="text-xs text-gray-500 mb-1">Indexes</div>
              <div className="text-lg font-semibold text-gold">3</div>
            </div>
          </div>
        </div>
      )}

      {/* Graph Visualization Placeholder */}
      <div className="bg-dark-800 border border-gold/20 rounded-lg p-8 text-center">
        <div className="text-xl text-gray-400 mb-2">Graph Visualization</div>
        <div className="text-sm text-gray-600">Interactive graph view coming soon</div>
      </div>
    </div>
  )
}
