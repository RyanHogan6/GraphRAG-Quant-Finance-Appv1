'use client'

import { useState, useEffect, useRef } from 'react'
import { GRAPH_SCHEMA, SchemaNode, isValidConnection } from '@/lib/schema'
import { motion, AnimatePresence } from 'framer-motion'

interface QueryBuilderProps {
    onQueryChange: (aql: string, description: string) => void
}

type Filter = {
    field: string
    operator: string
    value: string
}

type Enrichment = {
    targetKey: string // Key in GRAPH_SCHEMA (e.g., 'marketdata', 'predictionmarkets')
    filters?: Filter[] // Optional filters for this enrichment
}

// Category definition for entry points
type Category = {
    key: string
    label: string
    icon: string
    description: string
    useCases: string[]
    collections: string[]
}

const CATEGORIES: Category[] = [
    {
        key: 'core',
        label: 'Core Data',
        icon: '📊',
        description: 'Companies, stock prices, and economic indicators',
        useCases: [
            'Track company fundamentals',
            'Analyze stock price movements',
            'Correlate with macro indicators'
        ],
        collections: ['company', 'marketdata', 'economicdata']
    },
    {
        key: 'sec',
        label: 'SEC Filings',
        icon: '📄',
        description: 'Corporate filings, insider trades, and institutional holdings',
        useCases: [
            'Detect insider buying/selling',
            'Track activist investors (13D)',
            'Monitor hedge fund positions (13F)',
            'Analyze corporate sentiment'
        ],
        collections: ['sec']
    },
    {
        key: 'markets',
        label: 'Prediction Markets',
        icon: '🎲',
        description: 'Polymarket and Kalshi event-driven probabilities',
        useCases: [
            'Track whale trader positions',
            'Monitor event probabilities',
            'Find markets mentioning companies'
        ],
        collections: ['predictionmarkets', 'kalshi', 'polymarket_traders', 'polymarket_positions', 'polymarket_price_history']
    },
    {
        key: 'commodities',
        label: 'Commodities & Energy',
        icon: '🌾',
        description: 'Futures prices, CFTC positioning, EIA inventory data',
        useCases: [
            'Track crude oil and natural gas',
            'Monitor speculator positioning',
            'Analyze inventory builds/draws'
        ],
        collections: ['futures', 'cftc', 'eia_crude', 'eia_natgas_storage', 'eia_natgas_production', 'eia_lng']
    },
    {
        key: 'options',
        label: 'Options & Contracts',
        icon: '📈',
        description: 'Options flow unusual activity and government contracts',
        useCases: [
            'Find unusual call/put activity',
            'Track government contracts',
            'Detect potential insider trading'
        ],
        collections: ['options', 'awards']
    }
]

export default function QueryBuilder({ onQueryChange }: QueryBuilderProps) {
    // Steps: 0 = Category, 1 = Collection + Filter, 2 = Connections
    const [step, setStep] = useState(0)

    // State
    const [category, setCategory] = useState<string>('')
    const [source, setSource] = useState<string>('')
    const [filters, setFilters] = useState<Filter[]>([])
    const [enrichments, setEnrichments] = useState<Enrichment[]>([])
    const [limit, setLimit] = useState(20)
    const [showConnections, setShowConnections] = useState(false)

    // Company suggestions for autofill
    const [companies, setCompanies] = useState<{ ticker: string, company: string }[]>([])

    useEffect(() => {
        const fetchCompanies = async () => {
            try {
                const { api } = await import('@/lib/api')
                // Direct call to browse Company collection
                const res = await api.browseCollection('Company', 500)
                const data = Array.isArray(res) ? res : (res.documents || [])
                setCompanies(data.map((c: any) => ({ ticker: c.ticker, company: c.company })))
            } catch (e) {
                console.warn("QueryBuilder: Failed to pre-cache company list")
            }
        }
        fetchCompanies()
    }, [])

    // Derived
    const selectedCategory = CATEGORIES.find(c => c.key === category)
    const sourceNode = source ? GRAPH_SCHEMA[source] : null

    // Generate AQL whenever state changes
    useEffect(() => {
        if (!sourceNode) return

        let aql = `FOR doc IN ${sourceNode.collection}\n`
        let desc = `Find ${sourceNode.name}`

        // Filters
        filters.forEach(f => {
            if (!f.field || !f.value) return
            // Handle string vs number
            const isNum = !isNaN(Number(f.value))
            const val = isNum ? f.value : `"${f.value}"`
            aql += `  FILTER doc.${f.field} ${f.operator} ${val}\n`
            desc += ` where ${f.field} ${f.operator} ${f.value}`
        })

        // Enrichments (Graph Traversals using Schema Connections)
        enrichments.forEach(e => {
            const targetKey = e.targetKey
            const targetNode = GRAPH_SCHEMA[targetKey]

            // Find connection definition in schema
            const connection = sourceNode.connections.find(c => c.target === targetKey)

            if (!targetNode || !connection) {
                // Should not happen if UI is consistent with schema
                aql += `  LET ${targetNode?.collection || targetKey}_data = [] // Connection not defined in schema\n`
                return
            }

            aql += `\n  // Enrich with ${targetNode.name}\n`
            aql += `  LET ${targetNode.collection}_data = (\n`

            if (connection.type === 'direct') {
                if (connection.direction === 'INBOUND') {
                    // FOR t IN INBOUND doc edge_collection
                    aql += `    FOR t IN INBOUND doc ${connection.edge}\n`
                    aql += `      LIMIT 5 RETURN t\n`
                } else {
                    // FOR t IN OUTBOUND doc edge_collection
                    aql += `    FOR t IN OUTBOUND doc ${connection.edge}\n`
                    // Special handling for market data sort
                    if (targetKey === 'marketdata') {
                        aql += `      SORT t.date DESC LIMIT 1800 RETURN t\n`
                    } else if (targetKey === 'awards') {
                        aql += `      SORT t.start_date DESC LIMIT 5 RETURN t\n`
                    } else if (targetKey === 'sec') {
                        aql += `      SORT t.filing_date DESC LIMIT 3 // Limit heavy traversal\n`
                        aql += `      LET top_sentences = (\n`
                        aql += `        FOR s IN 1..2 OUTBOUND t has_section, has_sentence\n`
                        aql += `        SORT ABS(s.finbert_score) DESC LIMIT 5 RETURN { text: s.text, score: s.finbert_score }\n`
                        aql += `      )\n`
                        aql += `      RETURN MERGE(t, { top_sentences })\n`
                    } else {
                        aql += `      LIMIT 5 RETURN t\n`
                    }
                }
            } else if (connection.type === 'multi_hop') {
                // Special case for SEC sentences (multi-hop traversal)
                if (targetKey === 'sec_sentences') {
                    // Company -> HAS_FILING -> sec_filings -> has_section -> sec_sections -> has_sentence -> sec_sentences
                    // AQL graph traversal 3 hops
                    // 1..3 OUTBOUND doc HAS_FILING, has_section, has_sentence
                    // Note: Edges must be listed.
                    // Simplified: Just match by ticker if reliable, BUT the user wants graph traversal.
                    // Let's use the explicit pattern from example queries which is usually cleaner:
                    // 1. Get Filings -> 2. Get Sections -> 3. Get Sentences
                    // BUT to do it in one block efficiently:
                    // TRAVERSAL with depth 3? Or nested?
                    // Let's use the robust traversal if the edges are known.
                    // "FOR v, e, p IN 1..3 OUTBOUND doc HAS_FILING, has_section, has_sentence FILTER IS_SAME_COLLECTION('sec_sentences', v) LIMIT 5 RETURN v"
                    // This assumes the edge names are exactly right.
                    // schema says: Company -> (HAS_FILING) -> sec_filings
                    // sec_filings -> (has_section) -> sec_sections
                    // sec_sections -> (has_sentence) -> sec_sentences
                    aql += `    FOR v IN 1..3 OUTBOUND doc HAS_FILING, has_section, has_sentence\n`
                    aql += `      FILTER IS_SAME_COLLECTION('sec_sentences', v)\n`
                    aql += `      LIMIT 5 RETURN v\n`
                } else {
                    aql += `    RETURN {} // Multi-hop logic not implemented genrically yet\n`
                }
            }

            aql += `  )\n`
            desc += ` + ${targetNode.name}`
        })

        aql += `  LIMIT ${limit}\n`

        // Merge enrichments into return
        if (enrichments.length > 0) {
            const merges = enrichments.map(e => {
                const node = GRAPH_SCHEMA[e.targetKey]
                return node ? `${node.collection}: ${node.collection}_data` : ''
            }).filter(Boolean).join(', ')
            aql += `  RETURN MERGE(doc, { ${merges} })`
        } else {
            aql += `  RETURN doc`
        }

        onQueryChange(aql, desc)
    }, [source, filters, enrichments, limit])

    const addFilter = () => {
        if (!sourceNode) return
        setFilters([...filters, { field: sourceNode.keyFields[0], operator: '==', value: '' }])
    }

    const removeFilter = (idx: number) => {
        const newFilters = [...filters]
        newFilters.splice(idx, 1)
        setFilters(newFilters)
    }

    const updateFilter = (idx: number, field: keyof Filter, value: string) => {
        const newFilters = [...filters]
        newFilters[idx] = { ...newFilters[idx], [field]: value }
        setFilters(newFilters)
    }

    const toggleEnrichment = (targetKey: string) => {
        if (enrichments.find(e => e.targetKey === targetKey)) {
            setEnrichments(enrichments.filter(e => e.targetKey !== targetKey))
        } else {
            setEnrichments([...enrichments, { targetKey }])
        }
    }

    // UI Components
    return (
        <div className="bg-dark-900 border border-gold/20 p-4 rounded-lg space-y-4 shadow-2xl">

            {/* Step 0: Category Selection */}
            {step === 0 && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-3"
                >
                    <div className="text-center space-y-2 pb-2">
                        <h3 className="text-gold font-bold text-sm uppercase tracking-wider">Choose Your Starting Point</h3>
                        <p className="text-xs text-gray-400">Select a category to begin exploring the graph</p>
                    </div>

                    <div className="grid grid-cols-1 gap-3">
                        {CATEGORIES.map(cat => (
                            <motion.button
                                key={cat.key}
                                whileHover={{ scale: 1.02, translateY: -2 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => {
                                    setCategory(cat.key)
                                    setStep(1)
                                }}
                                className="group relative bg-gradient-to-br from-dark-800 to-dark-900 border border-gold/20 hover:border-gold/50 rounded-xl p-4 text-left transition-all shadow-lg hover:shadow-xl"
                            >
                                {/* Icon */}
                                <div className="absolute top-4 right-4 text-4xl opacity-20 group-hover:opacity-30 transition-opacity">
                                    {cat.icon}
                                </div>

                                {/* Content */}
                                <div className="relative space-y-2">
                                    <div className="flex items-center gap-2">
                                        <span className="text-2xl">{cat.icon}</span>
                                        <h4 className="text-gold font-bold text-base">{cat.label}</h4>
                                    </div>
                                    <p className="text-xs text-gray-300 leading-relaxed">{cat.description}</p>

                                    {/* Use Cases */}
                                    <div className="pt-2 space-y-1">
                                        {cat.useCases.slice(0, 3).map((useCase, idx) => (
                                            <div key={idx} className="flex items-center gap-2 text-xs text-gray-400">
                                                <span className="text-gold/60">→</span>
                                                <span>{useCase}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Arrow */}
                                <div className="absolute bottom-4 right-4 text-gold/40 group-hover:text-gold group-hover:translate-x-1 transition-all">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                    </svg>
                                </div>
                            </motion.button>
                        ))}
                    </div>
                </motion.div>
            )}

            {/* Step 1: Collection Selection + Filtering */}
            {step === 1 && selectedCategory && (
                <AnimatePresence>
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="space-y-4"
                    >
                        {/* Breadcrumb */}
                        <div className="flex items-center gap-2 text-xs">
                            <button
                                onClick={() => {
                                    setStep(0)
                                    setCategory('')
                                    setSource('')
                                    setFilters([])
                                    setEnrichments([])
                                }}
                                className="text-gray-400 hover:text-gold transition-colors"
                            >
                                ← Categories
                            </button>
                            <span className="text-gray-600">/</span>
                            <span className="text-gold flex items-center gap-1">
                                <span>{selectedCategory.icon}</span>
                                <span>{selectedCategory.label}</span>
                            </span>
                        </div>

                        {/* Collection Selection */}
                        <div className="space-y-2">
                            <label className="text-[10px] text-gold/60 font-bold uppercase tracking-widest pl-1">
                                1. Choose Collection
                            </label>
                            <div className="grid grid-cols-1 gap-2">
                                {selectedCategory.collections.map(key => {
                                    const node = GRAPH_SCHEMA[key]
                                    if (!node) return null
                                    const isSelected = source === key

                                    return (
                                        <button
                                            key={key}
                                            onClick={() => {
                                                setSource(key)
                                                setFilters([])
                                                setEnrichments([])
                                            }}
                                            className={`p-3 rounded-lg border text-left transition-all ${
                                                isSelected
                                                    ? 'bg-gold/20 border-gold/50'
                                                    : 'bg-dark-800 border-gold/20 hover:border-gold/40'
                                            }`}
                                        >
                                            <div className="text-sm font-semibold text-gold">{node.name}</div>
                                            <div className="text-xs text-gray-400">{node.description}</div>
                                        </button>
                                    )
                                })}
                            </div>
                        </div>

                        {/* Filtering Section (appears after collection selected) */}
                        {sourceNode && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                className="space-y-3 pt-3 border-t border-gold/10"
                            >
                                <div className="space-y-2">
                                    <div className="flex justify-between items-center px-1">
                                        <label className="text-[10px] text-gold/60 font-bold uppercase tracking-widest">
                                            2. Filter Data (Optional)
                                        </label>
                                        <button
                                            onClick={addFilter}
                                            className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                                        >
                                            + Add Filter
                                        </button>
                                    </div>

                                    {filters.length === 0 && (
                                        <div className="text-xs text-gray-500 italic px-2">No filters applied</div>
                                    )}

                                    {filters.map((filter, idx) => (
                                        <div key={idx} className="flex gap-1.5 items-center bg-dark-800/80 p-1.5 rounded border border-gold/10">
                                            <select
                                                value={filter.field}
                                                onChange={(e) => updateFilter(idx, 'field', e.target.value)}
                                                className="bg-dark-900 text-gray-200 text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none"
                                            >
                                                {sourceNode.keyFields.map(f => <option key={f} value={f}>{f}</option>)}
                                            </select>

                                            <select
                                                value={filter.operator}
                                                onChange={(e) => updateFilter(idx, 'operator', e.target.value)}
                                                className="bg-dark-900 text-gold text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none font-mono"
                                            >
                                                <option value="==">==</option>
                                                <option value="!=">!=</option>
                                                <option value=">">&gt;</option>
                                                <option value="<">&lt;</option>
                                                <option value=">=">&gt;=</option>
                                                <option value="<=">&lt;=</option>
                                                <option value="=~">contains</option>
                                            </select>

                                            <input
                                                type="text"
                                                list="company-list"
                                                value={filter.value}
                                                onChange={(e) => updateFilter(idx, 'value', e.target.value)}
                                                placeholder="Value..."
                                                className="bg-dark-900 text-gray-200 text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                            />

                                            <button onClick={() => removeFilter(idx)} className="text-gray-500 hover:text-red-400 px-1">×</button>
                                        </div>
                                    ))}
                                </div>

                                {/* Available Connections */}
                                {sourceNode.connections.length > 0 && (
                                    <div className="space-y-2 pt-3 border-t border-gold/10">
                                        <div className="flex justify-between items-center px-1">
                                            <label className="text-[10px] text-gold/60 font-bold uppercase tracking-widest">
                                                3. Expand via Graph Connections
                                            </label>
                                            <button
                                                onClick={() => setShowConnections(!showConnections)}
                                                className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1"
                                            >
                                                {showConnections ? '▼ Hide' : '▶ Show Available Connections'}
                                            </button>
                                        </div>

                                        {showConnections && (
                                            <motion.div
                                                initial={{ opacity: 0, height: 0 }}
                                                animate={{ opacity: 1, height: 'auto' }}
                                                className="space-y-2"
                                            >
                                                <div className="text-xs text-gray-400 px-2 pb-2">
                                                    These connections are available from {sourceNode.name}:
                                                </div>

                                                <div className="grid grid-cols-1 gap-2">
                                                    {sourceNode.connections.map(conn => {
                                                        const targetKey = conn.target
                                                        const target = GRAPH_SCHEMA[targetKey]
                                                        if (!target) return null
                                                        const isSelected = enrichments.some(e => e.targetKey === targetKey)

                                                        return (
                                                            <button
                                                                key={targetKey}
                                                                onClick={() => toggleEnrichment(targetKey)}
                                                                className={`p-3 rounded-lg border text-left transition-all ${
                                                                    isSelected
                                                                        ? 'bg-purple-500/20 border-purple-500/50'
                                                                        : 'bg-dark-800 border-gray-700 hover:border-purple-400/50'
                                                                }`}
                                                            >
                                                                <div className="flex items-start justify-between">
                                                                    <div className="flex-1 space-y-1">
                                                                        <div className="flex items-center gap-2">
                                                                            <div className={`w-2 h-2 rounded-full ${
                                                                                isSelected ? 'bg-purple-500 shadow-[0_0_5px_rgba(168,85,247,0.5)]' : 'bg-gray-700'
                                                                            }`} />
                                                                            <span className="text-sm font-semibold text-gold">{target.name}</span>
                                                                        </div>
                                                                        <div className="text-xs text-gray-400 pl-4">{target.description}</div>
                                                                        <div className="text-xs text-purple-400/70 pl-4 font-mono">
                                                                            via {conn.edge} ({conn.direction})
                                                                        </div>
                                                                    </div>
                                                                    {isSelected && (
                                                                        <svg className="w-5 h-5 text-purple-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                                        </svg>
                                                                    )}
                                                                </div>
                                                            </button>
                                                        )
                                                    })}
                                                </div>
                                            </motion.div>
                                        )}
                                    </div>
                                )}

                                {/* Limit Config */}
                                <div className="flex items-center justify-between gap-2 pt-3 border-t border-gold/10">
                                    <div className="text-xs text-gray-400">
                                        {enrichments.length > 0 && (
                                            <span className="text-purple-400">
                                                ⚡ {enrichments.length} connection{enrichments.length !== 1 ? 's' : ''} active
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-gray-500">Result limit:</span>
                                        <select
                                            value={limit}
                                            onChange={(e) => setLimit(Number(e.target.value))}
                                            className="bg-dark-800 text-xs border border-gray-700 rounded p-1 text-gray-300"
                                        >
                                            <option value={10}>10</option>
                                            <option value={20}>20</option>
                                            <option value={50}>50</option>
                                            <option value={100}>100</option>
                                        </select>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </motion.div>
                </AnimatePresence>
            )}

            <datalist id="company-list">
                {companies.map(c => (
                    <option key={c.ticker} value={c.ticker}>{c.company} ({c.ticker})</option>
                ))}
            </datalist>
        </div>
    )
}
