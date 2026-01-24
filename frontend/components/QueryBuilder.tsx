'use client'

import { useState, useEffect } from 'react'
import { GRAPH_SCHEMA, SchemaNode } from '@/lib/schema'
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
    targetKey: string
}

// Category definitions with color themes
const CATEGORIES = [
    {
        key: 'core',
        label: 'Core Data',
        icon: 'M3 7h18M3 12h18M3 17h18', // chart bars icon path
        color: 'gold',
        collections: ['company', 'marketdata', 'economicdata']
    },
    {
        key: 'sec',
        label: 'SEC Filings',
        icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z', // document icon path
        color: 'blue',
        collections: ['sec']
    },
    {
        key: 'markets',
        label: 'Prediction Markets',
        icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6', // trending up icon path
        color: 'purple',
        collections: ['predictionmarkets', 'kalshi', 'polymarket_traders', 'polymarket_positions', 'polymarket_price_history']
    },
    {
        key: 'commodities',
        label: 'Commodities',
        icon: 'M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4', // cube icon path
        color: 'amber',
        collections: ['futures', 'cftc', 'eia_crude', 'eia_natgas_storage', 'eia_natgas_production', 'eia_lng']
    },
    {
        key: 'options',
        label: 'Options & Awards',
        icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z', // chart bar icon path
        color: 'green',
        collections: ['options', 'awards']
    }
]

const COLOR_VARIANTS: Record<string, { border: string; bg: string; text: string; glow: string }> = {
    gold: { border: 'border-gold/30', bg: 'bg-gold/10', text: 'text-gold', glow: 'shadow-[0_0_20px_rgba(255,215,0,0.1)]' },
    blue: { border: 'border-blue-500/30', bg: 'bg-blue-500/10', text: 'text-blue-400', glow: 'shadow-[0_0_20px_rgba(59,130,246,0.1)]' },
    purple: { border: 'border-purple-500/30', bg: 'bg-purple-500/10', text: 'text-purple-400', glow: 'shadow-[0_0_20px_rgba(168,85,247,0.1)]' },
    amber: { border: 'border-amber-500/30', bg: 'bg-amber-500/10', text: 'text-amber-400', glow: 'shadow-[0_0_20px_rgba(245,158,11,0.1)]' },
    green: { border: 'border-green-500/30', bg: 'bg-green-500/10', text: 'text-green-400', glow: 'shadow-[0_0_20px_rgba(34,197,94,0.1)]' }
}

export default function QueryBuilder({ onQueryChange }: QueryBuilderProps) {
    // State
    const [category, setCategory] = useState<string>('')
    const [source, setSource] = useState<string>('')
    const [filters, setFilters] = useState<Filter[]>([])
    const [enrichments, setEnrichments] = useState<Enrichment[]>([])
    const [limit, setLimit] = useState(20)
    const [showGraphViz, setShowGraphViz] = useState(false)

    // Company suggestions for autofill
    const [companies, setCompanies] = useState<{ ticker: string, company: string }[]>([])

    useEffect(() => {
        const fetchCompanies = async () => {
            try {
                const { api } = await import('@/lib/api')
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
        <div className="space-y-6">
            {/* Step 1: Category Selection - Horizontal Cards */}
            <div>
                <div className="text-xs text-gray-400 uppercase tracking-wider mb-3 font-semibold">Select Data Category</div>
                <div className="grid grid-cols-5 gap-3">
                    {CATEGORIES.map(cat => {
                        const colors = COLOR_VARIANTS[cat.color]
                        const isSelected = category === cat.key
                        return (
                            <motion.button
                                key={cat.key}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => {
                                    setCategory(cat.key)
                                    setSource('')
                                    setFilters([])
                                    setEnrichments([])
                                }}
                                className={`relative p-4 rounded-xl border-2 transition-all ${
                                    isSelected
                                        ? `${colors.border} ${colors.bg} ${colors.glow}`
                                        : 'border-gray-700 bg-dark-800 hover:border-gray-600'
                                }`}
                            >
                                {/* Icon */}
                                <div className={`mb-2 ${isSelected ? colors.text : 'text-gray-500'}`}>
                                    <svg className="w-8 h-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={cat.icon} />
                                    </svg>
                                </div>
                                {/* Label */}
                                <div className={`text-xs font-semibold ${isSelected ? colors.text : 'text-gray-400'}`}>
                                    {cat.label}
                                </div>
                                {/* Selected Indicator */}
                                {isSelected && (
                                    <motion.div
                                        layoutId="categoryIndicator"
                                        className={`absolute top-2 right-2 w-2 h-2 rounded-full ${colors.text.replace('text-', 'bg-')}`}
                                    />
                                )}
                            </motion.button>
                        )
                    })}
                </div>
            </div>

            {/* Step 2: Collection Selection */}
            <AnimatePresence>
                {selectedCategory && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4"
                    >
                        <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Choose Collection</div>
                        <div className="grid grid-cols-3 gap-3">
                            {selectedCategory.collections.map(key => {
                                const node = GRAPH_SCHEMA[key]
                                if (!node) return null
                                const isSelected = source === key
                                const colors = COLOR_VARIANTS[selectedCategory.color]
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
                                                ? `${colors.border} ${colors.bg}`
                                                : 'border-gray-700 bg-dark-800 hover:border-gray-600'
                                        }`}
                                    >
                                        <div className={`text-sm font-semibold mb-1 ${isSelected ? colors.text : 'text-gray-300'}`}>
                                            {node.name}
                                        </div>
                                        <div className="text-xs text-gray-500 line-clamp-2">
                                            {node.description}
                                        </div>
                                    </button>
                                )
                            })}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Step 3: Filters & Connections */}
            <AnimatePresence>
                {sourceNode && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4"
                    >
                        {/* Filters */}
                        <div>
                            <div className="flex items-center justify-between mb-3">
                                <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Filters (Optional)</div>
                                <button
                                    onClick={addFilter}
                                    className="text-xs text-blue-400 hover:text-blue-300 font-semibold"
                                >
                                    + Add Filter
                                </button>
                            </div>
                            {filters.length > 0 ? (
                                <div className="space-y-2">
                                    {filters.map((filter, idx) => (
                                        <div key={idx} className="flex gap-2 items-center bg-dark-800 p-2 rounded-lg border border-gray-700">
                                            <select
                                                value={filter.field}
                                                onChange={(e) => updateFilter(idx, 'field', e.target.value)}
                                                className="bg-dark-900 text-gray-200 text-xs p-2 rounded border border-gray-700 focus:border-gold/50 outline-none"
                                            >
                                                {sourceNode.keyFields.map(f => <option key={f} value={f}>{f}</option>)}
                                            </select>
                                            <select
                                                value={filter.operator}
                                                onChange={(e) => updateFilter(idx, 'operator', e.target.value)}
                                                className="bg-dark-900 text-gray-200 text-xs p-2 rounded border border-gray-700 focus:border-gold/50 outline-none"
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
                                                placeholder="value"
                                                className="bg-dark-900 text-gray-200 text-xs p-2 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                            />
                                            <button
                                                onClick={() => removeFilter(idx)}
                                                className="text-gray-500 hover:text-red-400 px-2"
                                            >
                                                ×
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-xs text-gray-600 italic">No filters applied</div>
                            )}
                        </div>

                        {/* Connections */}
                        {sourceNode.connections.length > 0 && (
                            <div>
                                <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-3">Graph Connections</div>
                                <div className="grid grid-cols-3 gap-2">
                                    {sourceNode.connections.map(conn => {
                                        const target = GRAPH_SCHEMA[conn.target]
                                        if (!target) return null
                                        const isSelected = enrichments.some(e => e.targetKey === conn.target)
                                        return (
                                            <button
                                                key={conn.target}
                                                onClick={() => toggleEnrichment(conn.target)}
                                                className={`p-3 rounded-lg border text-left transition-all ${
                                                    isSelected
                                                        ? 'bg-purple-500/20 border-purple-500/50'
                                                        : 'bg-dark-800 border-gray-700 hover:border-purple-400/30'
                                                }`}
                                            >
                                                <div className={`text-xs font-semibold mb-1 ${isSelected ? 'text-purple-300' : 'text-gray-300'}`}>
                                                    {target.name}
                                                </div>
                                                <div className="text-xs text-gray-600 font-mono">
                                                    via {conn.edge}
                                                </div>
                                            </button>
                                        )
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Bottom Actions */}
                        <div className="flex items-center justify-between pt-4 border-t border-gray-800">
                            <div className="flex items-center gap-4">
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-gray-500">Result Limit:</span>
                                    <select
                                        value={limit}
                                        onChange={(e) => setLimit(Number(e.target.value))}
                                        className="bg-dark-800 text-xs border border-gray-700 rounded p-1.5 text-gray-300"
                                    >
                                        <option value={10}>10</option>
                                        <option value={20}>20</option>
                                        <option value={50}>50</option>
                                        <option value={100}>100</option>
                                    </select>
                                </div>
                                {enrichments.length > 0 && (
                                    <span className="text-xs text-purple-400">
                                        {enrichments.length} connection{enrichments.length !== 1 ? 's' : ''} active
                                    </span>
                                )}
                            </div>
                            <button
                                onClick={() => setShowGraphViz(!showGraphViz)}
                                className="text-xs text-gray-400 hover:text-gold transition-colors flex items-center gap-1"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                Visualize Graph
                            </button>
                        </div>

                        {/* Graph Visualization */}
                        <AnimatePresence>
                            {showGraphViz && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="bg-dark-800 border border-gold/20 rounded-lg p-4"
                                >
                                    <div className="text-xs text-gold mb-3 font-semibold">Query Graph Structure</div>
                                    <div className="flex items-center gap-3">
                                        {/* Source Node */}
                                        <div className="bg-gold/10 border border-gold/30 rounded-lg p-3 text-center">
                                            <div className="text-xs font-semibold text-gold">{sourceNode.name}</div>
                                            <div className="text-xs text-gray-500 mt-1">Source</div>
                                        </div>

                                        {/* Arrows and Connections */}
                                        {enrichments.map((enr, idx) => {
                                            const target = GRAPH_SCHEMA[enr.targetKey]
                                            if (!target) return null
                                            const conn = sourceNode.connections.find(c => c.target === enr.targetKey)
                                            return (
                                                <div key={idx} className="flex items-center gap-3">
                                                    <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                                    </svg>
                                                    <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 text-center">
                                                        <div className="text-xs font-semibold text-purple-300">{target.name}</div>
                                                        <div className="text-xs text-gray-500 mt-1">{conn?.edge}</div>
                                                    </div>
                                                </div>
                                            )
                                        })}
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                )}
            </AnimatePresence>

            <datalist id="company-list">
                {companies.map(c => (
                    <option key={c.ticker} value={c.ticker}>{c.company} ({c.ticker})</option>
                ))}
            </datalist>
        </div>
    )
}
