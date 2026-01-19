'use client'

import { useState, useEffect } from 'react'
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
}

export default function QueryBuilder({ onQueryChange }: QueryBuilderProps) {
    // Steps: 0 = Source, 1 = Filter, 2 = Enrich
    const [step, setStep] = useState(0)

    // State
    const [source, setSource] = useState<string>('')
    const [filters, setFilters] = useState<Filter[]>([])
    const [enrichments, setEnrichments] = useState<Enrichment[]>([])
    const [limit, setLimit] = useState(20)

    // Derived
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
                        aql += `      SORT t.date DESC LIMIT 30 RETURN t\n`
                    } else if (targetKey === 'awards') {
                        aql += `      SORT t.start_date DESC LIMIT 5 RETURN t\n`
                    } else if (targetKey === 'sec') {
                        aql += `      LET top_sentences = (\n`
                        aql += `        FOR s IN 1..2 OUTBOUND t has_section, has_sentence\n`
                        aql += `        FILTER s.finbert_score > 0.4 OR s.finbert_score < -0.4 // Optimization: Use index to filter significant sentiment first\n`
                        aql += `        SORT ABS(s.finbert_score) DESC LIMIT 5 RETURN { text: s.text, score: s.finbert_score }\n`
                        aql += `      )\n`
                        aql += `      LIMIT 5 RETURN MERGE(t, { top_sentences })\n`
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
        <div className="bg-dark-900/50 p-4 rounded-lg border border-gold/10 space-y-4">

            {/* Step 1: Source Selection */}
            <div className="space-y-2">
                <label className="text-xs text-gold font-semibold uppercase tracking-wider">1. Start With</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {Object.entries(GRAPH_SCHEMA)
                        .filter(([key]) => key !== 'sec' && key !== 'sec_sections')
                        // Only show nodes that act as roots/sources (usually Company)
                        // Or let user pick anything? For now, stick to user's flow
                        .map(([key, node]) => (
                            <button
                                key={key}
                                onClick={() => {
                                    setSource(key)
                                    setFilters([])
                                    setEnrichments([])
                                    setStep(1)
                                }}
                                className={`p-2 rounded text-xs md:text-sm border transition-all truncate text-left
                ${source === key
                                        ? 'bg-gold/20 border-gold text-gold'
                                        : 'bg-dark-800 border-gold/10 text-gray-400 hover:border-gold/30'
                                    }`}
                            >
                                <div className="font-semibold">{node.name}</div>
                                <div className="text-[10px] opacity-60 truncate">{node.description}</div>
                            </button>
                        ))}
                </div>
            </div>

            {sourceNode && (
                <AnimatePresence>
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="space-y-4 pt-2 border-t border-gold/10"
                    >

                        {/* Step 2: Filters */}
                        <div className="space-y-2">
                            <div className="flex justify-between items-center">
                                <label className="text-xs text-gold font-semibold uppercase tracking-wider">2. Filter Data (Where)</label>
                                <button
                                    onClick={addFilter}
                                    className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                                >
                                    + Add Filter
                                </button>
                            </div>

                            {filters.length === 0 && (
                                <div className="text-xs text-gray-500 italic px-2">No filters (Get all records)</div>
                            )}

                            {filters.map((filter, idx) => (
                                <div key={idx} className="flex gap-2 items-center bg-dark-800 p-2 rounded border border-gold/10">
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
                                        value={filter.value}
                                        onChange={(e) => updateFilter(idx, 'value', e.target.value)}
                                        placeholder="Value..."
                                        className="bg-dark-900 text-gray-200 text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                    />

                                    <button onClick={() => removeFilter(idx)} className="text-gray-500 hover:text-red-400 px-1">×</button>
                                </div>
                            ))}
                        </div>

                        {/* Step 3: Enrich */}
                        {sourceNode.connections.length > 0 && (
                            <div className="space-y-2 pt-2 border-t border-gold/10">
                                <label className="text-xs text-gold font-semibold uppercase tracking-wider">3. Enrich With (Connect)</label>
                                <div className="flex flex-wrap gap-2">
                                    {sourceNode.connections.map(conn => {
                                        const targetKey = conn.target
                                        const target = GRAPH_SCHEMA[targetKey]
                                        if (!target) return null
                                        const isSelected = enrichments.some(e => e.targetKey === targetKey)

                                        return (
                                            <button
                                                key={targetKey}
                                                onClick={() => toggleEnrichment(targetKey)}
                                                className={`px-3 py-1.5 rounded-full text-xs border flex items-center gap-2 transition-all
                          ${isSelected
                                                        ? 'bg-purple-500/20 border-purple-500 text-purple-300'
                                                        : 'bg-dark-800 border-gray-700 text-gray-400 hover:border-purple-500/50'
                                                    }`}
                                            >
                                                <span className={isSelected ? 'bg-purple-500 w-2 h-2 rounded-full' : 'bg-gray-600 w-2 h-2 rounded-full'} />
                                                {target.name}
                                            </button>
                                        )
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Limit Config */}
                        <div className="flex items-center justify-end gap-2 pt-2">
                            <span className="text-xs text-gray-500">Limit:</span>
                            <select
                                value={limit}
                                onChange={(e) => setLimit(Number(e.target.value))}
                                className="bg-dark-800 text-xs border border-gray-700 rounded p-1 text-gray-300"
                            >
                                <option value={10}>10</option>
                                <option value={50}>50</option>
                                <option value={100}>100</option>
                                <option value={500}>500</option>
                            </select>
                        </div>

                    </motion.div>
                </AnimatePresence>
            )}
        </div>
    )
}
