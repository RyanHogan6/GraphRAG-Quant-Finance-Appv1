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
    collection: string
    targetField: string
}

export default function VisualQueryBuilder({ onQueryChange }: QueryBuilderProps) {
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

        // Enrichments (Graph Traversals)
        enrichments.forEach(e => {
            const targetNode = GRAPH_SCHEMA[e.collection]
            if (!targetNode) return

            aql += `\n  // Enrich with ${targetNode.name}\n`

            // Connection logic depends on schema. For MVP we use standard edge patterns
            // This is a simplification. Real graph traversals might need specific edge collections.
            // We will try to infer connection based on IDs or semantic links.

            // Strategy: Simple Subquery Lookup
            // e.g. FOR m IN MarketData FILTER m.ticker == doc.ticker

            if (sourceNode.collection === 'Company' && targetNode.collection === 'MarketData') {
                aql += `  LET ${e.collection}_data = (\n`
                aql += `    FOR x IN MarketData FILTER x.ticker == doc.ticker SORT x.date DESC LIMIT 1 RETURN x\n`
                aql += `  )\n`
            } else if (sourceNode.collection === 'Company' && targetNode.collection === 'Award') {
                aql += `  LET ${e.collection}_data = (\n`
                aql += `    FOR x IN Award FILTER x.recipient_name == doc.name SORT x.award_amount_float DESC LIMIT 5 RETURN x\n`
                aql += `  )\n`
            } else {
                // Generic fallback or Graph Traversal
                // For MVP, if no specific logic, we skip or show warning
                aql += `  // Auto-connection logic for ${targetNode.name} not implemented in this MVP builder\n`
            }

            desc += ` + ${targetNode.name}`
        })

        aql += `  LIMIT ${limit}\n`

        // Merge enrichments into return
        if (enrichments.length > 0) {
            const merges = enrichments.map(e => `${e.collection}: ${e.collection}_data`).join(', ')
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

    const toggleEnrichment = (collectionKey: string) => {
        if (enrichments.find(e => e.collection === collectionKey)) {
            setEnrichments(enrichments.filter(e => e.collection !== collectionKey))
        } else {
            setEnrichments([...enrichments, { collection: collectionKey, targetField: 'data' }])
        }
    }

    // UI Components
    return (
        <div className="bg-dark-900/50 p-4 rounded-lg border border-gold/10 space-y-4">

            {/* Step 1: Source Selection */}
            <div className="space-y-2">
                <label className="text-xs text-gold font-semibold uppercase tracking-wider">1. Start With</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {Object.entries(GRAPH_SCHEMA).map(([key, node]) => (
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
                        {sourceNode.validConnections.length > 0 && (
                            <div className="space-y-2 pt-2 border-t border-gold/10">
                                <label className="text-xs text-gold font-semibold uppercase tracking-wider">3. Enrich With (Connect)</label>
                                <div className="flex flex-wrap gap-2">
                                    {sourceNode.validConnections.map(targetKey => {
                                        const target = GRAPH_SCHEMA[targetKey]
                                        if (!target) return null
                                        const isSelected = enrichments.some(e => e.collection === targetKey)

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
