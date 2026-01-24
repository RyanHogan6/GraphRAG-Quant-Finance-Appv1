'use client'

import { useState, useEffect } from 'react'
import { GRAPH_SCHEMA, SchemaNode } from '@/lib/schema'

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

// Simplified category grouping
const CATEGORY_GROUPS: Record<string, { label: string; collections: string[] }> = {
    core: { label: 'Core', collections: ['company', 'marketdata', 'economicdata'] },
    sec: { label: 'SEC', collections: ['sec'] },
    markets: { label: 'Markets', collections: ['predictionmarkets', 'kalshi', 'polymarket_traders', 'polymarket_positions', 'polymarket_price_history'] },
    commodities: { label: 'Commodities', collections: ['futures', 'cftc', 'eia_crude', 'eia_natgas_storage', 'eia_natgas_production', 'eia_lng'] },
    options: { label: 'Options', collections: ['options', 'awards'] }
}

export default function QueryBuilder({ onQueryChange }: QueryBuilderProps) {
    // State
    const [source, setSource] = useState<string>('')
    const [filters, setFilters] = useState<Filter[]>([])
    const [enrichments, setEnrichments] = useState<Enrichment[]>([])
    const [limit, setLimit] = useState(20)

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
        <div className="bg-dark-900 border border-gold/20 p-3 rounded-lg space-y-3">
            {/* Single Row Layout */}
            <div className="flex flex-wrap items-start gap-3">
                {/* Collection Selection */}
                <div className="flex-shrink-0" style={{ minWidth: '200px', maxWidth: '250px' }}>
                    <label className="text-[9px] text-gray-400 uppercase tracking-wider mb-1 block">Collection</label>
                    <select
                        value={source}
                        onChange={(e) => {
                            setSource(e.target.value)
                            setFilters([])
                            setEnrichments([])
                        }}
                        className="w-full bg-dark-800 text-gray-200 text-xs p-2 rounded border border-gray-700 focus:border-gold/50 outline-none"
                    >
                        <option value="">Select...</option>
                        {Object.entries(CATEGORY_GROUPS).map(([groupKey, group]) => (
                            <optgroup key={groupKey} label={group.label}>
                                {group.collections.map(key => {
                                    const node = GRAPH_SCHEMA[key]
                                    if (!node) return null
                                    return (
                                        <option key={key} value={key}>
                                            {node.name}
                                        </option>
                                    )
                                })}
                            </optgroup>
                        ))}
                    </select>
                </div>

                {/* Filters (if collection selected) */}
                {sourceNode && filters.length > 0 && (
                    <div className="flex-1 min-w-[300px]">
                        <label className="text-[9px] text-gray-400 uppercase tracking-wider mb-1 block">Filters</label>
                        <div className="space-y-1">
                            {filters.map((filter, idx) => (
                                <div key={idx} className="flex gap-1 items-center">
                                    <select
                                        value={filter.field}
                                        onChange={(e) => updateFilter(idx, 'field', e.target.value)}
                                        className="bg-dark-800 text-gray-200 text-xs p-1.5 rounded border border-gray-700 focus:border-gold/50 outline-none"
                                    >
                                        {sourceNode.keyFields.map(f => <option key={f} value={f}>{f}</option>)}
                                    </select>
                                    <select
                                        value={filter.operator}
                                        onChange={(e) => updateFilter(idx, 'operator', e.target.value)}
                                        className="bg-dark-800 text-gray-200 text-xs p-1.5 rounded border border-gray-700 focus:border-gold/50 outline-none"
                                    >
                                        <option value="==">==</option>
                                        <option value="!=">!=</option>
                                        <option value=">">&gt;</option>
                                        <option value="<">&lt;</option>
                                    </select>
                                    <input
                                        type="text"
                                        list="company-list"
                                        value={filter.value}
                                        onChange={(e) => updateFilter(idx, 'value', e.target.value)}
                                        placeholder="value"
                                        className="bg-dark-800 text-gray-200 text-xs p-1.5 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                    />
                                    <button onClick={() => removeFilter(idx)} className="text-gray-500 hover:text-red-400 text-sm px-1">×</button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Connections (if collection selected) */}
                {sourceNode && sourceNode.connections.length > 0 && (
                    <div className="flex-shrink-0" style={{ minWidth: '200px', maxWidth: '250px' }}>
                        <label className="text-[9px] text-gray-400 uppercase tracking-wider mb-1 block">Connections</label>
                        <div className="flex flex-wrap gap-1">
                            {sourceNode.connections.slice(0, 6).map(conn => {
                                const target = GRAPH_SCHEMA[conn.target]
                                if (!target) return null
                                const isSelected = enrichments.some(e => e.targetKey === conn.target)
                                return (
                                    <button
                                        key={conn.target}
                                        onClick={() => toggleEnrichment(conn.target)}
                                        className={`px-2 py-1 rounded text-[10px] border transition-all ${
                                            isSelected
                                                ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                                                : 'bg-dark-800 border-gray-700 text-gray-400 hover:border-purple-400/30'
                                        }`}
                                    >
                                        {target.name}
                                    </button>
                                )
                            })}
                        </div>
                    </div>
                )}

                {/* Limit */}
                {sourceNode && (
                    <div className="flex-shrink-0">
                        <label className="text-[9px] text-gray-400 uppercase tracking-wider mb-1 block">Limit</label>
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
                )}
            </div>

            {/* Add Filter Button */}
            {sourceNode && (
                <div className="flex items-center gap-2 pt-1 border-t border-gray-800">
                    <button
                        onClick={addFilter}
                        className="text-xs text-blue-400 hover:text-blue-300"
                    >
                        + Filter
                    </button>
                    {enrichments.length > 0 && (
                        <span className="text-xs text-gray-500">
                            · {enrichments.length} connection{enrichments.length !== 1 ? 's' : ''}
                        </span>
                    )}
                </div>
            )}

            <datalist id="company-list">
                {companies.map(c => (
                    <option key={c.ticker} value={c.ticker}>{c.company} ({c.ticker})</option>
                ))}
            </datalist>
        </div>
    )
}
