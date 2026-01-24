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
}

// Collection grouping structure
type CollectionGroup = {
    label: string
    icon: string
    collections: string[]
    subItems?: Array<{
        key: string
        label: string
        filter: { field: string; operator: string; value: string } | null
    }>
}

const COLLECTION_GROUPS: Record<string, CollectionGroup> = {
    core: {
        label: 'Core Data',
        icon: '📊',
        collections: ['company', 'marketdata', 'economicdata']
    },
    sec: {
        label: 'SEC Filings',
        icon: '📄',
        collections: ['sec'],
        subItems: [
            { key: 'sec-all', label: 'All SEC Filings', filter: null },
            { key: 'sec-form4', label: 'Form 4/5 - Insider Trades', filter: { field: 'type', operator: 'IN', value: '["4", "5"]' } },
            { key: 'sec-10k', label: '10-K - Annual Reports', filter: { field: 'type', operator: '==', value: '10-K' } },
            { key: 'sec-8k', label: '8-K - Material Events', filter: { field: 'type', operator: '==', value: '8-K' } },
            { key: 'sec-10q', label: '10-Q - Quarterly Reports', filter: { field: 'type', operator: '==', value: '10-Q' } },
            { key: 'sec-13d', label: 'SC 13D/G - Institutional', filter: { field: 'type', operator: 'IN', value: '["SC 13D", "SC 13G"]' } },
            { key: 'sec-13f', label: '13F-HR - Hedge Funds', filter: { field: 'type', operator: '==', value: '13F-HR' } }
        ]
    },
    markets: {
        label: 'Prediction Markets',
        icon: '🎲',
        collections: ['predictionmarkets', 'kalshi', 'polymarket_traders', 'polymarket_positions', 'polymarket_price_history']
    },
    commodities: {
        label: 'Commodities & Energy',
        icon: '🌾',
        collections: ['futures', 'cftc', 'eia_crude', 'eia_natgas_storage', 'eia_natgas_production', 'eia_lng']
    },
    options: {
        label: 'Options & Contracts',
        icon: '📈',
        collections: ['options', 'awards']
    }
}

export default function QueryBuilder({ onQueryChange }: QueryBuilderProps) {
    // Steps: 0 = Source, 1 = Filter, 2 = Enrich
    const [step, setStep] = useState(0)

    // State
    const [source, setSource] = useState<string>('')
    const [filters, setFilters] = useState<Filter[]>([])
    const [enrichments, setEnrichments] = useState<Enrichment[]>([])
    const [limit, setLimit] = useState(20)

    // Dropdown state
    const [dropdownOpen, setDropdownOpen] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const dropdownRef = useRef<HTMLDivElement>(null)

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

    // Close dropdown on click outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setDropdownOpen(false)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
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
        <div className="bg-dark-900 border border-gold/20 p-3 rounded-lg space-y-3 shadow-2xl">

            {/* Step 1: Source Selection - Smart Dropdown */}
            <div className="space-y-1.5 relative" ref={dropdownRef}>
                <label className="text-[10px] text-gold/60 font-bold uppercase tracking-widest pl-1">1. START WITH</label>

                {/* Dropdown Button */}
                <button
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                    className="w-full p-3 rounded-lg text-left border border-gold/20 bg-dark-800 hover:border-gold/40 transition-all flex items-center justify-between group"
                >
                    <div className="flex-1">
                        {source ? (
                            <div>
                                <div className="text-sm font-semibold text-gold">{sourceNode?.name}</div>
                                <div className="text-xs text-gray-400 truncate">{sourceNode?.description}</div>
                            </div>
                        ) : (
                            <div className="text-sm text-gray-400">Select a collection...</div>
                        )}
                    </div>
                    <svg className={`w-5 h-5 text-gold transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </button>

                {/* Dropdown Menu */}
                {dropdownOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="absolute z-50 mt-1 left-0 right-0 bg-dark-800 border border-gold/20 rounded-lg shadow-2xl max-h-96 overflow-hidden"
                    >
                        {/* Search Bar */}
                        <div className="sticky top-0 bg-dark-900 border-b border-gold/10 p-3">
                            <input
                                type="text"
                                placeholder="🔍 Search collections..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full bg-dark-800 text-gray-200 text-sm px-3 py-2 rounded border border-gray-700 focus:border-gold/50 outline-none"
                                autoFocus
                            />
                        </div>

                        {/* Collections List */}
                        <div className="overflow-y-auto max-h-80">
                            {Object.entries(COLLECTION_GROUPS).map(([groupKey, group]) => {
                                // Filter collections based on search
                                const visibleCollections = group.collections.filter(key => {
                                    const node = GRAPH_SCHEMA[key]
                                    if (!node) return false
                                    const searchLower = searchQuery.toLowerCase()
                                    return (
                                        node.name.toLowerCase().includes(searchLower) ||
                                        node.description.toLowerCase().includes(searchLower) ||
                                        node.collection.toLowerCase().includes(searchLower)
                                    )
                                })

                                if (visibleCollections.length === 0 && searchQuery) return null

                                return (
                                    <div key={groupKey} className="border-b border-gold/10 last:border-0">
                                        {/* Group Header */}
                                        <div className="px-4 py-2 bg-dark-900/50 text-xs font-semibold text-gold/60 uppercase tracking-wider flex items-center gap-2">
                                            <span>{group.icon}</span>
                                            <span>{group.label}</span>
                                        </div>

                                        {/* Collections in Group */}
                                        {visibleCollections.map(key => {
                                            const node = GRAPH_SCHEMA[key]
                                            if (!node) return null

                                            return (
                                                <div key={key}>
                                                    {/* Main Collection */}
                                                    <button
                                                        onClick={() => {
                                                            setSource(key)
                                                            setFilters([])
                                                            setEnrichments([])
                                                            setStep(1)
                                                            setDropdownOpen(false)
                                                            setSearchQuery('')
                                                        }}
                                                        className={`w-full px-4 py-2.5 text-left hover:bg-gold/10 transition-colors flex items-center justify-between group
                                                            ${source === key ? 'bg-gold/20' : ''}`}
                                                    >
                                                        <div className="flex-1">
                                                            <div className={`text-sm font-medium ${source === key ? 'text-gold' : 'text-gray-200'}`}>
                                                                {node.name}
                                                            </div>
                                                            <div className="text-xs text-gray-400 truncate">
                                                                {node.description}
                                                            </div>
                                                        </div>
                                                        {source === key && (
                                                            <svg className="w-5 h-5 text-gold" fill="currentColor" viewBox="0 0 20 20">
                                                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                            </svg>
                                                        )}
                                                    </button>

                                                    {/* SEC Sub-Items (Quick Filters) */}
                                                    {key === 'sec' && group.subItems && (
                                                        <div className="bg-dark-900/30">
                                                            {group.subItems.map(subItem => (
                                                                <button
                                                                    key={subItem.key}
                                                                    onClick={() => {
                                                                        setSource('sec')
                                                                        if (subItem.filter) {
                                                                            setFilters([subItem.filter])
                                                                        } else {
                                                                            setFilters([])
                                                                        }
                                                                        setEnrichments([])
                                                                        setStep(1)
                                                                        setDropdownOpen(false)
                                                                        setSearchQuery('')
                                                                    }}
                                                                    className="w-full px-8 py-2 text-left hover:bg-gold/5 transition-colors"
                                                                >
                                                                    <div className="text-xs text-gray-300 hover:text-gold transition-colors flex items-center gap-2">
                                                                        <span className="text-gold/40">└─</span>
                                                                        <span>{subItem.label}</span>
                                                                    </div>
                                                                </button>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            )
                                        })}
                                    </div>
                                )
                            })}

                            {/* No Results */}
                            {searchQuery && Object.entries(COLLECTION_GROUPS).every(([_, group]) =>
                                group.collections.every(key => {
                                    const node = GRAPH_SCHEMA[key]
                                    if (!node) return true
                                    const searchLower = searchQuery.toLowerCase()
                                    return !(
                                        node.name.toLowerCase().includes(searchLower) ||
                                        node.description.toLowerCase().includes(searchLower) ||
                                        node.collection.toLowerCase().includes(searchLower)
                                    )
                                })
                            ) && (
                                <div className="px-4 py-8 text-center text-gray-500 text-sm">
                                    No collections found for "{searchQuery}"
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </div>

            {sourceNode && (
                <AnimatePresence>
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="space-y-4 pt-2 border-t border-gold/10"
                    >

                        {/* Step 2: Filters */}
                        <div className="space-y-1.5">
                            <div className="flex justify-between items-center px-1">
                                <label className="text-[10px] text-gold/60 font-bold uppercase tracking-widest">2. FILTER DATA</label>
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

                        {/* Step 3: Enrich */}
                        {sourceNode.connections.length > 0 && (
                            <div className="space-y-1.5 pt-1 border-t border-gold/10">
                                <label className="text-[10px] text-gold/60 font-bold uppercase tracking-widest pl-1">3. ENRICH WITH</label>
                                <div className="flex flex-wrap gap-1.5">
                                    {sourceNode.connections.map(conn => {
                                        const targetKey = conn.target
                                        const target = GRAPH_SCHEMA[targetKey]
                                        if (!target) return null
                                        const isSelected = enrichments.some(e => e.targetKey === targetKey)

                                        return (
                                            <button
                                                key={targetKey}
                                                onClick={() => toggleEnrichment(targetKey)}
                                                className={`px-2.5 py-1 rounded-full text-[10px] border flex items-center gap-1.5 transition-all
                          ${isSelected
                                                        ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                                                        : 'bg-dark-800 border-gray-700 text-gray-500 hover:border-gold/30'
                                                    }`}
                                            >
                                                <span className={isSelected ? 'bg-purple-500 w-1.5 h-1.5 rounded-full shadow-[0_0_5px_rgba(168,85,247,0.5)]' : 'bg-gray-700 w-1.5 h-1.5 rounded-full'} />
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

            <datalist id="company-list">
                {companies.map(c => (
                    <option key={c.ticker} value={c.ticker}>{c.company} ({c.ticker})</option>
                ))}
            </datalist>
        </div>
    )
}
