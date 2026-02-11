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
    fieldType?: 'text' | 'number' | 'date' | 'boolean'
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

// Quick-start presets: one click sets source + enrichments
type QuickStartBundle = {
    id: string
    label: string
    icon: string
    source: string
    enrichments: string[]
}
const QUICK_START_BUNDLES: QuickStartBundle[] = [
    { id: 'commodities', label: 'Futures + EIA + CFTC', icon: '🌾', source: 'futures', enrichments: ['eia_natgas_storage', 'eia_crude', 'cftc', 'economicdata'] },
    { id: 'company-awards-sec', label: 'Company + Awards + SEC', icon: '🏢', source: 'company', enrichments: ['awards', 'sec'] },
    { id: 'company-market-options', label: 'Company + Market Data + Options', icon: '📈', source: 'company', enrichments: ['marketdata', 'options'] }
]

// Commodities demo: one row per date (last 90 days) with close + inventory for dual-chart visualization
const COMMODITIES_DEMO_AQL = `FOR f IN futures_prices
  FILTER f.commodity == "CRUDE_OIL"
  SORT f.date DESC
  LIMIT 90
  LET inv = (FOR e IN INBOUND f INVENTORY_AFFECTS_PRICE LIMIT 1 RETURN e)
  LET inv_val = inv[0] ? (inv[0].crude_stocks != null ? inv[0].crude_stocks : inv[0].value) : null
  RETURN {
    date: f.date,
    close: f.close,
    open: f.open,
    commodity: f.commodity,
    inventory_million_barrels: inv_val,
    report_date: inv[0] ? (inv[0].report_date || inv[0].date) : null
  }
`
const COMMODITIES_DEMO_DESC = 'Crude oil price vs EIA inventory (demo)'

export default function VisualQueryBuilder({ onQueryChange }: QueryBuilderProps) {
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

    // Semantic search state
    const [searchType, setSearchType] = useState<'keyword' | 'semantic'>('keyword')
    const [similarityThreshold, setSimilarityThreshold] = useState(0.75)

    // Derived
    const sourceNode = source ? GRAPH_SCHEMA[source] : null

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
            aql += `  LET ${targetNode.collection} = (\n`

            if (connection.type === 'direct') {
                if (connection.direction === 'INBOUND') {
                    // FOR t IN INBOUND doc edge_collection (sort by date when target has one)
                    aql += `    FOR t IN INBOUND doc ${connection.edge}\n`
                    const targetDateField = (targetKey === 'eia_crude' || targetKey === 'eia_natgas_storage') ? 'report_date' : (targetNode.keyFields.includes('date') ? 'date' : null)
                    if (targetDateField) {
                        aql += `      SORT t.${targetDateField} DESC\n`
                    }
                    aql += `      LIMIT 5 RETURN t\n`
                } else {
                    // FOR t IN OUTBOUND doc edge_collection
                    aql += `    FOR t IN OUTBOUND doc ${connection.edge}\n`
                    // Special handling for market data sort
                    if (targetKey === 'marketdata') {
                        aql += `      SORT t.date DESC LIMIT 30 RETURN t\n`
                    } else if (targetKey === 'awards') {
                        aql += `      SORT t.start_date DESC LIMIT 5 RETURN t\n`
                    } else if (targetKey === 'options') {
                        aql += `      SORT t.date DESC LIMIT 20 RETURN t\n`
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

        // Sort by most recent when source has a date field (so LIMIT returns newest first)
        const dateSortCollections: Record<string, string> = {
            'futures_prices': 'date',
            'MarketData': 'date',
            'eia_crude_inventory': 'date',
            'eia_natgas_storage': 'date',
            'eia_natgas_production': 'date',
            'eia_lng_exports': 'date',
            'options_flow': 'date',
            'commodity_positions': 'as_of_date'
        }
        const sortField = dateSortCollections[sourceNode.collection] || (sourceNode.collection === 'sec_filings' ? 'filing_date' : null)
        if (sortField) {
            aql += `  SORT doc.${sortField} DESC\n`
        }

        desc += ` (limit ${limit})`
        aql += `  LIMIT ${limit}\n`

        // Merge enrichments into return
        if (enrichments.length > 0) {
            const merges = enrichments.map(e => {
                const node = GRAPH_SCHEMA[e.targetKey]
                return node ? `${node.collection}` : ''
            }).filter(Boolean).join(', ')
            aql += `  RETURN MERGE(doc, { ${merges} })`
        } else {
            aql += `  RETURN doc`
        }

        onQueryChange(aql, desc)
    }, [source, filters, enrichments, limit])

    // Helper to detect field type
    const detectFieldType = (fieldName: string): 'text' | 'number' | 'date' | 'boolean' => {
        const lower = fieldName.toLowerCase()
        if (lower.includes('date') || lower.includes('time') || lower === 'year' || lower === 'month') return 'date'
        if (lower.includes('price') || lower.includes('amount') || lower.includes('volume') ||
            lower.includes('ratio') || lower.includes('yield') || lower.includes('rate') ||
            lower.includes('margin') || lower.includes('cap') || lower.includes('value')) return 'number'
        if (lower.includes('is_') || lower.includes('has_') || lower.includes('above_') ||
            lower.includes('_flag') || lower.includes('unusual')) return 'boolean'
        return 'text'
    }

    const addFilter = () => {
        if (!sourceNode) return
        const firstField = sourceNode.keyFields[0]
        const fieldType = detectFieldType(firstField)
        setFilters([...filters, { field: firstField, operator: '==', value: '', fieldType }])
    }

    const removeFilter = (idx: number) => {
        const newFilters = [...filters]
        newFilters.splice(idx, 1)
        setFilters(newFilters)
    }

    const updateFilter = (idx: number, field: keyof Filter, value: string) => {
        const newFilters = [...filters]
        newFilters[idx] = { ...newFilters[idx], [field]: value }

        // If changing field, detect new field type
        if (field === 'field') {
            newFilters[idx].fieldType = detectFieldType(value)
            // Reset operator to appropriate default for field type
            if (newFilters[idx].fieldType === 'date') {
                newFilters[idx].operator = '>='
            }
        }

        setFilters(newFilters)
    }

    const toggleEnrichment = (targetKey: string) => {
        if (enrichments.find(e => e.targetKey === targetKey)) {
            setEnrichments(enrichments.filter(e => e.targetKey !== targetKey))
        } else {
            setEnrichments([...enrichments, { targetKey }])
        }
    }

    const applyBundle = (bundle: QuickStartBundle) => {
        const node = GRAPH_SCHEMA[bundle.source]
        if (!node) return
        const validEnrichments = bundle.enrichments.filter(t => node.connections.some(c => c.target === t))
        setSource(bundle.source)
        setFilters([])
        setEnrichments(validEnrichments.map(targetKey => ({ targetKey })))
        setStep(1)
        setDropdownOpen(false)
    }

    // UI Components
    return (
        <div className="bg-dark-900/50 p-4 rounded-lg border border-gold/10 space-y-4">

            {/* Quick start presets */}
            <div className="space-y-2">
                <label className="text-xs text-gold font-semibold uppercase tracking-wider">Quick start</label>
                <div className="flex flex-wrap gap-2">
                    {QUICK_START_BUNDLES.map(bundle => (
                        <button
                            key={bundle.id}
                            onClick={() => applyBundle(bundle)}
                            className="px-3 py-2 rounded-lg text-xs border border-gold/30 bg-dark-800/80 hover:border-gold/50 hover:bg-gold/10 transition-all flex items-center gap-2 text-gray-200"
                        >
                            <span>{bundle.icon}</span>
                            <span>{bundle.label}</span>
                        </button>
                    ))}
                    <button
                        onClick={() => onQueryChange(COMMODITIES_DEMO_AQL, COMMODITIES_DEMO_DESC)}
                        className="px-3 py-2 rounded-lg text-xs border border-emerald-500/40 bg-emerald-500/10 hover:border-emerald-400/60 hover:bg-emerald-500/20 transition-all flex items-center gap-2 text-emerald-200"
                        title="Load query: EIA Crude Inventory → Futures Prices. Returns date, close, inventory for two charts."
                    >
                        <span>📊</span>
                        <span>Crude price vs inventory (demo)</span>
                    </button>
                </div>
            </div>

            {/* Step 1: Source Selection - Smart Dropdown */}
            <div className="space-y-2 relative" ref={dropdownRef}>
                <label className="text-xs text-gold font-semibold uppercase tracking-wider">1. Start With</label>
                {!source && (
                    <p className="text-xs text-gray-500 italic">Pick a collection, then add related data with Enrich.</p>
                )}

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
                                                            <div className={`text-sm font-medium ${source === key ? 'text-gold' : 'text-gray-200'} flex items-center gap-2`}>
                                                                <span>{node.name}</span>
                                                                {node.supportsSemanticSearch && (
                                                                    <span className="text-[10px] px-1.5 py-0.5 bg-blue-500/20 text-blue-300 rounded border border-blue-500/30 font-normal">
                                                                        🔍 Semantic Search
                                                                    </span>
                                                                )}
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

                            {filters.map((filter, idx) => {
                                const fieldType = filter.fieldType || 'text'
                                return (
                                    <div key={idx} className="flex gap-2 items-center bg-dark-800 p-2 rounded border border-gold/10">
                                        <select
                                            value={filter.field}
                                            onChange={(e) => updateFilter(idx, 'field', e.target.value)}
                                            className="bg-dark-900 text-gray-200 text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                        >
                                            {sourceNode.keyFields.map(f => (
                                                <option key={f} value={f}>
                                                    {f}
                                                    {f.toLowerCase().includes('date') && ' 📅'}
                                                </option>
                                            ))}
                                        </select>

                                        <select
                                            value={filter.operator}
                                            onChange={(e) => updateFilter(idx, 'operator', e.target.value)}
                                            className="bg-dark-900 text-gold text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none font-mono"
                                        >
                                            {fieldType === 'boolean' ? (
                                                <>
                                                    <option value="==">is</option>
                                                    <option value="!=">is not</option>
                                                </>
                                            ) : fieldType === 'text' ? (
                                                <>
                                                    <option value="==">==</option>
                                                    <option value="!=">!=</option>
                                                    <option value="=~">contains</option>
                                                </>
                                            ) : (
                                                <>
                                                    <option value="==">==</option>
                                                    <option value="!=">!=</option>
                                                    <option value=">">&gt;</option>
                                                    <option value="<">&lt;</option>
                                                    <option value=">=">&gt;=</option>
                                                    <option value="<=">&lt;=</option>
                                                </>
                                            )}
                                        </select>

                                        {fieldType === 'date' ? (
                                            <input
                                                type="date"
                                                value={filter.value}
                                                onChange={(e) => updateFilter(idx, 'value', e.target.value)}
                                                className="bg-dark-900 text-gray-200 text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                            />
                                        ) : fieldType === 'boolean' ? (
                                            <select
                                                value={filter.value}
                                                onChange={(e) => updateFilter(idx, 'value', e.target.value)}
                                                className="bg-dark-900 text-gray-200 text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                            >
                                                <option value="">Select...</option>
                                                <option value="true">true</option>
                                                <option value="false">false</option>
                                            </select>
                                        ) : fieldType === 'number' ? (
                                            <input
                                                type="number"
                                                value={filter.value}
                                                onChange={(e) => updateFilter(idx, 'value', e.target.value)}
                                                placeholder="Number..."
                                                step="any"
                                                className="bg-dark-900 text-gray-200 text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                            />
                                        ) : (
                                            <input
                                                type="text"
                                                value={filter.value}
                                                onChange={(e) => updateFilter(idx, 'value', e.target.value)}
                                                placeholder="Value..."
                                                className="bg-dark-900 text-gray-200 text-xs p-1 rounded border border-gray-700 focus:border-gold/50 outline-none flex-1"
                                            />
                                        )}

                                        <button onClick={() => removeFilter(idx)} className="text-gray-500 hover:text-red-400 px-1 text-lg">×</button>
                                    </div>
                                )
                            })}
                        </div>

                        {/* Step 3: Enrich */}
                        {sourceNode.connections.length > 0 && (
                            <div className="space-y-2 pt-2 border-t border-gold/10">
                                <label className="text-xs text-gold font-semibold uppercase tracking-wider">3. Enrich With (Connect)</label>
                                {source === 'futures' && (
                                    <p className="text-xs text-gray-500">Futures → EIA inventory/storage, CFTC positions, Economic data</p>
                                )}
                                {source === 'company' && (
                                    <p className="text-xs text-gray-500">Company → Market data, Awards, SEC, Options, Prediction markets</p>
                                )}
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
