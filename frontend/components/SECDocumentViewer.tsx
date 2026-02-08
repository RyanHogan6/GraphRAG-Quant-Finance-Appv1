'use client'

import { useState, useMemo } from 'react'
import SECFilingsExplorer from './SECFilingsExplorer'

export type SECDocumentTab = 'filings' | 'exhibits' | 'xbrl'

interface SECDocumentViewerProps {
    filings: any[]
    exhibits: any[]
    xbrlData: any[]
    ticker: string
    onSelectFiling?: (filing: any) => void
    showFullAmounts?: boolean
}

function getYearQuarter(dateStr: string | undefined): { year: number; quarter: number } | null {
    if (!dateStr) return null
    const d = new Date(dateStr)
    if (isNaN(d.getTime())) return null
    const year = d.getFullYear()
    const month = d.getMonth()
    const quarter = Math.floor(month / 3) + 1
    return { year, quarter }
}

function getFiscalYearQuarter(xbrl: any): { year: number; quarter: number } | null {
    const fy = xbrl.fiscal_year
    if (fy == null) return xbrl.filing_date ? getYearQuarter(xbrl.filing_date) : null
    const year = typeof fy === 'number' ? fy : parseInt(String(fy), 10)
    if (isNaN(year)) return null
    const ft = (xbrl.filing_type || '').toString()
    const q = ft.includes('10-Q') ? (ft.match(/Q(\d)/)?.[1] ? parseInt(ft.match(/Q(\d)/)?.[1] ?? '1', 10) : null) : 4
    return { year, quarter: q ?? 4 }
}

export default function SECDocumentViewer({
    filings,
    exhibits,
    xbrlData,
    ticker,
    onSelectFiling,
    showFullAmounts = true
}: SECDocumentViewerProps) {
    const [activeTab, setActiveTab] = useState<SECDocumentTab>('filings')
    const [filterYear, setFilterYear] = useState<string>('all')
    const [filterQuarter, setFilterQuarter] = useState<string>('all')
    const [expandedXbrlIndex, setExpandedXbrlIndex] = useState<number | null>(null)

    const years = useMemo(() => {
        const set = new Set<number>()
        filings.forEach((f: any) => {
            const yq = getYearQuarter(f.filing_date)
            if (yq) set.add(yq.year)
        })
        exhibits.forEach((ex: any) => {
            const yq = getYearQuarter(ex.filing_date || ex.filed_date || ex.parent_filing_date)
            if (yq) set.add(yq.year)
        })
        xbrlData.forEach((x: any) => {
            const yq = getFiscalYearQuarter(x)
            if (yq) set.add(yq.year)
        })
        return Array.from(set).sort((a, b) => b - a)
    }, [filings, exhibits, xbrlData])

    const filterByPeriod = useMemo(() => {
        const y = filterYear === 'all' ? null : parseInt(filterYear, 10)
        const q = filterQuarter === 'all' ? null : parseInt(filterQuarter, 10)
        return { year: y, quarter: q }
    }, [filterYear, filterQuarter])

    const filteredFilings = useMemo(() => {
        if (!filterByPeriod.year) return filings
        return filings.filter((f: any) => {
            const yq = getYearQuarter(f.filing_date)
            if (!yq) return false
            if (yq.year !== filterByPeriod.year) return false
            if (filterByPeriod.quarter != null && yq.quarter !== filterByPeriod.quarter) return false
            return true
        })
    }, [filings, filterByPeriod])

    const filteredExhibits = useMemo(() => {
        if (!filterByPeriod.year) return exhibits
        return exhibits.filter((ex: any) => {
            const yq = getYearQuarter(ex.filing_date || ex.filed_date || ex.parent_filing_date)
            if (!yq) return false
            if (yq.year !== filterByPeriod.year) return false
            if (filterByPeriod.quarter != null && yq.quarter !== filterByPeriod.quarter) return false
            return true
        })
    }, [exhibits, filterByPeriod])

    const filteredXbrl = useMemo(() => {
        if (!filterByPeriod.year) return xbrlData
        return xbrlData.filter((x: any) => {
            const yq = getFiscalYearQuarter(x)
            if (!yq) return false
            if (yq.year !== filterByPeriod.year) return false
            if (filterByPeriod.quarter != null && yq.quarter !== filterByPeriod.quarter) return false
            return true
        })
    }, [xbrlData, filterByPeriod])

    const secCompanyUrl = ticker ? `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=${encodeURIComponent(ticker)}` : null
    const exhibitTypeHint: Record<string, string> = {
        'EX-10': 'Material contracts (credit, employment, M&A, etc.)',
        'EX-4': 'Debt instruments, indentures',
        'EX-21': 'Subsidiaries list',
        'EX-99': 'Additional exhibits (press releases, etc.)'
    }
    const getCategoryHint = (ex: any) => {
        const cat = (ex.exhibit_category || ex.exhibit_type || '').toString().replace(/\d.*$/, '').trim()
        return exhibitTypeHint[cat] || ''
    }
    const exhibitDate = (ex: any) => ex.filing_date || ex.filed_date || ex.parent_filing_date || '—'

    type ExhibitGroup = { type: string; exhibits: any[] }
    const groupedExhibits = useMemo(() => {
        const initialGroups: ExhibitGroup[] = []
        return filteredExhibits.reduce((acc: ExhibitGroup[], ex: any) => {
            const type = (ex.exhibit_category || ex.exhibit_type || 'Other').toString().replace(/\d.*$/, '').trim() || 'Other'
            let group = acc.find(g => g.type === type)
            if (!group) {
                group = { type, exhibits: [] }
                acc.push(group)
            }
            group.exhibits.push(ex)
            return acc
        }, initialGroups).sort((a, b) => a.type.localeCompare(b.type))
    }, [filteredExhibits])

    const hasAny = filteredFilings.length > 0 || filteredExhibits.length > 0 || filteredXbrl.length > 0
    if (!hasAny && filings.length === 0 && exhibits.length === 0 && xbrlData.length === 0) {
        return (
            <div className="bg-dark-900/40 border border-white/10 rounded-xl p-6 text-center">
                <div className="text-gray-500 text-sm">No SEC documents available for {ticker}</div>
            </div>
        )
    }

    const tabs: { id: SECDocumentTab; label: string }[] = [
        { id: 'filings', label: 'Filings' },
        { id: 'exhibits', label: 'Exhibits & contracts' },
        { id: 'xbrl', label: 'Financial breakdowns (XBRL)' }
    ]

    return (
        <div className="bg-dark-900/40 border border-gold/10 rounded-xl overflow-hidden shadow-xl backdrop-blur-sm">
            {/* Toolbar: period filter + doc type tabs */}
            <div className="p-3 border-b border-white/10 bg-dark-800/50 space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] text-gray-500 uppercase tracking-wider">Find by period</span>
                    <select
                        value={filterYear}
                        onChange={(e) => setFilterYear(e.target.value)}
                        className="px-2 py-1.5 bg-dark-800 border border-white/10 rounded text-xs text-white focus:border-gold/30 outline-none"
                    >
                        <option value="all">All years</option>
                        {years.map(y => (
                            <option key={y} value={String(y)}>{y}</option>
                        ))}
                    </select>
                    <select
                        value={filterQuarter}
                        onChange={(e) => setFilterQuarter(e.target.value)}
                        className="px-2 py-1.5 bg-dark-800 border border-white/10 rounded text-xs text-white focus:border-gold/30 outline-none"
                    >
                        <option value="all">All quarters</option>
                        <option value="1">Q1</option>
                        <option value="2">Q2</option>
                        <option value="3">Q3</option>
                        <option value="4">Q4</option>
                    </select>
                </div>
                <div className="flex gap-1 border-b border-white/5 pb-0">
                    {tabs.map(({ id, label }) => (
                        <button
                            key={id}
                            type="button"
                            onClick={() => setActiveTab(id)}
                            className={`px-4 py-2 text-xs font-semibold uppercase tracking-wider transition-all border-b-2 -mb-px ${
                                activeTab === id
                                    ? 'text-gold border-gold'
                                    : 'text-gray-500 border-transparent hover:text-gray-300'
                            }`}
                        >
                            {label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="p-4 min-h-[200px]">
                {activeTab === 'filings' && (
                    <>
                        {filteredFilings.length === 0 ? (
                            <div className="text-center py-8 text-gray-500 text-xs">No filings for selected period</div>
                        ) : (
                            <SECFilingsExplorer filings={filteredFilings} ticker={ticker} onSelectFiling={onSelectFiling} />
                        )}
                    </>
                )}

                {activeTab === 'exhibits' && (
                    <>
                        {filteredExhibits.length === 0 ? (
                            <div className="text-center py-8 text-gray-500 text-xs">No exhibits for selected period</div>
                        ) : (
                            <div>
                                <p className="text-[10px] text-gray-500 mb-3">
                                    EX-10 = material contracts, EX-4 = debt instruments, EX-21 = subsidiaries, EX-99 = additional exhibits.
                                </p>
                                {secCompanyUrl && (
                                    <a href={secCompanyUrl} target="_blank" rel="noopener noreferrer" className="text-[10px] text-purple-400 hover:text-purple-300 mb-3 inline-block">
                                        View {ticker} on SEC EDGAR →
                                    </a>
                                )}
                                <div className="overflow-x-auto space-y-4">
                                    {groupedExhibits.map(({ type, exhibits: exList }) => (
                                        <div key={type}>
                                            <div className="text-[10px] text-purple-400/80 font-semibold uppercase tracking-wider mb-1.5">
                                                {type}
                                                {exhibitTypeHint[type] && <span className="text-gray-500 font-normal normal-case ml-1.5">— {exhibitTypeHint[type]}</span>}
                                            </div>
                                            <table className="w-full text-xs">
                                                <thead>
                                                    <tr className="border-b border-purple-500/10">
                                                        <th className="text-left text-gray-300 font-semibold pb-2 px-2">Type</th>
                                                        <th className="text-left text-gray-300 font-semibold pb-2 px-2">Category</th>
                                                        <th className="text-left text-gray-300 font-semibold pb-2 px-2">Description</th>
                                                        <th className="text-center text-gray-300 font-semibold pb-2 px-2">Date</th>
                                                        <th className="text-center text-gray-300 font-semibold pb-2 px-2">Sentiment</th>
                                                        {secCompanyUrl && <th className="text-center text-gray-300 font-semibold pb-2 px-2 w-20" />}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {exList.slice(0, 15).map((ex: any, i: number) => {
                                                        const sentiment = ex.finbert_score || 0
                                                        const sentimentColor = sentiment > 0.2 ? 'text-green-400' : sentiment < -0.2 ? 'text-red-400' : 'text-gray-400'
                                                        return (
                                                            <tr key={i} className="border-b border-white/5 hover:bg-purple-500/10 transition-colors">
                                                                <td className="py-2 px-2 text-purple-300 font-mono font-semibold">{ex.exhibit_type || '—'}</td>
                                                                <td className="py-2 px-2 text-gray-300 truncate max-w-[100px]" title={getCategoryHint(ex)}>{ex.contract_type || ex.exhibit_category || '—'}</td>
                                                                <td className="py-2 px-2 text-gray-300 truncate max-w-[300px]" title={ex.description || ''}>{ex.description || 'Exhibit'}</td>
                                                                <td className="py-2 px-2 text-center text-gray-400">{exhibitDate(ex)}</td>
                                                                <td className={`py-2 px-2 text-center font-mono ${sentimentColor}`}>
                                                                    {sentiment !== 0 ? (sentiment > 0 ? '+' : '') + sentiment.toFixed(3) : '—'}
                                                                </td>
                                                                {secCompanyUrl && (
                                                                    <td className="py-2 px-2 text-center">
                                                                        <a href={secCompanyUrl} target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:text-purple-300 text-[10px]">
                                                                            View on SEC
                                                                        </a>
                                                                    </td>
                                                                )}
                                                            </tr>
                                                        )
                                                    })}
                                                </tbody>
                                            </table>
                                            {exList.length > 15 && (
                                                <div className="text-[10px] text-gray-500 mt-1 px-2">+ {exList.length - 15} more in this group</div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </>
                )}

                {activeTab === 'xbrl' && (
                    <>
                        {filteredXbrl.length === 0 ? (
                            <div className="text-center py-8 text-gray-500 text-xs">No XBRL breakdowns for selected period</div>
                        ) : (
                            <div className="space-y-1">
                                {filteredXbrl.map((xbrl: any, i: number) => {
                                    const conceptCount = xbrl.concepts_found ?? [xbrl.debt, xbrl.costs, xbrl.revenue_segments].filter(Boolean).length
                                    const isExpanded = expandedXbrlIndex === i
                                    return (
                                        <div key={i} className="bg-dark-800/50 border border-cyan-500/10 rounded-lg overflow-hidden">
                                            <button
                                                type="button"
                                                onClick={() => setExpandedXbrlIndex(isExpanded ? null : i)}
                                                className="w-full flex items-center justify-between gap-2 px-4 py-2.5 text-left hover:bg-cyan-500/5 transition-colors"
                                            >
                                                <span className="text-xs font-semibold text-cyan-300">
                                                    {xbrl.filing_type} FY{xbrl.fiscal_year}
                                                    {xbrl.filing_date && <span className="text-gray-500 font-normal ml-1">(filed {xbrl.filing_date})</span>}
                                                </span>
                                                <span className="text-[10px] text-gray-500 font-mono flex-shrink-0">
                                                    {conceptCount} concepts
                                                </span>
                                                <span className={`text-cyan-400 text-[10px] transition-transform flex-shrink-0 ${isExpanded ? 'rotate-180' : ''}`}>▼</span>
                                            </button>
                                            {isExpanded && (
                                                <div className="px-4 pb-4 pt-0 border-t border-cyan-500/10 space-y-3">
                                                    {xbrl.has_segment_data && xbrl.revenue_segments && (
                                                        <div>
                                                            <div className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider mb-1">Revenue Segments</div>
                                                            <div className="grid grid-cols-2 gap-1">
                                                                {Object.entries(xbrl.revenue_segments).slice(0, 4).map(([key, value]: [string, any], j: number) => (
                                                                    <div key={j} className="flex justify-between px-2 py-1 bg-dark-900/50 rounded text-[10px]">
                                                                        <span className="text-gray-400 truncate">{key}</span>
                                                                        <span className="text-white font-mono">{showFullAmounts ? `$${Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(value / 1e6).toFixed(1)}M`}</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                    {xbrl.debt && Object.keys(xbrl.debt).length > 0 && (
                                                        <div>
                                                            <div className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider mb-1">Debt & Obligations</div>
                                                            <div className="grid grid-cols-2 gap-1">
                                                                {Object.entries(xbrl.debt).slice(0, 4).map(([key, value]: [string, any], j: number) => (
                                                                    <div key={j} className="flex justify-between px-2 py-1 bg-dark-900/50 rounded text-[10px]">
                                                                        <span className="text-gray-400">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                                                                        <span className="text-white font-mono">{showFullAmounts ? `$${Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(value / 1e6).toFixed(1)}M`}</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                    {xbrl.costs && Object.keys(xbrl.costs).length > 0 && (
                                                        <div>
                                                            <div className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider mb-1">Operating Costs</div>
                                                            <div className="grid grid-cols-2 gap-1">
                                                                {Object.entries(xbrl.costs).slice(0, 4).map(([key, value]: [string, any], j: number) => (
                                                                    <div key={j} className="flex justify-between px-2 py-1 bg-dark-900/50 rounded text-[10px]">
                                                                        <span className="text-gray-400">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                                                                        <span className="text-white font-mono">{showFullAmounts ? `$${Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(value / 1e6).toFixed(1)}M`}</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    )
                                })}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}
