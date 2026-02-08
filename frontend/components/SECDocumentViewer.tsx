'use client'

import { useState, useMemo } from 'react'

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

type UnifiedDoc = { kind: 'filing' | 'exhibit' | 'xbrl'; date: string; formType: string; docType: string; name: string; snippet: string; item: any }

export default function SECDocumentViewer({
    filings,
    exhibits,
    xbrlData,
    ticker,
    onSelectFiling,
    showFullAmounts = true
}: SECDocumentViewerProps) {
    const [filterYear, setFilterYear] = useState<string>('all')
    const [filterQuarter, setFilterQuarter] = useState<string>('all')
    const [filterFormType, setFilterFormType] = useState<string>('all')
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedDoc, setSelectedDoc] = useState<UnifiedDoc | null>(null)
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

    const unifiedList = useMemo((): UnifiedDoc[] => {
        const list: UnifiedDoc[] = []
        filteredFilings.forEach((f: any) => {
            const sent = (f.top_sentences || f.sec_sentences)?.[0]
            const snippet = (sent?.text || '').toString().slice(0, 60)
            list.push({
                kind: 'filing',
                date: f.filing_date || '',
                formType: f.type || f.form_type || 'Filing',
                docType: '',
                name: `${f.type || f.form_type || 'Filing'} ${f.filing_date || ''}`,
                snippet: snippet ? snippet + (snippet.length >= 60 ? '…' : '') : '—',
                item: f
            })
        })
        filteredExhibits.forEach((ex: any) => {
            const desc = (ex.description || ex.text || 'Exhibit').toString().slice(0, 60)
            list.push({
                kind: 'exhibit',
                date: ex.filing_date || ex.filed_date || '',
                formType: ex.filing_type || '',
                docType: ex.exhibit_type || 'Exhibit',
                name: ex.filename || ex.exhibit_type || 'Exhibit',
                snippet: desc + (desc.length >= 60 ? '…' : ''),
                item: ex
            })
        })
        filteredXbrl.forEach((x: any) => {
            const conceptCount = [x.debt, x.costs, x.revenue_segments].filter(Boolean).length
            list.push({
                kind: 'xbrl',
                date: x.filing_date || '',
                formType: x.filing_type || 'XBRL',
                docType: 'XBRL',
                name: `${x.filing_type} FY${x.fiscal_year ?? '—'}`,
                snippet: `${conceptCount} concepts`,
                item: x
            })
        })
        list.sort((a, b) => (b.date || '').localeCompare(a.date || ''))
        return list
    }, [filteredFilings, filteredExhibits, filteredXbrl])

    const formTypesForFilter = useMemo(() => {
        const set = new Set<string>()
        unifiedList.forEach(d => {
            if (d.formType) set.add(d.formType)
            if (d.docType) set.add(d.docType)
        })
        return ['all', ...Array.from(set).sort()]
    }, [unifiedList])

    const filteredUnifiedList = useMemo(() => {
        let out = unifiedList
        if (filterFormType !== 'all') {
            out = out.filter(d => d.formType === filterFormType || d.docType === filterFormType)
        }
        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase()
            out = out.filter(d =>
                (d.name || '').toLowerCase().includes(q) ||
                (d.snippet || '').toLowerCase().includes(q) ||
                (d.formType || '').toLowerCase().includes(q) ||
                (d.docType || '').toLowerCase().includes(q)
            )
        }
        return out
    }, [unifiedList, filterFormType, searchQuery])

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

    const handleSelectRow = (doc: UnifiedDoc) => {
        setSelectedDoc(doc)
        if (doc.kind === 'filing' && onSelectFiling) onSelectFiling(doc.item)
    }

    return (
        <div className="bg-dark-900/40 border border-gold/10 rounded-xl overflow-hidden shadow-xl backdrop-blur-sm flex flex-col max-h-[700px]">
            {/* Top: Filters */}
            <div className="p-3 border-b border-white/10 bg-dark-800/50 space-y-2 shrink-0">
                <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] text-gray-500 uppercase tracking-wider">Year</span>
                    <select
                        value={filterYear}
                        onChange={(e) => setFilterYear(e.target.value)}
                        className="px-2 py-1.5 bg-dark-800 border border-white/10 rounded text-xs text-white focus:border-gold/30 outline-none"
                    >
                        <option value="all">Any</option>
                        {years.map(y => (
                            <option key={y} value={String(y)}>{y}</option>
                        ))}
                    </select>
                    <span className="text-[10px] text-gray-500 uppercase tracking-wider ml-1">Quarter</span>
                    <select
                        value={filterQuarter}
                        onChange={(e) => setFilterQuarter(e.target.value)}
                        className="px-2 py-1.5 bg-dark-800 border border-white/10 rounded text-xs text-white focus:border-gold/30 outline-none"
                    >
                        <option value="all">Any</option>
                        <option value="1">Q1</option>
                        <option value="2">Q2</option>
                        <option value="3">Q3</option>
                        <option value="4">Q4</option>
                    </select>
                    <span className="text-[10px] text-gray-500 uppercase tracking-wider ml-1">Form / Doc type</span>
                    <select
                        value={filterFormType}
                        onChange={(e) => setFilterFormType(e.target.value)}
                        className="px-2 py-1.5 bg-dark-800 border border-white/10 rounded text-xs text-white focus:border-gold/30 outline-none"
                    >
                        <option value="all">Any</option>
                        {formTypesForFilter.filter(f => f !== 'all').map(ft => (
                            <option key={ft} value={ft}>{ft}</option>
                        ))}
                    </select>
                    <input
                        type="text"
                        placeholder="Search..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="ml-2 px-2 py-1.5 bg-dark-800 border border-white/10 rounded text-xs text-white placeholder-gray-500 focus:border-gold/30 outline-none w-32"
                    />
                </div>
            </div>

            {/* Middle: Unified document list */}
            <div className="flex-1 min-h-0 overflow-auto border-b border-white/10">
                <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-dark-800 z-10 border-b border-gold/20">
                        <tr>
                            <th className="text-left text-gray-300 font-semibold py-2 px-2 w-16">Type</th>
                            <th className="text-left text-gray-300 font-semibold py-2 px-2 w-20">Date</th>
                            <th className="text-left text-gray-300 font-semibold py-2 px-2 w-20">Form Type</th>
                            <th className="text-left text-gray-300 font-semibold py-2 px-2 w-24">Doc Type</th>
                            <th className="text-left text-gray-300 font-semibold py-2 px-2">Name / Description</th>
                            <th className="text-left text-gray-300 font-semibold py-2 px-2 max-w-[200px]">Text</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredUnifiedList.map((doc, i) => {
                            const isSelected = selectedDoc === doc
                            const typeLabel = doc.kind === 'filing' ? 'Filing' : doc.kind === 'exhibit' ? 'Exhibit' : 'XBRL'
                            return (
                                <tr
                                    key={`${doc.kind}-${i}-${doc.date}-${doc.name}`}
                                    onClick={() => handleSelectRow(doc)}
                                    className={`border-b border-white/5 cursor-pointer transition-colors ${isSelected ? 'bg-gold/20 border-l-2 border-l-gold' : 'hover:bg-white/5'}`}
                                >
                                    <td className="py-2 px-2 font-mono text-[10px] text-gray-400">{typeLabel}</td>
                                    <td className="py-2 px-2 text-gray-300">{doc.date || '—'}</td>
                                    <td className="py-2 px-2 text-gold/90 font-medium">{doc.formType || '—'}</td>
                                    <td className="py-2 px-2 text-gray-400">{doc.docType || '—'}</td>
                                    <td className="py-2 px-2 text-white truncate max-w-[180px]" title={doc.name}>{doc.name || '—'}</td>
                                    <td className="py-2 px-2 text-gray-500 truncate max-w-[200px]" title={doc.snippet}>{doc.snippet || '—'}</td>
                                </tr>
                            )
                        })}
                    </tbody>
                </table>
                {filteredUnifiedList.length === 0 && (
                    <div className="text-center py-6 text-gray-500 text-xs">No documents match filters</div>
                )}
            </div>

            {/* Bottom: Detail (left) + Content (right) */}
            <div className="flex flex-1 min-h-[200px] max-h-[280px] shrink-0">
                <div className="w-1/2 border-r border-white/10 overflow-y-auto p-4 bg-dark-800/30">
                    <h4 className="text-[10px] text-gold uppercase tracking-wider font-bold mb-2">Document details</h4>
                    {selectedDoc ? (() => {
                        const desc = selectedDoc.item.description != null ? selectedDoc.item.description : (selectedDoc.snippet || '—')
                        const name = selectedDoc.item.filename != null ? selectedDoc.item.filename : (selectedDoc.name || '—')
                        const periodLabel = selectedDoc.kind === 'xbrl' ? `FY${selectedDoc.item.fiscal_year != null ? selectedDoc.item.fiscal_year : '—'}` : (selectedDoc.date || '—')
                        const isExhibit = selectedDoc.kind === 'exhibit'
                        const exhibitSentiment = isExhibit && selectedDoc.item.finbert_score != null ? selectedDoc.item.finbert_score : null
                        const exhibitContractType = isExhibit ? (selectedDoc.item.contract_type || selectedDoc.item.exhibit_category || '—') : null
                        const exhibitSummaryRaw = isExhibit && selectedDoc.item.text ? selectedDoc.item.text : (isExhibit ? selectedDoc.item.description : null)
                        const exhibitSummary = exhibitSummaryRaw != null ? String(exhibitSummaryRaw).slice(0, 120) + (String(exhibitSummaryRaw).length > 120 ? '…' : '') : '—'
                        return (
                        <div className="space-y-1.5 text-xs">
                            <div><span className="text-gray-500">Form Type:</span> <span className="text-white">{selectedDoc.formType || '—'}</span></div>
                            <div><span className="text-gray-500">File Date:</span> <span className="text-white">{selectedDoc.date || '—'}</span></div>
                            {(selectedDoc.formType === '10-K' || selectedDoc.formType === '10-Q') && (
                                <div><span className="text-gray-500">Period:</span> <span className="text-white">{periodLabel}</span></div>
                            )}
                            <div><span className="text-gray-500">Doc Type:</span> <span className="text-white">{selectedDoc.docType || '—'}</span></div>
                            <div><span className="text-gray-500">Name:</span> <span className="text-white truncate block">{name}</span></div>
                            <div><span className="text-gray-500">Description:</span> <span className="text-gray-300 text-[10px] block line-clamp-2">{desc}</span></div>
                            {isExhibit && exhibitContractType !== null && (
                                <div><span className="text-gray-500">Contract type:</span> <span className="text-white">{exhibitContractType}</span></div>
                            )}
                            {isExhibit && exhibitSentiment !== null && (
                                <div><span className="text-gray-500">Sentiment:</span> <span className="text-white font-mono">{(exhibitSentiment as number).toFixed(3)}</span></div>
                            )}
                            {isExhibit && (
                                <div><span className="text-gray-500">Summary:</span> <span className="text-gray-300 text-[10px] block line-clamp-2">{exhibitSummary}</span></div>
                            )}
                            {selectedDoc.item.accession && (
                                <div><span className="text-gray-500">Accession:</span> <span className="text-gray-400 font-mono text-[10px]">{selectedDoc.item.accession}</span></div>
                            )}
                            {secCompanyUrl && (
                                <a href={secCompanyUrl} target="_blank" rel="noopener noreferrer" className="inline-block mt-2 text-gold hover:text-gold/80 text-[10px] font-semibold">
                                    Open on SEC EDGAR →
                                </a>
                            )}
                        </div>
                        )
                    })() : (
                        <p className="text-gray-500 text-xs">Select a row above to view details</p>
                    )}
                </div>
                <div className="w-1/2 overflow-y-auto p-4 bg-dark-900/50">
                    <h4 className="text-[10px] text-gold uppercase tracking-wider font-bold mb-2">Content</h4>
                            {selectedDoc ? (
                        <>
                            {selectedDoc.kind === 'xbrl' && (
                                <div className="space-y-2 text-xs">
                                    {selectedDoc.item.revenue_segments && Object.keys(selectedDoc.item.revenue_segments).length > 0 && (
                                        <div>
                                            <div className="text-[10px] text-cyan-400 font-bold uppercase mb-1">Revenue</div>
                                            <div className="grid grid-cols-2 gap-1">
                                                {Object.entries(selectedDoc.item.revenue_segments).slice(0, 3).map(([k, v]: [string, any]) => (
                                                    <div key={k} className="flex justify-between"><span className="text-gray-400 truncate">{k}</span><span className="text-white font-mono">{showFullAmounts ? `$${Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(v / 1e6).toFixed(1)}M`}</span></div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {selectedDoc.item.debt && Object.keys(selectedDoc.item.debt).length > 0 && (
                                        <div>
                                            <div className="text-[10px] text-cyan-400 font-bold uppercase mb-1">Debt</div>
                                            <div className="grid grid-cols-2 gap-1">
                                                {Object.entries(selectedDoc.item.debt).slice(0, 3).map(([k, v]: [string, any]) => (
                                                    <div key={k} className="flex justify-between"><span className="text-gray-400 truncate">{k.replace(/([A-Z])/g, ' $1').trim()}</span><span className="text-white font-mono">{showFullAmounts ? `$${Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(v / 1e6).toFixed(1)}M`}</span></div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {selectedDoc.kind === 'xbrl' && !(selectedDoc.item.revenue_segments && Object.keys(selectedDoc.item.revenue_segments).length > 0) && !(selectedDoc.item.debt && Object.keys(selectedDoc.item.debt).length > 0) && (
                                        <p className="text-gray-500">XBRL concepts available in database</p>
                                    )}
                                </div>
                            )}
                            {selectedDoc.kind === 'filing' && (
                                <p className="text-gray-500 text-xs">Full text is on SEC EDGAR. Use the link above to open the document.</p>
                            )}
                            {selectedDoc.kind === 'exhibit' && selectedDoc.item.text && (
                                <div className="space-y-2">
                                    <p className="text-[10px] text-gray-500 uppercase tracking-wider">In-house exhibit text</p>
                                    <pre className="whitespace-pre-wrap break-words text-xs text-gray-300 bg-dark-800/50 border border-white/10 rounded p-3 max-h-[220px] overflow-y-auto font-sans">
                                        {selectedDoc.item.text}
                                    </pre>
                                    {selectedDoc.item.truncated && (
                                        <p className="text-[10px] text-amber-400">Text truncated at 50,000 characters. View on SEC for full document.</p>
                                    )}
                                </div>
                            )}
                            {selectedDoc.kind === 'exhibit' && !selectedDoc.item.text && (
                                <p className="text-gray-500 text-xs">No exhibit text available. Use the link above to open on SEC EDGAR.</p>
                            )}
                            {secCompanyUrl && (
                                <p className="mt-4 pt-3 border-t border-white/10">
                                    <a href={secCompanyUrl} target="_blank" rel="noopener noreferrer" className="text-[10px] text-gray-500 hover:text-gold transition-colors">Open on SEC EDGAR</a>
                                </p>
                            )}
                        </>
                    ) : (
                        <p className="text-gray-500 text-xs">Select a row to see content or open on SEC</p>
                    )}
                </div>
            </div>
        </div>
    )
}
