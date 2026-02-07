'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface SECFilingsExplorerProps {
    filings: any[]
    ticker: string
    /** When provided, clicking a row calls this instead of opening the built-in detail modal (e.g. parent opens its own modal). */
    onSelectFiling?: (filing: any) => void
}

export default function SECFilingsExplorer({ filings, ticker, onSelectFiling }: SECFilingsExplorerProps) {
    const [selectedFormType, setSelectedFormType] = useState<string>('all')
    const [sortBy, setSortBy] = useState<'negative' | 'positive' | 'recent'>('recent')
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedFiling, setSelectedFiling] = useState<any | null>(null)

    // Get unique form types
    const formTypes = useMemo(() => {
        const types = new Set<string>(filings.map((f: any) => f.type || f.form_type).filter(Boolean))
        return ['all', ...Array.from(types).sort()] as string[]
    }, [filings])

    // Filter and sort filings
    const filteredFilings = useMemo(() => {
        let filtered = selectedFormType === 'all'
            ? filings
            : filings.filter((f: any) => (f.type || f.form_type) === selectedFormType)

        // Text search filter
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase()
            filtered = filtered.filter((f: any) => {
                // Search in filing type
                if ((f.type || f.form_type || '').toLowerCase().includes(query)) return true

                // Search in top sentences if available
                if (f.top_sentences && Array.isArray(f.top_sentences)) {
                    return f.top_sentences.some((s: any) =>
                        s.text && s.text.toLowerCase().includes(query)
                    )
                }

                return false
            })
        }

        // Sort
        const sorted = [...filtered].sort((a: any, b: any) => {
            if (sortBy === 'negative') {
                return (a.avg_finbert || 0) - (b.avg_finbert || 0)
            } else if (sortBy === 'positive') {
                return (b.avg_finbert || 0) - (a.avg_finbert || 0)
            } else {
                return new Date(b.filing_date || 0).getTime() - new Date(a.filing_date || 0).getTime()
            }
        })

        return sorted
    }, [filings, selectedFormType, sortBy, searchQuery])

    if (!filings || filings.length === 0) {
        return (
            <div className="bg-dark-900/40 border border-green-500/10 rounded-xl p-6 text-center">
                <div className="text-gray-500 text-sm">No SEC filings available for {ticker}</div>
            </div>
        )
    }

    return (
        <>
            <div className="bg-dark-900/40 border border-green-500/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
                {/* Header */}
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-bold text-green-400 uppercase tracking-[0.2em] flex items-center gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
                        SEC Filings Explorer
                    </h3>
                    <div className="text-xs text-gray-500">
                        {filteredFilings.length} filing{filteredFilings.length !== 1 ? 's' : ''}
                    </div>
                </div>

                {/* Search and Filters */}
                <div className="space-y-3 mb-4">
                    {/* Search Bar */}
                    <div className="relative">
                        <input
                            type="text"
                            placeholder="Search filing content or type..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full px-3 py-2 bg-dark-800 border border-green-500/20 rounded-lg text-xs text-white placeholder-gray-500 focus:border-green-500/50 outline-none pr-8"
                        />
                        {searchQuery && (
                            <button
                                onClick={() => setSearchQuery('')}
                                className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white"
                            >
                                ×
                            </button>
                        )}
                    </div>

                    {/* Form Type Filter */}
                    <div className="flex items-center gap-2 overflow-x-auto pb-2">
                        {formTypes.map((type) => (
                            <button
                                key={type}
                                onClick={() => setSelectedFormType(type)}
                                className={`px-3 py-1.5 rounded-full text-xs border whitespace-nowrap transition-all ${selectedFormType === type
                                    ? 'bg-green-500/20 border-green-500 text-green-300 font-semibold'
                                    : 'bg-dark-800 border-gray-700 text-gray-400 hover:border-green-500/50'
                                    }`}
                            >
                                {type === 'all' ? 'All Forms' : type}
                            </button>
                        ))}
                    </div>

                    {/* Sort Controls */}
                    <div className="flex items-center justify-between gap-2">
                        <div className="text-xs text-gray-500">Sort by:</div>
                        <div className="flex gap-2">
                            <button
                                onClick={() => setSortBy('recent')}
                                className={`px-3 py-1 rounded text-xs transition-all ${sortBy === 'recent'
                                    ? 'bg-green-500/20 text-green-300 font-semibold'
                                    : 'text-gray-400 hover:text-green-300'
                                    }`}
                            >
                                Recent
                            </button>
                            <button
                                onClick={() => setSortBy('positive')}
                                className={`px-3 py-1 rounded text-xs transition-all ${sortBy === 'positive'
                                    ? 'bg-green-500/20 text-green-300 font-semibold'
                                    : 'text-gray-400 hover:text-green-300'
                                    }`}
                            >
                                Most Positive
                            </button>
                            <button
                                onClick={() => setSortBy('negative')}
                                className={`px-3 py-1 rounded text-xs transition-all ${sortBy === 'negative'
                                    ? 'bg-green-500/20 text-green-300 font-semibold'
                                    : 'text-gray-400 hover:text-green-300'
                                    }`}
                            >
                                Most Negative
                            </button>
                        </div>
                    </div>
                </div>

                {/* Filings List */}
                <div className="space-y-2 max-h-96 overflow-y-auto">
                    {filteredFilings.length === 0 ? (
                        <div className="text-center py-8 text-gray-500 text-xs">
                            No filings match your search criteria
                        </div>
                    ) : (
                        filteredFilings.map((filing, idx) => {
                            const sentiment = filing.avg_finbert || 0
                            const sentimentColor = sentiment > 0.2 ? 'text-green-400' : sentiment < -0.2 ? 'text-red-400' : 'text-gray-400'
                            const sentimentBg = sentiment > 0.2 ? 'bg-green-500/10' : sentiment < -0.2 ? 'bg-red-500/10' : 'bg-gray-500/10'
                            const formType = filing.type || filing.form_type || '—'
                            const filedDate = filing.filing_date || 'Unknown date'

                            return (
                                <button
                                    key={idx}
                                    onClick={() => {
                                        if (onSelectFiling) {
                                            onSelectFiling(filing)
                                        } else {
                                            setSelectedFiling(filing)
                                        }
                                    }}
                                    className="w-full bg-dark-800/50 border border-green-500/10 rounded-lg p-3 text-left hover:border-green-500/30 hover:bg-dark-800 transition-all group"
                                >
                                    <div className="flex items-start justify-between gap-2">
                                        <div className="flex items-center gap-2 flex-wrap min-w-0">
                                            <span className="text-xs font-semibold text-green-300 shrink-0">
                                                {formType}
                                            </span>
                                            <span className="text-[10px] text-gray-500 shrink-0">·</span>
                                            <span className="text-[10px] text-gray-400 shrink-0">{filedDate}</span>
                                            <span className={`text-[10px] px-2 py-0.5 rounded shrink-0 ${sentimentBg} ${sentimentColor} font-mono`} title="Sentiment score">
                                                {sentiment > 0 ? '+' : ''}{sentiment.toFixed(2)}
                                            </span>
                                        </div>
                                    </div>

                                    {filing.top_sentences && filing.top_sentences.length > 0 && (
                                        <div className="mt-2 pt-2 border-t border-white/5">
                                            <div className="text-[10px] text-gray-400 italic line-clamp-2 group-hover:text-gray-300 transition-colors">
                                                "{filing.top_sentences[0].text?.substring(0, 150)}..."
                                            </div>
                                        </div>
                                    )}
                                </button>
                            )
                        })
                    )}
                </div>
            </div>

            {/* Filing Detail Modal (only when parent does not handle selection) */}
            <AnimatePresence>
                {selectedFiling && !onSelectFiling && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/95 backdrop-blur-xl z-[150] flex items-center justify-center p-4 md:p-8"
                        onClick={() => setSelectedFiling(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="bg-dark-800 border border-green-500/30 rounded-3xl w-full max-w-3xl max-h-[90vh] overflow-hidden flex flex-col shadow-[0_0_50px_rgba(34,197,94,0.1)]"
                            onClick={e => e.stopPropagation()}
                        >
                            {/* Header */}
                            <div className="p-6 border-b border-green-500/20 flex justify-between items-center bg-dark-900/80">
                                <div>
                                    <h4 className="text-green-400 font-bold uppercase tracking-[0.3em] text-[10px] mb-1">
                                        SEC Filing Analysis
                                    </h4>
                                    <div className="text-sm text-white font-bold">
                                        Form {selectedFiling.type || selectedFiling.form_type} - {ticker}
                                    </div>
                                </div>
                                <button
                                    onClick={() => setSelectedFiling(null)}
                                    className="p-2 bg-dark-700 rounded-full text-gray-400 hover:text-white border border-white/10 transition-all"
                                >
                                    ×
                                </button>
                            </div>

                            {/* Content */}
                            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                                {/* Filing Info */}
                                <div className="flex justify-between items-start">
                                    <div>
                                        <div className="text-2xl font-black text-white tracking-tighter">
                                            Form {selectedFiling.type || selectedFiling.form_type}
                                        </div>
                                        <div className="text-xs text-gray-400 mt-1 uppercase font-bold tracking-widest">
                                            Filed: {selectedFiling.filing_date || 'Unknown'}
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-1">
                                            Sentiment Score
                                        </div>
                                        <div
                                            className={`text-2xl font-mono font-black ${(selectedFiling.avg_finbert || 0) > 0
                                                ? 'text-green-400 shadow-[0_0_20px_rgba(74,222,128,0.2)]'
                                                : 'text-red-400 shadow-[0_0_20px_rgba(248,113,113,0.2)]'
                                                }`}
                                        >
                                            {(selectedFiling.avg_finbert || 0) > 0 ? '+' : ''}
                                            {(selectedFiling.avg_finbert || 0).toFixed(4)}
                                        </div>
                                    </div>
                                </div>

                                {/* Key Sentences */}
                                <div className="space-y-4">
                                    <h5 className="text-[10px] font-black text-green-400 uppercase tracking-[0.2em] border-b border-green-500/20 pb-2">
                                        Key Excerpts (Sorted by Sentiment Magnitude)
                                    </h5>

                                    {selectedFiling.top_sentences && selectedFiling.top_sentences.length > 0 ? (
                                        selectedFiling.top_sentences.map((s: any, j: number) => {
                                            const score = s.score || 0
                                            const scoreColor = score > 0.2 ? 'text-green-400' : score < -0.2 ? 'text-red-400' : 'text-gray-400'
                                            const barColor = score > 0.2 ? 'bg-green-500' : score < -0.2 ? 'bg-red-500' : 'bg-gray-500'

                                            return (
                                                <div
                                                    key={j}
                                                    className="bg-dark-900/50 p-5 rounded-2xl border border-white/5 relative group hover:border-green-500/20 transition-all"
                                                >
                                                    <div className={`absolute top-0 left-0 w-1 h-0 ${barColor} group-hover:h-full transition-all duration-300`} />
                                                    <p className="text-xs text-gray-300 leading-relaxed">"{s.text}"</p>
                                                    <div className="mt-3 flex items-center justify-between">
                                                        <div className="text-[10px] text-green-500 font-bold uppercase tracking-wider">
                                                            Sentiment Signal
                                                        </div>
                                                        <div className={`text-xs font-mono font-bold ${scoreColor}`}>
                                                            {score > 0 ? '+' : ''}{score.toFixed(3)}
                                                        </div>
                                                    </div>
                                                </div>
                                            )
                                        })
                                    ) : (
                                        <div className="text-xs text-gray-500 italic text-center py-8 border border-dashed border-gray-700 rounded-lg">
                                            No sentence-level analysis available for this filing.
                                            <div className="text-[10px] text-gray-600 mt-2">
                                                This may be a Form 4/5 insider trading report or a filing without extracted text.
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    )
}
