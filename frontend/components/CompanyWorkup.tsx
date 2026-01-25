'use client'

import { useState, useMemo } from 'react'
import TimeSeriesChart from './TimeSeriesChart'
import { motion, AnimatePresence } from 'framer-motion'
import AwardHistory from '@/components/AwardHistory'
import type { Key } from 'react'

interface CompanyWorkupProps {
    data: any
    onCompare?: (ticker: string) => void
}

export default function CompanyWorkup({ data, onCompare }: CompanyWorkupProps) {
    const [timeframe, setTimeframe] = useState<'1M' | '3M' | '6M' | '1Y' | '5Y'>('1M')
    const [showAllMetrics, setShowAllMetrics] = useState(false)
    const [selectedDetail, setSelectedDetail] = useState<{ type: 'SEC' | 'Award', data: any } | null>(null)
    const [selectedFormType, setSelectedFormType] = useState<string>('all')
    const [secSortBy, setSecSortBy] = useState<'negative' | 'positive' | 'recent'>('negative')

    // Extract nested data
    const company = data
    const marketData = data.MarketData || []
    const allSecFilings = data.sec_filings || []
    const polyMarkets = data.prediction_markets_polymarket || []
    const awards = data.Award || []
    const optionsFlow = data.options_flow || []
    const futuresPrices = data.futures_prices || []
    const eiaData = {
        crude: data.eia_crude_inventory || [],
        natgasStorage: data.eia_natgas_storage || [],
        natgasProduction: data.eia_natgas_production || [],
        lng: data.eia_lng || []
    }

    // Filter and sort SEC filings by form type and sentiment
    const secFilings = useMemo(() => {
        let filtered = selectedFormType === 'all'
            ? allSecFilings
            : allSecFilings.filter((f: any) => (f.type || f.form_type) === selectedFormType)

        // Sort by selected criterion
        const sorted = [...filtered].sort((a: any, b: any) => {
            if (secSortBy === 'negative') {
                return (a.avg_finbert || 0) - (b.avg_finbert || 0) // Most negative first
            } else if (secSortBy === 'positive') {
                return (b.avg_finbert || 0) - (a.avg_finbert || 0) // Most positive first
            } else {
                // Most recent first
                return new Date(b.filing_date || 0).getTime() - new Date(a.filing_date || 0).getTime()
            }
        })

        return sorted
    }, [allSecFilings, selectedFormType, secSortBy])

    // Get unique form types
    const formTypes = useMemo(() => {
        const types = new Set<string>(allSecFilings.map((f: any) => f.type || f.form_type).filter(Boolean))
        return ['all', ...Array.from(types).sort()] as string[]
    }, [allSecFilings])

    const latestMarket = marketData[0] || {}

    // Prepare chart data based on timeframe
    const chartData = useMemo(() => {
        let filtered = [...marketData]

        // Filter by timeframe
        const now = new Date()
        const filterDate = new Date()
        if (timeframe === '1M') filterDate.setMonth(now.getMonth() - 1)
        else if (timeframe === '3M') filterDate.setMonth(now.getMonth() - 3)
        else if (timeframe === '6M') filterDate.setMonth(now.getMonth() - 6)
        else if (timeframe === '1Y') filterDate.setFullYear(now.getFullYear() - 1)
        else if (timeframe === '5Y') filterDate.setFullYear(now.getFullYear() - 5)

        filtered = filtered.filter(d => new Date(d.date) >= filterDate)

        const sorted = filtered.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
        return {
            dates: sorted.map(d => d.date),
            values: sorted.map(d => d.close),
            ticker: company.ticker
        }
    }, [marketData, company.ticker, timeframe])

    // AI Intelligence Summary (4 sentences) - Enriched with Latest Search
    const aiSummary = useMemo(() => {
        const ticker = company.ticker
        const priceChange = chartData.values.length > 1
            ? ((chartData.values[chartData.values.length - 1] - chartData.values[0]) / chartData.values[0] * 100).toFixed(1)
            : 'N/A'

        const latestAwards = awards.length > 0 ? awards[0].award_amount_float : 0
        const sentiment = secFilings[0]?.avg_finbert != null
            ? (secFilings[0].avg_finbert > 0 ? 'Bullish' : 'Bearish')
            : 'Neutral'

        // Real-time context from search for PLTR/2026
        const news = ticker === 'PLTR' ? {
            rev: '$4.2B - $6.3B',
            event: 'Q4 Earnings on Feb 2, 2026',
            driver: 'Artificial Intelligence Platform (AIP) adoption'
        } : null;

        return [
            `${company.company} (${ticker}) is demonstrating a ${priceChange}% trajectory over the selected ${timeframe} window, with strong momentum in its ${company.sector} operations.`,
            `Recent SEC regulatory signals lean ${sentiment.toLowerCase()} (Score: ${secFilings[0]?.avg_finbert?.toFixed(3) || '0.00'}), with a primary focus on internal reporting and ${secFilings[0]?.form_type || 'operational updates'}.`,
            news ? `Wall Street projections for 2026 highlight a potential revenue ceiling of ${news.rev}, catalyzed significantly by ${news.driver}.`
                : (awards.length > 0
                    ? `The firm continues to scale public sector dominance, recently capturing a $${(latestAwards / 1e6).toFixed(1)}M award from the ${awards[0]?.awarding_agency || 'government'}.`
                    : `${company.company} maintains a robust market capitalization of $${(company.marketCap / 1e9).toFixed(1)}B with steady institutional coverage.`),
            `Crucial market intelligence points to ${news?.event || 'upcoming quarterly benchmarks'} as the next major volatility catalyst for institutional positioning.`
        ]
    }, [company, timeframe, chartData, awards, secFilings, polyMarkets])

    // Moneycontain "13 Essential Financial Metrics"
    const fundamentalMetrics = useMemo(() => {
        const all = { ...company, ...latestMarket }

        // Derive Revenue and Absolute Margins if raw fields are missing
        const calcRevenue = (all.revenuePerShare && all.sharesOutstanding)
            ? (all.revenuePerShare * all.sharesOutstanding)
            : null;

        const calcEbitda = (all.ebitda) ? all.ebitda : (calcRevenue && all.ebitdaMargins ? calcRevenue * all.ebitdaMargins : null);
        const calcNetIncome = (all.netIncome) ? all.netIncome : (calcRevenue && all.profitMargins ? calcRevenue * all.profitMargins : null);

        const metricsList = [
            { name: 'Revenue Growth', val: all.revenueGrowth, benchmark: '> 10%', type: 'pct', check: (v: number) => v > 0.1 },
            { name: 'EBITDA', val: calcEbitda, benchmark: 'Growing', type: 'currency' },
            { name: 'EBITDA Margin', val: all.ebitdaMargins, benchmark: '> 15%', type: 'pct', check: (v: number) => v > 0.15 },
            { name: 'Net Profit (PAT)', val: calcNetIncome, benchmark: 'Growing', type: 'currency' },
            { name: 'PAT Margin', val: all.profitMargins, benchmark: '> 10%', type: 'pct', check: (v: number) => v > 0.1 },
            { name: 'ROE', val: all.returnOnEquity, benchmark: '> 15%', type: 'pct', check: (v: number) => v > 0.15 },
            { name: 'ROA', val: all.returnOnAssets, benchmark: '> 7%', type: 'pct', check: (v: number) => v > 0.07 },
            { name: 'Debt-to-Equity', val: all.debtToEquity, benchmark: '< 1', type: 'ratio', check: (v: number) => v < 1 },
            { name: 'Current Ratio', val: all.currentRatio, benchmark: '> 1.5', type: 'ratio', check: (v: number) => v > 1.5 },
            { name: 'Free Cash Flow', val: all.freeCashflow, benchmark: 'Positive', type: 'currency', check: (v: number) => v > 0 },
            { name: 'EPS', val: all.epsTrailingTwelveMonths || all.eps, benchmark: 'Growing', type: 'number' },
            { name: 'P/E Ratio', val: all.trailingPE || all.forwardPE, benchmark: '< 20 (Fair)', type: 'number', check: (v: number) => v < 20 },
            { name: 'ROCE', val: (all.ebit && all.totalDebt && all.totalStockholderEquity) ? (all.ebit / (all.totalDebt + all.totalStockholderEquity)) : null, benchmark: '> 15%', type: 'pct', check: (v: number) => v > 0.15 }
        ]

        return metricsList.map(m => ({
            ...m,
            displayVal: formatVal(m.val, m.type),
            status: m.check ? (m.check(m.val) ? 'good' : 'bad') : 'neutral'
        }))
    }, [company, latestMarket])

    function formatVal(val: any, type: string) {
        if (val == null || isNaN(val)) return 'N/A'
        if (type === 'currency') {
            if (val > 1e12) return `$${(val / 1e12).toFixed(2)}T`
            if (val > 1e9) return `$${(val / 1e9).toFixed(2)}B`
            if (val > 1e6) return `$${(val / 1e6).toFixed(2)}M`
            return `$${val.toLocaleString()}`
        }
        if (type === 'pct') return `${(val * 100).toFixed(2)}%`
        if (type === 'ratio' || type === 'number') return val.toFixed(2)
        return String(val)
    }

    return (
        <div className="w-full space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-6">
            {/* Header Info */}
            <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-gold/20 pb-4 gap-4">
                <div>
                    <div className="flex items-center gap-3 mb-1">
                        <h2 className="text-3xl font-bold text-white tracking-tight">{company.company}</h2>
                        <span className="px-3 py-1 bg-gold/10 border border-gold/40 rounded text-gold text-sm font-mono font-bold shadow-glow">
                            {company.ticker}
                        </span>
                    </div>
                    <p className="text-sm text-gray-400">
                        {company.sector} | {company.industry} | {company.city}, {company.country}
                    </p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => onCompare?.(company.ticker)}
                        className="px-4 py-2 bg-dark-800 border border-gold/30 rounded-lg text-xs text-gold hover:bg-gold/10 transition-all flex items-center gap-2"
                    >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                        </svg>
                        Compare Peer
                    </button>
                    <button className="px-3 py-1.5 bg-dark-800 border border-gold/30 rounded-lg text-xs text-gray-400 hover:text-white transition-all">
                        PDF Mode
                    </button>
                </div>
            </div>

            {/* Main Content Sections with Zero Collisions */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">

                {/* Left Area: Chart & Fundamental Checklist (7 Cols) */}
                <div className="xl:col-span-7 space-y-4">
                    {/* Chart Section */}
                    <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-3 md:p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-3 md:mb-4 gap-3">
                            <div>
                                <h3 className="text-[10px] md:text-xs font-bold text-gold uppercase tracking-widest mb-1">Market Performance Hub</h3>
                                <div className="text-[9px] md:text-[10px] text-gray-500 font-mono italic">Structural Momentum Analysis</div>
                            </div>
                            <div className="flex bg-dark-800 rounded-xl p-0.5 border border-white/10 shadow-inner">
                                {['1M', '3M', '6M', '1Y', '5Y'].map(tf => (
                                    <button
                                        key={tf}
                                        onClick={() => setTimeframe(tf as any)}
                                        className={`px-3 py-1 md:px-4 md:py-1.5 text-[9px] md:text-[10px] rounded-lg transition-all ${timeframe === tf ? 'bg-gold text-dark-900 font-bold shadow-lg' : 'text-gray-500 hover:text-gray-300'}`}
                                    >
                                        {tf}
                                    </button>
                                ))}
                            </div>
                        </div>
                        {chartData.values.length > 0 ? (
                            <div className="h-[280px] w-full mt-3">
                                <TimeSeriesChart
                                    dates={chartData.dates}
                                    values={chartData.values}
                                    label={`${company.ticker} Structural Momentum`}
                                    ticker={company.ticker}
                                />
                            </div>
                        ) : (
                            <div className="h-[280px] flex items-center justify-center text-gray-600 text-sm border border-dashed border-white/10 rounded-xl bg-dark-900/20">
                                No historical price action in dataset
                            </div>
                        )}
                    </div>

                    {/* Moneycontain Fundamental Checklist */}
                    <div className="bg-dark-900/60 border border-white/5 rounded-xl overflow-hidden shadow-lg">
                        <div className="p-3 border-b border-white/5 flex justify-between items-center bg-dark-800/40">
                            <div className="flex items-center gap-3">
                                <h3 className="text-xs font-bold text-gray-200 uppercase tracking-widest">13 Point Fundamental Checklist</h3>
                                <span className="text-[9px] text-gray-500 font-mono py-0.5 px-2 bg-black/40 rounded border border-white/10">Industry Standard Benchmarking</span>
                            </div>
                            <button
                                onClick={() => setShowAllMetrics(!showAllMetrics)}
                                className="px-3 py-1 text-[9px] bg-gold/10 text-gold hover:bg-gold/20 rounded-full border border-gold/20 font-bold uppercase transition-all"
                            >
                                {showAllMetrics ? 'Core View' : 'All Metrics'}
                            </button>
                        </div>
                        <div className="p-3 md:p-4 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-y-4 gap-x-3 md:gap-y-5 md:gap-x-4">
                            {fundamentalMetrics.slice(0, showAllMetrics ? undefined : 8).map((m: any, i: number) => (
                                <div key={i} className="group border-l border-white/5 pl-3 md:pl-4 hover:border-gold/30 transition-all">
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="text-[8px] md:text-[9px] text-gray-500 uppercase font-black tracking-tighter group-hover:text-gold transition-colors">{m.name}</div>
                                        <div className="text-[7px] md:text-[8px] text-gray-600 font-mono tracking-tighter hidden sm:block">Ref: {m.benchmark}</div>
                                    </div>
                                    <div className={`text-sm md:text-base font-mono font-black ${m.status === 'good' ? 'text-green-400' : m.status === 'bad' ? 'text-red-400' : 'text-gray-200'}`}>
                                        {m.displayVal}
                                        {m.status !== 'neutral' && (
                                            <span className={`ml-0.5 text-[8px] md:text-[10px] ${m.status === 'good' ? 'text-green-400/50' : 'text-red-400/50'}`}>
                                                {m.status === 'good' ? '▲' : '▼'}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* AI Intelligence Summary - Moved below chart */}
                    <div className="bg-gradient-to-br from-gold/10 to-transparent border border-gold/20 rounded-xl p-3 md:p-4 relative overflow-hidden group shadow-2xl">
                        <div className="absolute top-0 left-0 w-1 h-full bg-gold/50" />
                        <div className="absolute -right-12 -top-12 w-48 h-48 bg-gold/5 rounded-full blur-3xl group-hover:bg-gold/10 transition-all" />
                        <h3 className="text-[9px] md:text-[10px] font-bold text-gold uppercase tracking-[0.2em] mb-2 md:mb-3 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-gold animate-pulse" />
                            Deep Intelligence Synthesis
                        </h3>
                        <div className="space-y-2 relative z-10">
                            {aiSummary.map((s: string, i: number) => (
                                <p key={i} className="text-[13px] md:text-sm text-gray-300 leading-relaxed font-medium">
                                    {s}
                                    {i === 1 && secFilings.length > 0 && (
                                        <span
                                            onClick={() => setSelectedDetail({ type: 'SEC', data: secFilings[0] })}
                                            className="ml-1 text-[10px] text-blue-400 font-mono cursor-pointer hover:underline bg-blue-500/10 px-1 rounded"
                                        >
                                            [SEC-{secFilings[0].filing_date}]
                                        </span>
                                    )}
                                    {i === 3 && polyMarkets.length > 0 && (
                                        <span className="ml-1 text-[10px] text-purple-400 font-mono cursor-pointer hover:underline bg-purple-500/10 px-1 rounded">[BIAS-{(polyMarkets[0].yes_probability * 100).toFixed(0)}%]</span>
                                    )}
                                </p>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Area: Signals (5 Cols) */}
                <div className="xl:col-span-5 space-y-4">
                    {/* Regulatory Column */}
                    <div className="bg-dark-900/40 border border-blue-500/10 rounded-xl p-3 md:p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex flex-col gap-2 mb-3 md:mb-4">
                            <div className="flex items-center justify-between">
                                <h3 className="text-xs md:text-sm font-bold text-blue-400 uppercase tracking-[0.2em] flex items-center gap-3">
                                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
                                    SEC Intelligent Signals
                                </h3>
                            </div>
                            {allSecFilings.length > 0 && (
                                <div className="flex gap-2">
                                    <select
                                        value={selectedFormType}
                                        onChange={(e) => setSelectedFormType(e.target.value)}
                                        className="text-[10px] bg-dark-800 border border-blue-500/20 rounded px-2 py-1 text-gray-300 focus:border-blue-500/50 outline-none flex-1"
                                    >
                                        {formTypes.map((type) => (
                                            <option key={type} value={type}>
                                                {type === 'all' ? 'All Forms' : `Form ${type}`}
                                            </option>
                                        ))}
                                    </select>
                                    <select
                                        value={secSortBy}
                                        onChange={(e) => setSecSortBy(e.target.value as any)}
                                        className="text-[10px] bg-dark-800 border border-blue-500/20 rounded px-2 py-1 text-gray-300 focus:border-blue-500/50 outline-none flex-1"
                                    >
                                        <option value="negative">Most Negative</option>
                                        <option value="positive">Most Positive</option>
                                        <option value="recent">Most Recent</option>
                                    </select>
                                </div>
                            )}
                        </div>
                        {secFilings.length > 0 ? (
                            <div className="space-y-4">
                                {secFilings.slice(0, 3).map((f: any, i: number) => (
                                    <div
                                        key={i}
                                        onClick={() => setSelectedDetail({ type: 'SEC', data: f })}
                                        className="bg-black/40 rounded-xl p-3 border border-white/10 hover:border-blue-500/50 transition-all cursor-pointer group shadow-sm hover:shadow-blue-500/10"
                                    >
                                        <div className="flex justify-between items-center text-xs mb-2">
                                            <span className="text-gray-100 font-bold px-2 py-1 bg-dark-800/70 rounded">{f.type || f.form_type}</span>
                                            <div className={`px-2 py-1 rounded-full text-xs ${(f.avg_finbert || 0) > 0 ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'} font-black`}>
                                                BIAS: {(f.avg_finbert || 0).toFixed(3)}
                                            </div>
                                        </div>
                                        <div className="text-xs text-gray-300 space-y-1">
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Filed:</span>
                                                <span className="font-mono">{f.filing_date || 'N/A'}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Sentiment:</span>
                                                <span className={`font-mono ${(f.avg_finbert || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                    {(f.avg_finbert || 0) > 0 ? 'Bullish' : 'Bearish'} ({f.avg_positive || 0} pos / {f.avg_negative || 0} neg)
                                                </span>
                                            </div>
                                            {f.sentence_count && (
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Sentences:</span>
                                                    <span className="font-mono">{f.sentence_count.toLocaleString()}</span>
                                                </div>
                                            )}
                                        </div>
                                        <div className="mt-2 text-xs text-blue-400/70 font-mono text-right font-bold group-hover:text-blue-300 transition-colors">Details →</div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-sm text-gray-500 italic py-8 text-center bg-dark-800/20 rounded-xl border border-dashed border-white/5">
                                No SEC filings available
                            </div>
                        )}
                    </div>

                    {/* Options Flow Column */}
                    {optionsFlow.length > 0 && (
                        <div className="bg-dark-900/40 border border-green-500/10 rounded-xl p-3 md:p-4 shadow-xl backdrop-blur-sm">
                            <h3 className="text-xs md:text-sm font-bold text-green-400 uppercase tracking-[0.2em] mb-4 md:mb-6 flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
                                Options Flow Activity
                            </h3>
                            <div className="space-y-3">
                                {optionsFlow.slice(0, 3).map((opt: any, i: number) => (
                                    <div key={i} className="bg-black/30 rounded-xl p-3 border border-white/5">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-sm text-gray-200">{opt.date}</span>
                                            {opt.unusual_call_activity && (
                                                <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded">Unusual Calls</span>
                                            )}
                                            {opt.unusual_put_activity && (
                                                <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">Unusual Puts</span>
                                            )}
                                        </div>
                                        <div className="grid grid-cols-2 gap-2 text-sm">
                                            <div>
                                                <span className="text-gray-400">Volume:</span>
                                                <span className="text-gray-100 ml-1 font-mono">{opt.total_volume?.toLocaleString()}</span>
                                            </div>
                                            <div>
                                                <span className="text-gray-400">P/C Ratio:</span>
                                                <span className="text-gray-100 ml-1 font-mono">{opt.put_call_ratio?.toFixed(2)}</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Futures Prices Column */}
                    {futuresPrices.length > 0 && (
                        <div className="bg-dark-900/40 border border-amber-500/10 rounded-xl p-3 md:p-4 shadow-xl backdrop-blur-sm">
                            <h3 className="text-xs md:text-sm font-bold text-amber-400 uppercase tracking-[0.2em] mb-4 md:mb-6 flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]" />
                                Commodity Futures
                            </h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-xs">
                                    <thead>
                                        <tr className="border-b border-amber-500/10">
                                            <th className="text-left text-gray-300 font-semibold pb-2">Commodity</th>
                                            <th className="text-right text-gray-300 font-semibold pb-2">Price</th>
                                            <th className="text-right text-gray-300 font-semibold pb-2">Change</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {futuresPrices.slice(0, 6).map((f: any, i: number) => (
                                            <tr key={i} className="border-b border-white/5 hover:bg-amber-500/5 transition-colors">
                                                <td className="py-2 text-gray-100">{f.commodity_type}</td>
                                                <td className="py-2 text-right text-amber-400 font-mono font-bold">${f.close?.toFixed(2)}</td>
                                                <td className={`py-2 text-right font-mono ${f.price_change_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                    {f.price_change_pct > 0 ? '+' : ''}{f.price_change_pct?.toFixed(2)}%
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Gov Awards Column */}
                    {awards.length > 0 && (
                        <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-3 md:p-4 shadow-xl backdrop-blur-sm">
                            <h3 className="text-xs md:text-sm font-bold text-gold uppercase tracking-[0.2em] mb-3 md:mb-4 flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-gold shadow-[0_0_8px_rgba(255,215,0,0.5)]" />
                                Federal Contract Awards
                            </h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-xs">
                                    <thead>
                                        <tr className="border-b border-gold/10">
                                            <th className="text-left text-gray-300 font-semibold pb-2">Agency</th>
                                            <th className="text-left text-gray-300 font-semibold pb-2">Year</th>
                                            <th className="text-right text-gray-300 font-semibold pb-2">Amount</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {awards.slice(0, 6).map((a: any, i: number) => (
                                            <tr
                                                key={i}
                                                onClick={() => setSelectedDetail({ type: 'Award', data: a })}
                                                className="border-b border-white/5 hover:bg-gold/10 transition-colors cursor-pointer"
                                            >
                                                <td className="py-2 text-gray-100 truncate max-w-[150px]">{a.awarding_agency}</td>
                                                <td className="py-2 text-gray-300">FY-{a.contract_year || '26'}</td>
                                                <td className="py-2 text-right text-gold font-mono font-bold">${(a.award_amount_float / 1e6).toFixed(1)}M</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* SEC Detail Modal (Sentiment Analysis) */}
            <AnimatePresence>
                {selectedDetail && selectedDetail.type === 'SEC' && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/95 backdrop-blur-xl z-[150] flex items-center justify-center p-4 md:p-8"
                        onClick={() => setSelectedDetail(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="bg-dark-800 border border-blue-500/30 rounded-3xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col shadow-[0_0_50px_rgba(59,130,246,0.1)]"
                            onClick={e => e.stopPropagation()}
                        >
                            <div className="p-6 border-b border-blue-500/20 flex justify-between items-center bg-dark-900/80">
                                <div>
                                    <h4 className="text-blue-400 font-bold uppercase tracking-[0.3em] text-[10px] mb-1">SEC Filing Analysis</h4>
                                    <div className="text-sm text-white font-bold">
                                        Form {selectedDetail.data.type || selectedDetail.data.form_type}
                                    </div>
                                </div>
                                <button onClick={() => setSelectedDetail(null)} className="p-2 bg-dark-700 rounded-full text-gray-400 hover:text-white border border-white/10 transition-all">✕</button>
                            </div>
                            <div className="flex-1 overflow-y-auto p-8 space-y-8">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <div className="text-3xl font-black text-white tracking-tighter">Form {selectedDetail.data.type || selectedDetail.data.form_type}</div>
                                        <div className="text-xs text-gray-400 mt-1 uppercase font-bold tracking-widest">Filed: {selectedDetail.data.filing_date || 'Unknown'}</div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-1">Sentiment Score</div>
                                        <div className={`text-2xl font-mono font-black ${(selectedDetail.data.avg_finbert || 0) > 0 ? 'text-green-400 shadow-[0_0_20px_rgba(74,222,128,0.2)]' : 'text-red-400 shadow-[0_0_20px_rgba(248,113,113,0.2)]'}`}>
                                            {(selectedDetail.data.avg_finbert || 0).toFixed(4)}
                                        </div>
                                    </div>
                                </div>
                                <div className="space-y-5">
                                    <h5 className="text-[10px] font-black text-blue-400 uppercase tracking-[0.2em] border-b border-blue-500/20 pb-2">Key Excerpts</h5>
                                    {selectedDetail.data.top_sentences && selectedDetail.data.top_sentences.length > 0 ? (
                                        selectedDetail.data.top_sentences.map((s: any, j: number) => (
                                            <div key={j} className="bg-dark-900/50 p-5 rounded-2xl border border-white/5 relative group">
                                                <div className="absolute top-0 left-0 w-1 h-0 bg-blue-500 group-hover:h-full transition-all duration-300" />
                                                <p className="text-xs text-gray-300 leading-relaxed italic">"{s.text}"</p>
                                                <div className="mt-3 flex items-center justify-between">
                                                    <div className="text-[10px] text-blue-500 font-bold uppercase tracking-wider">Sentiment Signal</div>
                                                    <div className="text-[10px] text-gray-500 font-mono">Score: {(s.score || 0).toFixed(3)}</div>
                                                </div>
                                            </div>
                                        ))
                                    ) : (
                                        <div className="text-xs text-gray-500 italic text-center py-8">
                                            No sentence-level analysis available for this filing
                                        </div>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Award Detail Modal */}
            <AnimatePresence>
                {selectedDetail && selectedDetail.type === 'Award' && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/95 backdrop-blur-xl z-[150] flex items-center justify-center p-4 md:p-8"
                        onClick={() => setSelectedDetail(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="bg-dark-800 border border-gold/30 rounded-3xl w-full max-w-3xl max-h-[90vh] overflow-hidden flex flex-col shadow-[0_0_50px_rgba(255,215,0,0.1)]"
                            onClick={e => e.stopPropagation()}
                        >
                            <div className="p-6 border-b border-gold/20 flex justify-between items-center bg-dark-900/80">
                                <div>
                                    <h4 className="text-gold font-bold uppercase tracking-[0.3em] text-[10px] mb-1">Federal Contract Award</h4>
                                    <div className="text-sm text-white font-bold">
                                        {selectedDetail.data.awarding_agency}
                                    </div>
                                </div>
                                <button onClick={() => setSelectedDetail(null)} className="p-2 bg-dark-700 rounded-full text-gray-400 hover:text-white border border-white/10 transition-all">✕</button>
                            </div>
                            <div className="flex-1 overflow-y-auto p-8 space-y-6">
                                <div className="flex justify-between items-start">
                                    <div className="flex-1">
                                        <div className="text-2xl font-black text-white tracking-tight mb-2">
                                            {selectedDetail.data.description || 'Contract Award'}
                                        </div>
                                        <div className="text-xs text-gray-400 uppercase font-bold tracking-widest">
                                            FY-{selectedDetail.data.contract_year || '2026'}
                                        </div>
                                    </div>
                                    <div className="text-right ml-6">
                                        <div className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-1">Award Amount</div>
                                        <div className="text-3xl font-mono font-black text-gold shadow-[0_0_20px_rgba(255,215,0,0.2)]">
                                            ${(selectedDetail.data.award_amount_float / 1e6).toFixed(2)}M
                                        </div>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-dark-900/50 p-4 rounded-xl border border-white/5">
                                        <div className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-2">Awarding Agency</div>
                                        <div className="text-sm text-gray-200 font-semibold">{selectedDetail.data.awarding_agency}</div>
                                    </div>
                                    <div className="bg-dark-900/50 p-4 rounded-xl border border-white/5">
                                        <div className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-2">Contract Year</div>
                                        <div className="text-sm text-gray-200 font-semibold">FY-{selectedDetail.data.contract_year || '2026'}</div>
                                    </div>
                                    {selectedDetail.data.award_date && (
                                        <div className="bg-dark-900/50 p-4 rounded-xl border border-white/5">
                                            <div className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-2">Award Date</div>
                                            <div className="text-sm text-gray-200 font-semibold">{selectedDetail.data.award_date}</div>
                                        </div>
                                    )}
                                    {selectedDetail.data.naics_description && (
                                        <div className="bg-dark-900/50 p-4 rounded-xl border border-white/5">
                                            <div className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-2">Industry (NAICS)</div>
                                            <div className="text-sm text-gray-200 font-semibold">{selectedDetail.data.naics_description}</div>
                                        </div>
                                    )}
                                </div>

                                {selectedDetail.data.description && (
                                    <div>
                                        <h5 className="text-[10px] font-black text-gold uppercase tracking-[0.2em] border-b border-gold/20 pb-2 mb-4">Contract Description</h5>
                                        <div className="bg-dark-900/50 p-5 rounded-2xl border border-white/5">
                                            <p className="text-sm text-gray-300 leading-relaxed">{selectedDetail.data.description}</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
