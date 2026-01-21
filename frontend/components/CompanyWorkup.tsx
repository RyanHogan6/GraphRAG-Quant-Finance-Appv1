'use client'

import { useState, useMemo } from 'react'
import TimeSeriesChart from './TimeSeriesChart'
import { motion } from 'framer-motion'

interface CompanyWorkupProps {
    data: any
    onCompare?: (ticker: string) => void
}

export default function CompanyWorkup({ data, onCompare }: CompanyWorkupProps) {
    const [timeframe, setTimeframe] = useState<'1M' | '3M' | '6M' | '1Y'>('1M')

    // Extract nested data
    const company = data
    const marketData = data.MarketData || []
    const secFilings = data.sec_filings || []
    const polyMarkets = data.prediction_markets_polymarket || []
    const awards = data.Award || []

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

        filtered = filtered.filter(d => new Date(d.date) >= filterDate)

        const sorted = filtered.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
        return {
            dates: sorted.map(d => d.date),
            values: sorted.map(d => d.close),
            ticker: company.ticker
        }
    }, [marketData, company.ticker, timeframe])

    // Sentiment aggregation
    const sentiment = useMemo(() => {
        if (!secFilings.length) return null
        const scores = secFilings.map((f: any) => f.avg_finbert).filter((s: any) => s != null)
        if (!scores.length) return null
        const avg = scores.reduce((a: number, b: number) => a + b, 0) / scores.length
        return {
            score: avg,
            label: avg > 0.2 ? 'Bullish' : avg < -0.2 ? 'Bearish' : 'Neutral',
            color: avg > 0.2 ? 'text-green-400' : avg < -0.2 ? 'text-red-400' : 'text-gray-400'
        }
    }, [secFilings])

    // Polymarket probability (average of active markets)
    const predictionBias = useMemo(() => {
        if (!polyMarkets.length) return null
        const probs = polyMarkets.map((m: any) => m.yes_probability).filter((p: any) => p != null)
        if (!probs.length) return null
        const avg = (probs.reduce((a: number, b: number) => a + b, 0) / probs.length) * 100
        return avg.toFixed(1)
    }, [polyMarkets])

    return (
        <div className="w-full space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Header Info */}
            <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-gold/20 pb-4 gap-4">
                <div>
                    <div className="flex items-center gap-3 mb-1">
                        <h2 className="text-2xl font-bold text-white tracking-tight">{company.company}</h2>
                        <span className="px-2 py-0.5 bg-gold/10 border border-gold/30 rounded text-gold text-sm font-mono font-bold">
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
                        className="px-3 py-1.5 bg-dark-800 border border-gold/30 rounded-lg text-xs text-gold hover:bg-gold/10 transition-all flex items-center gap-2"
                    >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                        </svg>
                        Compare
                    </button>
                    <button className="px-3 py-1.5 bg-dark-800 border border-gold/30 rounded-lg text-xs text-gray-400 hover:text-white transition-all">
                        Export Report
                    </button>
                </div>
            </div>

            {/* Stats Quick Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-dark-900/50 border border-white/5 rounded-xl p-3">
                    <div className="text-[10px] text-gray-500 uppercase font-bold tracking-wider mb-1">Market Cap</div>
                    <div className="text-sm font-mono font-bold text-gray-200">
                        ${(company.marketCap / 1e9).toFixed(2)}B
                    </div>
                </div>
                <div className="bg-dark-900/50 border border-white/5 rounded-xl p-3">
                    <div className="text-[10px] text-gray-500 uppercase font-bold tracking-wider mb-1">Employees</div>
                    <div className="text-sm font-mono font-bold text-gray-200">
                        {company.fullTimeEmployees?.toLocaleString()}
                    </div>
                </div>
                <div className="bg-dark-900/50 border border-white/5 rounded-xl p-3">
                    <div className="text-[10px] text-gray-500 uppercase font-bold tracking-wider mb-1">SEC Sentiment</div>
                    <div className={`text-sm font-bold ${sentiment?.color || 'text-gray-400'}`}>
                        {sentiment?.label || 'N/A'}
                    </div>
                </div>
                <div className="bg-dark-900/50 border border-white/5 rounded-xl p-3">
                    <div className="text-[10px] text-gray-500 uppercase font-bold tracking-wider mb-1">Poly Bias</div>
                    <div className="text-sm font-mono font-bold text-purple-400">
                        {predictionBias ? `${predictionBias}% Bull` : 'N/A'}
                    </div>
                </div>
            </div>

            {/* Chart Section */}
            <div className="bg-dark-900/30 border border-gold/10 rounded-2xl p-4">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-xs font-bold text-gold uppercase tracking-tighter">Market Intelligence (Price Action)</h3>
                    <div className="flex bg-dark-800 rounded-lg p-0.5 border border-white/10">
                        {['1M', '3M', '6M', '1Y'].map(tf => (
                            <button
                                key={tf}
                                onClick={() => setTimeframe(tf as any)}
                                className={`px-2 py-1 text-[10px] rounded-md transition-all ${timeframe === tf ? 'bg-gold/20 text-gold font-bold' : 'text-gray-500 hover:text-gray-300'}`}
                            >
                                {tf}
                            </button>
                        ))}
                    </div>
                </div>
                {chartData.values.length > 0 ? (
                    <div className="h-[250px] w-full">
                        <TimeSeriesChart
                            dates={chartData.dates}
                            values={chartData.values}
                            label={`${company.ticker} Close Price`}
                            ticker={company.ticker}
                        />
                    </div>
                ) : (
                    <div className="h-[200px] flex items-center justify-center text-gray-600 text-sm border border-dashed border-white/10 rounded-xl">
                        No historical price data available in query result
                    </div>
                )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* SEC Deep Dive */}
                <div className="bg-dark-900/50 border border-white/5 rounded-2xl p-4">
                    <h3 className="text-xs font-bold text-blue-400 uppercase tracking-tighter mb-4">Regulatory Signals (SEC)</h3>
                    {secFilings.length > 0 ? (
                        <div className="space-y-3">
                            {secFilings.slice(0, 2).map((f: any, i: number) => (
                                <div key={i} className="bg-black/20 rounded-lg p-3 border border-white/5">
                                    <div className="flex justify-between text-[11px] mb-2">
                                        <span className="text-gray-400">{f.form_type || '10-K'} • {f.filing_date}</span>
                                        <span className={f.avg_finbert > 0 ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>
                                            Score: {f.avg_finbert?.toFixed(3)}
                                        </span>
                                    </div>
                                    {f.top_sentences?.length > 0 && (
                                        <div className="text-[10px] text-gray-500 italic leading-tight border-l-2 border-gold/20 pl-2">
                                            "{f.top_sentences[0].text.substring(0, 120)}..."
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-xs text-gray-600 italic">No recent SEC filings in dataset</div>
                    )}
                </div>

                {/* Prediction & Contracts */}
                <div className="bg-dark-900/50 border border-white/5 rounded-2xl p-4">
                    <h3 className="text-xs font-bold text-purple-400 uppercase tracking-tighter mb-4">Market Sentiment & Awards</h3>
                    <div className="space-y-4">
                        {/* Prediction Probability */}
                        <div>
                            <div className="flex justify-between text-[10px] text-gray-500 mb-1 font-bold">
                                <span>POLYMARKET ODDS</span>
                                <span className="text-purple-400">{predictionBias}% BULLISH</span>
                            </div>
                            <div className="w-full bg-dark-800 h-1.5 rounded-full overflow-hidden border border-white/5">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${predictionBias || 0}%` }}
                                    className="h-full bg-gradient-to-r from-purple-600 to-purple-400"
                                />
                            </div>
                        </div>

                        {/* Awards List */}
                        <div className="pt-2">
                            <div className="text-[10px] text-gray-500 uppercase font-bold mb-2">Recent Gov Contracts</div>
                            {awards.length > 0 ? (
                                <div className="space-y-2">
                                    {awards.slice(0, 2).map((a: any, i: number) => (
                                        <div key={i} className="flex justify-between items-center text-[10px] bg-gold/5 p-2 rounded border border-gold/10">
                                            <span className="text-gray-300 truncate max-w-[150px]">{a.description}</span>
                                            <span className="text-gold font-bold">${(a.award_amount_float / 1e6).toFixed(1)}M</span>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-[10px] text-gray-600 italic">No government awards found</div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
