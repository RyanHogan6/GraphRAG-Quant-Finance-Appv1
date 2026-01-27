'use client'

import { useState, useMemo } from 'react'
import CompanyWorkup from './CompanyWorkup'
import ResultsTable from './ResultsTable'

interface EnrichedResultsDisplayProps {
    data: any
    question?: string
}

export default function EnrichedResultsDisplay({ data, question }: EnrichedResultsDisplayProps) {
    const [expandedSection, setExpandedSection] = useState<string | null>(null)

    // Detect what enrichments are present
    const enrichments = useMemo(() => {
        const detected: Array<{ key: string; name: string; icon: string; data: any[]; color: string }> = []

        if (data.options_flow && Array.isArray(data.options_flow) && data.options_flow.length > 0) {
            detected.push({
                key: 'options',
                name: 'Options Flow',
                icon: '📈',
                data: data.options_flow,
                color: 'blue'
            })
        }

        if (data.Award && Array.isArray(data.Award) && data.Award.length > 0) {
            detected.push({
                key: 'awards',
                name: 'Gov Contracts',
                icon: '🏛️',
                data: data.Award,
                color: 'purple'
            })
        }

        if (data.sec_filings && Array.isArray(data.sec_filings) && data.sec_filings.length > 0) {
            detected.push({
                key: 'sec',
                name: 'SEC Filings',
                icon: '📄',
                data: data.sec_filings,
                color: 'green'
            })
        }

        if (data.prediction_markets_polymarket && Array.isArray(data.prediction_markets_polymarket) && data.prediction_markets_polymarket.length > 0) {
            detected.push({
                key: 'markets',
                name: 'Prediction Markets',
                icon: '🎲',
                data: data.prediction_markets_polymarket,
                color: 'orange'
            })
        }

        if (data.MarketData && Array.isArray(data.MarketData) && data.MarketData.length > 0) {
            detected.push({
                key: 'marketdata',
                name: 'Market Data',
                icon: '📊',
                data: data.MarketData,
                color: 'gold'
            })
        }

        return detected
    }, [data])

    // Generate LLM inference/context
    const inference = useMemo(() => {
        if (enrichments.length === 0) return null

        const ticker = data.ticker || 'Company'
        const enrichmentNames = enrichments.map(e => e.name).join(', ')

        // Analyze the data
        const insights: string[] = []

        // Options insights
        const options = enrichments.find(e => e.key === 'options')
        if (options && options.data.length > 0) {
            const latestOptions = options.data[0]
            const putCallRatio = latestOptions.put_call_volume_ratio || 0
            const unusualActivity = latestOptions.unusual_total_activity

            if (unusualActivity) {
                insights.push(`🚨 <strong>Unusual options activity detected</strong> on ${latestOptions.date || 'recent date'}`)
            }

            if (putCallRatio > 1.5) {
                insights.push(`📉 <strong>Bearish options sentiment</strong> with Put/Call ratio of ${putCallRatio.toFixed(2)}`)
            } else if (putCallRatio < 0.7) {
                insights.push(`📈 <strong>Bullish options sentiment</strong> with Put/Call ratio of ${putCallRatio.toFixed(2)}`)
            }
        }

        // Awards insights
        const awards = enrichments.find(e => e.key === 'awards')
        if (awards && awards.data.length > 0) {
            const totalValue = awards.data.reduce((sum: number, a: any) => sum + (a.award_amount_float || 0), 0)
            const recentAwards = awards.data.filter((a: any) => {
                if (!a.start_date) return false
                const date = new Date(a.start_date)
                const sixMonthsAgo = new Date()
                sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6)
                return date >= sixMonthsAgo
            }).length

            insights.push(`🏛️ <strong>${awards.data.length} government contracts</strong> worth $${(totalValue / 1e6).toFixed(1)}M total`)
            if (recentAwards > 0) {
                insights.push(`✨ <strong>${recentAwards} new contracts</strong> awarded in the last 6 months`)
            }
        }

        // SEC insights
        const sec = enrichments.find(e => e.key === 'sec')
        if (sec && sec.data.length > 0) {
            const avgSentiment = sec.data.reduce((sum: number, f: any) => sum + (f.avg_finbert || 0), 0) / sec.data.length
            const recentFilings = sec.data.slice(0, 3)
            const formTypes = [...new Set(sec.data.map((f: any) => f.type || f.form_type))].slice(0, 3).join(', ')

            if (avgSentiment > 0.2) {
                insights.push(`💚 <strong>Positive SEC sentiment</strong> (avg ${avgSentiment.toFixed(3)})`)
            } else if (avgSentiment < -0.2) {
                insights.push(`💔 <strong>Negative SEC sentiment</strong> (avg ${avgSentiment.toFixed(3)})`)
            }

            insights.push(`📄 <strong>${sec.data.length} SEC filings</strong> including ${formTypes}`)
        }

        return {
            summary: `Found ${enrichments.length} data sources for <strong>${ticker}</strong>: ${enrichmentNames}`,
            insights
        }
    }, [data, enrichments])

    if (enrichments.length === 0) {
        return <CompanyWorkup data={data} />
    }

    const colorClasses: Record<string, { bg: string; border: string; text: string; badge: string }> = {
        blue: { bg: 'bg-blue-500/5', border: 'border-blue-500/30', text: 'text-blue-400', badge: 'bg-blue-500/20' },
        purple: { bg: 'bg-purple-500/5', border: 'border-purple-500/30', text: 'text-purple-400', badge: 'bg-purple-500/20' },
        green: { bg: 'bg-green-500/5', border: 'border-green-500/30', text: 'text-green-400', badge: 'bg-green-500/20' },
        orange: { bg: 'bg-orange-500/5', border: 'border-orange-500/30', text: 'text-orange-400', badge: 'bg-orange-500/20' },
        gold: { bg: 'bg-gold/5', border: 'border-gold/30', text: 'text-gold', badge: 'bg-gold/20' }
    }

    return (
        <div className="space-y-4">
            {/* LLM Inference Section */}
            {inference && (
                <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                        <div className="text-2xl">🤖</div>
                        <div className="flex-1 space-y-2">
                            <div className="text-sm text-gray-300" dangerouslySetInnerHTML={{ __html: inference.summary }} />
                            {inference.insights.length > 0 && (
                                <div className="space-y-1.5 pt-2 border-t border-blue-500/20">
                                    {inference.insights.map((insight, idx) => (
                                        <div key={idx} className="text-xs text-gray-400" dangerouslySetInnerHTML={{ __html: insight }} />
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Enriched Data Drill-Down Sections */}
            <div className="space-y-3">
                {enrichments.map((enrichment) => {
                    const isExpanded = expandedSection === enrichment.key
                    const colors = colorClasses[enrichment.color]

                    return (
                        <div key={enrichment.key} className={`border rounded-lg overflow-hidden ${colors.border} ${colors.bg}`}>
                            <button
                                onClick={() => setExpandedSection(isExpanded ? null : enrichment.key)}
                                className={`w-full px-4 py-3 flex items-center justify-between hover:bg-white/5 transition-colors`}
                            >
                                <div className="flex items-center gap-3">
                                    <span className="text-xl">{enrichment.icon}</span>
                                    <div className="text-left">
                                        <div className={`text-sm font-semibold ${colors.text}`}>
                                            {enrichment.name}
                                        </div>
                                        <div className="text-xs text-gray-500">
                                            {enrichment.data.length} record{enrichment.data.length !== 1 ? 's' : ''}
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={`text-xs px-2 py-1 rounded ${colors.badge} ${colors.text} font-mono`}>
                                        {enrichment.data.length}
                                    </span>
                                    <svg
                                        className={`w-5 h-5 ${colors.text} transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                    </svg>
                                </div>
                            </button>

                            {isExpanded && (
                                <div className="px-4 pb-4 border-t border-white/10">
                                    <div className="mt-3">
                                        {enrichment.key === 'options' && (
                                            <OptionsFlowTable data={enrichment.data} />
                                        )}
                                        {enrichment.key === 'awards' && (
                                            <AwardsTable data={enrichment.data} />
                                        )}
                                        {enrichment.key === 'sec' && (
                                            <SECFilingsTable data={enrichment.data} />
                                        )}
                                        {enrichment.key === 'markets' && (
                                            <ResultsTable data={enrichment.data} maxRows={10} />
                                        )}
                                        {enrichment.key === 'marketdata' && (
                                            <ResultsTable data={enrichment.data} maxRows={10} />
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            {/* Company Overview (Default Workup) */}
            <div className="pt-4 border-t border-gold/20">
                <div className="text-xs text-gold uppercase font-bold tracking-widest opacity-70 mb-4">
                    Company Overview
                </div>
                <CompanyWorkup data={data} />
            </div>
        </div>
    )
}

// Specialized table components
function OptionsFlowTable({ data }: { data: any[] }) {
    return (
        <div className="overflow-x-auto">
            <table className="w-full text-xs">
                <thead>
                    <tr className="border-b border-white/10">
                        <th className="text-left py-2 text-gray-400 font-semibold">Date</th>
                        <th className="text-right py-2 text-gray-400 font-semibold">Stock Price</th>
                        <th className="text-right py-2 text-gray-400 font-semibold">Call Vol</th>
                        <th className="text-right py-2 text-gray-400 font-semibold">Put Vol</th>
                        <th className="text-right py-2 text-gray-400 font-semibold">P/C Ratio</th>
                        <th className="text-right py-2 text-gray-400 font-semibold">Unusual</th>
                    </tr>
                </thead>
                <tbody>
                    {data.slice(0, 10).map((row, idx) => (
                        <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                            <td className="py-2 text-gray-300">{row.date || 'N/A'}</td>
                            <td className="py-2 text-right text-gold">${row.stock_price?.toFixed(2) || 'N/A'}</td>
                            <td className="py-2 text-right text-green-400">{row.call_volume?.toLocaleString() || 0}</td>
                            <td className="py-2 text-right text-red-400">{row.put_volume?.toLocaleString() || 0}</td>
                            <td className="py-2 text-right text-gray-300">{row.put_call_volume_ratio?.toFixed(2) || 'N/A'}</td>
                            <td className="py-2 text-right">
                                {row.unusual_total_activity ? (
                                    <span className="px-2 py-0.5 bg-orange-500/20 text-orange-400 rounded text-[10px] font-semibold">
                                        UNUSUAL
                                    </span>
                                ) : (
                                    <span className="text-gray-600">-</span>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

function AwardsTable({ data }: { data: any[] }) {
    return (
        <div className="space-y-2">
            {data.slice(0, 5).map((award, idx) => (
                <div key={idx} className="bg-dark-900/50 border border-purple-500/20 rounded p-3">
                    <div className="flex justify-between items-start mb-2">
                        <div className="text-xs font-semibold text-purple-300">
                            {award.awarding_agency || 'Unknown Agency'}
                        </div>
                        <div className="text-xs font-mono text-gold">
                            ${(award.award_amount_float / 1e6).toFixed(2)}M
                        </div>
                    </div>
                    <div className="text-xs text-gray-400 line-clamp-2">
                        {award.description || 'No description available'}
                    </div>
                    <div className="text-[10px] text-gray-500 mt-2">
                        {award.start_date || 'Date unknown'}
                    </div>
                </div>
            ))}
        </div>
    )
}

function SECFilingsTable({ data }: { data: any[] }) {
    return (
        <div className="space-y-2">
            {data.slice(0, 5).map((filing, idx) => {
                const sentiment = filing.avg_finbert || 0
                const sentimentColor = sentiment > 0.2 ? 'text-green-400' : sentiment < -0.2 ? 'text-red-400' : 'text-gray-400'
                const sentimentIcon = sentiment > 0.2 ? '📈' : sentiment < -0.2 ? '📉' : '➡️'

                return (
                    <div key={idx} className="bg-dark-900/50 border border-green-500/20 rounded p-3">
                        <div className="flex justify-between items-start mb-2">
                            <div className="flex items-center gap-2">
                                <span className="text-xs font-semibold text-green-300">
                                    {filing.type || filing.form_type || 'Unknown'}
                                </span>
                                <span className={`text-xs ${sentimentColor} flex items-center gap-1`}>
                                    {sentimentIcon}
                                    <span>{sentiment.toFixed(3)}</span>
                                </span>
                            </div>
                            <div className="text-[10px] text-gray-500">
                                {filing.filing_date || 'Date unknown'}
                            </div>
                        </div>
                        {filing.top_sentences && filing.top_sentences.length > 0 && (
                            <div className="mt-2 space-y-1">
                                {filing.top_sentences.slice(0, 2).map((sent: any, sIdx: number) => (
                                    <div key={sIdx} className="text-[10px] text-gray-400 italic border-l-2 border-green-500/30 pl-2">
                                        "{sent.text?.substring(0, 120)}..." ({sent.score?.toFixed(2)})
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )
            })}
        </div>
    )
}
