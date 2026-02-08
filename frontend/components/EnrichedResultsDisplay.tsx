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
    const [showRawData, setShowRawData] = useState(false)

    // Detect what enrichments are present
    const enrichments = useMemo(() => {
        const detected: Array<{ key: string; name: string; icon: string; data: any[]; color: string }> = []

        if (data.options_flow && Array.isArray(data.options_flow) && data.options_flow.length > 0) {
            detected.push({
                key: 'options',
                name: 'Options Flow',
                icon: 'OPTS',
                data: data.options_flow,
                color: 'blue'
            })
        }

        if (data.Award && Array.isArray(data.Award) && data.Award.length > 0) {
            detected.push({
                key: 'awards',
                name: 'Gov Contracts',
                icon: 'GOV',
                data: data.Award,
                color: 'purple'
            })
        }

        if (data.sec_filings && Array.isArray(data.sec_filings) && data.sec_filings.length > 0) {
            detected.push({
                key: 'sec',
                name: 'SEC Filings',
                icon: 'SEC',
                data: data.sec_filings,
                color: 'green'
            })
        }

        if (data.prediction_markets_polymarket && Array.isArray(data.prediction_markets_polymarket) && data.prediction_markets_polymarket.length > 0) {
            detected.push({
                key: 'markets',
                name: 'Prediction Markets',
                icon: 'PRED',
                data: data.prediction_markets_polymarket,
                color: 'orange'
            })
        }

        if (data.MarketData && Array.isArray(data.MarketData) && data.MarketData.length > 0) {
            detected.push({
                key: 'marketdata',
                name: 'Market Data',
                icon: 'MKT',
                data: data.MarketData,
                color: 'gold'
            })
        }

        return detected
    }, [data])

    // Generate intelligent AI analysis with correlations
    const aiAnalysis = useMemo(() => {
        if (enrichments.length === 0) return null

        const ticker = data.ticker || 'Company'
        const analysis: string[] = []

        // Get data references
        const options = enrichments.find(e => e.key === 'options')
        const awards = enrichments.find(e => e.key === 'awards')
        const sec = enrichments.find(e => e.key === 'sec')
        const markets = enrichments.find(e => e.key === 'markets')
        const marketData = enrichments.find(e => e.key === 'marketdata')

        // Cross-correlation analysis

        // 1. Options + SEC Correlation
        if (options && sec) {
            const latestOptions = options.data[0]
            const avgSecSentiment = sec.data.reduce((sum: number, f: any) => sum + (f.avg_finbert || 0), 0) / sec.data.length
            const putCallRatio = latestOptions.put_call_volume_ratio || 0

            if (avgSecSentiment < -0.2 && putCallRatio > 1.3) {
                analysis.push(`<strong class="text-red-400">Warning Signal:</strong> SEC filings show negative sentiment (${avgSecSentiment.toFixed(3)}) while options traders are positioning bearish with P/C ratio of ${putCallRatio.toFixed(2)}. This confluence suggests institutional concern about near-term performance.`)
            } else if (avgSecSentiment > 0.2 && putCallRatio < 0.7) {
                analysis.push(`<strong class="text-green-400">Bullish Alignment:</strong> Positive SEC disclosure sentiment (${avgSecSentiment.toFixed(3)}) aligns with bullish options positioning (P/C ratio ${putCallRatio.toFixed(2)}), indicating market confidence in management guidance.`)
            } else if (Math.abs(avgSecSentiment) > 0.2 && Math.abs(putCallRatio - 1.0) > 0.3) {
                analysis.push(`<strong class="text-yellow-400">Divergence Detected:</strong> SEC sentiment (${avgSecSentiment.toFixed(3)}) diverges from options flow (P/C ratio ${putCallRatio.toFixed(2)}). This mismatch may indicate market mispricing or delayed reaction to regulatory disclosures.`)
            }
        }

        // 2. Options + Awards Correlation (Insider trading potential)
        if (options && awards) {
            const recentAwards = awards.data.filter((a: any) => {
                if (!a.start_date) return false
                const date = new Date(a.start_date)
                const threeMonthsAgo = new Date()
                threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3)
                return date >= threeMonthsAgo
            })

            if (recentAwards.length > 0) {
                const totalValue = recentAwards.reduce((sum: number, a: any) => sum + (a.award_amount_float || 0), 0)
                const latestOptions = options.data[0]

                if (latestOptions.unusual_total_activity) {
                    analysis.push(`<strong class="text-orange-400">Insider Trading Watch:</strong> Unusual options activity detected following ${recentAwards.length} recent government contracts worth $${Number(totalValue).toLocaleString('en-US', { maximumFractionDigits: 0 })}. Monitor for Form 4 insider disclosures in next 2 business days per SEC Rule 10b5-1 requirements.`)
                } else if (recentAwards.length >= 2) {
                    analysis.push(`<strong class="text-blue-400">Contract Momentum:</strong> ${ticker} secured ${recentAwards.length} government contracts totaling $${Number(totalValue).toLocaleString('en-US', { maximumFractionDigits: 0 })} in the last 90 days. Options flow remains within normal ranges, suggesting awards are priced in or contract values are immaterial to market cap.`)
                }
            }
        }

        // 3. SEC + Awards Correlation (Claim verification)
        if (sec && awards) {
            const recentSec = sec.data.filter((f: any) => {
                const date = new Date(f.filing_date)
                const sixMonthsAgo = new Date()
                sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6)
                return date >= sixMonthsAgo
            })

            if (recentSec.length > 0 && awards.data.length > 0) {
                const totalAwardValue = awards.data.reduce((sum: number, a: any) => sum + (a.award_amount_float || 0), 0)
                analysis.push(`<strong class="text-purple-400">Cross-Domain Validation:</strong> ${ticker} discloses ${recentSec.length} SEC filings in the last 6 months while holding $${Number(totalAwardValue).toLocaleString('en-US', { maximumFractionDigits: 0 })} in government contracts. This enables verification of revenue recognition claims in 10-K/10-Q against actual awarded contract values.`)
            }
        }

        // 4. Market Sentiment Analysis
        if (sec) {
            const formTypes = Array.from(new Set(sec.data.map((f: any) => f.type || f.form_type)))
            const avgSentiment = sec.data.reduce((sum: number, f: any) => sum + (f.avg_finbert || 0), 0) / sec.data.length
            const recentFilings = sec.data.slice(0, 3)

            const form8Ks = sec.data.filter((f: any) => (f.type || f.form_type) === '8-K')
            if (form8Ks.length > 0) {
                analysis.push(`<strong class="text-cyan-400">Material Events Disclosure:</strong> ${form8Ks.length} Form 8-K filings (material events) detected. Review for M&A announcements, executive changes, or earnings restatements that may drive volatility.`)
            }
        }

        // 5. Options Activity Insights
        if (options) {
            const latestOptions = options.data[0]
            const totalVolume = (latestOptions.call_volume || 0) + (latestOptions.put_volume || 0)
            const callPremium = latestOptions.call_premium || 0
            const putPremium = latestOptions.put_premium || 0

            if (totalVolume > 1000000) {
                analysis.push(`<strong class="text-blue-400">High Volume Alert:</strong> ${totalVolume.toLocaleString()} total options contracts traded on ${latestOptions.date}. Call premium: $${(callPremium / 1e6).toFixed(2)}M, Put premium: $${(putPremium / 1e6).toFixed(2)}M. Large institutional positioning indicates expected near-term catalyst.`)
            }

            if (latestOptions.iv_rank > 75) {
                analysis.push(`<strong class="text-yellow-400">Elevated IV:</strong> Implied volatility rank at ${latestOptions.iv_rank}/100 suggests traders pricing in significant price movement. Consider straddle/strangle strategies or wait for IV crush post-event.`)
            }
        }

        // 6. Awards Analysis
        if (awards) {
            const agencies = Array.from(new Set(awards.data.map((a: any) => a.awarding_agency)))
            const totalValue = awards.data.reduce((sum: number, a: any) => sum + (a.award_amount_float || 0), 0)

            if (agencies.length > 5) {
                analysis.push(`<strong class="text-purple-400">Diversified Government Revenue:</strong> ${ticker} works with ${agencies.length} different federal agencies, reducing single-client concentration risk. Total contract value: $${(totalValue / 1e6).toFixed(1)}M across ${awards.data.length} awards.`)
            }
        }

        return analysis.length > 0 ? analysis : null
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
            {/* AI Analysis Section */}
            {aiAnalysis && aiAnalysis.length > 0 && (
                <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-lg p-5">
                    <div className="flex items-start gap-3 mb-4">
                        <div className="text-sm font-bold text-blue-400 px-2 py-1 bg-blue-500/20 rounded">AI ANALYSIS</div>
                        <div className="flex-1">
                            <div className="text-xs text-gray-400">
                                Cross-domain correlation analysis across {enrichments.length} data sources
                            </div>
                        </div>
                    </div>
                    <div className="space-y-3">
                        {aiAnalysis.map((insight, idx) => (
                            <div key={idx} className="bg-dark-900/50 border border-blue-500/20 rounded-lg p-4">
                                <div className="text-sm text-gray-300 leading-relaxed" dangerouslySetInnerHTML={{ __html: insight }} />
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Raw Data Toggle */}
            <button
                onClick={() => setShowRawData(!showRawData)}
                className="w-full px-4 py-2 bg-dark-800 border border-gold/20 rounded-lg text-sm text-gray-400 hover:text-gold hover:border-gold/40 transition-all flex items-center justify-between"
            >
                <span>{showRawData ? 'Hide' : 'Show'} Raw Data Browser</span>
                <svg
                    className={`w-4 h-4 transition-transform ${showRawData ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {/* Enriched Data Drill-Down Sections (Collapsible) */}
            {showRawData && (
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
                                        <span className={`text-xs font-bold ${colors.text} px-2 py-1 ${colors.badge} rounded`}>{enrichment.icon}</span>
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
            )}

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
                const sentimentLabel = sentiment > 0.2 ? 'POS' : sentiment < -0.2 ? 'NEG' : 'NEU'

                return (
                    <div key={idx} className="bg-dark-900/50 border border-green-500/20 rounded p-3">
                        <div className="flex justify-between items-start mb-2">
                            <div className="flex items-center gap-2">
                                <span className="text-xs font-semibold text-green-300">
                                    {filing.type || filing.form_type || 'Unknown'}
                                </span>
                                <span className={`text-xs ${sentimentColor} flex items-center gap-1`}>
                                    <span className="font-bold">{sentimentLabel}</span>
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
