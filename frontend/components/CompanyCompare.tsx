'use client'

import CompanyWorkup from './CompanyWorkup'

interface CompanyCompareProps {
    companyA: any
    companyB: any
}

export default function CompanyCompare({ companyA, companyB }: CompanyCompareProps) {
    // Simple correlation calculation (direction of sentiment/price)
    const calculateCorrelation = () => {
        const sentA = companyA.sec_filings?.[0]?.avg_finbert || 0
        const sentB = companyB.sec_filings?.[0]?.avg_finbert || 0

        const directionMatch = (sentA > 0 && sentB > 0) || (sentA < 0 && sentB < 0)

        return {
            sentiment: directionMatch ? 'Positive Correlation' : 'Divergent Sentiment',
            color: directionMatch ? 'text-green-400' : 'text-orange-400',
            description: directionMatch
                ? `Both firms share a similar regulatory sentiment profile (${sentA > 0 ? 'Bullish' : 'Bearish'}).`
                : `Regulatory sentiment is diverging between these peers.`
        }
    }

    const correlation = calculateCorrelation()

    return (
        <div className="w-full space-y-6">
            {/* Correlation Header */}
            <div className="bg-dark-900/80 border border-gold/30 rounded-2xl p-4 flex items-center justify-between">
                <div>
                    <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">Comparative Analysis</h3>
                    <div className={`text-lg font-bold ${correlation.color}`}>{correlation.sentiment}</div>
                    <p className="text-xs text-gray-400 mt-1">{correlation.description}</p>
                </div>
                <div className="hidden md:block">
                    <div className="flex -space-x-3">
                        <div className="w-10 h-10 rounded-full bg-gold/20 border border-gold/40 flex items-center justify-center text-gold font-bold text-xs z-10">
                            {companyA.ticker}
                        </div>
                        <div className="w-10 h-10 rounded-full bg-purple-500/20 border border-purple-500/40 flex items-center justify-center text-purple-400 font-bold text-xs">
                            {companyB.ticker}
                        </div>
                    </div>
                </div>
            </div>

            {/* Side-by-Side Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 items-start">
                <div className="bg-black/20 rounded-3xl p-1 border border-white/5">
                    <CompanyWorkup data={companyA} />
                </div>
                <div className="bg-black/20 rounded-3xl p-1 border border-white/5">
                    <CompanyWorkup data={companyB} />
                </div>
            </div>
        </div>
    )
}
