'use client'

import { useState, useEffect } from 'react'

interface AwardHistoryProps {
    recipientName: string
    awardAmount: number
    startDate: string
    agency: string
}

export default function AwardHistory({ recipientName, awardAmount, startDate, agency }: AwardHistoryProps) {
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [summary, setSummary] = useState<any>(null)
    const [transactions, setTransactions] = useState<any[]>([])

    useEffect(() => {
        const fetchAwardDetails = async () => {
            setLoading(true)
            setError(null)
            try {
                if (!recipientName || isNaN(awardAmount)) {
                    setError("Insufficient award telemetry to initiate deep tracking.")
                    setLoading(false)
                    return
                }

                // Ensure startDate is in YYYY-MM-DD format
                let formattedDate: string | null = startDate;
                if (!startDate || startDate === 'N/A') {
                    formattedDate = null;
                } else {
                    try {
                        formattedDate = new Date(startDate).toISOString().split('T')[0];
                    } catch (e) {
                        formattedDate = null;
                    }
                }

                // 1. Search for the award to get the generated_internal_id
                const searchFilters: any = {
                    keywords: [recipientName],
                    award_amounts: [{ lower_bound: Math.max(0, awardAmount * 0.8), upper_bound: awardAmount * 1.2 }]
                };

                if (formattedDate) {
                    const d = new Date(formattedDate);
                    const prevMonth = new Date(d);
                    prevMonth.setMonth(d.getMonth() - 1);
                    const nextMonth = new Date(d);
                    nextMonth.setMonth(d.getMonth() + 1);

                    searchFilters.time_period = [{
                        start_date: prevMonth.toISOString().split('T')[0],
                        end_date: nextMonth.toISOString().split('T')[0]
                    }];
                }

                const searchRes = await fetch('https://api.usaspending.gov/api/v2/search/spending_by_award/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filters: searchFilters,
                        fields: ["Award ID", "Recipient Name", "Award Amount", "Awarding Agency", "generated_internal_id"],
                        limit: 1
                    })
                })

                const searchData = await searchRes.json()
                const internalId = searchData.results?.[0]?.generated_internal_id

                if (!internalId) {
                    // Fallback to searching just by keyword and agency if exact match fails
                    setError("Direct match not found in USAspending index. Try searching by Recipient on site.")
                    setLoading(false)
                    return
                }

                // 2. Fetch Transaction History
                const transRes = await fetch('https://api.usaspending.gov/api/v2/transactions/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        items: [internalId],
                        limit: 10,
                        sort: "action_date",
                        order: "desc"
                    })
                })
                const transData = await transRes.json()
                setTransactions(transData.results || [])

                // 3. Fetch specific award metadata (Competitive status, etc)
                // Note: USAspending has multiple v2 endpoints, /api/v2/awards/{id}/ is most direct
                const detailRes = await fetch(`https://api.usaspending.gov/api/v2/awards/${internalId}/`)
                const detailData = await detailRes.json()
                setSummary(detailData)

            } catch (err: any) {
                console.error("USAspending Fetch Error:", err)
                setError("Network error connecting to USAspending.gov")
            } finally {
                setLoading(false)
            }
        }

        if (recipientName && awardAmount) {
            fetchAwardDetails()
        }
    }, [recipientName, awardAmount, startDate])

    if (loading) return (
        <div className="flex flex-col items-center justify-center py-12 space-y-4">
            <div className="w-8 h-8 border-2 border-gold/30 border-t-gold rounded-full animate-spin" />
            <div className="text-[10px] text-gold uppercase tracking-[0.2em] animate-pulse">Initializing Advanced Telemetry...</div>
        </div>
    )

    if (error) return (
        <div className="bg-red-500/10 border border-red-500/20 p-4 rounded-xl text-center">
            <p className="text-[11px] text-red-400 font-bold mb-2">TELEMETRY SYNC INTERRUPTED</p>
            <p className="text-[10px] text-gray-500">{error}</p>
        </div>
    )

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            {/* Competitive Status & Risk Profile */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-black/40 p-4 rounded-2xl border border-white/5">
                    <div className="text-[9px] text-gray-500 uppercase font-black mb-1">Competition</div>
                    <div className={`text-xs font-bold ${summary?.description?.toLowerCase().includes('sole') ? 'text-red-400' : 'text-green-400'}`}>
                        {summary?.latest_transaction?.competed_status || 'Full & Open'}
                    </div>
                </div>
                <div className="bg-black/40 p-4 rounded-2xl border border-white/5">
                    <div className="text-[9px] text-gray-500 uppercase font-black mb-1">Contract Type</div>
                    <div className="text-xs text-gray-200 font-bold truncate">
                        {summary?.type_description || 'Standard procurement'}
                    </div>
                </div>
                <div className="bg-black/40 p-4 rounded-2xl border border-white/5">
                    <div className="text-[9px] text-gray-500 uppercase font-black mb-1">Funding Agency</div>
                    <div className="text-xs text-gold font-bold truncate">
                        {summary?.funding_agency?.office_name || 'Departmental Central'}
                    </div>
                </div>
            </div>

            {/* Transaction Ledger */}
            <div>
                <h5 className="text-[10px] font-black text-gold uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-gold" />
                    Funding Transaction Ledger
                </h5>
                <div className="space-y-2 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
                    {transactions.map((t, i) => (
                        <div key={i} className="bg-dark-900/60 p-3 rounded-xl border border-white/5 flex justify-between items-center group hover:border-gold/20 transition-all">
                            <div>
                                <div className="text-[10px] text-white font-bold">{t.action_date}</div>
                                <div className="text-[9px] text-gray-500 italic mt-0.5 line-clamp-1">{t.description || "Obligation adjustment"}</div>
                            </div>
                            <div className={`text-[11px] font-mono font-black ${t.federal_action_obligation >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {t.federal_action_obligation >= 0 ? '+' : ''}${t.federal_action_obligation.toLocaleString()}
                            </div>
                        </div>
                    ))}
                    {transactions.length === 0 && <div className="text-[10px] text-gray-600 italic">No incremental modifications detected.</div>}
                </div>
            </div>

            {/* Sub-award Summary */}
            {summary?.subaward_count > 0 && (
                <div className="bg-blue-500/5 border border-blue-500/10 p-5 rounded-2xl">
                    <div className="flex justify-between items-center mb-1">
                        <div className="text-[10px] text-blue-400 font-bold uppercase tracking-widest">Sub-Award Network</div>
                        <div className="text-xs text-white font-black">{summary.subaward_count} ENTITIES</div>
                    </div>
                    <div className="text-[10px] text-gray-400">
                        Total pass-through volume: <span className="text-blue-400 font-bold font-mono">${summary.total_subaward_amount?.toLocaleString()}</span>
                    </div>
                </div>
            )}

            <div className="text-[9px] text-gray-600 text-center italic mt-4">
                Powered by USAspending.gov Real-Time API
            </div>
        </div>
    )
}
