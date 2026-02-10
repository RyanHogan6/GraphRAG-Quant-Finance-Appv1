'use client'

import { useState, useMemo } from 'react'
import TimeSeriesChart from './TimeSeriesChart'
import { motion, AnimatePresence } from 'framer-motion'
import SECFilingsExplorer from './SECFilingsExplorer'
import SECDocumentViewer from './SECDocumentViewer'
import SectorComparison from './SectorComparison'
import SentimentIndicators from './company/SentimentIndicators'
import PerformanceSummaryCard from './PerformanceSummaryCard'
import { secFilingDocumentUrl, secCompanyUrl } from '@/lib/secUrls'
import type { Key } from 'react'

const API_BASE = typeof process !== 'undefined' ? (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') : 'http://localhost:8000'

interface CompanyWorkupProps {
    data: any
    onCompare?: (ticker: string) => void
    peerData?: any  // Peer company data for comparison
    comparisonMode?: boolean  // Enable comparison view
}

export default function CompanyWorkup({ data, onCompare, peerData, comparisonMode = false }: CompanyWorkupProps) {
    const [timeframe, setTimeframe] = useState<'1W' | '1M' | '3M' | '6M' | '1Y' | '5Y' | 'ALL'>('1M')
    const [showAllMetrics, setShowAllMetrics] = useState(false)
    const [selectedDetail, setSelectedDetail] = useState<{ type: 'SEC' | 'Award', data: any } | null>(null)
    const [selectedFormType, setSelectedFormType] = useState<string>('all')
    const [secSortBy, setSecSortBy] = useState<'negative' | 'positive' | 'recent'>('negative')
    const [selectedXbrlIndex, setSelectedXbrlIndex] = useState(0)
    const [selectedStatementTab, setSelectedStatementTab] = useState<'income' | 'balance' | 'cashflow'>('income')
    const [showMoreSecSentences, setShowMoreSecSentences] = useState(false)
    const [secSentenceSearch, setSecSentenceSearch] = useState('')
    const [secSentenceSentiment, setSecSentenceSentiment] = useState<'all' | 'bullish' | 'bearish' | 'neutral'>('all')
    const [showMoreSecModalExcerpts, setShowMoreSecModalExcerpts] = useState(false)
    const [expandedExcerptIndex, setExpandedExcerptIndex] = useState<number | null>(null)
    const [showFullAmounts, setShowFullAmounts] = useState(true)
    const [sectorPeers, setSectorPeers] = useState<any[] | null>(null)
    const [sectorPeersLoading, setSectorPeersLoading] = useState(false)
    const [sectorPeersError, setSectorPeersError] = useState<string | null>(null)
    const [sectorSectionExpanded, setSectorSectionExpanded] = useState(true)
    const closeDetail = () => {
        setSelectedDetail(null)
        setShowMoreSecModalExcerpts(false)
        setExpandedExcerptIndex(null)
    }
    const openSecDetail = (filing: any) => {
        setShowMoreSecModalExcerpts(false)
        setExpandedExcerptIndex(null)
        setSelectedDetail({ type: 'SEC', data: filing })
    }

    const loadSectorPeers = async () => {
        const sector = data.sector
        if (!sector) return
        setSectorPeersLoading(true)
        setSectorPeers(null)
        setSectorPeersError(null)
        try {
            const res = await fetch(`${API_BASE}/api/query/companies/by-sector?${new URLSearchParams({ sector })}`)
            if (!res.ok) throw new Error(res.statusText)
            const json = await res.json()
            const list = json?.companies ?? []
            setSectorPeers(Array.isArray(list) ? list : [])
        } catch (e) {
            console.error('[CompanyWorkup] Sector peers fetch failed:', e)
            setSectorPeers([])
            setSectorPeersError('Failed to load sector peers.')
        } finally {
            setSectorPeersLoading(false)
        }
    }

    // Extract nested data - handle both Company-centric and Filing-centric queries - handle both Company-centric and Filing-centric queries
    const company = data

    // If query returned SEC filings (filing-centric), flatten the nested data
    const isFiling = data.type || data.filing_date  // Detect if root is a filing

    const marketData = data.MarketData || []
    const allSecFilings = data.sec_filings || (isFiling ? [data] : [])
    const secExhibits = data.sec_exhibits || (isFiling && data.sec_exhibits_data ? data.sec_exhibits_data : [])
    const secXbrlData = data.sec_xbrl_data || (isFiling && data.sec_xbrl_data_data ? data.sec_xbrl_data_data : [])
    const secSentences = data.sec_sentences || (isFiling && data.sec_sentences_data ? data.sec_sentences_data : [])
    const secSections = data.sec_sections || (isFiling && data.sec_sections_data ? data.sec_sections_data : [])
    const polyMarkets = data.prediction_markets_polymarket || []
    const awards = data.Award || []
    const optionsFlow = data.options_flow || (isFiling && data.options_flow_data ? data.options_flow_data : [])
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

    // Filter SEC sentences by keyword and sentiment (discoverable 4.5M-backed list)
    const filteredSecSentences = useMemo(() => {
        let list = secSentences as any[]
        const q = (secSentenceSearch || '').trim().toLowerCase()
        if (q) list = list.filter((s: any) => (s.text || '').toLowerCase().includes(q))
        if (secSentenceSentiment !== 'all') {
            list = list.filter((s: any) => {
                const score = s.finbert_score ?? 0
                const bucket = score > 0.2 ? 'bullish' : score < -0.2 ? 'bearish' : 'neutral'
                return bucket === secSentenceSentiment
            })
        }
        return list
    }, [secSentences, secSentenceSearch, secSentenceSentiment])

    // Get unique form types
    const formTypes = useMemo(() => {
        const types = new Set<string>(allSecFilings.map((f: any) => f.type || f.form_type).filter(Boolean))
        return ['all', ...Array.from(types).sort()] as string[]
    }, [allSecFilings])

    const latestMarket = marketData[0] || {}

    // Prepare chart data based on timeframe
    const chartData = useMemo(() => {
        // Helper to filter and sort market data by timeframe
        // Sanitize price series: cap single-point spikes (likely bad data) so chart is readable
        const sanitizeCloseValues = (values: number[]): number[] => {
            if (values.length <= 1) return values
            const out = [...values]
            const median = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] || 0
            const maxReasonable = Math.max(median * 3, 1000) // cap at 3x median or 1000
            for (let i = 1; i < out.length; i++) {
                const prev = out[i - 1]
                const curr = out[i]
                if (curr == null || curr <= 0) {
                    out[i] = prev
                    continue
                }
                // Single-day move > 100% or value way above rest of series → treat as bad data
                if (prev > 0 && (curr > prev * 2 || curr > maxReasonable)) {
                    out[i] = prev
                }
            }
            return out
        }

        const prepareSeriesData = (data: any[], ticker: string, color: string) => {
            let filtered = [...data]

            // Filter by timeframe
            const now = new Date()
            const filterDate = new Date()
            if (timeframe === '1W') filterDate.setDate(now.getDate() - 7)
            else if (timeframe === '1M') filterDate.setMonth(now.getMonth() - 1)
            else if (timeframe === '3M') filterDate.setMonth(now.getMonth() - 3)
            else if (timeframe === '6M') filterDate.setMonth(now.getMonth() - 6)
            else if (timeframe === '1Y') filterDate.setFullYear(now.getFullYear() - 1)
            else if (timeframe === '5Y') filterDate.setFullYear(now.getFullYear() - 5)
            else if (timeframe === 'ALL') filterDate.setFullYear(1900)  // Show all data

            filtered = filtered.filter(d => new Date(d.date) >= filterDate)
            const sorted = filtered.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
            const rawValues = sorted.map(d => typeof d.close === 'number' ? d.close : Number(d.close) || 0)
            const values = sanitizeCloseValues(rawValues)
            const volumes = sorted.map(d => Number((d as any).volume) || 0)

            return {
                dates: sorted.map(d => d.date),
                values,
                volumes,
                label: ticker,
                color,
                ticker
            }
        }

        // Primary company data
        const primarySeries = prepareSeriesData(marketData, company.ticker, '#D4AF37')

        // If in comparison mode with peer data, create multi-series array
        if (comparisonMode && peerData?.MarketData) {
            const peerSeries = prepareSeriesData(peerData.MarketData, peerData.ticker, '#3B82F6')
            return {
                series: [primarySeries, peerSeries],
                // Legacy fields for backward compatibility
                dates: primarySeries.dates,
                values: primarySeries.values,
                ticker: company.ticker
            }
        }

        // Single company mode - return legacy format
        return {
            dates: primarySeries.dates,
            values: primarySeries.values,
            volumes: primarySeries.volumes,
            ticker: company.ticker
        }
    }, [marketData, company.ticker, timeframe, comparisonMode, peerData])

    // AI Intelligence Summary (4 sentences) - Enriched with Latest Search
    const aiSummary = useMemo(() => {
        const ticker = company.ticker
        const priceChangeNum = chartData.values.length > 1
            ? ((chartData.values[chartData.values.length - 1] - chartData.values[0]) / chartData.values[0] * 100)
            : 0
        const priceChange = priceChangeNum !== 0 ? priceChangeNum.toFixed(1) : 'N/A'

        const latestAwards = awards.length > 0 ? awards[0].award_amount_float : 0
        const sentiment = secFilings[0]?.avg_finbert != null
            ? (secFilings[0].avg_finbert > 0.05 ? 'Bullish' : secFilings[0].avg_finbert < -0.05 ? 'Bearish' : 'Neutral')
            : 'Neutral'

        // Real-time context from search for PLTR/2026
        const news = ticker === 'PLTR' ? {
            rev: '$4.2B - $6.3B',
            event: 'Q4 Earnings on Feb 2, 2026',
            driver: 'Artificial Intelligence Platform (AIP) adoption'
        } : null;

        const latestOptions = optionsFlow[0]
        const putCallRatio = latestOptions?.put_call_ratio ?? latestOptions?.put_call_volume_ratio ?? 0
        const optionsSignal = putCallRatio > 1.5 ? 'bearish positioning with elevated put activity' : putCallRatio < 0.5 ? 'bullish positioning with strong call demand' : 'neutral options flow'

        const avgRsi = latestMarket?.rsi || 50
        const technicalSignal = avgRsi > 70 ? 'overbought territory (RSI: ' + avgRsi.toFixed(0) + ')' : avgRsi < 30 ? 'oversold territory (RSI: ' + avgRsi.toFixed(0) + ')' : 'balanced momentum (RSI: ' + avgRsi.toFixed(0) + ')'

        const totalAwardValue = awards.reduce((sum: number, a: any) => sum + (a.award_amount_float || 0), 0)

        // Calculate market cap from available data
        const marketCapB = company.marketCap
            ? (company.marketCap / 1e9).toFixed(2)
            : (latestMarket?.close && latestMarket?.sharesOutstanding)
                ? ((latestMarket.close * latestMarket.sharesOutstanding) / 1e9).toFixed(2)
                : 'N/A'

        // Best 4 points only; one short sentence per bullet
        const points = [
            `Price: ${priceChange}% over ${timeframe}; $${latestMarket?.close?.toFixed(2)}; market cap ${marketCapB !== 'N/A' ? '$' + marketCapB + 'B' : 'N/A'}. ${company.sector}.`,
            `SEC: ${sentiment} (FinBERT ${secFilings[0]?.avg_finbert?.toFixed(2) ?? 'N/A'}); ${secFilings.length} filings.`,
            awards.length > 0
                ? `Contracts: $${(totalAwardValue / 1e6).toFixed(1)}M across ${awards.length} awards; recent ${awards[0]?.awarding_agency || 'DoD'} $${(awards[0]?.award_amount_float / 1e6).toFixed(1)}M.`
                : `${company.company}: ${company.sector} exposure; commercial focus.`,
            optionsFlow.length > 0
                ? `Options: P/C ${putCallRatio.toFixed(2)}, IV ${((latestOptions?.implied_volatility ?? latestOptions?.iv_avg) != null ? (Number(latestOptions.implied_volatility ?? latestOptions.iv_avg) * 100).toFixed(1) : 'N/A')}%; ${putCallRatio > 1.5 ? 'bearish' : putCallRatio < 0.5 ? 'bullish' : 'neutral'}.`
                : `Options: no flow data yet for ${ticker}.`,
            `Technical: ${technicalSignal}. ${latestMarket?.sma_50 && latestMarket?.sma_200 ? (latestMarket.sma_50 > latestMarket.sma_200 ? 'Golden cross.' : 'Death cross.') : ''}`,
            polyMarkets.length > 0
                ? `Prediction: ${(polyMarkets[0].yes_probability * 100).toFixed(0)}% — ${polyMarkets[0].question?.slice(0, 50)}…`
                : news ? `Catalyst: ${news.event}. ${news.driver}.` : null
        ].filter(Boolean)
        return points.slice(0, 4)
    }, [company, timeframe, chartData, awards, secFilings, polyMarkets, secXbrlData, secExhibits])

    // Indicator-style synthesis cards (same data as aiSummary, compact)
    const synthesisIndicators = useMemo(() => {
        const ticker = company.ticker
        const priceChangeNum = chartData.values.length > 1
            ? ((chartData.values[chartData.values.length - 1] - chartData.values[0]) / chartData.values[0] * 100)
            : 0
        const priceChange = priceChangeNum !== 0 ? priceChangeNum.toFixed(1) : 'N/A'
        const sentiment = secFilings[0]?.avg_finbert != null
            ? (secFilings[0].avg_finbert > 0.05 ? 'Bullish' : secFilings[0].avg_finbert < -0.05 ? 'Bearish' : 'Neutral')
            : 'Neutral'
        const latestOptions = optionsFlow[0]
        const putCallRatio = latestOptions?.put_call_ratio ?? latestOptions?.put_call_volume_ratio ?? 0
        const totalAwardValue = awards.reduce((sum: number, a: any) => sum + (a.award_amount_float || 0), 0)
        const marketCapB = company.marketCap
            ? (company.marketCap / 1e9).toFixed(2)
            : (latestMarket?.close && latestMarket?.sharesOutstanding)
                ? ((latestMarket.close * latestMarket.sharesOutstanding) / 1e9).toFixed(2)
                : 'N/A'
        const avgRsi = latestMarket?.rsi || 50
        return [
            { title: 'Price', value: `${priceChange}% · $${latestMarket?.close?.toFixed(2) ?? '—'} · $${marketCapB}B cap`, secChip: false, awardChip: false },
            { title: 'SEC', value: `${sentiment} · ${secFilings.length} filings`, secChip: true, awardChip: false },
            { title: awards.length > 0 ? 'Contracts' : 'Options', value: awards.length > 0 ? `$${(totalAwardValue / 1e6).toFixed(1)}M · ${awards.length} awards` : (optionsFlow.length > 0 ? `P/C ${Number(putCallRatio).toFixed(2)}` : '—'), secChip: false, awardChip: awards.length > 0 },
            { title: 'Technical', value: `RSI ${avgRsi.toFixed(0)}${latestMarket?.sma_50 && latestMarket?.sma_200 ? (latestMarket.sma_50 > latestMarket.sma_200 ? ' · Golden cross' : ' · Death cross') : ''}`, secChip: false, awardChip: false }
        ]
    }, [company, chartData, awards, secFilings, optionsFlow, latestMarket])

    // Moneycontain "13 Essential Financial Metrics"
    const fundamentalMetrics = useMemo(() => {
        const calculateMetrics = (companyData: any, marketData: any, xbrlData: any[]) => {
            // Merge company and market data
            const all = { ...companyData, ...marketData }

            // Try both camelCase and snake_case field names for compatibility
            const getValue = (camelCase: string, snakeCase: string) => {
                return all[camelCase] ?? all[snakeCase] ?? null
            }

            // Get latest XBRL data
            const latestXbrl = xbrlData?.[0] || null
            const prevXbrl = xbrlData?.[1] || null

            /** Resolve a numeric value from XBRL doc: buckets (costs/debt/equity/cashflow) or all_concepts (may be array of {value, context}) */
            const conceptVal = (x: any, name: string): number | null => {
                if (!x) return null
                const fromBucket = x.costs?.[name] ?? x.debt?.[name] ?? x.equity?.[name] ?? x.cashflow?.[name]
                if (typeof fromBucket === 'number') return fromBucket
                const fromAll = x.all_concepts?.[name]
                if (typeof fromAll === 'number') return fromAll
                if (Array.isArray(fromAll) && fromAll.length) {
                    const v = fromAll[0]?.value
                    return typeof v === 'number' ? v : null
                }
                return null
            }

            // Extract XBRL financials (use conceptVal so all_concepts array shape is handled)
            const revRaw = conceptVal(latestXbrl, 'Revenues') ?? conceptVal(latestXbrl, 'RevenueFromContractWithCustomerExcludingAssessedTax') ?? Object.values(latestXbrl?.revenue_segments || {})[0] ?? null
            const prevRevRaw = conceptVal(prevXbrl, 'Revenues') ?? conceptVal(prevXbrl, 'RevenueFromContractWithCustomerExcludingAssessedTax') ?? Object.values(prevXbrl?.revenue_segments || {})[0] ?? null
            const xbrlRevenue: number | null = typeof revRaw === 'number' ? revRaw : null
            const xbrlPrevRevenue: number | null = typeof prevRevRaw === 'number' ? prevRevRaw : null
            const xbrlNetIncome = conceptVal(latestXbrl, 'NetIncomeLoss')
            const xbrlEquity = conceptVal(latestXbrl, 'StockholdersEquity') ?? conceptVal(latestXbrl, 'Equity')
            const xbrlDebt = conceptVal(latestXbrl, 'LongTermDebt') ?? conceptVal(latestXbrl, 'DebtCurrent')
            const xbrlFreeCashflow = conceptVal(latestXbrl, 'NetCashProvidedByUsedInOperatingActivities')

            // Calculate fallbacks from XBRL
            const xbrlRevenueGrowth = (xbrlRevenue != null && xbrlPrevRevenue != null && xbrlPrevRevenue !== 0)
                ? (xbrlRevenue - xbrlPrevRevenue) / xbrlPrevRevenue
                : null
            const xbrlProfitMargin = (xbrlNetIncome != null && xbrlRevenue != null && xbrlRevenue !== 0)
                ? xbrlNetIncome / xbrlRevenue
                : null
            const xbrlROE = (xbrlNetIncome != null && xbrlEquity != null && xbrlEquity !== 0)
                ? xbrlNetIncome / xbrlEquity
                : null
            const xbrlDebtToEquity = (xbrlDebt != null && xbrlEquity != null && xbrlEquity !== 0)
                ? xbrlDebt / xbrlEquity
                : null

            // Build metrics list with robust fallbacks
            const metricsList = [
                {
                    name: 'Revenue Growth',
                    val: getValue('revenueGrowth', 'revenue_growth') ?? xbrlRevenueGrowth,
                    benchmark: '> 10%', type: 'pct', check: (v: number) => v > 0.1
                },
                {
                    name: 'EBITDA Margin',
                    val: getValue('ebitdaMargins', 'ebitda_margins'),
                    benchmark: '> 15%', type: 'pct', check: (v: number) => v > 0.15
                },
                {
                    name: 'PAT Margin',
                    val: getValue('profitMargins', 'profit_margins') ?? xbrlProfitMargin,
                    benchmark: '> 10%', type: 'pct', check: (v: number) => v > 0.1
                },
                {
                    name: 'ROE',
                    val: getValue('returnOnEquity', 'return_on_equity') ?? xbrlROE,
                    benchmark: '> 15%', type: 'pct', check: (v: number) => v > 0.15
                },
                {
                    name: 'ROA',
                    val: getValue('returnOnAssets', 'return_on_assets'),
                    benchmark: '> 7%', type: 'pct', check: (v: number) => v > 0.07
                },
                {
                    name: 'Debt-to-Equity',
                    val: getValue('debtToEquity', 'debt_to_equity') ?? xbrlDebtToEquity,
                    benchmark: '< 1', type: 'ratio', check: (v: number) => v < 1
                },
                {
                    name: 'Current Ratio',
                    val: getValue('currentRatio', 'current_ratio'),
                    benchmark: '> 1.5', type: 'ratio', check: (v: number) => v > 1.5
                },
                {
                    name: 'Free Cash Flow',
                    val: getValue('freeCashflow', 'free_cashflow') ?? xbrlFreeCashflow,
                    benchmark: 'Positive', type: 'currency', check: (v: number) => v > 0
                },
                {
                    name: 'EPS',
                    val: getValue('trailingEps', 'trailing_eps') || getValue('epsTrailingTwelveMonths', 'eps_trailing_twelve_months') || getValue('forwardEps', 'forward_eps'),
                    benchmark: 'Growing', type: 'number'
                },
                {
                    name: 'P/E Ratio',
                    val: getValue('trailingPE', 'trailing_pe') || getValue('forwardPE', 'forward_pe'),
                    benchmark: '< 20 (Fair)', type: 'number', check: (v: number) => v < 20
                },
                {
                    name: 'Operating Margin',
                    val: getValue('operatingMargins', 'operating_margins'),
                    benchmark: '> 12%', type: 'pct', check: (v: number) => v > 0.12
                },
                {
                    name: 'Gross Margin',
                    val: getValue('grossMargins', 'gross_margins'),
                    benchmark: '> 30%', type: 'pct', check: (v: number) => v > 0.30
                },
                {
                    name: 'Quick Ratio',
                    val: getValue('quickRatio', 'quick_ratio'),
                    benchmark: '> 1', type: 'ratio', check: (v: number) => v > 1
                }
            ]

            return metricsList.map(m => ({
                ...m,
                displayVal: formatVal(m.val, m.type),
                status: m.val != null && m.check ? (m.check(m.val) ? 'good' : 'bad') : 'neutral'
            }))
        }

        return calculateMetrics(company, latestMarket, secXbrlData)
    }, [company, latestMarket, secXbrlData, showFullAmounts])

    // Peer fundamental metrics (if in comparison mode)
    const peerFundamentalMetrics = useMemo(() => {
        if (!comparisonMode || !peerData) return null

        const peerLatestMarket = peerData.MarketData?.[0] || {}
        const peerXbrlData = peerData.sec_xbrl_data || []

        const calculateMetrics = (companyData: any, marketData: any, xbrlData: any[]) => {
            const all = { ...companyData, ...marketData }

            const getValue = (camelCase: string, snakeCase: string) => {
                return all[camelCase] ?? all[snakeCase] ?? null
            }

            const latestXbrl = xbrlData?.[0] || null
            const prevXbrl = xbrlData?.[1] || null

            const conceptVal = (x: any, name: string): number | null => {
                if (!x) return null
                const fromBucket = x.costs?.[name] ?? x.debt?.[name] ?? x.equity?.[name] ?? x.cashflow?.[name]
                if (typeof fromBucket === 'number') return fromBucket
                const fromAll = x.all_concepts?.[name]
                if (typeof fromAll === 'number') return fromAll
                if (Array.isArray(fromAll) && fromAll.length) {
                    const v = fromAll[0]?.value
                    return typeof v === 'number' ? v : null
                }
                return null
            }

            const revRaw = conceptVal(latestXbrl, 'Revenues') ?? conceptVal(latestXbrl, 'RevenueFromContractWithCustomerExcludingAssessedTax') ?? Object.values(latestXbrl?.revenue_segments || {})[0] ?? null
            const prevRevRaw = conceptVal(prevXbrl, 'Revenues') ?? conceptVal(prevXbrl, 'RevenueFromContractWithCustomerExcludingAssessedTax') ?? Object.values(prevXbrl?.revenue_segments || {})[0] ?? null
            const xbrlRevenue: number | null = typeof revRaw === 'number' ? revRaw : null
            const xbrlPrevRevenue: number | null = typeof prevRevRaw === 'number' ? prevRevRaw : null
            const xbrlNetIncome = conceptVal(latestXbrl, 'NetIncomeLoss')
            const xbrlEquity = conceptVal(latestXbrl, 'StockholdersEquity') ?? conceptVal(latestXbrl, 'Equity')
            const xbrlDebt = conceptVal(latestXbrl, 'LongTermDebt') ?? conceptVal(latestXbrl, 'DebtCurrent')
            const xbrlFreeCashflow = conceptVal(latestXbrl, 'NetCashProvidedByUsedInOperatingActivities')

            const xbrlRevenueGrowth = (xbrlRevenue != null && xbrlPrevRevenue != null && xbrlPrevRevenue !== 0)
                ? (xbrlRevenue - xbrlPrevRevenue) / xbrlPrevRevenue
                : null
            const xbrlProfitMargin = (xbrlNetIncome != null && xbrlRevenue != null && xbrlRevenue !== 0)
                ? xbrlNetIncome / xbrlRevenue
                : null
            const xbrlROE = (xbrlNetIncome != null && xbrlEquity != null && xbrlEquity !== 0)
                ? xbrlNetIncome / xbrlEquity
                : null
            const xbrlDebtToEquity = (xbrlDebt != null && xbrlEquity != null && xbrlEquity !== 0)
                ? xbrlDebt / xbrlEquity
                : null

            const metricsList = [
                { name: 'Revenue Growth', val: getValue('revenueGrowth', 'revenue_growth') ?? xbrlRevenueGrowth, type: 'pct', check: (v: number) => v > 0.1 },
                { name: 'EBITDA Margin', val: getValue('ebitdaMargins', 'ebitda_margins'), type: 'pct', check: (v: number) => v > 0.15 },
                { name: 'PAT Margin', val: getValue('profitMargins', 'profit_margins') ?? xbrlProfitMargin, type: 'pct', check: (v: number) => v > 0.1 },
                { name: 'ROE', val: getValue('returnOnEquity', 'return_on_equity') ?? xbrlROE, type: 'pct', check: (v: number) => v > 0.15 },
                { name: 'ROA', val: getValue('returnOnAssets', 'return_on_assets'), type: 'pct', check: (v: number) => v > 0.07 },
                { name: 'Debt-to-Equity', val: getValue('debtToEquity', 'debt_to_equity') ?? xbrlDebtToEquity, type: 'ratio', check: (v: number) => v < 1 },
                { name: 'Current Ratio', val: getValue('currentRatio', 'current_ratio'), type: 'ratio', check: (v: number) => v > 1.5 },
                { name: 'Free Cash Flow', val: getValue('freeCashflow', 'free_cashflow') ?? xbrlFreeCashflow, type: 'currency', check: (v: number) => v > 0 },
                { name: 'EPS', val: getValue('trailingEps', 'trailing_eps') || getValue('epsTrailingTwelveMonths', 'eps_trailing_twelve_months') || getValue('forwardEps', 'forward_eps'), type: 'number' },
                { name: 'P/E Ratio', val: getValue('trailingPE', 'trailing_pe') || getValue('forwardPE', 'forward_pe'), type: 'number', check: (v: number) => v < 20 },
                { name: 'Operating Margin', val: getValue('operatingMargins', 'operating_margins'), type: 'pct', check: (v: number) => v > 0.12 },
                { name: 'Gross Margin', val: getValue('grossMargins', 'gross_margins'), type: 'pct', check: (v: number) => v > 0.30 },
                { name: 'Quick Ratio', val: getValue('quickRatio', 'quick_ratio'), type: 'ratio', check: (v: number) => v > 1 }
            ]

            return metricsList.map(m => ({
                ...m,
                displayVal: formatVal(m.val, m.type),
                status: m.val != null && m.check ? (m.check(m.val) ? 'good' : 'bad') : 'neutral'
            }))
        }

        return calculateMetrics(peerData, peerLatestMarket, peerXbrlData)
    }, [peerData, comparisonMode, showFullAmounts])

    // Award KPIs for Federal Contract Awards section (Total Value, Count, Avg, Largest, FY Range, Top Agency)
    const awardKpis = useMemo(() => {
        if (!awards.length) return null
        const totalValue = awards.reduce((sum: number, a: any) => sum + (a.award_amount_float || 0), 0)
        const count = awards.length
        const amounts = awards.map((a: any) => a.award_amount_float || 0).filter((n: number) => n > 0)
        const largest = amounts.length ? Math.max(...amounts) : 0
        const years = awards.map((a: any) => a.contract_year).filter(Boolean)
        const fyMin = years.length ? Math.min(...years) : null
        const fyMax = years.length ? Math.max(...years) : null
        const fyRange = fyMin != null && fyMax != null ? `FY${fyMin} – FY${fyMax}` : '—'
        const byAgency = awards.reduce((acc: Record<string, { total: number; count: number }>, a: any) => {
            const ag = a.awarding_agency || 'Unknown'
            if (!acc[ag]) acc[ag] = { total: 0, count: 0 }
            acc[ag].total += a.award_amount_float || 0
            acc[ag].count += 1
            return acc
        }, {})
        type AgencyStat = { total: number; count: number }
        const topAgencyEntry = (Object.entries(byAgency) as [string, AgencyStat][]).sort((a, b) => b[1].total - a[1].total)[0]
        const topAgency = topAgencyEntry ? `${topAgencyEntry[0]} (${topAgencyEntry[1].count})` : '—'
        return {
            totalValue,
            count,
            avgAward: count ? totalValue / count : 0,
            largest,
            fyRange,
            topAgency
        }
    }, [awards])

    function formatVal(val: any, type: string) {
        if (val == null || isNaN(val)) return 'N/A'
        if (type === 'currency') {
            if (showFullAmounts) return `$${Number(val).toLocaleString('en-US', { maximumFractionDigits: 0 })}`
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
                            {company.ticker || data.ticker || '—'}
                        </span>
                    </div>
                    <p className="text-sm text-gray-400">
                        {[company.sector || data.sector, company.industry || data.industry, (company.city || company.country) ? [company.city, company.country].filter(Boolean).join(', ') : null].filter(Boolean).join(' | ') || '—'}
                    </p>
                </div>
                <div className="flex gap-2">
                    {comparisonMode && peerData && (
                        <button
                            onClick={() => onCompare?.(company.ticker)}
                            className="px-4 py-2 bg-red-900/20 border border-red-500/30 rounded-lg text-xs text-red-400 hover:bg-red-900/30 transition-all flex items-center gap-2"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                            Exit Comparison
                        </button>
                    )}
                </div>
            </div>

            {/* Stack up in sector - compare to sector peers (collapsible) */}
            {(company.sector || data.sector) && (
                <div className="rounded-lg border border-gold/20 bg-gold/5 px-4 py-3">
                    <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Sector</span>
                            <span className="text-xs text-gold">{company.sector || data.sector}</span>
                        </div>
                        <button
                            type="button"
                            onClick={() => setSectorSectionExpanded(!sectorSectionExpanded)}
                            className="text-[10px] font-semibold text-gold hover:text-gold/80 border border-gold/30 rounded px-2 py-1 transition-all"
                        >
                            {sectorSectionExpanded ? 'Collapse' : 'Expand'}
                        </button>
                    </div>
                    {sectorSectionExpanded && (
                        <>
                            <div className="mt-2 flex items-center gap-2">
                                <button
                                    type="button"
                                    onClick={loadSectorPeers}
                                    disabled={sectorPeersLoading}
                                    className="text-xs font-semibold text-gold hover:text-gold/80 border border-gold/30 rounded-lg px-3 py-1.5 bg-gold/10 hover:bg-gold/20 transition-all disabled:opacity-50"
                                >
                                    {sectorPeersLoading ? 'Loading…' : 'Compare to sector peers'}
                                </button>
                                {onCompare && (company.ticker || data.ticker) && (
                                    <button type="button" onClick={() => onCompare?.(company.ticker || data.ticker)} className="text-[11px] text-amber-400 hover:text-amber-300">
                                        Compare to another company →
                                    </button>
                                )}
                            </div>
                            {sectorPeers && sectorPeers.length > 0 && (
                                <div className="mt-4">
                                    <SectorComparison companies={sectorPeers} title={`${company.sector || data.sector} peers`} />
                                </div>
                            )}
                            {sectorPeersError && <p className="mt-2 text-[11px] text-red-400">{sectorPeersError}</p>}
                            {sectorPeers && sectorPeers.length === 0 && !sectorPeersLoading && !sectorPeersError && (
                                <p className="mt-2 text-[11px] text-gray-500">No sector peers returned. Try again.</p>
                            )}
                        </>
                    )}
                </div>
            )}

            {/* 3 Sentiment Indicators - Quick Insights */}
            <SentimentIndicators
                marketData={marketData}
                secFilings={allSecFilings}
                optionsFlow={optionsFlow}
                xbrlData={secXbrlData}
            />

            {/* 13 Point Fundamental Checklist - above chart */}
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
                        {comparisonMode && peerFundamentalMetrics ? (
                            <div className="p-3 md:p-4 space-y-2">
                                {fundamentalMetrics.slice(0, showAllMetrics ? undefined : 8).map((m: any, i: number) => {
                                    const peerMetric = peerFundamentalMetrics[i]
                                    const primaryBetter = m.val != null && peerMetric.val != null && (
                                        (m.type === 'ratio' && m.name.includes('Debt')) ? m.val < peerMetric.val : m.val > peerMetric.val
                                    )
                                    const peerBetter = m.val != null && peerMetric.val != null && (
                                        (m.type === 'ratio' && m.name.includes('Debt')) ? peerMetric.val < m.val : peerMetric.val > m.val
                                    )

                                    return (
                                        <div key={i} className="grid grid-cols-7 gap-2 items-center border-b border-white/5 pb-2 hover:bg-white/5 transition-all px-2 rounded">
                                            <div className="col-span-2 text-[9px] text-gray-400 uppercase font-bold">{m.name}</div>
                                            <div className={`col-span-2 text-right font-mono font-bold text-base ${
                                                m.displayVal === 'N/A' ? 'text-gray-600' :
                                                primaryBetter ? 'text-gold drop-shadow-[0_0_8px_rgba(212,175,55,0.5)]' : 'text-gray-300'
                                            }`}>
                                                {m.displayVal}
                                                {primaryBetter && m.displayVal !== 'N/A' && <span className="ml-1 text-[10px] text-gold/50">✓</span>}
                                            </div>
                                            <div className="col-span-1 text-center text-[9px] text-gray-600">vs</div>
                                            <div className={`col-span-2 text-left font-mono font-bold text-base ${
                                                peerMetric.displayVal === 'N/A' ? 'text-gray-600' :
                                                peerBetter ? 'text-blue-400 drop-shadow-[0_0_8px_rgba(96,165,250,0.5)]' : 'text-gray-300'
                                            }`}>
                                                {peerMetric.displayVal}
                                                {peerBetter && peerMetric.displayVal !== 'N/A' && <span className="ml-1 text-[10px] text-blue-400/50">✓</span>}
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        ) : (
                            <div className="p-3 md:p-4 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-y-4 gap-x-3 md:gap-y-5 md:gap-x-4">
                                {fundamentalMetrics.slice(0, showAllMetrics ? undefined : 8).map((m: any, i: number) => (
                                    <div key={i} className="group border-l border-white/5 pl-3 md:pl-4 hover:border-gold/30 transition-all">
                                        <div className="flex items-center justify-between mb-1">
                                            <div className="text-[8px] md:text-[9px] text-gray-500 uppercase font-black tracking-tighter group-hover:text-gold transition-colors">{m.name}</div>
                                            <div className="text-[7px] md:text-[8px] text-gray-600 font-mono tracking-tighter hidden sm:block">Ref: {m.benchmark}</div>
                                        </div>
                                        <div className={`text-lg md:text-xl font-mono font-black tracking-tight ${
                                            m.displayVal === 'N/A'
                                                ? 'text-gray-600'
                                                : m.status === 'good' ? 'text-green-400 drop-shadow-[0_0_8px_rgba(74,222,128,0.5)]'
                                                : m.status === 'bad' ? 'text-red-400 drop-shadow-[0_0_8px_rgba(248,113,113,0.5)]'
                                                : 'text-gray-100 drop-shadow-[0_0_6px_rgba(255,255,255,0.3)]'
                                        }`}>
                                            {m.displayVal}
                                            {m.status !== 'neutral' && m.displayVal !== 'N/A' && (
                                                <span className={`ml-1 text-[10px] md:text-xs ${m.status === 'good' ? 'text-green-400/70' : 'text-red-400/70'}`}>
                                                    {m.status === 'good' ? '▲' : '▼'}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

            {/* Main Content Sections - Full Width Layout */}
            <div className="space-y-4">

                {/* Performance Summary Card - first section */}
                {!comparisonMode && chartData.values.length > 0 && (
                    <PerformanceSummaryCard
                        dates={chartData.dates}
                        values={chartData.values}
                        volumes={(chartData as any).volumes}
                        ticker={company.ticker}
                    />
                )}

                {/* Chart Section - Full Width */}
                <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-3 md:p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-3 md:mb-4 gap-3">
                            <div>
                                <h3 className="text-[10px] md:text-xs font-bold text-gold uppercase tracking-widest mb-1">Market Performance Hub</h3>
                                <div className="text-[9px] md:text-[10px] text-gray-500 font-mono italic">Structural Momentum Analysis</div>
                            </div>
                            <div className="flex gap-2">
                                <div className="flex bg-dark-800 rounded-xl p-0.5 border border-white/10 shadow-inner">
                                    {['1W', '1M', '3M', '6M', '1Y', '5Y', 'ALL'].map(tf => (
                                        <button
                                            key={tf}
                                            onClick={() => setTimeframe(tf as any)}
                                            className={`px-3 py-1 md:px-4 md:py-1.5 text-[9px] md:text-[10px] rounded-lg transition-all ${timeframe === tf ? 'bg-gold text-dark-900 font-bold shadow-lg' : 'text-gray-500 hover:text-gray-300'}`}
                                        >
                                            {tf}
                                        </button>
                                    ))}
                                </div>
                                {comparisonMode && peerData && (
                                    <div className="flex items-center gap-2 px-3 py-1 bg-dark-800 rounded-xl border border-blue-500/30">
                                        <span className="text-[9px] text-gray-400">vs</span>
                                        <span className="text-[10px] font-bold text-blue-400">{peerData.ticker}</span>
                                    </div>
                                )}
                            </div>
                        </div>
                        {chartData.values.length > 0 ? (
                            <div className="h-[280px] w-full mt-3">
                                <TimeSeriesChart
                                    series={chartData.series}
                                    dates={chartData.dates}
                                    values={chartData.values}
                                    label={comparisonMode ? 'Peer Comparison' : `${company.ticker} Structural Momentum`}
                                    ticker={company.ticker}
                                />
                            </div>
                        ) : (
                            <div className="h-[280px] flex items-center justify-center text-gray-600 text-sm border border-dashed border-white/10 rounded-xl bg-dark-900/20">
                                No historical price action in dataset
                            </div>
                        )}
                    </div>

                {/* Deep Intelligence Synthesis - paragraph-style analysis below chart */}
                <div className="bg-dark-900/40 border border-gold/20 rounded-xl p-3 relative overflow-hidden">
                    <h3 className="text-[9px] md:text-[10px] font-bold text-gold uppercase tracking-[0.2em] mb-3 flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-gold" />
                        Deep Intelligence Synthesis
                    </h3>
                    <div className="space-y-2.5 text-xs md:text-sm text-gray-300 leading-relaxed">
                        {aiSummary.map((line, i) => (
                            <div key={i} className="flex flex-wrap items-baseline gap-1.5">
                                <span className="w-1.5 h-1.5 rounded-full bg-gold shrink-0 mt-1.5" />
                                <span className="flex-1">{line}</span>
                                {i === 1 && secFilings.length > 0 && (
                                    <span onClick={() => openSecDetail(secFilings[0])} className="text-[10px] text-blue-400 cursor-pointer hover:underline bg-blue-500/10 px-1 rounded shrink-0">[SEC]</span>
                                )}
                                {i === 2 && awards.length > 0 && (
                                    <span onClick={() => setSelectedDetail({ type: 'Award', data: awards[0] })} className="text-[10px] text-gold cursor-pointer hover:underline bg-gold/10 px-1 rounded shrink-0">[Award]</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

            </div>

            {/* Options Activity - tracker-style summary + last 50 with scroll */}
            {optionsFlow.length > 0 && (() => {
                const latest = optionsFlow[0]
                // Window averages for summary bar (not just latest day)
                const n = optionsFlow.length
                const sumPut = optionsFlow.reduce((s: number, r: any) => s + (r.put_volume ?? 0), 0)
                const sumCall = optionsFlow.reduce((s: number, r: any) => s + (r.call_volume ?? 0), 0)
                const avgPc = sumCall > 0 ? sumPut / sumCall : (latest.put_call_ratio ?? latest.put_call_volume_ratio)
                const pc = avgPc != null ? avgPc : (latest.put_call_ratio ?? latest.put_call_volume_ratio)
                const stance = pc != null ? (pc > 1 ? 'Bearish' : pc < 0.7 ? 'Bullish' : 'Neutral') : null
                const putVol = Math.round(sumPut / n)
                const callVol = Math.round(sumCall / n)
                const sumPutPrem = optionsFlow.reduce((s: number, r: any) => s + (Number(r.put_premium) || 0), 0)
                const sumCallPrem = optionsFlow.reduce((s: number, r: any) => s + (Number(r.call_premium) || 0), 0)
                const putPrem = sumPutPrem
                const callPrem = sumCallPrem
                const ivVals = optionsFlow.map((r: any) => r.implied_volatility ?? r.iv_avg).filter((v: any) => v != null)
                const avgIv = ivVals.length ? ivVals.reduce((a: number, b: any) => a + Number(b), 0) / ivVals.length : (latest.implied_volatility ?? latest.iv_avg)
                const fmtNum = (n: number) => n.toLocaleString('en-US', { maximumFractionDigits: 0 })
                const fmtDollars = (n: number) => n >= 1e9 ? `$${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `$${(n / 1e6).toFixed(2)}M` : `$${fmtNum(n)}`
                return (
                <div className="mt-4">
                    <div className="bg-dark-900/40 border border-green-500/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-sm font-bold text-green-400 uppercase tracking-[0.2em] flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
                                Options Activity
                            </h3>
                            <span className="text-[10px] text-gray-500 font-mono">
                                {stance ? `${stance} · P/C ${Number(pc).toFixed(2)}` : `Last ${optionsFlow.length} days`}
                            </span>
                        </div>
                        {optionsFlow.length < 7 && (
                            <p className="text-[10px] text-amber-400/90 mb-3 italic">Limited history ({optionsFlow.length} days). Averages will stabilize with more data.</p>
                        )}
                        {/* Summary bar - window averages */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4 p-3 rounded-lg bg-dark-800/50 border border-green-500/10">
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider">Put/Call (avg)</div>
                                <div className="font-mono text-white font-semibold">{pc != null ? Number(pc).toFixed(2) : '—'}</div>
                                {stance && <div className={`text-[10px] ${stance === 'Bullish' ? 'text-green-400' : stance === 'Bearish' ? 'text-red-400' : 'text-gray-400'}`}>{stance}</div>}
                            </div>
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider">Puts (avg)</div>
                                <div className="font-mono text-red-300">{fmtNum(putVol)} contracts</div>
                                {putPrem > 0 && <div className="font-mono text-[10px] text-gray-400">{fmtDollars(putPrem)}</div>}
                            </div>
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider">Calls (avg)</div>
                                <div className="font-mono text-green-300">{fmtNum(callVol)} contracts</div>
                                {callPrem > 0 && <div className="font-mono text-[10px] text-gray-400">{fmtDollars(callPrem)}</div>}
                            </div>
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider">IV (avg)</div>
                                <div className="font-mono text-white">{avgIv != null ? (Number(avgIv) * 100).toFixed(1) + '%' : '—'}</div>
                            </div>
                        </div>
                        <div className="max-h-[400px] overflow-y-auto overflow-x-auto border border-green-500/20 rounded-lg">
                            <table className="w-full text-xs border-collapse">
                                <thead className="sticky top-0 bg-dark-900 z-10 border-b-2 border-green-500/30">
                                    <tr>
                                        <th className="text-left text-gray-300 font-semibold pb-2 pt-2 px-2">Date</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 pt-2 px-2">P/C Ratio</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 pt-2 px-2">Call Vol</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 pt-2 px-2">Put Vol</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 pt-2 px-2">Total Vol</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 pt-2 px-2">Call $</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 pt-2 px-2">Put $</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 pt-2 px-2">IV</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {optionsFlow.map((row: any, i: number) => {
                                        const rowPc = row.put_call_ratio ?? row.put_call_volume_ratio
                                        const pcNum = rowPc != null ? Number(rowPc) : null
                                        const pcClass = pcNum != null ? (pcNum > 1 ? 'text-red-400 font-semibold' : pcNum < 0.7 ? 'text-green-400 font-semibold' : 'text-gray-300') : 'text-gray-200'
                                        const iv = row.implied_volatility ?? row.iv_avg
                                        const rowCallPrem = row.call_premium != null ? Number(row.call_premium) : null
                                        const rowPutPrem = row.put_premium != null ? Number(row.put_premium) : null
                                        const stripe = i % 2 === 1 ? 'bg-dark-800/40' : ''
                                        return (
                                        <tr key={i} className={`border-b border-white/10 hover:bg-green-500/10 transition-colors ${stripe}`}>
                                            <td className="py-2 px-2 text-gray-300 font-mono">{row.date || '—'}</td>
                                            <td className={`py-2 px-2 text-right font-mono ${pcClass}`}>{rowPc != null ? Number(rowPc).toFixed(2) : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-green-300/90">{row.call_volume != null ? row.call_volume.toLocaleString() : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-red-300/90">{row.put_volume != null ? row.put_volume.toLocaleString() : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-gray-200">{row.total_volume != null ? row.total_volume.toLocaleString() : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-green-300/90">{rowCallPrem != null ? rowCallPrem.toLocaleString('en-US', { maximumFractionDigits: 0 }) : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-red-300/90">{rowPutPrem != null ? rowPutPrem.toLocaleString('en-US', { maximumFractionDigits: 0 }) : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-gray-200">{iv != null ? (Number(iv) * 100).toFixed(1) + '%' : '—'}</td>
                                        </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                        {optionsFlow.length <= 3 && optionsFlow.length > 0 && (
                            <p className="mt-2 text-[10px] text-gray-500">Showing available history ({optionsFlow.length} days). More data will appear as the pipeline runs daily.</p>
                        )}
                    </div>
                </div>
                )
            })()}

            {/* Full Financial Statements Viewer (first in deep section) */}
            {secXbrlData.length > 0 && (
                <div className="mt-4">
                    <div className="bg-gradient-to-br from-emerald-900/20 via-dark-900/40 to-blue-900/20 border border-emerald-500/20 rounded-xl shadow-2xl backdrop-blur-sm overflow-hidden">
                        {/* Header */}
                        <div className="p-4 border-b border-emerald-500/10 bg-dark-800/50">
                            <div className="flex items-center justify-between mb-3">
                                <h3 className="text-sm font-bold text-emerald-400 uppercase tracking-[0.2em] flex items-center gap-3">
                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                                    Financial Statements
                                </h3>
                                <div className="text-xs text-gray-500">
                                    XBRL Data
                                </div>
                            </div>

                            {/* Filing years - period selector */}
                            <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-2">Filing years</div>
                            <div className="flex items-center gap-2 overflow-x-auto pb-1">
                                {secXbrlData.map((xbrl: any, i: number) => (
                                    <button
                                        key={i}
                                        onClick={() => setSelectedXbrlIndex(i)}
                                        className={`px-3 py-2 rounded-lg text-xs font-semibold text-left transition-all flex flex-col items-start min-w-0 ${
                                            selectedXbrlIndex === i
                                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-[0_0_12px_rgba(16,185,129,0.3)]'
                                                : 'bg-dark-800/50 text-gray-400 border border-white/5 hover:border-emerald-500/20'
                                        }`}
                                    >
                                        <span className="whitespace-nowrap">{xbrl.filing_type} FY{xbrl.fiscal_year ?? '—'}</span>
                                        <span className="text-[10px] text-gray-500 font-normal mt-0.5">{xbrl.filing_date ? `Filed ${xbrl.filing_date}` : '—'}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Statement Tabs */}
                        <div className="flex border-b border-emerald-500/10 bg-dark-800/30">
                            {[
                                { key: 'income', label: 'Income Statement' },
                                { key: 'balance', label: 'Balance Sheet' },
                                { key: 'cashflow', label: 'Cash Flow' }
                            ].map((tab) => (
                                <button
                                    key={tab.key}
                                    onClick={() => setSelectedStatementTab(tab.key as any)}
                                    className={`flex-1 px-4 py-3 text-xs font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${
                                        selectedStatementTab === tab.key
                                            ? 'text-emerald-300 bg-emerald-500/10 border-b-2 border-emerald-500'
                                            : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                                    }`}
                                >
                                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${selectedStatementTab === tab.key ? 'bg-emerald-400' : 'bg-gray-500'}`} />
                                    {tab.label}
                                </button>
                            ))}
                        </div>

                        {/* Statement Content - accounting-style: label left, value right, monospace values, — for missing */}
                        <div className="p-4">
                            {(() => {
                                const selectedXbrl = secXbrlData[selectedXbrlIndex]
                                if (!selectedXbrl) return <div className="text-gray-500 text-sm">No data available</div>

                                /** Get numeric value for an XBRL concept from costs, debt, equity, cashflow, or all_concepts (array of {value, context}) */
                                const getConcept = (conceptName: string): number | null | undefined => {
                                    const x = selectedXbrl as any
                                    const fromBucket = x.costs?.[conceptName] ?? x.debt?.[conceptName] ?? x.equity?.[conceptName] ?? x.cashflow?.[conceptName]
                                    if (fromBucket != null && typeof fromBucket === 'number') return fromBucket
                                    const fromAll = x.all_concepts?.[conceptName]
                                    if (fromAll == null) return undefined
                                    if (typeof fromAll === 'number') return fromAll
                                    if (Array.isArray(fromAll) && fromAll.length) {
                                        const vals = fromAll.map((e: any) => (e && typeof e.value === 'number') ? e.value : null).filter((v: number | null) => v != null) as number[]
                                        if (vals.length) return Math.abs(vals[0]) >= Math.abs(vals[vals.length - 1]) ? vals[0] : vals[vals.length - 1]
                                    }
                                    return undefined
                                }
                                /** Try multiple concept names (fallbacks for alternate GAAP names) */
                                const getConceptAny = (names: string[]): number | null | undefined => {
                                    for (const name of names) {
                                        const v = getConcept(name)
                                        if (v != null) return v
                                    }
                                    return undefined
                                }

                                const fmt = (val: number | null | undefined, currency = true, unit?: string, signed = false): string => {
                                    if (val == null) return '—'
                                    if (currency) {
                                        if (showFullAmounts) {
                                            const n = Number(val)
                                            const str = n.toLocaleString('en-US', { maximumFractionDigits: 0 })
                                            return signed ? (val >= 0 ? `+$${str}` : `-$${Math.abs(n).toLocaleString('en-US', { maximumFractionDigits: 0 })}`) : `$${str}`
                                        }
                                        const s = (val / 1e6).toFixed(1)
                                        return signed ? (val >= 0 ? `+$${s}M` : `-$${(Math.abs(val) / 1e6).toFixed(1)}M`) : `$${s}M`
                                    }
                                    if (unit === 'M') return showFullAmounts ? val.toLocaleString('en-US', { maximumFractionDigits: 0 }) : `${(val / 1e6).toFixed(1)}M`
                                    if (typeof val === 'number' && !currency) return val.toFixed(2)
                                    return String(val)
                                }
                                const row = (label: string, value: number | null | undefined, opts?: { highlight?: boolean; large?: boolean; currency?: boolean; unit?: string; signed?: boolean }) => {
                                    if (value == null || value === undefined) return null
                                    return (
                                        <div key={label} className={`flex justify-between items-center py-1.5 px-2 border-b border-white/5 last:border-0 ${opts?.highlight ? 'bg-emerald-500/5' : ''}`}>
                                            <span className={`text-xs ${opts?.highlight ? 'text-emerald-400 font-semibold' : 'text-gray-300'}`}>{label}</span>
                                            <span className={`font-mono text-sm tabular-nums text-right min-w-[7rem] ${opts?.highlight ? 'text-emerald-300 font-bold' : 'text-gray-200'}`}>
                                                {fmt(value, opts?.currency !== false, opts?.unit, opts?.signed)}
                                            </span>
                                        </div>
                                    )
                                }

                                // Income Statement
                                if (selectedStatementTab === 'income') {
                                    const rev = getConcept('Revenues') ?? getConcept('RevenueFromContractWithCustomerExcludingAssessedTax') ?? getConcept('RevenueFromContractWithCustomerIncludingAssessedTax')
                                    const cogs = getConcept('CostOfRevenue') ?? getConcept('CostOfGoodsAndServicesSold')
                                    const gross = (rev != null && cogs != null) ? rev - cogs : null
                                    return (
                                        <div className="space-y-4">
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Revenue</h4>
                                                <div className="space-y-0">
                                                    {row('Revenue', rev ?? undefined, { highlight: true })}
                                                    {row('Cost of Revenue', cogs ?? undefined)}
                                                    {row('Gross Profit', gross, { highlight: true })}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Operating</h4>
                                                <div className="space-y-0">
                                                    {row('R&D Expense', getConceptAny(['ResearchAndDevelopmentExpense']) ?? undefined)}
                                                    {row('SG&A Expense', getConceptAny(['SellingGeneralAndAdministrativeExpense', 'SellingAndMarketingExpense']) ?? undefined)}
                                                    {row('Operating Expense', getConceptAny(['OperatingExpense', 'OperatingExpenses']) ?? undefined)}
                                                    {row('Operating Income', getConceptAny(['OperatingIncomeLoss', 'OperatingIncome']) ?? undefined, { highlight: true })}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Net Income</h4>
                                                <div className="space-y-0">
                                                    {row('Interest Expense', getConceptAny(['InterestExpense', 'InterestExpenseDebt']) ?? undefined)}
                                                    {row('Income Tax', getConcept('IncomeTaxExpenseBenefit') ?? undefined)}
                                                    {row('Net Income', getConceptAny(['NetIncomeLoss', 'ProfitLoss']) ?? undefined, { highlight: true, large: true })}
                                                    {row('EPS (Basic)', getConceptAny(['EarningsPerShareBasic', 'NetIncomeLossPerShare']) ?? undefined, { currency: false })}
                                                    {row('EPS (Diluted)', getConceptAny(['EarningsPerShareDiluted', 'NetIncomeLossPerShare']) ?? undefined, { currency: false })}
                                                </div>
                                            </div>
                                        </div>
                                    )
                                }

                                // Balance Sheet - Assets / Liabilities and Equity with total tie-in
                                if (selectedStatementTab === 'balance') {
                                    const totalAssets = getConceptAny(['Assets', 'TotalAssets']) ?? undefined
                                    const currentDebt = getConceptAny(['DebtCurrent', 'CurrentDebt', 'LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths']) ?? undefined
                                    const ltDebt = getConceptAny(['LongTermDebt', 'LongTermDebtAndCapitalLeaseObligations'])
                                    const totalDebt = (currentDebt != null || ltDebt != null) ? (currentDebt || 0) + (ltDebt || 0) : null
                                    const equity = getConceptAny(['StockholdersEquity', 'Equity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'])
                                    const totalLiabEquity = (totalDebt != null && equity != null) ? totalDebt + equity : (equity != null ? equity : null)
                                    return (
                                        <div className="space-y-4">
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Assets</h4>
                                                <div className="space-y-0">
                                                    {row('Cash & Equivalents', getConceptAny(['CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashAndCashEquivalents']) ?? undefined)}
                                                    {row('Total Assets', totalAssets, { highlight: true })}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Liabilities and Equity</h4>
                                                <div className="space-y-0">
                                                    {row('Current Debt', currentDebt ?? undefined)}
                                                    {row('Long-Term Debt', ltDebt ?? undefined)}
                                                    {row('Total Debt', totalDebt, { highlight: true })}
                                                    {row('Stockholders Equity', equity ?? undefined, { highlight: true })}
                                                    {row('Retained Earnings', getConceptAny(['RetainedEarningsAccumulatedDeficit', 'RetainedEarnings']) ?? undefined)}
                                                    {row('Treasury Stock', getConceptAny(['TreasuryStockValue', 'TreasuryStockCommonValue', 'TreasuryStock']) ?? undefined)}
                                                    {row('Shares Outstanding', getConceptAny(['CommonStockSharesOutstanding', 'CommonStockSharesIssued']) ?? undefined, { currency: false, unit: 'M' })}
                                                    {totalLiabEquity != null && totalAssets != null && (
                                                        <div className="flex justify-between items-center py-1.5 px-2 mt-2 pt-2 border-t border-emerald-500/20">
                                                            <span className="text-xs font-bold text-emerald-400">Total liabilities and equity</span>
                                                            <span className="font-mono text-sm font-bold text-emerald-300 tabular-nums text-right min-w-[7rem]">{fmt(totalLiabEquity)}</span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )
                                }

                                // Cash Flow Statement
                                if (selectedStatementTab === 'cashflow') {
                                    const opCf = getConcept('NetCashProvidedByUsedInOperatingActivities')
                                    const invCf = getConcept('NetCashProvidedByUsedInInvestingActivities')
                                    const finCf = getConcept('NetCashProvidedByUsedInFinancingActivities')
                                    const fcf = (opCf != null && invCf != null) ? opCf + invCf : null
                                    return (
                                        <div className="space-y-4">
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Operating Activities</h4>
                                                <div className="space-y-0">
                                                    {row('Operating Cash Flow', opCf ?? undefined, { highlight: true, signed: true })}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Investing Activities</h4>
                                                <div className="space-y-0">
                                                    {row('Investing Cash Flow', invCf ?? undefined, { highlight: true, signed: true })}
                                                    {row('Business Acquisitions', getConcept('PaymentsToAcquireBusinessesNetOfCashAcquired') ?? undefined, { signed: true })}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Financing Activities</h4>
                                                <div className="space-y-0">
                                                    {row('Financing Cash Flow', finCf ?? undefined, { highlight: true, signed: true })}
                                                    {row('Dividends Paid', getConceptAny(['PaymentsOfDividends', 'DividendsPaid', 'PaymentOfDividends']) ?? undefined, { signed: true })}
                                                    {row('Stock Repurchases', getConceptAny(['PaymentsForRepurchaseOfCommonStock', 'PaymentsForRepurchaseOfEquity']) ?? undefined, { signed: true })}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Net Change</h4>
                                                <div className="space-y-0">
                                                    {row('Free Cash Flow', fcf, { highlight: true, signed: true })}
                                                </div>
                                            </div>
                                        </div>
                                    )
                                }
                            })()}
                        </div>
                    </div>
                </div>
            )}

            {/* SEC Sentences - Discoverable list with search and sentiment filter */}
            {secSentences.length > 0 && (
                <div id="sec-sentences-section" className="mt-4">
                    <div className="bg-dark-900/40 border border-indigo-500/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                            <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-[0.2em] flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
                                SEC Sentences
                            </h3>
                            <div className="flex items-center gap-2 flex-wrap">
                                <input
                                    type="text"
                                    placeholder="Search sentences..."
                                    value={secSentenceSearch}
                                    onChange={(e) => setSecSentenceSearch(e.target.value)}
                                    className="px-2 py-1.5 text-xs bg-dark-800 border border-white/10 rounded text-gray-200 placeholder-gray-500 w-40 focus:border-indigo-500/50 focus:outline-none"
                                />
                                {(['all', 'bullish', 'bearish', 'neutral'] as const).map((sent) => (
                                    <button
                                        key={sent}
                                        type="button"
                                        onClick={() => setSecSentenceSentiment(sent)}
                                        className={`px-2 py-1 text-[10px] font-semibold rounded capitalize ${secSentenceSentiment === sent ? 'bg-indigo-500/30 text-indigo-300 border border-indigo-500/50' : 'bg-dark-800 text-gray-400 border border-white/10 hover:text-gray-200'}`}
                                    >
                                        {sent}
                                    </button>
                                ))}
                            </div>
                            <div className="text-xs text-gray-500">
                                {filteredSecSentences.length} of {secSentences.length} sentence{secSentences.length !== 1 ? 's' : ''}
                            </div>
                        </div>
                        <div className="space-y-2">
                            {(showMoreSecSentences ? filteredSecSentences : filteredSecSentences.slice(0, 10)).map((sentence: any, i: number) => {
                                const score = sentence.finbert_score || 0
                                const sentiment = score > 0.2 ? 'Bullish' : score < -0.2 ? 'Bearish' : 'Neutral'
                                const color = score > 0.2 ? 'text-green-400' : score < -0.2 ? 'text-red-400' : 'text-gray-400'
                                const bgColor = score > 0.2 ? 'bg-green-500/10 border-green-500/20' : score < -0.2 ? 'bg-red-500/10 border-red-500/20' : 'bg-gray-500/10 border-gray-500/20'

                                return (
                                    <div key={i} className={`p-3 rounded-lg border ${bgColor} hover:bg-white/5 transition-all`}>
                                        <div className="flex items-start gap-3">
                                            <div className={`flex-shrink-0 px-2 py-1 rounded text-[10px] font-bold ${color}`}>
                                                {sentiment}
                                                <div className="text-[9px] opacity-70">{score.toFixed(3)}</div>
                                            </div>
                                            <div className="flex-1">
                                                <p className="text-xs text-gray-300 leading-relaxed">
                                                    {sentence.text}
                                                </p>
                                                {sentence.section_type && (
                                                    <div className="mt-1 text-[9px] text-gray-500">
                                                        Section: {sentence.section_type}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )
                            })}
                            {filteredSecSentences.length > 10 && (
                                <button
                                    type="button"
                                    onClick={() => setShowMoreSecSentences(!showMoreSecSentences)}
                                    className="w-full py-2 text-[11px] font-semibold text-indigo-400 hover:text-indigo-300 border border-indigo-500/20 rounded-lg hover:bg-indigo-500/10 transition-colors"
                                >
                                    {showMoreSecSentences ? 'Show less' : `Show more (${filteredSecSentences.length - 10} more)`}
                                </button>
                            )}
                            {filteredSecSentences.length === 0 && (
                                <p className="text-xs text-gray-500 py-4 text-center">No sentences match your search or filter.</p>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Unified SEC Document Viewer (Filings | Exhibits | XBRL breakdown) */}
            {(allSecFilings.length > 0 || secExhibits.length > 0 || secXbrlData.length > 0) && (
                <div className="mt-4">
                    <SECDocumentViewer
                        filings={allSecFilings}
                        exhibits={secExhibits}
                        xbrlData={secXbrlData}
                        ticker={company.ticker || data.ticker || ''}
                        onSelectFiling={openSecDetail}
                        showFullAmounts={showFullAmounts}
                    />
                </div>
            )}

            {/* Awards Table - KPI strip + reference-style table */}
            {awards.length > 0 && awardKpis && (
                <div className="mt-4">
                    <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-sm font-bold text-gold uppercase tracking-[0.2em] flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-gold shadow-[0_0_8px_rgba(255,215,0,0.5)]" />
                                Federal Contract Awards
                            </h3>
                            <div className="text-xs text-gray-500">
                                {awards.length} award{awards.length !== 1 ? 's' : ''}
                            </div>
                        </div>

                        {/* Key Performance Indicators strip */}
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 mb-4 p-3 rounded-lg border border-gold/10 bg-dark-800/50">
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Total Value</div>
                                <div className="text-sm font-bold text-gold font-mono">
                                    {showFullAmounts ? `$${Number(awardKpis.totalValue).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(awardKpis.totalValue / 1e6).toFixed(1)}M`}
                                </div>
                            </div>
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Award Count</div>
                                <div className="text-sm font-bold text-white">{awardKpis.count}</div>
                            </div>
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Avg Award</div>
                                <div className="text-sm font-bold text-gray-200 font-mono">
                                    {showFullAmounts ? `$${Number(awardKpis.avgAward).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(awardKpis.avgAward / 1e6).toFixed(1)}M`}
                                </div>
                            </div>
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Largest Award</div>
                                <div className="text-sm font-bold text-gold font-mono">
                                    {showFullAmounts ? `$${Number(awardKpis.largest).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(awardKpis.largest / 1e6).toFixed(1)}M`}
                                </div>
                            </div>
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">FY Range</div>
                                <div className="text-sm font-bold text-gray-200">{awardKpis.fyRange}</div>
                            </div>
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Top Agency</div>
                                <div className="text-xs font-bold text-gray-300 truncate" title={awardKpis.topAgency}>{awardKpis.topAgency}</div>
                            </div>
                        </div>

                        {/* Table - distinct header, alternating rows, copy on Amount, scroll */}
                        <div className="rounded-lg border border-gold/20 overflow-hidden">
                            <div className="max-h-[400px] overflow-y-auto overflow-x-auto">
                                <table className="w-full text-[11px]">
                                    <thead className="sticky top-0 z-10 bg-gold/20 border-b border-gold/30">
                                        <tr>
                                            <th className="text-left text-gold font-semibold uppercase tracking-wider py-2.5 px-2">Agency</th>
                                            <th className="text-left text-gold font-semibold uppercase tracking-wider py-2.5 px-2">Description</th>
                                            <th className="text-center text-gold font-semibold uppercase tracking-wider py-2.5 px-2">FY</th>
                                            <th className="text-center text-gold font-semibold uppercase tracking-wider py-2.5 px-2">Date</th>
                                            <th className="text-right text-gold font-semibold uppercase tracking-wider py-2.5 px-2">Amount</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {awards.map((a: any, i: number) => {
                                            const amtStr = showFullAmounts ? `$${Number(a.award_amount_float).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(a.award_amount_float / 1e6).toFixed(1)}M`
                                            const descStr = (a.description || 'Contract Award').toString()
                                            return (
                                                <tr
                                                    key={i}
                                                    onClick={() => setSelectedDetail({ type: 'Award', data: a })}
                                                    className={`border-b border-white/5 hover:bg-gold/10 transition-colors cursor-pointer ${i % 2 === 1 ? 'bg-white/[0.02]' : ''}`}
                                                >
                                                    <td className="py-1.5 px-2 text-gray-100 truncate max-w-[150px]">{a.awarding_agency}</td>
                                                    <td className="py-1.5 px-2 text-gray-300 truncate max-w-[300px]">
                                                        <span className="inline-flex items-center gap-1 max-w-full">
                                                            <span className="truncate">{descStr}</span>
                                                            <button
                                                                type="button"
                                                                onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(descStr).catch(() => {}); }}
                                                                className="opacity-60 hover:opacity-100 text-gray-400 hover:text-gold p-0.5 rounded flex-shrink-0"
                                                                title="Copy description"
                                                            >
                                                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                                                            </button>
                                                        </span>
                                                    </td>
                                                    <td className="py-1.5 px-2 text-center text-gray-300">FY-{a.contract_year || '—'}</td>
                                                    <td className="py-1.5 px-2 text-center text-gray-400 text-[10px]">{a.award_date || a.start_date || '—'}</td>
                                                    <td className="py-1.5 px-2 text-right text-gold font-mono font-bold">
                                                        <span className="inline-flex items-center gap-1">
                                                            {amtStr}
                                                            <button
                                                                type="button"
                                                                onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(amtStr).catch(() => {}); }}
                                                                className="opacity-60 hover:opacity-100 text-gray-400 hover:text-gold p-0.5 rounded"
                                                                title="Copy amount"
                                                            >
                                                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                                                            </button>
                                                        </span>
                                                    </td>
                                                </tr>
                                            )
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* SEC Detail Modal (Sentiment Analysis) */}
            <AnimatePresence>
                {selectedDetail && selectedDetail.type === 'SEC' && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/95 backdrop-blur-xl z-[150] flex items-center justify-center p-4 md:p-8"
                        onClick={closeDetail}
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
                                        {company.ticker && <span className="text-gray-400 font-normal ml-2">· {company.ticker}</span>}
                                    </div>
                                </div>
                                <button onClick={closeDetail} className="px-3 py-1.5 text-xs font-medium bg-dark-700 rounded-lg text-gray-400 hover:text-white border border-white/10 transition-all">Close</button>
                            </div>
                            <div className="flex-1 overflow-y-auto p-6 space-y-5">
                                {(() => {
                                    const avg = selectedDetail.data.avg_finbert ?? 0;
                                    const sentimentLabel = avg > 0.05 ? 'Positive' : avg < -0.05 ? 'Negative' : 'Neutral';
                                    const sentimentClass = avg > 0.05 ? 'bg-green-500/20 text-green-400 border-green-500/40' : avg < -0.05 ? 'bg-red-500/20 text-red-400 border-red-500/40' : 'bg-gray-500/20 text-gray-400 border-gray-500/40';
                                    const formType = selectedDetail.data.type || selectedDetail.data.form_type || 'Filing';
                                    const filedDate = selectedDetail.data.filing_date ?? '—';
                                    const directFilingUrl = secFilingDocumentUrl(selectedDetail.data.accession);
                                    const secCompanyLink = secCompanyUrl(company.ticker);
                                    return (
                                        <>
                                            <div className="flex flex-wrap items-center gap-2">
                                                <span className={`text-xs font-bold px-2.5 py-1 rounded border ${sentimentClass}`}>
                                                    {sentimentLabel} ({(avg).toFixed(3)})
                                                </span>
                                                <span className="text-xs text-gray-500">Filed: {filedDate}</span>
                                            </div>
                                            <p className="text-xs text-gray-400">
                                                {formType} filed on {filedDate}{company.ticker ? ` for ${company.ticker}` : ''}.
                                            </p>
                                            <div className="flex flex-wrap gap-x-4 gap-y-1">
                                                {directFilingUrl && (
                                                    <a href={directFilingUrl} target="_blank" rel="noopener noreferrer" className="text-[10px] text-gold hover:text-gold/80 font-semibold">View this filing on SEC →</a>
                                                )}
                                                {secCompanyLink && (
                                                    <a href={secCompanyLink} target="_blank" rel="noopener noreferrer" className="text-[10px] text-gray-500 hover:text-blue-400 transition-colors">All filings (EDGAR)</a>
                                                )}
                                            </div>
                                            <div>
                                                <h5 className="text-[10px] font-bold text-blue-400 uppercase tracking-wider mb-3">Key Excerpts (by sentiment)</h5>
                                                {(selectedDetail.data.top_sentences || selectedDetail.data.sec_sentences)?.length > 0 ? (
                                                    <div className="space-y-2">
                                                        {(selectedDetail.data.top_sentences || selectedDetail.data.sec_sentences)
                                                            .slice(0, showMoreSecModalExcerpts ? undefined : 5)
                                                            .map((s: any, j: number) => {
                                                                const score = s.score ?? s.finbert_score ?? 0;
                                                                const isPositive = score > 0.2;
                                                                const isNegative = score < -0.2;
                                                                const expanded = expandedExcerptIndex === j;
                                                                const text = (s.text || '').trim();
                                                                const trunc = text.length > 80 ? text.slice(0, 80) + '…' : text;
                                                                return (
                                                                    <div
                                                                        key={j}
                                                                        className={`rounded-lg border px-3 py-2 text-left ${isPositive ? 'border-green-500/30 bg-green-500/5' : isNegative ? 'border-red-500/30 bg-red-500/5' : 'border-white/10 bg-dark-900/50'}`}
                                                                    >
                                                                        <div className="flex items-start justify-between gap-2">
                                                                            <p className="text-xs text-gray-300 flex-1 italic">"{expanded ? text : trunc}"</p>
                                                                            <span className={`text-[10px] font-mono font-bold shrink-0 ${isPositive ? 'text-green-400' : isNegative ? 'text-red-400' : 'text-gray-400'}`}>
                                                                                {score > 0 ? '+' : ''}{(score).toFixed(2)}
                                                                            </span>
                                                                        </div>
                                                                        {text.length > 80 && (
                                                                            <button
                                                                                type="button"
                                                                                onClick={() => setExpandedExcerptIndex(expanded ? null : j)}
                                                                                className="mt-1.5 text-[10px] text-blue-400 hover:text-blue-300"
                                                                            >
                                                                                {expanded ? 'Show less' : 'Show more'}
                                                                            </button>
                                                                        )}
                                                                    </div>
                                                                );
                                                            })}
                                                        {(selectedDetail.data.top_sentences || selectedDetail.data.sec_sentences).length > 5 && (
                                                            <button
                                                                type="button"
                                                                onClick={() => setShowMoreSecModalExcerpts(!showMoreSecModalExcerpts)}
                                                                className="w-full py-2 text-xs text-blue-400 hover:text-blue-300 border border-blue-500/20 rounded-lg mt-2"
                                                            >
                                                                {showMoreSecModalExcerpts ? 'Show less' : `Show more (${(selectedDetail.data.top_sentences || selectedDetail.data.sec_sentences).length - 5} more excerpts)`}
                                                            </button>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <div className="text-xs text-gray-500 text-center py-6 border border-dashed border-gray-700 rounded-lg space-y-2">
                                                        <p className="italic">No sentence-level excerpts for this filing.</p>
                                                        <p className="text-[10px]">View SEC Sentences below for ticker-level sentiment.</p>
                                                        <button
                                                            type="button"
                                                            onClick={() => { closeDetail(); document.getElementById('sec-sentences-section')?.scrollIntoView({ behavior: 'smooth' }) }}
                                                            className="text-[10px] text-blue-400 hover:text-blue-300 underline"
                                                        >
                                                            Go to SEC Sentences
                                                        </button>
                                                        {secFilingUrl && (
                                                            <span className="block mt-2">
                                                                <a href={secFilingUrl} target="_blank" rel="noopener noreferrer" className="text-[10px] text-gray-500 hover:text-blue-400 transition-colors">Open on SEC EDGAR</a>
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        </>
                                    );
                                })()}
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
                        onClick={closeDetail}
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
                                <button onClick={closeDetail} className="px-3 py-1.5 text-xs font-medium bg-dark-700 rounded-lg text-gray-400 hover:text-white border border-white/10 transition-all">Close</button>
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
                                            {showFullAmounts ? `$${Number(selectedDetail.data.award_amount_float).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(selectedDetail.data.award_amount_float / 1e6).toFixed(2)}M`}
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
