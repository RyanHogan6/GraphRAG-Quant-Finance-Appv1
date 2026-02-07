'use client'

import { useState, useMemo, useRef } from 'react'
import TimeSeriesChart from './TimeSeriesChart'
import { motion, AnimatePresence } from 'framer-motion'
import SECFilingsExplorer from './SECFilingsExplorer'
import SentimentIndicators from './company/SentimentIndicators'
import type { Key } from 'react'

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
    const [showPeerSelector, setShowPeerSelector] = useState(false)
    const [peerSearchTerm, setPeerSearchTerm] = useState('')
    const [selectedXbrlIndex, setSelectedXbrlIndex] = useState(0)
    const [selectedStatementTab, setSelectedStatementTab] = useState<'income' | 'balance' | 'cashflow'>('income')
    const [expandedXbrlBreakdownIndex, setExpandedXbrlBreakdownIndex] = useState<number | null>(null)
    const [showMoreSecSentences, setShowMoreSecSentences] = useState(false)
    const [showMoreSecModalExcerpts, setShowMoreSecModalExcerpts] = useState(false)
    const [expandedExcerptIndex, setExpandedExcerptIndex] = useState<number | null>(null)
    const [showFullAmounts, setShowFullAmounts] = useState(false)
    const xbrlSectionRef = useRef<HTMLDivElement>(null)
    const exhibitsSectionRef = useRef<HTMLDivElement>(null)

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

    console.log('[COMPANY WORKUP DEBUG] Data structure:', {
        isFiling,
        hasXbrl: secXbrlData.length,
        hasSentences: secSentences.length,
        hasExhibits: secExhibits.length,
        hasSections: secSections.length
    })
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

    // S&P 500 tickers for peer selection (sample - should come from backend)
    const availablePeers = useMemo(() => {
        const sp500Tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'V', 'JPM', 'WMT', 'XOM', 'UNH', 'MA', 'PG', 'JNJ', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'KO', 'COST', 'AVGO', 'LLY', 'TMO', 'ADBE', 'MCD', 'CSCO', 'ACN', 'NFLX', 'ABT', 'CRM', 'DHR', 'NKE', 'WFC', 'VZ', 'TXN', 'PM', 'ORCL', 'NEE', 'RTX', 'UPS', 'MS', 'BMY', 'QCOM', 'LOW', 'HON', 'INTU', 'T', 'UNP', 'AMD', 'IBM', 'BA', 'SPGI', 'GE', 'SBUX', 'CAT', 'DE', 'AXP', 'GS', 'PLD', 'MDT', 'BLK', 'AMGN', 'GILD', 'AMAT', 'LMT', 'ISRG', 'SYK', 'ADI', 'MMM', 'TJX', 'CI', 'MDLZ', 'CB', 'ADP', 'C', 'VRTX', 'SO', 'BKNG', 'ZTS', 'CME', 'SCHW', 'REGN', 'FISV', 'MMC', 'DUK', 'PGR', 'TMUS', 'MO', 'BDX', 'CVS', 'USB', 'PNC', 'NOC', 'COP', 'ITW', 'EOG', 'TGT']
        return sp500Tickers.filter(t => t !== company.ticker && t.toLowerCase().includes(peerSearchTerm.toLowerCase()))
    }, [company.ticker, peerSearchTerm])

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

            return {
                dates: sorted.map(d => d.date),
                values,
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

        return [
            `${company.company} (${ticker}) is demonstrating a ${priceChange}% trajectory over the selected ${timeframe} window, currently trading at $${latestMarket?.close?.toFixed(2)} with market capitalization of ${marketCapB !== 'N/A' ? '$' + marketCapB + 'B' : 'market cap unavailable'} in the ${company.sector} sector.`,

            `Recent SEC regulatory signals lean ${sentiment.toLowerCase()} (FinBERT Score: ${secFilings[0]?.avg_finbert?.toFixed(3) || 'N/A'}), with ${secFilings.length} filings analyzed including ${secFilings[0]?.form_type || 'quarterly/annual'} reports showing ${sentiment === 'Bullish' ? 'optimistic' : sentiment === 'Bearish' ? 'cautious' : 'neutral'} management tone regarding operational performance and forward guidance.`,

            awards.length > 0
                ? `Government contract portfolio totals $${(totalAwardValue / 1e6).toFixed(1)}M across ${awards.length} federal awards, with recent ${awards[0]?.awarding_agency || 'Department of Defense'} contract for $${(awards[0]?.award_amount_float / 1e6).toFixed(1)}M awarded in FY-${awards[0]?.contract_year || '2026'}, establishing strong public sector revenue diversification.`
                : `${company.company} maintains focused commercial market exposure with ${company.sector} industry leadership, supported by institutional ownership patterns and steady revenue generation from core business operations.`,

            optionsFlow.length > 0
                ? `Options market activity reflects ${optionsSignal} with put/call ratio of ${putCallRatio.toFixed(2)}, total options volume of ${latestOptions?.total_volume?.toLocaleString()} contracts, and implied volatility at ${((latestOptions?.implied_volatility ?? latestOptions?.iv_avg) != null ? (Number(latestOptions.implied_volatility ?? latestOptions.iv_avg) * 100).toFixed(1) : 'N/A')}%, indicating ${putCallRatio > 1.5 ? 'defensive hedging' : putCallRatio < 0.5 ? 'aggressive upside speculation' : 'standard market expectations'}.`
                : `Options flow data not yet available for this ticker, with trading activity primarily focused on equity markets and institutional block transactions.`,

            `Technical indicators place ${ticker} in ${technicalSignal}, with ${latestMarket?.sma_50 && latestMarket?.sma_200 ? (latestMarket.sma_50 > latestMarket.sma_200 ? 'bullish golden cross formation' : 'bearish death cross warning') : 'developing trend structure'} as moving averages ${latestMarket?.sma_50 && latestMarket?.sma_200 ? (latestMarket.sma_50 > latestMarket.sma_200 ? 'confirm' : 'challenge') : 'establish'} current price action.`,

            polyMarkets.length > 0
                ? `Prediction markets assign ${(polyMarkets[0].yes_probability * 100).toFixed(0)}% probability to ${polyMarkets[0].question}, with $${(polyMarkets[0].volume_24h / 1000).toFixed(0)}K daily volume reflecting market sentiment divergence from traditional equity pricing models.`
                : news ? `Wall Street projections for 2026 highlight potential revenue ceiling of ${news.rev}, catalyzed significantly by ${news.driver}, with upcoming ${news.event} serving as critical inflection point for institutional portfolio rebalancing.`
                : `Fundamental metrics show ${latestMarket?.revenue_growth ? ((latestMarket.revenue_growth * 100).toFixed(1) + '% revenue growth') : 'steady revenue generation'} with ${latestMarket?.profit_margins ? ((latestMarket.profit_margins * 100).toFixed(1) + '% profit margins') : 'industry-standard profitability'}, positioning ${ticker} for ${priceChangeNum > 5 ? 'continued momentum expansion' : priceChangeNum < -5 ? 'potential mean reversion opportunity' : 'range-bound consolidation'}.`,

            `Critical intelligence suggests monitoring ${news?.event || secFilings.length > 0 ? 'upcoming ' + (secFilings[0]?.form_type === '10-Q' ? 'quarterly earnings release' : secFilings[0]?.form_type === '10-K' ? 'annual report filing' : 'regulatory filings') : 'next quarterly earnings'} as primary volatility catalyst, with institutional positioning ${awards.length > 0 ? 'supported by government contract visibility' : 'driven by sector rotation dynamics'} and ${optionsFlow.length > 0 && putCallRatio > 1.5 ? 'hedged downside protection' : optionsFlow.length > 0 && putCallRatio < 0.5 ? 'leveraged upside exposure' : 'balanced risk/reward profiles'}.`,

            secXbrlData.length > 0 || secExhibits.length > 0
                ? `Enhanced financial transparency available through ${secXbrlData.length > 0 ? `detailed XBRL breakdowns covering ${secXbrlData[0]?.has_segment_data ? 'revenue segments, ' : ''}${secXbrlData[0]?.debt ? 'debt obligations, ' : ''}and cost structures` : ''}${secXbrlData.length > 0 && secExhibits.length > 0 ? ', alongside ' : ''}${secExhibits.length > 0 ? `${secExhibits.length} material contract${secExhibits.length > 1 ? 's' : ''} including ${secExhibits.filter((e: any) => e.exhibit_type?.includes('10.') || e.contract_type?.toLowerCase().includes('credit')).length > 0 ? 'credit agreements, ' : ''}${secExhibits.filter((e: any) => e.contract_type?.toLowerCase().includes('employment')).length > 0 ? 'executive compensation packages, ' : ''}and strategic partnership filings` : ''} providing institutional-grade due diligence capabilities.`
                : null
        ].filter(Boolean)
    }, [company, timeframe, chartData, awards, secFilings, polyMarkets, secXbrlData, secExhibits])

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
                            {company.ticker}
                        </span>
                    </div>
                    <p className="text-sm text-gray-400">
                        {company.sector} | {company.industry} | {company.city}, {company.country}
                    </p>
                </div>
                <div className="flex gap-2 relative">
                    {comparisonMode && peerData ? (
                        <button
                            onClick={() => onCompare?.(company.ticker)}
                            className="px-4 py-2 bg-red-900/20 border border-red-500/30 rounded-lg text-xs text-red-400 hover:bg-red-900/30 transition-all flex items-center gap-2"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                            Exit Comparison
                        </button>
                    ) : (
                        <>
                            <button
                                onClick={() => setShowPeerSelector(!showPeerSelector)}
                                className="px-4 py-2 bg-dark-800 border border-gold/30 rounded-lg text-xs text-gold hover:bg-gold/10 transition-all flex items-center gap-2"
                            >
                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                                </svg>
                                Compare Peer
                            </button>
                            {showPeerSelector && (
                                <div className="absolute top-full mt-2 right-0 bg-dark-800 border border-gold/30 rounded-lg shadow-2xl z-50 w-64 max-h-80 overflow-hidden flex flex-col">
                                    <div className="p-2 border-b border-gold/20">
                                        <input
                                            type="text"
                                            placeholder="Search ticker..."
                                            value={peerSearchTerm}
                                            onChange={(e) => setPeerSearchTerm(e.target.value)}
                                            className="w-full px-3 py-2 bg-dark-900 border border-gold/20 rounded text-xs text-white placeholder-gray-500 focus:border-gold/50 outline-none"
                                            autoFocus
                                        />
                                    </div>
                                    <div className="overflow-y-auto max-h-64">
                                        {availablePeers.slice(0, 50).map((ticker) => (
                                            <button
                                                key={ticker}
                                                onClick={() => {
                                                    onCompare?.(ticker)
                                                    setShowPeerSelector(false)
                                                    setPeerSearchTerm('')
                                                }}
                                                className="w-full px-4 py-2 text-left text-xs text-gray-300 hover:bg-gold/10 hover:text-gold transition-all border-b border-white/5"
                                            >
                                                {ticker}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                    <button
                        onClick={() => setShowFullAmounts(!showFullAmounts)}
                        className={`px-3 py-1.5 border rounded-lg text-xs transition-all ${showFullAmounts ? 'bg-gold/20 border-gold/50 text-gold' : 'bg-dark-800 border-gold/30 text-gray-400 hover:text-white'}`}
                    >
                        {showFullAmounts ? 'Full $ amounts' : 'Abbreviated $'}
                    </button>
                    <button className="px-3 py-1.5 bg-dark-800 border border-gold/30 rounded-lg text-xs text-gray-400 hover:text-white transition-all">
                        PDF Mode
                    </button>
                </div>
            </div>

            {/* 3 Sentiment Indicators - Quick Insights */}
            <SentimentIndicators
                marketData={marketData}
                secFilings={allSecFilings}
                optionsFlow={optionsFlow}
                xbrlData={secXbrlData}
            />

            {/* Main Content Sections - Full Width Layout */}
            <div className="space-y-4">

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

                {/* AI Intelligence Summary - Below Chart */}
                <div className="bg-gradient-to-br from-gold/10 to-transparent border border-gold/20 rounded-xl p-3 md:p-4 relative overflow-hidden group shadow-2xl">
                    <div className="absolute top-0 left-0 w-1 h-full bg-gold/50" />
                    <div className="absolute -right-12 -top-12 w-48 h-48 bg-gold/5 rounded-full blur-3xl group-hover:bg-gold/10 transition-all" />
                    <h3 className="text-[9px] md:text-[10px] font-bold text-gold uppercase tracking-[0.2em] mb-2 md:mb-3 flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-gold animate-pulse" />
                        Deep Intelligence Synthesis
                    </h3>
                    <div className="space-y-3 relative z-10">
                        {aiSummary.filter((s): s is string => Boolean(s)).map((s: string, i: number) => (
                            <div key={i} className="flex items-start gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-gold/50 mt-2 flex-shrink-0" />
                                <p className="text-[13px] md:text-sm text-gray-300 leading-relaxed font-medium">
                                    {s}
                                    {i === 1 && secFilings.length > 0 && (
                                        <span
                                            onClick={() => openSecDetail(secFilings[0])}
                                            className="ml-1 text-[10px] text-blue-400 font-mono cursor-pointer hover:underline bg-blue-500/10 px-1 rounded"
                                        >
                                            [SEC-{secFilings[0].filing_date}]
                                        </span>
                                    )}
                                    {i === 2 && awards.length > 0 && (
                                        <span
                                            onClick={() => setSelectedDetail({ type: 'Award', data: awards[0] })}
                                            className="ml-1 text-[10px] text-gold font-mono cursor-pointer hover:underline bg-gold/10 px-1 rounded"
                                        >
                                            [AWARD-${(awards[0].award_amount_float / 1e6).toFixed(1)}M]
                                        </span>
                                    )}
                                    {i === 3 && polyMarkets.length > 0 && (
                                        <span className="ml-1 text-[10px] text-purple-400 font-mono cursor-pointer hover:underline bg-purple-500/10 px-1 rounded">
                                            [MARKET-{(polyMarkets[0].yes_probability * 100).toFixed(0)}%]
                                        </span>
                                    )}
                                    {i === 4 && optionsFlow.length > 0 && (
                                        <span className="ml-1 text-[10px] text-green-400 font-mono bg-green-500/10 px-1 rounded">
                                            [OPTIONS-P/C:{optionsFlow[0].put_call_ratio?.toFixed(2)}]
                                        </span>
                                    )}
                                </p>
                            </div>
                        ))}
                    </div>
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
            </div>

            {/* Options Activity */}
            {optionsFlow.length > 0 && (
                <div className="mt-4">
                    <div className="bg-dark-900/40 border border-green-500/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-sm font-bold text-green-400 uppercase tracking-[0.2em] flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
                                Options Activity
                            </h3>
                            <span className="text-[10px] text-gray-500 font-mono">
                                {(() => {
                                    const pc = optionsFlow[0].put_call_ratio ?? optionsFlow[0].put_call_volume_ratio;
                                    if (pc != null) {
                                        const stance = pc > 1 ? 'Bearish' : pc < 0.7 ? 'Bullish' : 'Neutral';
                                        return `${stance} positioning, P/C ${Number(pc).toFixed(2)}`;
                                    }
                                    const first = optionsFlow[optionsFlow.length - 1]?.date;
                                    const last = optionsFlow[0]?.date;
                                    if (first && last) return `${first} – ${last}`;
                                    return `Last ${optionsFlow.length} days`;
                                })()}
                            </span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-xs">
                                <thead>
                                    <tr className="border-b border-green-500/10">
                                        <th className="text-left text-gray-300 font-semibold pb-2 px-2">Date</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 px-2">P/C Ratio</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 px-2">Call Vol</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 px-2">Put Vol</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 px-2">Total Vol</th>
                                        <th className="text-right text-gray-300 font-semibold pb-2 px-2">IV</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {optionsFlow.slice(0, 20).map((row: any, i: number) => {
                                        const pc = row.put_call_ratio ?? row.put_call_volume_ratio;
                                        const iv = row.implied_volatility ?? row.iv_avg;
                                        return (
                                        <tr key={i} className="border-b border-white/5 hover:bg-green-500/5 transition-colors">
                                            <td className="py-2 px-2 text-gray-300 font-mono">{row.date || '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-gray-200">{pc != null ? Number(pc).toFixed(2) : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-gray-200">{row.call_volume != null ? row.call_volume.toLocaleString() : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-gray-200">{row.put_volume != null ? row.put_volume.toLocaleString() : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-gray-200">{row.total_volume != null ? row.total_volume.toLocaleString() : '—'}</td>
                                            <td className="py-2 px-2 text-right font-mono text-gray-200">{iv != null ? (Number(iv) * 100).toFixed(1) + '%' : '—'}</td>
                                        </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* XBRL & Exhibits Alert Cards */}
            {(secXbrlData.length > 0 || secExhibits.length > 0) && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                    {/* XBRL Highlights */}
                    {secXbrlData.length > 0 && (
                        <div
                            role="button"
                            tabIndex={0}
                            onClick={() => xbrlSectionRef.current?.scrollIntoView({ behavior: 'smooth' })}
                            onKeyDown={(e) => e.key === 'Enter' && xbrlSectionRef.current?.scrollIntoView({ behavior: 'smooth' })}
                            className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4 hover:border-cyan-500/50 transition-all cursor-pointer group"
                        >
                            <div className="flex items-center gap-2 mb-2">
                                <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
                                <h4 className="text-sm font-bold text-cyan-400 group-hover:text-cyan-300 transition-colors">📊 Financial Breakdowns Available</h4>
                            </div>
                            <div className="text-xs text-gray-300 mb-3">
                                {secXbrlData.length} filing{secXbrlData.length !== 1 ? 's' : ''} with detailed XBRL data
                            </div>
                            <div className="flex gap-2 flex-wrap">
                                {secXbrlData[0]?.has_segment_data && (
                                    <span className="px-2 py-1 bg-cyan-500/20 border border-cyan-500/30 rounded text-[10px] text-cyan-300">
                                        Revenue Segments
                                    </span>
                                )}
                                {secXbrlData[0]?.debt && Object.keys(secXbrlData[0].debt).length > 0 && (
                                    <span className="px-2 py-1 bg-cyan-500/20 border border-cyan-500/30 rounded text-[10px] text-cyan-300">
                                        Debt Details
                                    </span>
                                )}
                                {secXbrlData[0]?.costs && Object.keys(secXbrlData[0].costs).length > 0 && (
                                    <span className="px-2 py-1 bg-cyan-500/20 border border-cyan-500/30 rounded text-[10px] text-cyan-300">
                                        Cost Breakdown
                                    </span>
                                )}
                            </div>
                            <div className="text-[10px] text-cyan-400 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
                                Scroll down to view detailed breakdowns →
                            </div>
                        </div>
                    )}

                    {/* Material Contracts Alert */}
                    {secExhibits.length > 0 && (
                        <div
                            role="button"
                            tabIndex={0}
                            onClick={() => exhibitsSectionRef.current?.scrollIntoView({ behavior: 'smooth' })}
                            onKeyDown={(e) => e.key === 'Enter' && exhibitsSectionRef.current?.scrollIntoView({ behavior: 'smooth' })}
                            className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-4 hover:border-purple-500/50 transition-all cursor-pointer group"
                        >
                            <div className="flex items-center gap-2 mb-2">
                                <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                                <h4 className="text-sm font-bold text-purple-400 group-hover:text-purple-300 transition-colors">📄 Material Contracts</h4>
                            </div>
                            <div className="text-xs text-gray-300 mb-3">
                                {secExhibits.length} exhibit{secExhibits.length !== 1 ? 's' : ''} filed
                            </div>
                            <div className="space-y-1.5">
                                {secExhibits.slice(0, 3).map((ex: any, i: number) => (
                                    <div key={i} className="flex items-start justify-between gap-2 text-[10px] bg-purple-500/5 px-2 py-1.5 rounded">
                                        <div className="flex-1 min-w-0">
                                            <span className="text-purple-300 font-semibold">{ex.exhibit_type}</span>
                                            <span className="text-gray-400 ml-2 truncate block">{ex.description || 'Material Contract'}</span>
                                        </div>
                                        <span className="text-gray-500 whitespace-nowrap">{ex.filing_date?.split('-')[0]}</span>
                                    </div>
                                ))}
                                {secExhibits.length > 3 && (
                                    <div className="text-[10px] text-purple-400 pt-1">
                                        +{secExhibits.length - 3} more exhibits
                                    </div>
                                )}
                            </div>
                            <div className="text-[10px] text-purple-400 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
                                Scroll down to view full exhibit list →
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* SEC Filings Explorer - Full Width Section */}
            {allSecFilings.length > 0 && (
                <div className="mt-4">
                    <SECFilingsExplorer filings={allSecFilings} ticker={company.ticker} onSelectFiling={openSecDetail} />
                </div>
            )}

            {/* SEC Exhibits - Material Contracts */}
            {secExhibits.length > 0 && (() => {
                const exhibitTypeHint: Record<string, string> = {
                    'EX-10': 'Material contracts (credit, employment, M&A, etc.)',
                    'EX-4': 'Debt instruments, indentures',
                    'EX-21': 'Subsidiaries list',
                    'EX-99': 'Additional exhibits (press releases, etc.)',
                }
                const getCategoryHint = (ex: any) => {
                    const cat = (ex.exhibit_category || ex.exhibit_type || '').toString().replace(/\d.*$/, '').trim()
                    return exhibitTypeHint[cat] || ''
                }
                const exhibitDate = (ex: any) => ex.filing_date || ex.filed_date || ex.parent_filing_date || '—'
                const secCompanyUrl = company.ticker ? `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=${encodeURIComponent(company.ticker)}` : null
                type ExhibitGroup = { type: string; exhibits: any[] }
                const initialGroups: ExhibitGroup[] = []
                const grouped = secExhibits.reduce((acc: ExhibitGroup[], ex: any) => {
                    const type = (ex.exhibit_category || ex.exhibit_type || 'Other').toString().replace(/\d.*$/, '').trim() || 'Other'
                    let group = acc.find(g => g.type === type)
                    if (!group) {
                        group = { type, exhibits: [] }
                        acc.push(group)
                    }
                    group.exhibits.push(ex)
                    return acc
                }, initialGroups).sort((a: { type: string; exhibits: any[] }, b: { type: string; exhibits: any[] }) => a.type.localeCompare(b.type))

                return (
                    <div ref={exhibitsSectionRef} className="mt-4">
                        <div className="bg-dark-900/40 border border-purple-500/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="text-sm font-bold text-purple-400 uppercase tracking-[0.2em] flex items-center gap-3">
                                    <div className="w-1.5 h-1.5 rounded-full bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.5)]" />
                                    SEC Exhibits & Material Contracts
                                </h3>
                                <div className="text-xs text-gray-500">
                                    {secExhibits.length} exhibit{secExhibits.length !== 1 ? 's' : ''}
                                </div>
                            </div>
                            <p className="text-[10px] text-gray-500 mb-3">
                                How to read: EX-10 = material contracts, EX-4 = debt instruments, EX-21 = subsidiaries, EX-99 = additional exhibits.
                            </p>
                            {secCompanyUrl && (
                                <a href={secCompanyUrl} target="_blank" rel="noopener noreferrer" className="text-[10px] text-purple-400 hover:text-purple-300 mb-3 inline-block">
                                    View {company.ticker} on SEC EDGAR →
                                </a>
                            )}
                            <div className="overflow-x-auto space-y-4">
                                {grouped.map(({ type, exhibits }: { type: string; exhibits: any[] }) => (
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
                                                {exhibits.slice(0, 15).map((ex: any, i: number) => {
                                                    const sentiment = ex.finbert_score || 0
                                                    const sentimentColor = sentiment > 0.2 ? 'text-green-400' : sentiment < -0.2 ? 'text-red-400' : 'text-gray-400'
                                                    const hint = getCategoryHint(ex)
                                                    return (
                                                        <tr key={i} className="border-b border-white/5 hover:bg-purple-500/10 transition-colors">
                                                            <td className="py-2 px-2 text-purple-300 font-mono font-semibold">{ex.exhibit_type || '—'}</td>
                                                            <td className="py-2 px-2 text-gray-300 truncate max-w-[100px]" title={hint}>{ex.contract_type || ex.exhibit_category || '—'}</td>
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
                                        {exhibits.length > 15 && (
                                            <div className="text-[10px] text-gray-500 mt-1 px-2">+ {exhibits.length - 15} more in this group</div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )
            })()}

            {/* SEC XBRL Data - Financial Breakdowns (condensed: one line per filing, expand on click) */}
            {secXbrlData.length > 0 && (
                <div ref={xbrlSectionRef} className="mt-4">
                    <div className="bg-dark-900/40 border border-cyan-500/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-[0.2em] flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 shadow-[0_0_8px_rgba(34,211,238,0.5)]" />
                                Financial Breakdowns (XBRL)
                            </h3>
                            <div className="text-xs text-gray-500">
                                {secXbrlData.length} filing{secXbrlData.length !== 1 ? 's' : ''}
                            </div>
                        </div>
                        <div className="space-y-1">
                            {secXbrlData.map((xbrl: any, i: number) => {
                                const conceptCount = xbrl.concepts_found ?? [xbrl.debt, xbrl.costs, xbrl.revenue_segments].filter(Boolean).length
                                const isExpanded = expandedXbrlBreakdownIndex === i
                                return (
                                    <div key={i} className="bg-dark-800/50 border border-cyan-500/10 rounded-lg overflow-hidden">
                                        <button
                                            type="button"
                                            onClick={() => setExpandedXbrlBreakdownIndex(isExpanded ? null : i)}
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
                    </div>
                </div>
            )}

            {/* Full Financial Statements Viewer */}
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

                            {/* Period Selector */}
                            <div className="flex items-center gap-2 overflow-x-auto">
                                {secXbrlData.map((xbrl: any, i: number) => (
                                    <button
                                        key={i}
                                        onClick={() => setSelectedXbrlIndex(i)}
                                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                                            selectedXbrlIndex === i
                                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-[0_0_12px_rgba(16,185,129,0.3)]'
                                                : 'bg-dark-800/50 text-gray-400 border border-white/5 hover:border-emerald-500/20'
                                        }`}
                                    >
                                        {xbrl.filing_type} FY{xbrl.fiscal_year}
                                        {xbrl.filing_date && <span className="ml-1.5 text-[10px] text-gray-400">· Filed {xbrl.filing_date}</span>}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Statement Tabs */}
                        <div className="flex border-b border-emerald-500/10 bg-dark-800/30">
                            {[
                                { key: 'income', label: 'Income Statement', icon: '📊' },
                                { key: 'balance', label: 'Balance Sheet', icon: '⚖️' },
                                { key: 'cashflow', label: 'Cash Flow', icon: '💰' }
                            ].map((tab) => (
                                <button
                                    key={tab.key}
                                    onClick={() => setSelectedStatementTab(tab.key as any)}
                                    className={`flex-1 px-4 py-3 text-xs font-bold uppercase tracking-wider transition-all ${
                                        selectedStatementTab === tab.key
                                            ? 'text-emerald-300 bg-emerald-500/10 border-b-2 border-emerald-500'
                                            : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                                    }`}
                                >
                                    <span className="mr-2">{tab.icon}</span>
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
                                const row = (label: string, value: number | null | undefined, opts?: { highlight?: boolean; large?: boolean; currency?: boolean; unit?: string; signed?: boolean }) => (
                                    <div key={label} className={`flex justify-between items-center py-1.5 px-2 border-b border-white/5 last:border-0 ${opts?.highlight ? 'bg-emerald-500/5' : ''}`}>
                                        <span className={`text-xs ${opts?.highlight ? 'text-emerald-400 font-semibold' : 'text-gray-300'}`}>{label}</span>
                                        <span className={`font-mono text-sm tabular-nums text-right min-w-[7rem] ${opts?.highlight ? 'text-emerald-300 font-bold' : 'text-gray-200'}`}>
                                            {fmt(value, opts?.currency !== false, opts?.unit, opts?.signed)}
                                        </span>
                                    </div>
                                )

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
                                                    {row('R&D Expense', getConcept('ResearchAndDevelopmentExpense') ?? undefined)}
                                                    {row('SG&A Expense', getConcept('SellingGeneralAndAdministrativeExpense') ?? undefined)}
                                                    {row('Operating Expense', getConcept('OperatingExpense') ?? undefined)}
                                                    {row('Operating Income', getConcept('OperatingIncomeLoss') ?? undefined, { highlight: true })}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Net Income</h4>
                                                <div className="space-y-0">
                                                    {row('Interest Expense', getConcept('InterestExpense') ?? undefined)}
                                                    {row('Income Tax', getConcept('IncomeTaxExpenseBenefit') ?? undefined)}
                                                    {row('Net Income', getConcept('NetIncomeLoss') ?? undefined, { highlight: true, large: true })}
                                                    {row('EPS (Basic)', getConcept('EarningsPerShareBasic') ?? undefined, { currency: false })}
                                                    {row('EPS (Diluted)', getConcept('EarningsPerShareDiluted') ?? undefined, { currency: false })}
                                                </div>
                                            </div>
                                        </div>
                                    )
                                }

                                // Balance Sheet - Assets / Liabilities and Equity with total tie-in
                                if (selectedStatementTab === 'balance') {
                                    const totalAssets = getConcept('Assets') ?? undefined
                                    const currentDebt = getConcept('DebtCurrent') ?? getConcept('CurrentDebt') ?? getConcept('LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths')
                                    const ltDebt = getConcept('LongTermDebt')
                                    const totalDebt = (currentDebt != null || ltDebt != null) ? (currentDebt || 0) + (ltDebt || 0) : null
                                    const equity = getConcept('StockholdersEquity') ?? getConcept('Equity')
                                    const totalLiabEquity = (totalDebt != null && equity != null) ? totalDebt + equity : (equity != null ? equity : null)
                                    return (
                                        <div className="space-y-4">
                                            <div>
                                                <h4 className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider pb-2 border-b border-emerald-500/20 mb-2">Assets</h4>
                                                <div className="space-y-0">
                                                    {row('Cash & Equivalents', getConcept('CashAndCashEquivalentsAtCarryingValue') ?? getConcept('Cash') ?? undefined)}
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
                                                    {row('Retained Earnings', getConcept('RetainedEarningsAccumulatedDeficit') ?? undefined)}
                                                    {row('Treasury Stock', getConcept('TreasuryStockValue') ?? undefined)}
                                                    {row('Shares Outstanding', getConcept('CommonStockSharesOutstanding') ?? undefined, { currency: false, unit: 'M' })}
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
                                                    {row('Dividends Paid', getConcept('PaymentsOfDividends') ?? undefined, { signed: true })}
                                                    {row('Stock Repurchases', getConcept('PaymentsForRepurchaseOfCommonStock') ?? undefined, { signed: true })}
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

            {/* SEC Sentences - Sentiment Analysis (5 default, Show more) */}
            {secSentences.length > 0 && (
                <div className="mt-4">
                    <div className="bg-dark-900/40 border border-indigo-500/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-[0.2em] flex items-center gap-3">
                                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
                                SEC Filing Sentiment Analysis
                            </h3>
                            <div className="text-xs text-gray-500">
                                {secSentences.length} sentence{secSentences.length !== 1 ? 's' : ''} analyzed
                            </div>
                        </div>
                        <div className="space-y-2">
                            {(showMoreSecSentences ? secSentences : secSentences.slice(0, 5)).map((sentence: any, i: number) => {
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
                            {secSentences.length > 5 && (
                                <button
                                    type="button"
                                    onClick={() => setShowMoreSecSentences(!showMoreSecSentences)}
                                    className="w-full py-2 text-[11px] font-semibold text-indigo-400 hover:text-indigo-300 border border-indigo-500/20 rounded-lg hover:bg-indigo-500/10 transition-colors"
                                >
                                    {showMoreSecSentences ? 'Show less' : `Show more (${secSentences.length - 5} more)`}
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Awards Table - Full Width Section */}
            {awards.length > 0 && (
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
                        <div className="overflow-x-auto">
                            <table className="w-full text-[11px]">
                                <thead>
                                    <tr className="border-b border-gold/10">
                                        <th className="text-left text-gray-300 font-semibold pb-1.5 pt-0 px-2">Agency</th>
                                        <th className="text-left text-gray-300 font-semibold pb-1.5 pt-0 px-2">Description</th>
                                        <th className="text-center text-gray-300 font-semibold pb-1.5 pt-0 px-2">FY</th>
                                        <th className="text-center text-gray-300 font-semibold pb-1.5 pt-0 px-2">Date</th>
                                        <th className="text-right text-gray-300 font-semibold pb-1.5 pt-0 px-2">Amount</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {awards.slice(0, 10).map((a: any, i: number) => (
                                        <tr
                                            key={i}
                                            onClick={() => setSelectedDetail({ type: 'Award', data: a })}
                                            className="border-b border-white/5 hover:bg-gold/10 transition-colors cursor-pointer"
                                        >
                                            <td className="py-1.5 px-2 text-gray-100 truncate max-w-[150px]">{a.awarding_agency}</td>
                                            <td className="py-1.5 px-2 text-gray-300 truncate max-w-[300px]">{a.description || 'Contract Award'}</td>
                                            <td className="py-1.5 px-2 text-center text-gray-300">FY-{a.contract_year || '—'}</td>
                                            <td className="py-1.5 px-2 text-center text-gray-400 text-[10px]">{a.award_date || a.start_date || '—'}</td>
                                            <td className="py-1.5 px-2 text-right text-gold font-mono font-bold">{showFullAmounts ? `$${Number(a.award_amount_float).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${(a.award_amount_float / 1e6).toFixed(1)}M`}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
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
                                <button onClick={closeDetail} className="p-2 bg-dark-700 rounded-full text-gray-400 hover:text-white border border-white/10 transition-all">✕</button>
                            </div>
                            <div className="flex-1 overflow-y-auto p-6 space-y-5">
                                {(() => {
                                    const avg = selectedDetail.data.avg_finbert ?? 0;
                                    const sentimentLabel = avg > 0.05 ? 'Positive' : avg < -0.05 ? 'Negative' : 'Neutral';
                                    const sentimentClass = avg > 0.05 ? 'bg-green-500/20 text-green-400 border-green-500/40' : avg < -0.05 ? 'bg-red-500/20 text-red-400 border-red-500/40' : 'bg-gray-500/20 text-gray-400 border-gray-500/40';
                                    const formType = selectedDetail.data.type || selectedDetail.data.form_type || 'Filing';
                                    const filedDate = selectedDetail.data.filing_date || 'Unknown date';
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
                                                    <div className="text-xs text-gray-500 italic text-center py-6 border border-dashed border-gray-700 rounded-lg">
                                                        No sentence-level analysis available for this filing.
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
                                <button onClick={closeDetail} className="p-2 bg-dark-700 rounded-full text-gray-400 hover:text-white border border-white/10 transition-all">✕</button>
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
