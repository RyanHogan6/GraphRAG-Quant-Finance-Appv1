'use client'

import { useState, useMemo } from 'react'
import TimeSeriesChart from './TimeSeriesChart'

interface CompanyWorkupProps {
  data: any
  onCompare?: (ticker: string) => void
  peerData?: any
  comparisonMode?: boolean
}

export default function CompanyWorkup({ data, onCompare, peerData, comparisonMode = false }: CompanyWorkupProps) {
  const [timeframe, setTimeframe] = useState<'1M' | '3M' | '6M' | '1Y'>('1M')

  const company = data
  const isFiling = !!(data.type || data.filing_date)
  const marketData = data.MarketData || []
  const secFilings = data.sec_filings || (isFiling ? [data] : [])
  const secXbrlData = data.sec_xbrl_data || (isFiling && data.sec_xbrl_data_data ? data.sec_xbrl_data_data : [])
  const secExhibits = data.sec_exhibits || (isFiling && data.sec_exhibits_data ? data.sec_exhibits_data : [])
  const awards = data.Award || []
  const optionsFlow = data.options_flow || []
  const polyMarkets = data.prediction_markets_polymarket || []

  const latestMarket = marketData[0] || {}
  const latestOptions = optionsFlow[0]
  const ticker = company.ticker || data.ticker || '—'
  const sector = company.sector || data.sector
  const industry = company.industry || data.industry

  const chartSeries = useMemo(() => {
    const now = new Date()
    const cut = new Date(now)
    if (timeframe === '1M') cut.setMonth(now.getMonth() - 1)
    else if (timeframe === '3M') cut.setMonth(now.getMonth() - 3)
    else if (timeframe === '6M') cut.setMonth(now.getMonth() - 6)
    else cut.setFullYear(now.getFullYear() - 1)
    const filtered = marketData
      .filter((d: any) => new Date(d.date) >= cut)
      .sort((a: any, b: any) => new Date(a.date).getTime() - new Date(b.date).getTime())
    const values = filtered.map((d: any) => (typeof d.close === 'number' ? d.close : Number(d.close)) || 0)
    return { dates: filtered.map((d: any) => d.date), values, label: `${ticker}`, ticker }
  }, [marketData, timeframe, ticker])

  const totalAwardValue = useMemo(() => awards.reduce((s: number, a: any) => s + (a.award_amount_float || 0), 0), [awards])

  const fmt = (val: number | null | undefined, asCurrency = true) => {
    if (val == null) return '—'
    if (!asCurrency) return String(val)
    if (Math.abs(val) >= 1e9) return `$${(val / 1e9).toFixed(2)}B`
    if (Math.abs(val) >= 1e6) return `$${(val / 1e6).toFixed(1)}M`
    return `$${Number(val).toLocaleString()}`
  }

  const getConcept = (xbrl: any, name: string): number | null | undefined => {
    if (!xbrl) return undefined
    const fromBucket = xbrl.costs?.[name] ?? xbrl.debt?.[name] ?? xbrl.equity?.[name] ?? xbrl.cashflow?.[name]
    if (typeof fromBucket === 'number') return fromBucket
    const fromAll = xbrl.all_concepts?.[name]
    if (typeof fromAll === 'number') return fromAll
    if (Array.isArray(fromAll) && fromAll.length) {
      const v = fromAll[0]?.value
      return typeof v === 'number' ? v : null
    }
    return undefined
  }

  const latestXbrl = secXbrlData[0]
  const rev = latestXbrl ? (getConcept(latestXbrl, 'Revenues') ?? getConcept(latestXbrl, 'RevenueFromContractWithCustomerExcludingAssessedTax')) : null
  const netIncome = latestXbrl ? getConcept(latestXbrl, 'NetIncomeLoss') : null
  const opCash = latestXbrl ? getConcept(latestXbrl, 'NetCashProvidedByUsedInOperatingActivities') : null

  return (
    <div className="w-full space-y-6 pb-8 text-sm">
      {/* 1. Identity and context */}
      <section className="border-b border-white/10 pb-4">
        <div className="flex flex-wrap items-baseline gap-2">
          <h1 className="text-2xl font-bold text-white tracking-tight">{company.company || 'Company'}</h1>
          <span className="font-mono text-base text-amber-400/90 font-semibold">{ticker}</span>
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0 text-xs text-gray-400">
          {sector && <span>{sector}</span>}
          {industry && <span>{industry}</span>}
          {(company.city || company.country) && (
            <span>{[company.city, company.country].filter(Boolean).join(', ')}</span>
          )}
        </div>
        <p className="mt-2 text-xs text-gray-500 max-w-2xl">
          {sector && industry ? `${company.company || ticker} is a ${industry} company in the ${sector} sector.` : 'Company overview.'}
        </p>
      </section>

      {/* 2. Sector stacking (optional placeholder) */}
      {sector && (
        <section className="rounded-lg border border-white/10 bg-white/[0.02] px-4 py-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Sector</span>
            <span className="text-xs text-amber-400/80">{sector}</span>
          </div>
          <p className="mt-2 text-[11px] text-gray-500">
            Compare to sector — coming soon. Use &quot;Compare peer&quot; for now.
          </p>
          {onCompare && (
            <button
              type="button"
              onClick={() => onCompare?.(ticker)}
              className="mt-2 text-[11px] text-amber-400 hover:text-amber-300"
            >
              Compare to another company →
            </button>
          )}
        </section>
      )}

      {/* 3. Market view */}
      <section className="rounded-xl border border-white/10 bg-dark-900/40 p-4">
        <h2 className="text-xs font-bold text-gray-300 uppercase tracking-wider mb-3">Market view</h2>
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {['1M', '3M', '6M', '1Y'].map((tf) => (
            <button
              key={tf}
              type="button"
              onClick={() => setTimeframe(tf as any)}
              className={`px-3 py-1 rounded text-xs font-medium ${timeframe === tf ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40' : 'text-gray-500 border border-white/10 hover:text-gray-300'}`}
            >
              {tf}
            </button>
          ))}
        </div>
        {chartSeries.values.length > 0 ? (
          <div className="h-64 w-full">
            <TimeSeriesChart
              dates={chartSeries.dates}
              values={chartSeries.values}
              label={chartSeries.label}
              ticker={chartSeries.ticker}
            />
          </div>
        ) : (
          <div className="h-64 flex items-center justify-center text-gray-500 text-xs border border-dashed border-white/10 rounded-lg">
            No price data
          </div>
        )}
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          {latestMarket?.close != null && (
            <div>
              <div className="text-gray-500 uppercase tracking-wider">Price</div>
              <div className="font-mono text-white">${Number(latestMarket.close).toFixed(2)}</div>
            </div>
          )}
          {latestOptions && (
            <>
              <div>
                <div className="text-gray-500 uppercase tracking-wider">P/C ratio</div>
                <div className="font-mono text-white">
                  {(latestOptions.put_call_ratio ?? latestOptions.put_call_volume_ratio) != null
                    ? Number(latestOptions.put_call_ratio ?? latestOptions.put_call_volume_ratio).toFixed(2)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="text-gray-500 uppercase tracking-wider">IV</div>
                <div className="font-mono text-white">
                  {(latestOptions.implied_volatility ?? latestOptions.iv_avg) != null
                    ? `${(Number(latestOptions.implied_volatility ?? latestOptions.iv_avg) * 100).toFixed(1)}%`
                    : '—'}
                </div>
              </div>
            </>
          )}
          {polyMarkets[0] && (
            <div>
              <div className="text-gray-500 uppercase tracking-wider">Prediction</div>
              <div className="font-mono text-white">{(polyMarkets[0].yes_probability * 100).toFixed(0)}%</div>
            </div>
          )}
        </div>
      </section>

      {/* 4. Regulatory and disclosure */}
      {secFilings.length > 0 && (
        <section className="rounded-xl border border-white/10 bg-dark-900/40 p-4">
          <h2 className="text-xs font-bold text-gray-300 uppercase tracking-wider mb-3">Regulatory & disclosure</h2>
          <ul className="space-y-1.5">
            {secFilings.slice(0, 5).map((f: any, i: number) => (
              <li key={i} className="flex items-center justify-between text-xs">
                <span className="font-mono text-amber-400/90">{f.type || f.form_type}</span>
                <span className="text-gray-500">{f.filing_date || '—'}</span>
              </li>
            ))}
          </ul>
          {secFilings[0]?.avg_finbert != null && (
            <p className="mt-2 text-[11px] text-gray-400">
              Latest filing sentiment: {(secFilings[0].avg_finbert > 0.05 ? 'Positive' : secFilings[0].avg_finbert < -0.05 ? 'Negative' : 'Neutral')} ({(secFilings[0].avg_finbert).toFixed(3)})
            </p>
          )}
          {(secFilings[0]?.top_sentences?.length || secFilings[0]?.sec_sentences?.length) ? (
            <div className="mt-3 space-y-1">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider">Key excerpt</div>
              <p className="text-xs text-gray-300 italic">
                &quot;{(secFilings[0].top_sentences?.[0]?.text || secFilings[0].sec_sentences?.[0]?.text || '').slice(0, 120)}...&quot;
              </p>
            </div>
          ) : null}
        </section>
      )}

      {/* 5. Financials */}
      {(secXbrlData.length > 0 || latestMarket?.revenue_growth != null) && (
        <section className="rounded-xl border border-white/10 bg-dark-900/40 p-4">
          <h2 className="text-xs font-bold text-gray-300 uppercase tracking-wider mb-3">Financials</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {rev != null && (
              <div>
                <div className="text-gray-500 uppercase tracking-wider">Revenue</div>
                <div className="font-mono text-white">{fmt(rev)}</div>
              </div>
            )}
            {netIncome != null && (
              <div>
                <div className="text-gray-500 uppercase tracking-wider">Net income</div>
                <div className="font-mono text-white">{fmt(netIncome)}</div>
              </div>
            )}
            {opCash != null && (
              <div>
                <div className="text-gray-500 uppercase tracking-wider">Operating CF</div>
                <div className="font-mono text-white">{fmt(opCash)}</div>
              </div>
            )}
            {latestMarket?.profit_margins != null && (
              <div>
                <div className="text-gray-500 uppercase tracking-wider">Profit margin</div>
                <div className="font-mono text-white">{(latestMarket.profit_margins * 100).toFixed(1)}%</div>
              </div>
            )}
          </div>
          {secXbrlData.length > 0 && (
            <p className="mt-2 text-[11px] text-gray-500">
              {secXbrlData.length} filing{secXbrlData.length !== 1 ? 's' : ''} with XBRL data. Drill into statements in full workup.
            </p>
          )}
        </section>
      )}

      {/* 6. Government and contracts */}
      {(awards.length > 0 || secExhibits.length > 0) && (
        <section className="rounded-xl border border-white/10 bg-dark-900/40 p-4">
          <h2 className="text-xs font-bold text-gray-300 uppercase tracking-wider mb-3">Government & contracts</h2>
          {awards.length > 0 && (
            <div className="mb-3">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                <span><strong className="text-white">{awards.length}</strong> award{awards.length !== 1 ? 's' : ''}</span>
                <span className="text-gray-400">Total: {fmt(totalAwardValue)}</span>
              </div>
              <ul className="mt-2 space-y-1 text-[11px] text-gray-400">
                {awards.slice(0, 3).map((a: any, i: number) => (
                  <li key={i}>{a.awarding_agency} — {fmt(a.award_amount_float)} (FY-{a.contract_year || '—'})</li>
                ))}
              </ul>
            </div>
          )}
          {secExhibits.length > 0 && (
            <div className="flex items-center justify-between text-xs">
              <span><strong className="text-white">{secExhibits.length}</strong> exhibit{secExhibits.length !== 1 ? 's' : ''}</span>
              {ticker && (
                <a
                  href={`https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=${encodeURIComponent(ticker)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-amber-400 hover:text-amber-300"
                >
                  View on SEC EDGAR →
                </a>
              )}
            </div>
          )}
        </section>
      )}

      {/* Comparison strip */}
      {comparisonMode && peerData && (
        <section className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4">
          <h2 className="text-xs font-bold text-blue-400 uppercase tracking-wider mb-3">vs {peerData.ticker}</h2>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <div className="text-gray-500 mb-1">Price</div>
              <div className="font-mono text-white">{latestMarket?.close != null ? `$${Number(latestMarket.close).toFixed(2)}` : '—'}</div>
            </div>
            <div>
              <div className="text-gray-500 mb-1">{peerData.ticker} price</div>
              <div className="font-mono text-blue-300">
                {peerData.MarketData?.[0]?.close != null ? `$${Number(peerData.MarketData[0].close).toFixed(2)}` : '—'}
              </div>
            </div>
          </div>
          {onCompare && (
            <button
              type="button"
              onClick={() => onCompare?.(ticker)}
              className="mt-3 text-[11px] text-blue-400 hover:text-blue-300"
            >
              Exit comparison
            </button>
          )}
        </section>
      )}

      {/* Empty state when nothing to show */}
      {secFilings.length === 0 && secXbrlData.length === 0 && awards.length === 0 && secExhibits.length === 0 && marketData.length === 0 && (
        <p className="text-xs text-gray-500 text-center py-8">No enriched data for this company yet.</p>
      )}
    </div>
  )
}
