'use client'

import { useParams, useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import Link from 'next/link'
import { api } from '@/lib/api'

export default function MarketWorkupPage() {
  const params = useParams()
  const searchParams = useSearchParams()
  const marketId = typeof params?.marketId === 'string' ? params.marketId : ''
  const platform = (searchParams?.get('platform') as 'kalshi' | 'polymarket') || 'kalshi'

  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!marketId) {
      setLoading(false)
      setError('Missing market ID')
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    api.getMarketWorkup(marketId, platform)
      .then((payload) => {
        if (!cancelled) setData(payload)
      })
      .catch((e) => {
        if (!cancelled) setError(e.message || 'Failed to load workup')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [marketId, platform])

  if (!marketId) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-4">Missing market ID.</p>
          <Link href="/research" className="text-gold hover:underline">Back to Research</Link>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-2 border-gold/40 border-t-gold rounded-full animate-spin mb-4" />
          <p className="text-gray-400">Loading market workup…</p>
          <Link href="/research" className="text-gold hover:underline mt-2 inline-block">Back to Research</Link>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-300 mb-4">{error || 'No data'}</p>
          <Link href="/research" className="text-gold hover:underline">Back to Research</Link>
        </div>
      </div>
    )
  }

  const market = data.market || {}
  const yesPct = market.yes_probability != null ? (typeof market.yes_probability === 'number' ? market.yes_probability * 100 : market.yes_probability) : null

  const Card = ({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) => (
    <div className={`rounded-lg border border-gold/20 bg-dark-800 overflow-hidden ${className}`}>
      <h3 className="px-4 py-3 border-b border-gold/20 text-gold font-mono font-medium">{title}</h3>
      <div className="p-4 text-gray-300 text-sm">{children}</div>
    </div>
  )

  return (
    <div className="min-h-screen bg-dark-900 text-gray-100">
      <div className="container mx-auto px-4 md:px-6 py-8">
        <div className="mb-6">
          <Link href="/research" className="text-gold/80 hover:text-gold text-sm font-medium">← Back to Research</Link>
        </div>

        {/* Header */}
        <div className="rounded-lg border border-gold/20 bg-dark-800 p-6 mb-8">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <span className="text-xs uppercase tracking-wider text-gold/70 border border-gold/30 px-2 py-0.5 rounded">{platform}</span>
              <span className="ml-2 text-xs text-gray-500">{data.theme || 'other'}</span>
            </div>
            {yesPct != null && (
              <div className="text-right">
                <span className="text-2xl font-mono font-bold text-gold">{Number(yesPct).toFixed(0)}%</span>
                <span className="text-gray-400 ml-1">Yes</span>
              </div>
            )}
          </div>
          <h1 className="text-xl md:text-2xl font-semibold text-gray-100 mt-3">{market.question || 'Market'}</h1>
          <div className="flex flex-wrap gap-4 mt-2 text-sm text-gray-400">
            {market.volume != null && <span>Volume: {Number(market.volume).toLocaleString()}</span>}
            {market.open_interest != null && <span>Open interest: {Number(market.open_interest).toLocaleString()}</span>}
            {market.end_date && <span>End: {market.end_date}</span>}
          </div>
        </div>

        {/* Panels */}
        <div className="grid gap-6 md:grid-cols-1 lg:grid-cols-2">
          <Card title="Macro context">
            {data.macro_data?.length ? (
              <ul className="space-y-1">
                {(data.macro_data as any[]).slice(0, 10).map((row: any, i: number) => (
                  <li key={i} className="flex justify-between gap-2">
                    <span>{row.date}</span>
                    <span className="text-gold/90">
                      {row.federal_funds_rate != null && `Fed: ${row.federal_funds_rate}% `}
                      {row.unemployment_rate != null && `Unemp: ${row.unemployment_rate}% `}
                      {row.sandp_500_index != null && `S&P: ${Number(row.sandp_500_index).toFixed(0)}`}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500">No macro data for this theme.</p>
            )}
          </Card>

          <Card title="Options flow signal">
            {data.options_signal?.length ? (
              <ul className="space-y-1">
                {(data.options_signal as any[]).slice(0, 12).map((row: any, i: number) => (
                  <li key={i} className="flex justify-between gap-2">
                    <span>{row.ticker} · {row.date}</span>
                    <span className="text-gold/90">
                      P/C: {row.put_call_volume_ratio != null ? Number(row.put_call_volume_ratio).toFixed(2) : '—'}
                      {row.unusual_call_activity === 1 && ' · unusual call'}
                      {row.unusual_total_activity === 1 && ' · unusual total'}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500">No options data for this theme.</p>
            )}
          </Card>

          <Card title="SEC sentiment (sector)">
            {data.sec_sentiment?.length ? (
              <ul className="space-y-1">
                {(data.sec_sentiment as any[]).map((row: any, i: number) => (
                  <li key={i}>
                    {row.sector}: avg FinBERT {row.avg_finbert != null ? Number(row.avg_finbert).toFixed(3) : '—'} ({row.sentence_count ?? 0} sentences)
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500">No SEC sentiment for this theme.</p>
            )}
          </Card>

          <Card title="Government contract signal">
            {data.contract_signal?.length ? (
              <ul className="space-y-1">
                {(data.contract_signal as any[]).map((row: any, i: number) => (
                  <li key={i}>
                    {row.week}: {row.award_count} awards, ${row.total_value != null ? (Number(row.total_value) / 1e6).toFixed(1) : '—'}M
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500">No contract data (shown for government-spending theme).</p>
            )}
          </Card>

          {data.theme === 'government_spending' && (
            <Card title="Congressional trading">
              {data.congressional_trades?.length ? (
                <ul className="space-y-1">
                  {(data.congressional_trades as any[]).slice(0, 15).map((row: any, i: number) => (
                    <li key={i}>
                      {row.politician_name} ({row.chamber}) · {row.date} · {row.transaction_type} {row.ticker} · {row.amount_range}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-500">No congressional trade data for this theme.</p>
              )}
            </Card>
          )}

          <Card title="Recent news" className="lg:col-span-2">
            {data.news_context?.summary ? (
              <div>
                <p className="whitespace-pre-wrap text-gray-300">{data.news_context.summary}</p>
                {data.news_context.sources?.length > 0 && (
                  <ul className="mt-3 space-y-1 text-gold/80 text-xs">
                    {data.news_context.sources.slice(0, 5).map((url: string, i: number) => (
                      <li key={i}>
                        <a href={url} target="_blank" rel="noopener noreferrer" className="hover:underline truncate block max-w-full">{url}</a>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ) : (
              <p className="text-gray-500">No recent news (Perplexity may be unavailable).</p>
            )}
          </Card>

          <Card title="Cross-source insight" className="lg:col-span-2">
            {data.cross_source_insight ? (
              <p className="whitespace-pre-wrap text-gray-200">{data.cross_source_insight}</p>
            ) : (
              <p className="text-gray-500">Synthesis unavailable.</p>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
