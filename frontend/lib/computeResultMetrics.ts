/**
 * Compute summary metrics for a result set by display family.
 * Used by ResultSummaryStrip above charts/tables (VQB + NL).
 */

import type { DisplayFamily } from './displayFamily'

export interface ResultMetric {
  label: string
  value: string | number
  /** Optional: good | bad | neutral for color */
  status?: 'good' | 'bad' | 'neutral'
}

function fmtNum(n: number, decimals = 2): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return n.toLocaleString(undefined, { maximumFractionDigits: decimals })
}

function fmtPct(n: number): string {
  return `${(n * 100).toFixed(2)}%`
}

/** Infer date key and primary numeric key from first row */
function inferKeys(results: Record<string, any>[]): { dateKey: string | null; numericKeys: string[] } {
  if (!results.length) return { dateKey: null, numericKeys: [] }
  const keys = Object.keys(results[0]).filter(k => !k.startsWith('_'))
  const dateCandidates = ['date', 'Date', 'report_date', 'as_of_date', 'filing_date', 'datetime']
  const dateKey = dateCandidates.find(k => keys.includes(k)) ?? null
  const skip = new Set(['year', 'month', 'day_of_week', 'day_of_month', 'outcome_index'])
  const numericKeys = keys.filter(k => {
    const v = results[0][k]
    return typeof v === 'number' && !Number.isNaN(v) && !skip.has(k)
  })
  return { dateKey, numericKeys }
}

export function computeResultMetrics(
  results: any[],
  displayFamily: DisplayFamily,
  _queryPlan?: { chart_data?: any }
): ResultMetric[] {
  if (!results?.length) return []
  const rows = results as Record<string, any>[]
  const { dateKey, numericKeys } = inferKeys(rows)

  switch (displayFamily) {
    case 'time_series':
    case 'ohlc_candlestick':
    case 'economic_data': {
      if (!dateKey) return []
      const sorted = [...rows].sort((a, b) => String(a[dateKey]).localeCompare(String(b[dateKey])))
      const primaryKey = numericKeys.find(k => ['close', 'Close', 'value', 'open'].includes(k)) ?? numericKeys[0]
      if (!primaryKey) return []
      const values = sorted.map(r => Number(r[primaryKey])).filter(n => !Number.isNaN(n))
      if (!values.length) return []
      const min = Math.min(...values)
      const max = Math.max(...values)
      const mean = values.reduce((a, b) => a + b, 0) / values.length
      const start = sorted[0]?.[dateKey]
      const end = sorted[sorted.length - 1]?.[dateKey]
      const pctChange = values.length >= 2 && values[0] !== 0
        ? (values[values.length - 1] - values[0]) / values[0]
        : null
      const volatility = values.length >= 5
        ? Math.sqrt(values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length) / (mean || 1)
        : null
      const out: ResultMetric[] = [
        { label: 'Min', value: typeof min === 'number' ? (min >= 1e6 ? fmtNum(min) : min.toFixed(2)) : '—' },
        { label: 'Max', value: typeof max === 'number' ? (max >= 1e6 ? fmtNum(max) : max.toFixed(2)) : '—' },
        { label: 'Avg', value: (mean >= 1e6 ? fmtNum(mean) : mean.toFixed(2)) as string },
        { label: 'Start', value: start != null ? String(start).slice(0, 10) : '—' },
        { label: 'End', value: end != null ? String(end).slice(0, 10) : '—' }
      ]
      if (pctChange != null) {
        out.push({
          label: '% Chg',
          value: fmtPct(pctChange),
          status: pctChange >= 0 ? 'good' : 'bad'
        })
      }
      if (volatility != null) out.push({ label: 'Vol', value: fmtPct(volatility) })
      return out
    }

    case 'awards_list': {
      const amounts = rows.map(r => r.award_amount_float ?? r.award_amount ?? 0).filter((n: number) => typeof n === 'number' && n > 0)
      const total = amounts.reduce((a, b) => a + b, 0)
      const count = rows.length
      const avg = count ? total / count : 0
      const largest = amounts.length ? Math.max(...amounts) : 0
      const agencies = rows.map(r => r.awarding_agency || r.agency).filter(Boolean)
      const byAgency: Record<string, number> = {}
      agencies.forEach((ag, i) => {
        const amt = amounts[i] ?? 0
        byAgency[ag] = (byAgency[ag] || 0) + amt
      })
      const topAgency = Object.entries(byAgency).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—'
      return [
        { label: 'Total', value: `$${fmtNum(total)}` },
        { label: 'Count', value: count },
        { label: 'Avg', value: `$${fmtNum(avg)}` },
        { label: 'Largest', value: `$${fmtNum(largest)}` },
        { label: 'Top Agency', value: topAgency.length > 20 ? topAgency.slice(0, 17) + '…' : topAgency }
      ]
    }

    case 'options_flow_list': {
      const ratios = rows.map(r => r.put_call_volume_ratio ?? r.put_call_ratio).filter((n: any) => typeof n === 'number' && !Number.isNaN(n))
      const unusual = rows.filter(r => r.unusual_total_activity || r.unusual_call_volume || r.unusual_put_volume).length
      const tickersUnusual = new Set(rows.filter(r => r.unusual_total_activity || r.unusual_call_volume || r.unusual_put_volume).map(r => r.ticker).filter(Boolean))
      const avgPc = ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : null
      return [
        { label: 'Rows', value: rows.length },
        { label: 'Avg P/C Ratio', value: avgPc != null ? avgPc.toFixed(2) : '—' },
        { label: 'Unusual Count', value: unusual },
        { label: 'Tickers w/ Unusual', value: tickersUnusual.size }
      ]
    }

    case 'futures_commodities':
    case 'eia_energy': {
      const closeKey = numericKeys.find(k => ['close', 'Close', 'crude_stocks', 'total_stocks', 'value'].includes(k)) ?? numericKeys[0]
      if (!closeKey) return []
      const withDate = dateKey ? [...rows].sort((a, b) => String(a[dateKey]).localeCompare(String(b[dateKey]))) : rows
      const values = withDate.map(r => Number(r[closeKey])).filter(n => !Number.isNaN(n))
      if (!values.length) return []
      const latest = values[values.length - 1]
      const fiveAgo = values.length >= 5 ? values[values.length - 5] : values[0]
      const twentyAgo = values.length >= 20 ? values[values.length - 20] : values[0]
      const chg5 = fiveAgo !== 0 ? (latest - fiveAgo) / fiveAgo : null
      const chg20 = twentyAgo !== 0 ? (latest - twentyAgo) / twentyAgo : null
      const out: ResultMetric[] = [
        { label: 'Latest', value: latest >= 1e6 ? fmtNum(latest) : latest.toFixed(2) },
        { label: 'Rows', value: rows.length }
      ]
      if (chg5 != null) out.push({ label: '5d Chg', value: fmtPct(chg5), status: chg5 >= 0 ? 'good' : 'bad' })
      if (chg20 != null) out.push({ label: '20d Chg', value: fmtPct(chg20), status: chg20 >= 0 ? 'good' : 'bad' })
      return out
    }

    case 'prediction_markets_list': {
      const probs = rows.map(r => r.yes_probability ?? r.yes_price).filter((n: any) => typeof n === 'number' && !Number.isNaN(n))
      const volumes = rows.map(r => r.volume_24h ?? r.volume).filter((n: any) => typeof n === 'number' && !Number.isNaN(n))
      const avgProb = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null
      const totalVol = volumes.reduce((a, b) => a + b, 0)
      const byCat: Record<string, number> = {}
      rows.forEach(r => {
        const c = r.category ?? r.category_name ?? 'Other'
        byCat[c] = (byCat[c] || 0) + 1
      })
      const topCat = Object.entries(byCat).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—'
      return [
        { label: 'Markets', value: rows.length },
        { label: 'Avg Prob', value: avgProb != null ? fmtPct(avgProb) : '—' },
        { label: 'Total Vol', value: totalVol >= 1e6 ? fmtNum(totalVol) : String(totalVol) },
        { label: 'Top Category', value: topCat.length > 12 ? topCat.slice(0, 10) + '…' : topCat }
      ]
    }

    case 'company_screener':
    case 'sec_filings_list':
    case 'generic': {
      if (!numericKeys.length) return [{ label: 'Rows', value: rows.length }]
      const firstNum = numericKeys[0]
      const nums = rows.map(r => Number(r[firstNum])).filter(n => !Number.isNaN(n))
      const sum = nums.reduce((a, b) => a + b, 0)
      const avg = nums.length ? sum / nums.length : 0
      const lk = labelKey(rows)
      const topN = rows.length >= 5 ? rows.slice(0, 5).map(r => r[lk] ?? r.ticker ?? '—').join(', ') : null
      const out: ResultMetric[] = [
        { label: 'Rows', value: rows.length },
        { label: 'Sum', value: sum >= 1e6 ? fmtNum(sum) : sum.toFixed(2) },
        { label: 'Avg', value: avg >= 1e6 ? fmtNum(avg) : avg.toFixed(2) }
      ]
      if (topN) out.push({ label: 'Top 5', value: topN.length > 30 ? topN.slice(0, 27) + '…' : topN })
      return out
    }

    case 'positioning_cot': {
      const oi = rows.map(r => r.Open_Interest_All ?? r.open_interest).filter((n: any) => typeof n === 'number')
      const totalOI = oi.reduce((a, b) => a + b, 0)
      const avgOI = oi.length ? totalOI / oi.length : 0
      const markets = new Set(rows.map(r => r.Market_and_Exchange_Names).filter(Boolean))
      return [
        { label: 'Reports', value: rows.length },
        { label: 'Avg OI', value: fmtNum(avgOI) },
        { label: 'Markets', value: markets.size }
      ]
    }

    case 'probability_timeline': {
      const probs = rows.map(r => r.yes_price ?? r.yes_probability).filter((n: any) => typeof n === 'number' && !Number.isNaN(n))
      const avg = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null
      const min = probs.length ? Math.min(...probs) : null
      const max = probs.length ? Math.max(...probs) : null
      return [
        { label: 'Points', value: rows.length },
        { label: 'Avg Prob', value: avg != null ? fmtPct(avg) : '—' },
        { label: 'Range', value: min != null && max != null ? `${fmtPct(min)} – ${fmtPct(max)}` : '—' }
      ]
    }

    default:
      return [{ label: 'Rows', value: rows.length }]
  }
}

function labelKey(rows: Record<string, any>[]): string {
  if (!rows.length) return 'label'
  const k = ['ticker', 'company', 'question', 'title', 'commodity'].find(x => rows[0][x] != null)
  return k ?? 'label'
}

