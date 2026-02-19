'use client'

import { useState, useEffect } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

const API_BASE = process.env.NEXT_PUBLIC_MEDICAID_API_URL || 'http://localhost:8001'

type Stats = {
  collections: Record<string, number>
  edges: Record<string, number>
  graph_name: string
  graph_exists: boolean
} | null

type RiskProvider = {
  npi: string
  _key: string
  name?: string
  risk_score: number
  total_spending?: number
  claim_count?: number
  in_LEIE?: boolean
}

type RiskScores = { providers: RiskProvider[]; total: number } | null

type OigProvider = {
  npi: string
  _key: string
  name?: string
  exclusion_type?: string
  exclusion_date?: string
  waiver_date?: string
  in_LEIE?: boolean
}

type OigResponse = { providers: OigProvider[]; total: number } | null

type Cluster = {
  address_key: string
  address_1?: string
  city?: string
  state?: string
  zip?: string
  provider_count: number
  npis: string[]
}

type ClustersResponse = { clusters: Cluster[]; total: number } | null

type SpendingProvider = {
  npi: string
  _key: string
  name?: string
  total_spending?: number
  claim_count?: number
  in_LEIE?: boolean
}

type SpendingResponse = {
  by_spending: SpendingProvider[]
  by_claims: SpendingProvider[]
} | null

export default function MedicaidDashboardPage() {
  const [stats, setStats] = useState<Stats>(null)
  const [riskScores, setRiskScores] = useState<RiskScores>(null)
  const [oig, setOig] = useState<OigResponse>(null)
  const [clusters, setClusters] = useState<ClustersResponse>(null)
  const [spending, setSpending] = useState<SpendingResponse>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [riskTop, setRiskTop] = useState(50)

  useEffect(() => {
    const fetchAll = async () => {
      setLoading(true)
      setError(null)
      try {
        const [sRes, rRes, oRes, cRes, spRes] = await Promise.all([
          fetch(`${API_BASE}/stats`),
          fetch(`${API_BASE}/risk-scores?top=${riskTop}`),
          fetch(`${API_BASE}/oig-providers`),
          fetch(`${API_BASE}/same-address-clusters`),
          fetch(`${API_BASE}/spending-top?limit=100`),
        ])
        if (!sRes.ok) throw new Error(`API ${sRes.status}: ${sRes.statusText}`)
        setStats(await sRes.json())
        setRiskScores(await rRes.json())
        setOig(await oRes.json())
        setClusters(await cRes.json())
        setSpending(await spRes.json())
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load dashboard')
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
  }, [riskTop])

  if (loading && !stats) {
    return (
      <div className="container mx-auto px-6 py-8">
        <div className="text-gold text-xl">Loading Medicaid dashboard...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto px-6 py-8">
        <div className="bg-dark-800 border border-red-500/50 rounded-lg p-6 text-red-400">
          <strong>Error:</strong> {error}. Ensure the Medicaid API is running: <code className="text-gray-400">cd medicaid_fraud && uvicorn api.main:app --port 8001</code>
        </div>
      </div>
    )
  }

  const riskHistogram = (riskScores?.providers ?? []).length
    ? (() => {
        const bins: { bin: string; count: number }[] = []
        const scores = riskScores!.providers.map((p) => p.risk_score)
        const min = Math.min(...scores)
        const max = Math.max(...scores)
        const step = (max - min) / 10 || 0.1
        for (let i = 0; i < 10; i++) {
          const lo = min + i * step
          const hi = i === 9 ? max + 0.001 : min + (i + 1) * step
          const count = scores.filter((s) => s >= lo && s < hi).length
          bins.push({ bin: lo.toFixed(2), count })
        }
        return bins
      })()
    : []

  const spendingBars = (spending?.by_spending ?? [])
    .slice(0, 15)
    .map((p) => ({
      name: (p.name || p.npi || p._key).slice(0, 20),
      value: p.total_spending ?? 0,
    }))

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gold mb-2">Medicaid Fraud Dashboard</h1>
        <p className="text-gray-500">Pipeline overview, graph stats, ML risk scores, OIG exclusions, same-address clusters, spending</p>
      </div>

      {/* Overview / Stats */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gold mb-4">Pipeline overview</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {stats &&
            Object.entries(stats.collections).map(([name, count]) => (
              <div key={name} className="bg-dark-800 border border-gold/20 rounded-lg p-4">
                <div className="text-gray-500 text-sm truncate">{name}</div>
                <div className="text-2xl font-bold text-gold">{typeof count === 'number' ? count.toLocaleString() : count}</div>
              </div>
            ))}
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          {stats &&
            Object.entries(stats.edges).map(([name, count]) => (
              <div key={name} className="bg-dark-800 border border-gold/20 rounded-lg p-4">
                <div className="text-gray-500 text-sm">edges {name}</div>
                <div className="text-2xl font-bold text-gold">{typeof count === 'number' ? count.toLocaleString() : count}</div>
              </div>
            ))}
          {stats && (
            <div className="bg-dark-800 border border-gold/20 rounded-lg p-4">
              <div className="text-gray-500 text-sm">Graph</div>
              <div className="text-lg font-bold text-gold">{stats.graph_exists ? stats.graph_name : '—'}</div>
            </div>
          )}
        </div>
      </section>

      {/* ML Risk scores */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gold mb-4">ML risk scores</h2>
        <div className="mb-4 flex items-center gap-4">
          <label className="text-gray-400 text-sm">Top</label>
          <select
            value={riskTop}
            onChange={(e) => setRiskTop(Number(e.target.value))}
            className="bg-dark-800 border border-gold/20 rounded px-3 py-1 text-gold"
          >
            {[25, 50, 100, 200].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
          <span className="text-gray-500 text-sm">({riskScores?.total ?? 0} with risk_score)</span>
        </div>
        {riskHistogram.length > 0 && (
          <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 mb-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskHistogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(212,175,55,0.1)" />
                <XAxis dataKey="bin" stroke="#9ca3af" fontSize={11} />
                <YAxis stroke="#9ca3af" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid rgba(212,175,55,0.3)' }} />
                <Bar dataKey="count" fill="rgba(212,175,55,0.6)" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full border border-gold/20 rounded-lg overflow-hidden">
            <thead>
              <tr className="bg-dark-700 text-gold text-left text-sm">
                <th className="p-3">NPI</th>
                <th className="p-3">Name</th>
                <th className="p-3">Risk score</th>
                <th className="p-3">Spending</th>
                <th className="p-3">Claims</th>
                <th className="p-3">in_LEIE</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              {(riskScores?.providers ?? []).map((p) => (
                <tr key={p._key} className="border-t border-gold/10 hover:bg-dark-700/50">
                  <td className="p-3 font-mono text-sm">{p.npi}</td>
                  <td className="p-3 max-w-[200px] truncate">{p.name || '—'}</td>
                  <td className="p-3">{typeof p.risk_score === 'number' ? p.risk_score.toFixed(4) : p.risk_score}</td>
                  <td className="p-3">{p.total_spending != null ? Number(p.total_spending).toLocaleString() : '—'}</td>
                  <td className="p-3">{p.claim_count != null ? Number(p.claim_count).toLocaleString() : '—'}</td>
                  <td className="p-3">{p.in_LEIE ? 'Yes' : 'No'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* OIG providers */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gold mb-4">OIG exclusions (LEIE)</h2>
        <p className="text-gray-500 text-sm mb-2">Providers with LISTED_IN_OIG edge ({oig?.total ?? 0})</p>
        <div className="overflow-x-auto">
          <table className="w-full border border-gold/20 rounded-lg overflow-hidden">
            <thead>
              <tr className="bg-dark-700 text-gold text-left text-sm">
                <th className="p-3">NPI</th>
                <th className="p-3">Name</th>
                <th className="p-3">Exclusion type</th>
                <th className="p-3">Exclusion date</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              {(oig?.providers ?? []).map((p) => (
                <tr key={p._key} className="border-t border-gold/10 hover:bg-dark-700/50">
                  <td className="p-3 font-mono text-sm">{p.npi}</td>
                  <td className="p-3 max-w-[200px] truncate">{p.name || '—'}</td>
                  <td className="p-3">{p.exclusion_type || '—'}</td>
                  <td className="p-3">{p.exclusion_date || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Same-address clusters */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gold mb-4">Same-address clusters</h2>
        <p className="text-gray-500 text-sm mb-2">Addresses with more than one provider ({clusters?.total ?? 0})</p>
        <div className="overflow-x-auto">
          <table className="w-full border border-gold/20 rounded-lg overflow-hidden">
            <thead>
              <tr className="bg-dark-700 text-gold text-left text-sm">
                <th className="p-3">Address</th>
                <th className="p-3">City / State / ZIP</th>
                <th className="p-3">Provider count</th>
                <th className="p-3">NPIs</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              {(clusters?.clusters ?? []).slice(0, 100).map((c) => (
                <tr key={c.address_key} className="border-t border-gold/10 hover:bg-dark-700/50">
                  <td className="p-3 max-w-[180px] truncate">{c.address_1 || c.address_key}</td>
                  <td className="p-3">{[c.city, c.state, c.zip].filter(Boolean).join(', ') || '—'}</td>
                  <td className="p-3">{c.provider_count}</td>
                  <td className="p-3 font-mono text-xs max-w-[300px] truncate">{c.npis?.join(', ') || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Spending */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gold mb-4">Spending (top by total_spending)</h2>
        {spendingBars.length > 0 && (
          <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 mb-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={spendingBars} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(212,175,55,0.1)" />
                <XAxis type="number" stroke="#9ca3af" fontSize={11} tickFormatter={(v) => `${(v / 1e6).toFixed(1)}M`} />
                <YAxis type="category" dataKey="name" stroke="#9ca3af" fontSize={10} width={75} />
                <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid rgba(212,175,55,0.3)' }} formatter={(v: number) => v.toLocaleString()} />
                <Bar dataKey="value" fill="rgba(212,175,55,0.6)" radius={[0, 2, 2, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full border border-gold/20 rounded-lg overflow-hidden">
            <thead>
              <tr className="bg-dark-700 text-gold text-left text-sm">
                <th className="p-3">NPI</th>
                <th className="p-3">Name</th>
                <th className="p-3">Total spending</th>
                <th className="p-3">Claim count</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              {(spending?.by_spending ?? []).slice(0, 50).map((p) => (
                <tr key={p._key} className="border-t border-gold/10 hover:bg-dark-700/50">
                  <td className="p-3 font-mono text-sm">{p.npi}</td>
                  <td className="p-3 max-w-[200px] truncate">{p.name || '—'}</td>
                  <td className="p-3">{p.total_spending != null ? Number(p.total_spending).toLocaleString() : '—'}</td>
                  <td className="p-3">{p.claim_count != null ? Number(p.claim_count).toLocaleString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
