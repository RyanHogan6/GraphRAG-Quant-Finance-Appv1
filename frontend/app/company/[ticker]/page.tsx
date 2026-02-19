'use client'

import { useParams, useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import Link from 'next/link'
import CompanyWorkup from '@/components/CompanyWorkup'

const API_BASE = typeof process !== 'undefined' ? (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') : 'http://localhost:8000'

export default function CompanyPage() {
  const params = useParams()
  const router = useRouter()
  const ticker = typeof params?.ticker === 'string' ? params.ticker.toUpperCase() : ''
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!ticker) {
      setLoading(false)
      setError('Missing ticker')
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    fetch(`${API_BASE}/api/query/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: `company workup for ${ticker}`,
        conversation_history: [],
      }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(res.status === 404 ? 'Company not found' : res.statusText)
        return res.json()
      })
      .then((json) => {
        if (cancelled) return
        const results = json?.results
        if (Array.isArray(results) && results.length > 0) {
          setData(results[0])
        } else {
          setError('No data returned for this company')
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e.message || 'Failed to load company')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => { cancelled = true }
  }, [ticker])

  if (!ticker) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-4">Missing ticker in URL.</p>
          <Link href="/" className="text-gold hover:underline">Back to Query</Link>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-2 border-gold/40 border-t-gold rounded-full animate-spin mb-4" />
          <p className="text-gray-400">Loading company workup for {ticker}…</p>
          <Link href="/" className="text-gold hover:underline mt-2 inline-block">Back to Query</Link>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-4">{error || 'Company not found'}</p>
          <Link href="/" className="text-gold hover:underline">Back to Query</Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen py-6 px-4 md:px-6">
      <div className="container mx-auto max-w-6xl">
        <div className="flex items-center justify-between mb-4">
          <Link
            href="/"
            className="text-sm text-gray-400 hover:text-gold transition-colors"
          >
            ← Back to Query
          </Link>
        </div>
        <CompanyWorkup data={data} />
      </div>
    </div>
  )
}
