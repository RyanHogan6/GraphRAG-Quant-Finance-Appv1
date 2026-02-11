'use client'

import { useEffect, useRef, useState } from 'react'

export interface ProbabilityRow {
  datetime?: string
  date?: string
  timestamp?: string
  yes_price?: number
  yes_probability?: number
  no_price?: number
  volume?: number
}

interface ProbabilityTimelineChartProps {
  data: ProbabilityRow[]
  title?: string
  /** Show no_price as second series (default true when present) */
  showNoPrice?: boolean
  height?: number
}

const YES_COLOR = '#10B981'
const NO_COLOR = '#EF4444'
const GOLD = '#D4AF37'

export default function ProbabilityTimelineChart({
  data,
  title = 'Probability',
  showNoPrice = true,
  height = 280
}: ProbabilityTimelineChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  if (!data?.length) return null

  const dateKey = (['datetime', 'date', 'timestamp'] as const).find(k => data[0][k] != null) ?? 'date'
  const sorted = [...data].sort((a, b) => String(a[dateKey]).localeCompare(String(b[dateKey])))
  const dates = sorted.map(r => String(r[dateKey] ?? '').slice(0, 19))
  const yesValues = sorted.map(r => {
    const v = r.yes_price ?? r.yes_probability
    return typeof v === 'number' && !Number.isNaN(v) ? Math.min(1, Math.max(0, v)) : 0
  })
  const noValues = sorted.map(r => {
    const v = r.no_price
    return typeof v === 'number' && !Number.isNaN(v) ? Math.min(1, Math.max(0, v)) : 0
  })
  const hasNo = showNoPrice && noValues.some(v => v > 0)

  const padding = { top: 28, right: 40, bottom: 44, left: 52 }
  const chartHeight = height - padding.top - padding.bottom
  const chartWidth = 400

  useEffect(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const w = rect.width
    const h = rect.height
    const cw = w - padding.left - padding.right
    const ch = h - padding.top - padding.bottom

    ctx.clearRect(0, 0, w, h)

    // Y: 0–100%
    const yScale = (v: number) => padding.top + ch - v * ch
    const xScale = (i: number) => padding.left + (i / (sorted.length - 1 || 1)) * cw

    // Grid at 0, 25, 50, 75, 100
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.12)'
    ctx.lineWidth = 1
    for (const pct of [0.25, 0.5, 0.75]) {
      const y = yScale(pct)
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(w - padding.right, y)
      ctx.stroke()
    }

    // Y-axis
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.3)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(padding.left, padding.top)
    ctx.lineTo(padding.left, padding.top + ch)
    ctx.stroke()
    ctx.fillStyle = '#999'
    ctx.font = '10px monospace'
    ctx.textAlign = 'right'
    for (const pct of [0, 0.25, 0.5, 0.75, 1]) {
      const y = yScale(pct)
      ctx.fillText(`${Math.round(pct * 100)}%`, padding.left - 8, y + 3)
    }

    // Yes area (0–100% band)
    ctx.fillStyle = 'rgba(16, 185, 129, 0.15)'
    ctx.beginPath()
    yesValues.forEach((v, i) => {
      const x = xScale(i)
      const y = yScale(v)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.lineTo(xScale(yesValues.length - 1), padding.top + ch)
    ctx.lineTo(padding.left, padding.top + ch)
    ctx.closePath()
    ctx.fill()

    // Yes line
    ctx.strokeStyle = YES_COLOR
    ctx.lineWidth = 2.5
    ctx.beginPath()
    yesValues.forEach((v, i) => {
      const x = xScale(i)
      const y = yScale(v)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // No line (optional)
    if (hasNo) {
      ctx.strokeStyle = NO_COLOR
      ctx.lineWidth = 1.5
      ctx.beginPath()
      noValues.forEach((v, i) => {
        const x = xScale(i)
        const y = yScale(v)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
    }

    // X-axis labels
    ctx.fillStyle = '#999'
    ctx.font = '9px monospace'
    ctx.textAlign = 'center'
    const step = Math.max(1, Math.floor(sorted.length / 5))
    for (let i = 0; i < sorted.length; i += step) {
      const x = xScale(i)
      const d = new Date(dates[i])
      ctx.fillText(d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), x, padding.top + ch + 18)
    }

    // Title and legend
    ctx.fillStyle = GOLD
    ctx.font = 'bold 12px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(title, w / 2, 14)
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillStyle = YES_COLOR
    ctx.fillText('Yes', padding.left, h - 8)
    if (hasNo) {
      ctx.fillStyle = NO_COLOR
      ctx.fillText('No', padding.left + 50, h - 8)
    }

    // Hover tooltip
    if (hoveredIndex !== null && hoveredIndex >= 0 && hoveredIndex < sorted.length) {
      const x = xScale(hoveredIndex)
      ctx.setLineDash([4, 4])
      ctx.strokeStyle = 'rgba(212, 175, 55, 0.5)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, padding.top + ch)
      ctx.stroke()
      ctx.setLineDash([])

      const dateStr = new Date(dates[hoveredIndex]).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
      const yesPct = Math.round(yesValues[hoveredIndex] * 100)
      const noPct = hasNo ? Math.round(noValues[hoveredIndex] * 100) : null
      const lines = [dateStr, `Yes: ${yesPct}%`, ...(noPct != null ? [`No: ${noPct}%`] : [])]
      const tw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 16
      const th = 8 + lines.length * 14
      let tx = x - tw / 2
      if (tx < padding.left) tx = padding.left
      if (tx + tw > w - padding.right) tx = w - padding.right - tw
      const ty = padding.top + 6
      ctx.fillStyle = 'rgba(0,0,0,0.9)'
      ctx.strokeStyle = GOLD
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.roundRect(tx, ty, tw, th, 4)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = '#E5E7EB'
      ctx.font = '10px monospace'
      ctx.textAlign = 'left'
      lines.forEach((line, j) => {
        ctx.fillText(line, tx + 8, ty + 18 + j * 14)
      })
    }
  }, [data, sorted, yesValues, noValues, hasNo, dates, hoveredIndex, height])

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const cw = rect.width - padding.left - padding.right
    const i = Math.round(((x - padding.left) / cw) * (sorted.length - 1))
    if (i >= 0 && i < sorted.length) setHoveredIndex(i)
    else setHoveredIndex(null)
  }

  return (
    <div className="w-full bg-dark-800 border border-gold/20 rounded-lg p-4 my-4">
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredIndex(null)}
        className="w-full cursor-crosshair"
        style={{ height: `${height}px` }}
      />
    </div>
  )
}
