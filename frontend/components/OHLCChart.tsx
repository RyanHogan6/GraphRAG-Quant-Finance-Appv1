'use client'

import { useEffect, useRef, useState } from 'react'

export interface OHLCRow {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface OHLCChartProps {
  data: OHLCRow[]
  title?: string
  ticker?: string
  /** Show volume as subplot (default true when volume present) */
  showVolume?: boolean
  height?: number
}

const UP_COLOR = '#10B981'
const DOWN_COLOR = '#EF4444'
const GOLD = '#D4AF37'

export default function OHLCChart({ data, title, ticker, showVolume = true, height = 320 }: OHLCChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  if (!data?.length) return null

  const sorted = [...data].sort((a, b) => String(a.date).localeCompare(String(b.date)))
  const dates = sorted.map(r => String(r.date).slice(0, 10))
  const hasVolume = showVolume && sorted.some(r => typeof r.volume === 'number' && r.volume > 0)
  const volumeHeight = hasVolume ? 60 : 0
  const chartHeight = height - 50 - volumeHeight
  const padding = { top: 24, right: 40, bottom: 28, left: 64 }

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
    const chartW = w - padding.left - padding.right
    const mainH = hasVolume ? h - padding.top - padding.bottom - volumeHeight : h - padding.top - padding.bottom

    ctx.clearRect(0, 0, w, h)

    const lows = sorted.map(r => r.low)
    const highs = sorted.map(r => r.high)
    const minPrice = Math.min(...lows)
    const maxPrice = Math.max(...highs)
    const range = maxPrice - minPrice || 1
    const pad = range * 0.05
    const yMin = minPrice - pad
    const yMax = maxPrice + pad
    const yRange = yMax - yMin

    const xScale = (i: number) => padding.left + (i / (sorted.length - 1 || 1)) * chartW
    const yScale = (v: number) => padding.top + mainH - ((v - yMin) / yRange) * mainH

    // Grid
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (mainH / 4) * i
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(w - padding.right, y)
      ctx.stroke()
    }

    // Y-axis (price)
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.3)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(padding.left, padding.top)
    ctx.lineTo(padding.left, padding.top + mainH)
    ctx.stroke()
    ctx.fillStyle = '#999'
    ctx.font = '11px monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) {
      const v = yMin + (yRange / 4) * (4 - i)
      const y = padding.top + (mainH / 4) * i
      ctx.fillText(`$${v.toFixed(2)}`, padding.left - 8, y + 4)
    }

    // Candlesticks
    const barW = Math.max(2, (chartW / sorted.length) * 0.6)
    const wickW = 1
    sorted.forEach((row, i) => {
      const x = xScale(i)
      const openY = yScale(row.open)
      const closeY = yScale(row.close)
      const highY = yScale(row.high)
      const lowY = yScale(row.low)
      const isUp = row.close >= row.open
      ctx.strokeStyle = isUp ? UP_COLOR : DOWN_COLOR
      ctx.fillStyle = isUp ? UP_COLOR : DOWN_COLOR

      // Wick (high–low)
      ctx.lineWidth = wickW
      ctx.beginPath()
      ctx.moveTo(x, highY)
      ctx.lineTo(x, lowY)
      ctx.stroke()

      // Body (open–close)
      const bodyTop = Math.min(openY, closeY)
      const bodyH = Math.max(2, Math.abs(closeY - openY))
      ctx.fillRect(x - barW / 2, bodyTop, barW, bodyH)
      ctx.strokeRect(x - barW / 2, bodyTop, barW, bodyH)
    })

    // X-axis labels
    ctx.fillStyle = '#999'
    ctx.font = '9px monospace'
    ctx.textAlign = 'center'
    const step = Math.max(1, Math.floor(sorted.length / 5))
    for (let i = 0; i < sorted.length; i += step) {
      const x = xScale(i)
      const d = new Date(dates[i])
      ctx.fillText(d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), x, padding.top + mainH + 16)
    }

    // Title
    if (title || ticker) {
      ctx.fillStyle = GOLD
      ctx.font = 'bold 13px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(title || (ticker ? `${ticker} Price` : 'OHLC'), w / 2, 14)
    }

    // Start/end and % change
    const first = sorted[0]
    const last = sorted[sorted.length - 1]
    const change = last.close - first.open
    const changePct = (change / first.open) * 100
    ctx.font = 'bold 11px monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = '#9CA3AF'
    ctx.fillText(`Start: $${first.open.toFixed(2)}`, padding.left, h - 8)
    ctx.textAlign = 'right'
    ctx.fillStyle = change >= 0 ? UP_COLOR : DOWN_COLOR
    ctx.fillText(`${change >= 0 ? '+' : ''}$${change.toFixed(2)} (${changePct.toFixed(2)}%)`, w - padding.right, h - 8)

    // Volume subplot
    if (hasVolume) {
      const volY0 = padding.top + mainH + 8
      const volH = volumeHeight - 16
      const maxVol = Math.max(...sorted.map(r => (r.volume as number) || 0), 1)
      sorted.forEach((row, i) => {
        const vol = (row.volume as number) || 0
        const barH = (vol / maxVol) * volH
        const isUp = row.close >= row.open
        ctx.fillStyle = isUp ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)'
        ctx.fillRect(xScale(i) - barW / 2, volY0 + volH - barH, barW, barH)
      })
    }

    // Hover tooltip
    if (hoveredIndex !== null && hoveredIndex >= 0 && hoveredIndex < sorted.length) {
      const row = sorted[hoveredIndex]
      const x = xScale(hoveredIndex)
      ctx.setLineDash([4, 4])
      ctx.strokeStyle = 'rgba(212, 175, 55, 0.5)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, padding.top + mainH)
      ctx.stroke()
      ctx.setLineDash([])

      const dateStr = new Date(row.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      const lines = [`O ${row.open.toFixed(2)}  H ${row.high.toFixed(2)}`, `L ${row.low.toFixed(2)}  C ${row.close.toFixed(2)}`]
      if (typeof row.volume === 'number') lines.push(`Vol ${row.volume.toLocaleString()}`)
      const tw = Math.max(ctx.measureText(dateStr).width, ...lines.map(l => ctx.measureText(l).width)) + 16
      const th = 20 + lines.length * 14 + 8
      let tx = x - tw / 2
      if (tx < padding.left) tx = padding.left
      if (tx + tw > w - padding.right) tx = w - padding.right - tw
      const ty = padding.top + 8
      ctx.fillStyle = 'rgba(0,0,0,0.9)'
      ctx.strokeStyle = GOLD
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.roundRect(tx, ty, tw, th, 4)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = '#9CA3AF'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(dateStr, tx + tw / 2, ty + 12)
      ctx.textAlign = 'left'
      ctx.font = '10px monospace'
      lines.forEach((line, j) => {
        ctx.fillStyle = '#E5E7EB'
        ctx.fillText(line, tx + 8, ty + 26 + j * 14)
      })
    }
  }, [data, sorted, hasVolume, hoveredIndex, height])

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const chartW = rect.width - padding.left - padding.right
    const i = Math.round(((x - padding.left) / chartW) * (sorted.length - 1))
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
