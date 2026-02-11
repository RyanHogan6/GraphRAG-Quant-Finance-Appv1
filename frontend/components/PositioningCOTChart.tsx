'use client'

import { useEffect, useRef, useState } from 'react'

export interface COTRow {
  as_of_date?: string
  date?: string
  Open_Interest_All?: number
  Commercial_Positions_Long_All?: number
  Commercial_Positions_Short_All?: number
  Noncommercial_Positions_Long_All?: number
  Noncommercial_Positions_Short_All?: number
  Market_and_Exchange_Names?: string
}

interface PositioningCOTChartProps {
  data: COTRow[]
  title?: string
  /** Show open interest as second axis or line */
  showOI?: boolean
  height?: number
}

const COMMERCIAL_COLOR = '#3B82F6'
const NONCOMMERCIAL_COLOR = '#D4AF37'
const OI_COLOR = 'rgba(148, 163, 184, 0.8)'

export default function PositioningCOTChart({ data, title = 'COT Positioning', showOI = true, height = 300 }: PositioningCOTChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  if (!data?.length) return null

  const sorted = [...data].sort((a, b) => {
    const dA = String(a.as_of_date ?? a.date ?? '').slice(0, 10)
    const dB = String(b.as_of_date ?? b.date ?? '').slice(0, 10)
    return dA.localeCompare(dB)
  })

  const dates = sorted.map(r => String(r.as_of_date ?? r.date ?? '').slice(0, 10))
  const netCommercial = sorted.map(r => {
    const long = Number(r.Commercial_Positions_Long_All ?? 0)
    const short = Number(r.Commercial_Positions_Short_All ?? 0)
    return long - short
  })
  const netNoncommercial = sorted.map(r => {
    const long = Number(r.Noncommercial_Positions_Long_All ?? 0)
    const short = Number(r.Noncommercial_Positions_Short_All ?? 0)
    return long - short
  })
  const oi = sorted.map(r => Number(r.Open_Interest_All ?? 0))

  const padding = { top: 28, right: 44, bottom: 44, left: 64 }
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

    const allValues = [...netCommercial, ...netNoncommercial]
    const minNet = Math.min(...allValues, 0)
    const maxNet = Math.max(...allValues, 0)
    const rangeNet = maxNet - minNet || 1
    const padNet = rangeNet * 0.1
    const yMinNet = minNet - padNet
    const yMaxNet = maxNet + padNet
    const yRangeNet = yMaxNet - yMinNet

    const maxOI = Math.max(...oi, 1)
    const xScale = (i: number) => padding.left + (i / (sorted.length - 1 || 1)) * cw
    const yScaleNet = (v: number) => padding.top + ch - ((v - yMinNet) / yRangeNet) * ch
    const yScaleOI = (v: number) => padding.top + ch - (v / maxOI) * ch

    // Grid
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (ch / 4) * i
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(w - padding.right, y)
      ctx.stroke()
    }

    // Y-axis (net position)
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.3)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(padding.left, padding.top)
    ctx.lineTo(padding.left, padding.top + ch)
    ctx.stroke()
    ctx.fillStyle = '#999'
    ctx.font = '10px monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) {
      const v = yMinNet + (yRangeNet / 4) * (4 - i)
      const y = padding.top + (ch / 4) * i
      ctx.fillText(v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(0)}K` : String(Math.round(v)), padding.left - 8, y + 3)
    }

    // Zero line
    const zeroY = yScaleNet(0)
    ctx.strokeStyle = 'rgba(255,255,255,0.2)'
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(padding.left, zeroY)
    ctx.lineTo(w - padding.right, zeroY)
    ctx.stroke()
    ctx.setLineDash([])

    // Open Interest (right axis, thin line)
    if (showOI && maxOI > 0) {
      ctx.strokeStyle = OI_COLOR
      ctx.lineWidth = 1.5
      ctx.beginPath()
      oi.forEach((v, i) => {
        const x = xScale(i)
        const y = yScaleOI(v)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
    }

    // Net Commercial
    ctx.strokeStyle = COMMERCIAL_COLOR
    ctx.lineWidth = 2.5
    ctx.beginPath()
    netCommercial.forEach((v, i) => {
      const x = xScale(i)
      const y = yScaleNet(v)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Net Noncommercial
    ctx.strokeStyle = NONCOMMERCIAL_COLOR
    ctx.lineWidth = 2.5
    ctx.beginPath()
    netNoncommercial.forEach((v, i) => {
      const x = xScale(i)
      const y = yScaleNet(v)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // X-axis labels
    ctx.fillStyle = '#999'
    ctx.font = '9px monospace'
    ctx.textAlign = 'center'
    const step = Math.max(1, Math.floor(sorted.length / 5))
    for (let i = 0; i < sorted.length; i += step) {
      const x = xScale(i)
      const d = new Date(dates[i])
      ctx.fillText(d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' }), x, padding.top + ch + 18)
    }

    // Title and legend
    ctx.fillStyle = '#D4AF37'
    ctx.font = 'bold 12px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(title, w / 2, 14)
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillStyle = COMMERCIAL_COLOR
    ctx.fillText('Net Commercial', padding.left, h - 8)
    ctx.fillStyle = NONCOMMERCIAL_COLOR
    ctx.fillText('Net Noncommercial', padding.left + 110, h - 8)
    if (showOI) {
      ctx.fillStyle = OI_COLOR
      ctx.fillText('Open Interest', padding.left + 230, h - 8)
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

      const dateStr = new Date(dates[hoveredIndex]).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      const comm = netCommercial[hoveredIndex]
      const non = netNoncommercial[hoveredIndex]
      const oiVal = oi[hoveredIndex]
      const lines = [
        dateStr,
        `Comm: ${comm >= 1e6 ? `${(comm / 1e6).toFixed(2)}M` : comm.toLocaleString()}`,
        `NonComm: ${non >= 1e6 ? `${(non / 1e6).toFixed(2)}M` : non.toLocaleString()}`,
        `OI: ${oiVal >= 1e6 ? `${(oiVal / 1e6).toFixed(2)}M` : oiVal.toLocaleString()}`
      ]
      const tw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 16
      const th = 8 + lines.length * 14
      let tx = x - tw / 2
      if (tx < padding.left) tx = padding.left
      if (tx + tw > w - padding.right) tx = w - padding.right - tw
      const ty = padding.top + 6
      ctx.fillStyle = 'rgba(0,0,0,0.9)'
      ctx.strokeStyle = '#D4AF37'
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
  }, [data, sorted, netCommercial, netNoncommercial, oi, dates, showOI, hoveredIndex, height])

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
