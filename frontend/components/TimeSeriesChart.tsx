'use client'

import { useEffect, useRef, useState } from 'react'

interface TimeSeriesChartProps {
  dates: string[]
  values: number[]
  label: string
  ticker?: string
}

export default function TimeSeriesChart({ dates, values, label, ticker }: TimeSeriesChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !dates.length || !values.length) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const width = rect.width
    const height = rect.height
    const padding = { top: 30, right: 40, bottom: 50, left: 60 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Calculate scales
    const minValue = Math.min(...values)
    const maxValue = Math.max(...values)
    const valueRange = maxValue - minValue
    const valuePadding = valueRange * 0.1 // 10% padding
    const chartMinValue = minValue - valuePadding
    const chartMaxValue = maxValue + valuePadding
    const chartValueRange = chartMaxValue - chartMinValue

    // Scale functions
    const xScale = (index: number) => padding.left + (index / (dates.length - 1)) * chartWidth
    const yScale = (value: number) =>
      padding.top + chartHeight - ((value - chartMinValue) / chartValueRange) * chartHeight

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 215, 0, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.stroke()
    }

    // Draw axes
    ctx.strokeStyle = 'rgba(255, 215, 0, 0.3)'
    ctx.lineWidth = 2
    // Y-axis
    ctx.beginPath()
    ctx.moveTo(padding.left, padding.top)
    ctx.lineTo(padding.left, height - padding.bottom)
    ctx.stroke()
    // X-axis
    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(width - padding.right, height - padding.bottom)
    ctx.stroke()

    // Draw Y-axis labels (prices)
    ctx.fillStyle = '#999'
    ctx.font = '11px monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 5; i++) {
      const value = chartMinValue + (chartValueRange / 5) * i
      const y = padding.top + chartHeight - (chartHeight / 5) * i
      ctx.fillText(`$${value.toFixed(2)}`, padding.left - 10, y + 4)
    }

    // Draw X-axis labels (dates)
    ctx.textAlign = 'center'

    // Calculate 4-5 evenly spaced labels depending on data length
    const labelCount = dates.length > 10 ? 4 : 2
    const dateLabels = []
    for (let i = 0; i <= labelCount; i++) {
      dateLabels.push(Math.floor((dates.length - 1) * (i / labelCount)))
    }

    // Determine if we should show the year (if range > 12 months)
    const startDate = new Date(dates[0])
    const endDate = new Date(dates[dates.length - 1])
    const showYear = (endDate.getFullYear() - startDate.getFullYear()) >= 1

    dateLabels.forEach(index => {
      if (index < dates.length) {
        const x = xScale(index)
        const d = new Date(dates[index])
        const dateStr = showYear
          ? d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
          : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        ctx.fillText(dateStr, x, height - padding.bottom + 20)
      }
    })

    // Draw line
    ctx.strokeStyle = '#FFD700' // Gold color
    ctx.lineWidth = 2.5
    ctx.beginPath()
    values.forEach((value, index) => {
      const x = xScale(index)
      const y = yScale(value)
      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()

    // Draw area under line
    ctx.fillStyle = 'rgba(255, 215, 0, 0.1)'
    ctx.beginPath()
    values.forEach((value, index) => {
      const x = xScale(index)
      const y = yScale(value)
      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.lineTo(xScale(values.length - 1), height - padding.bottom)
    ctx.lineTo(xScale(0), height - padding.bottom)
    ctx.closePath()
    ctx.fill()

    // Draw points
    ctx.fillStyle = '#FFD700'
    values.forEach((value, index) => {
      const x = xScale(index)
      const y = yScale(value)
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw title
    ctx.fillStyle = '#FFD700'
    ctx.font = 'bold 14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(label, width / 2, 18)

    // Draw start and end values
    ctx.font = 'bold 12px monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = '#9CA3AF'
    ctx.fillText(`Start: $${values[0].toFixed(2)}`, padding.left, height - 10)
    ctx.textAlign = 'right'
    const change = values[values.length - 1] - values[0]
    const changePercent = (change / values[0]) * 100
    const changeColor = change >= 0 ? '#10B981' : '#EF4444'
    ctx.fillStyle = changeColor
    const changeText = `${change >= 0 ? '+' : ''}$${change.toFixed(2)} (${changePercent.toFixed(2)}%)`
    ctx.fillText(changeText, width - padding.right, height - 10)

    // Draw Hover State (Crosshair + Tooltip)
    if (hoveredIndex !== null && hoveredIndex < values.length) {
      const x = xScale(hoveredIndex)
      const y = yScale(values[hoveredIndex])
      const date = new Date(dates[hoveredIndex])
      const price = values[hoveredIndex]

      // 1. Draw dashed vertical line
      ctx.setLineDash([5, 5])
      ctx.strokeStyle = 'rgba(255, 215, 0, 0.5)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, y)
      ctx.lineTo(x, height - padding.bottom)
      ctx.stroke()
      ctx.setLineDash([]) // Reset line dash

      // 2. Highlight point
      ctx.fillStyle = '#FFF'
      ctx.strokeStyle = '#FFD700'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()

      // 3. Draw tooltip
      const tooltipPadding = 8
      const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      const priceStr = `$${price.toFixed(2)}`

      ctx.font = 'bold 12px monospace'
      const priceWidth = ctx.measureText(priceStr).width
      ctx.font = '10px sans-serif'
      const dateWidth = ctx.measureText(dateStr).width
      const tooltipWidth = Math.max(priceWidth, dateWidth) + tooltipPadding * 2
      const tooltipHeight = 40

      // Tooltip position (keep within bounds)
      let tooltipX = x - tooltipWidth / 2
      if (tooltipX < padding.left) tooltipX = padding.left
      if (tooltipX + tooltipWidth > width - padding.right) tooltipX = width - padding.right - tooltipWidth

      const tooltipY = y - tooltipHeight - 10

      // Draw tooltip box
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
      ctx.strokeStyle = '#FFD700'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 4)
      ctx.fill()
      ctx.stroke()

      // Draw tooltip text
      ctx.fillStyle = '#FFD700'
      ctx.font = 'bold 12px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(priceStr, tooltipX + tooltipWidth / 2, tooltipY + 18)

      ctx.fillStyle = '#9CA3AF'
      ctx.font = '10px sans-serif'
      ctx.fillText(dateStr, tooltipX + tooltipWidth / 2, tooltipY + 34)
    }

  }, [dates, values, label, hoveredIndex])

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !dates.length) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left

    const padding = { left: 60, right: 40 }
    const chartWidth = rect.width - padding.left - padding.right

    // Reverse xScale to get index
    const relativeX = x - padding.left
    const index = Math.round((relativeX / chartWidth) * (dates.length - 1))

    if (index >= 0 && index < dates.length) {
      setHoveredIndex(index)
    } else {
      setHoveredIndex(null)
    }
  }

  return (
    <div className="w-full bg-dark-800 border border-gold/20 rounded-lg p-4 my-4">
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredIndex(null)}
        className="w-full cursor-crosshair"
        style={{ height: '280px' }}
      />
    </div>
  )
}
