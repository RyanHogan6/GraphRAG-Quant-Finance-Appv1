'use client'

import { useEffect, useRef, useState } from 'react'

interface Series {
  dates: string[]
  values: number[]
  label: string
  color: string
  ticker?: string
}

interface TimeSeriesChartProps {
  dates?: string[]  // Legacy single series
  values?: number[]  // Legacy single series
  label?: string  // Legacy single series
  ticker?: string  // Legacy single series
  series?: Series[]  // New: support multiple series for comparison
}

export default function TimeSeriesChart({ dates, values, label, ticker, series }: TimeSeriesChartProps) {
  // Convert legacy props to series format
  const chartSeries = series || (dates && values ? [{
    dates,
    values,
    label: label || '',
    color: '#D4AF37',  // Gold
    ticker
  }] : [])

  if (!chartSeries.length) return null
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  useEffect(() => {
    if (!canvasRef.current || chartSeries.length === 0) return

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

    // Combine all dates from all series to get full range
    const allDates = Array.from(new Set(chartSeries.flatMap(s => s.dates))).sort()

    // Calculate min/max across ALL series
    const allValues = chartSeries.flatMap(s => s.values)
    const minValue = Math.min(...allValues)
    const maxValue = Math.max(...allValues)
    const valueRange = maxValue - minValue
    const valuePadding = valueRange * 0.1 // 10% padding
    const chartMinValue = minValue - valuePadding
    const chartMaxValue = maxValue + valuePadding
    const chartValueRange = chartMaxValue - chartMinValue

    // Scale functions
    const xScale = (index: number) => padding.left + (index / (allDates.length - 1)) * chartWidth
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
    const labelCount = allDates.length > 10 ? 4 : 2
    const dateLabels = []
    for (let i = 0; i <= labelCount; i++) {
      dateLabels.push(Math.floor((allDates.length - 1) * (i / labelCount)))
    }

    // Determine if we should show the year (if range > 12 months)
    const startDate = new Date(allDates[0])
    const endDate = new Date(allDates[allDates.length - 1])
    const showYear = (endDate.getFullYear() - startDate.getFullYear()) >= 1

    dateLabels.forEach(index => {
      if (index < allDates.length) {
        const x = xScale(index)
        const d = new Date(allDates[index])
        const dateStr = showYear
          ? d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
          : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        ctx.fillText(dateStr, x, height - padding.bottom + 20)
      }
    })

    // Draw each series (lines, areas, points)
    chartSeries.forEach((series, seriesIndex) => {
      // Draw line
      ctx.strokeStyle = series.color
      ctx.lineWidth = 2.5
      ctx.beginPath()
      series.values.forEach((value, index) => {
        // Map series date to allDates index for correct x position
        const dateIndex = allDates.indexOf(series.dates[index])
        const x = xScale(dateIndex)
        const y = yScale(value)
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      ctx.stroke()

      // Draw area under line with transparency
      const rgb = series.color === '#FFD700'
        ? '255, 215, 0'
        : series.color === '#3B82F6'
        ? '59, 130, 246'
        : '212, 175, 55' // fallback gold
      ctx.fillStyle = `rgba(${rgb}, 0.1)`
      ctx.beginPath()
      series.values.forEach((value, index) => {
        const dateIndex = allDates.indexOf(series.dates[index])
        const x = xScale(dateIndex)
        const y = yScale(value)
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      const lastDateIndex = allDates.indexOf(series.dates[series.dates.length - 1])
      const firstDateIndex = allDates.indexOf(series.dates[0])
      ctx.lineTo(xScale(lastDateIndex), height - padding.bottom)
      ctx.lineTo(xScale(firstDateIndex), height - padding.bottom)
      ctx.closePath()
      ctx.fill()

      // Draw points
      ctx.fillStyle = series.color
      series.values.forEach((value, index) => {
        const dateIndex = allDates.indexOf(series.dates[index])
        const x = xScale(dateIndex)
        const y = yScale(value)
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fill()
      })
    })

    // Draw title or legend
    if (chartSeries.length === 1) {
      // Single series - show title and start/end values
      ctx.fillStyle = '#FFD700'
      ctx.font = 'bold 14px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(chartSeries[0].label, width / 2, 18)

      // Draw start and end values
      ctx.font = 'bold 12px monospace'
      ctx.textAlign = 'left'
      ctx.fillStyle = '#9CA3AF'
      ctx.fillText(`Start: $${chartSeries[0].values[0].toFixed(2)}`, padding.left, height - 10)
      ctx.textAlign = 'right'
      const change = chartSeries[0].values[chartSeries[0].values.length - 1] - chartSeries[0].values[0]
      const changePercent = (change / chartSeries[0].values[0]) * 100
      const changeColor = change >= 0 ? '#10B981' : '#EF4444'
      ctx.fillStyle = changeColor
      const changeText = `${change >= 0 ? '+' : ''}$${change.toFixed(2)} (${changePercent.toFixed(2)}%)`
      ctx.fillText(changeText, width - padding.right, height - 10)
    } else {
      // Multiple series - show legend
      ctx.font = 'bold 12px sans-serif'
      ctx.textAlign = 'left'
      let legendX = padding.left
      const legendY = 18

      chartSeries.forEach((series, index) => {
        // Draw color indicator
        ctx.fillStyle = series.color
        ctx.beginPath()
        ctx.arc(legendX + 6, legendY - 4, 5, 0, Math.PI * 2)
        ctx.fill()

        // Draw label
        ctx.fillStyle = '#9CA3AF'
        const labelText = series.ticker || series.label
        ctx.fillText(labelText, legendX + 16, legendY)

        // Calculate change for this series
        const change = series.values[series.values.length - 1] - series.values[0]
        const changePercent = (change / series.values[0]) * 100
        const changeColor = change >= 0 ? '#10B981' : '#EF4444'
        ctx.fillStyle = changeColor
        const changeText = ` ${change >= 0 ? '+' : ''}${changePercent.toFixed(1)}%`
        const labelWidth = ctx.measureText(labelText).width
        ctx.fillText(changeText, legendX + 16 + labelWidth, legendY)

        // Move to next legend item position
        const totalWidth = ctx.measureText(labelText + changeText).width + 40
        legendX += totalWidth
      })
    }

    // Draw Hover State (Crosshair + Tooltip)
    if (hoveredIndex !== null && hoveredIndex < allDates.length) {
      const x = xScale(hoveredIndex)
      const date = new Date(allDates[hoveredIndex])

      // 1. Draw dashed vertical line from top to bottom
      ctx.setLineDash([5, 5])
      ctx.strokeStyle = 'rgba(255, 215, 0, 0.5)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, height - padding.bottom)
      ctx.stroke()
      ctx.setLineDash([]) // Reset line dash

      // 2. Highlight points for all series at this x position
      chartSeries.forEach(series => {
        const seriesDateIndex = series.dates.indexOf(allDates[hoveredIndex])
        if (seriesDateIndex !== -1) {
          const value = series.values[seriesDateIndex]
          const y = yScale(value)

          ctx.fillStyle = '#FFF'
          ctx.strokeStyle = series.color
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.arc(x, y, 5, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()
        }
      })

      // 3. Draw tooltip with all series values
      const tooltipPadding = 8
      const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })

      // Build tooltip content for all series
      const seriesData = chartSeries.map(series => {
        const seriesDateIndex = series.dates.indexOf(allDates[hoveredIndex])
        if (seriesDateIndex !== -1) {
          return {
            ticker: series.ticker || series.label,
            value: series.values[seriesDateIndex],
            color: series.color
          }
        }
        return null
      }).filter(Boolean)

      if (seriesData.length > 0) {
        ctx.font = 'bold 11px monospace'
        const maxWidth = Math.max(
          ctx.measureText(dateStr).width,
          ...seriesData.map(s => ctx.measureText(`${s.ticker}: $${s.value.toFixed(2)}`).width)
        )
        const tooltipWidth = maxWidth + tooltipPadding * 2
        const tooltipHeight = 20 + seriesData.length * 18 + 10

        // Tooltip position (keep within bounds)
        let tooltipX = x - tooltipWidth / 2
        if (tooltipX < padding.left) tooltipX = padding.left
        if (tooltipX + tooltipWidth > width - padding.right) tooltipX = width - padding.right - tooltipWidth

        const tooltipY = padding.top + 10

        // Draw tooltip box
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)'
        ctx.strokeStyle = '#FFD700'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 4)
        ctx.fill()
        ctx.stroke()

        // Draw date
        ctx.fillStyle = '#9CA3AF'
        ctx.font = '10px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(dateStr, tooltipX + tooltipWidth / 2, tooltipY + 14)

        // Draw each series value
        ctx.font = 'bold 11px monospace'
        ctx.textAlign = 'left'
        seriesData.forEach((s, i) => {
          ctx.fillStyle = s.color
          ctx.fillText(`${s.ticker}: $${s.value.toFixed(2)}`, tooltipX + tooltipPadding, tooltipY + 30 + i * 18)
        })
      }
    }

  }, [chartSeries, hoveredIndex])

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || chartSeries.length === 0) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left

    const padding = { left: 60, right: 40 }
    const chartWidth = rect.width - padding.left - padding.right

    // Get combined dates from all series
    const allDates = Array.from(new Set(chartSeries.flatMap(s => s.dates))).sort()

    // Reverse xScale to get index
    const relativeX = x - padding.left
    const index = Math.round((relativeX / chartWidth) * (allDates.length - 1))

    if (index >= 0 && index < allDates.length) {
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
