interface ProbabilityChartProps {
  yesProb: number
  noProb: number
  marketData?: any
}

export default function ProbabilityChart({ yesProb, noProb, marketData }: ProbabilityChartProps) {
  // Generate mock historical data based on current probability
  // In the future, this will use real historical data from the database
  const generateMockHistory = () => {
    const points = 30 // 30 data points
    const history = []
    const variation = 15 // Max variation from current price

    for (let i = 0; i < points; i++) {
      const progress = i / points
      // Create a trend that ends at current probability
      const randomWalk = Math.sin(progress * Math.PI * 2) * variation
      const trending = yesProb + randomWalk * (1 - progress)
      history.push({
        x: i,
        yes: Math.max(0, Math.min(100, trending)),
        no: Math.max(0, Math.min(100, 100 - trending))
      })
    }
    // Ensure last point is exactly current probability
    history[points - 1] = { x: points - 1, yes: yesProb, no: noProb }
    return history
  }

  const data = generateMockHistory()
  const width = 600
  const height = 200
  const padding = 40

  // Scale functions
  const xScale = (x: number) => padding + (x / (data.length - 1)) * (width - 2 * padding)
  const yScale = (y: number) => height - padding - (y / 100) * (height - 2 * padding)

  // Generate SVG path
  const yesPath = data.map((d, i) =>
    `${i === 0 ? 'M' : 'L'} ${xScale(d.x)} ${yScale(d.yes)}`
  ).join(' ')

  const noPath = data.map((d, i) =>
    `${i === 0 ? 'M' : 'L'} ${xScale(d.x)} ${yScale(d.no)}`
  ).join(' ')

  return (
    <div className="bg-dark-700 border border-gold/20 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gold mb-4">PROBABILITY TREND</h3>
      <div className="relative">
        <svg width={width} height={height} className="w-full h-auto">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(y => (
            <g key={y}>
              <line
                x1={padding}
                y1={yScale(y)}
                x2={width - padding}
                y2={yScale(y)}
                stroke="#2a2a2a"
                strokeWidth="1"
              />
              <text
                x={padding - 10}
                y={yScale(y)}
                fill="#6b7280"
                fontSize="10"
                textAnchor="end"
                dominantBaseline="middle"
              >
                {y}%
              </text>
            </g>
          ))}

          {/* Yes probability line */}
          <path
            d={yesPath}
            fill="none"
            stroke="#4ade80"
            strokeWidth="2"
            opacity="0.8"
          />

          {/* No probability line */}
          <path
            d={noPath}
            fill="none"
            stroke="#f87171"
            strokeWidth="2"
            opacity="0.8"
          />

          {/* Current point markers */}
          <circle
            cx={xScale(data[data.length - 1].x)}
            cy={yScale(yesProb)}
            r="4"
            fill="#4ade80"
          />
          <circle
            cx={xScale(data[data.length - 1].x)}
            cy={yScale(noProb)}
            r="4"
            fill="#f87171"
          />

          {/* Axis labels */}
          <text
            x={width / 2}
            y={height - 5}
            fill="#6b7280"
            fontSize="12"
            textAnchor="middle"
          >
            Time
          </text>
        </svg>

        {/* Legend */}
        <div className="flex items-center justify-center space-x-6 mt-4">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            <span className="text-xs text-gray-400">Yes: {yesProb}%</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-400 rounded-full"></div>
            <span className="text-xs text-gray-400">No: {noProb}%</span>
          </div>
        </div>

        <div className="text-xs text-gray-500 text-center mt-2">
          * Historical trend visualization (30-day simulated)
        </div>
      </div>
    </div>
  )
}
