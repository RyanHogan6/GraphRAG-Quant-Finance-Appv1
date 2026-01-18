export interface Market {
  id: string
  question: string
  icon: string // emoji or icon identifier
  category: string
  yes_prob: number
  no_prob: number
  volume_24h: number
  liquidity: number
  end_date: string
  outcomes?: Array<{ name: string; prob: number }>
  outcome_yes?: string
  outcome_no?: string
  description?: string
  traders: number
  probability_confidence?: number
  days_until_end?: number
  activity_score?: number
  liquidity_score?: number
  volume_per_day?: number
}

export interface WhaleTrader {
  address: string
  volume: number
  profit: number
  trades: number
  activity: string
  profit_ratio: number
  win_rate: number
}

export interface Category {
  name: string
  count: number
  icon: string
}
