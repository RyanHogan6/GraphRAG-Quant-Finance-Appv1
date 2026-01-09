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
  description?: string
  traders: number
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

// Mock Categories
export const mockCategories: Category[] = [
  { name: 'Politics', count: 1245, icon: '🏛️' },
  { name: 'Crypto', count: 892, icon: '₿' },
  { name: 'Sports', count: 567, icon: '⚽' },
  { name: 'Finance', count: 445, icon: '💹' },
  { name: 'Tech', count: 334, icon: '💻' },
  { name: 'Entertainment', count: 223, icon: '🎬' },
  { name: 'Science', count: 156, icon: '🔬' },
  { name: 'World', count: 445, icon: '🌍' },
]

// Mock Markets
export const mockMarkets: Market[] = [
  {
    id: '1',
    question: 'Will Trump acquire Greenland before 2027?',
    icon: '🇬🇱',
    category: 'Politics',
    yes_prob: 17,
    no_prob: 83,
    volume_24h: 124500,
    liquidity: 423062,
    end_date: '2027-01-01',
    traders: 1245,
    description: 'Resolves YES if Trump officially acquires Greenland'
  },
  {
    id: '2',
    question: 'Israel strikes Iran by January 31, 2026?',
    icon: '🇮🇱',
    category: 'World',
    yes_prob: 31,
    no_prob: 69,
    volume_24h: 89250,
    liquidity: 321944,
    end_date: '2026-01-31',
    traders: 892,
    description: 'Resolves YES if Israel conducts military strike on Iran'
  },
  {
    id: '3',
    question: 'Will the Iranian regime fall before 2027?',
    icon: '🇮🇷',
    category: 'World',
    yes_prob: 41,
    no_prob: 59,
    volume_24h: 67800,
    liquidity: 201637,
    end_date: '2027-01-01',
    traders: 678,
  },
  {
    id: '4',
    question: 'Bitcoin above $150k by end of 2026?',
    icon: '₿',
    category: 'Crypto',
    yes_prob: 28,
    no_prob: 72,
    volume_24h: 345600,
    liquidity: 1201692,
    end_date: '2026-12-31',
    traders: 3456,
  },
  {
    id: '5',
    question: 'Will Trump pardon Ghislaine Maxwell by end of 2026?',
    icon: '⚖️',
    category: 'Politics',
    yes_prob: 5,
    no_prob: 95,
    volume_24h: 45300,
    liquidity: 156487,
    end_date: '2026-12-31',
    traders: 453,
  },
  {
    id: '6',
    question: 'Fed increases interest rates by 25+ bps after January 2026 meeting?',
    icon: '💹',
    category: 'Finance',
    yes_prob: 3,
    no_prob: 97,
    volume_24h: 89400,
    liquidity: 380284,
    end_date: '2026-02-01',
    traders: 894,
  },
  {
    id: '7',
    question: 'Super Bowl Champion 2026',
    icon: '🏈',
    category: 'Sports',
    yes_prob: 19,
    no_prob: 81,
    volume_24h: 156800,
    liquidity: 651001,
    end_date: '2026-02-14',
    traders: 1568,
    outcomes: [
      { name: 'Seattle', prob: 19 },
      { name: 'Los Angeles R', prob: 17 },
    ]
  },
  {
    id: '8',
    question: 'Supreme Court rules in favor of Trump\'s tariffs?',
    icon: '⚖️',
    category: 'Politics',
    yes_prob: 26,
    no_prob: 74,
    volume_24h: 67200,
    liquidity: 224757,
    end_date: '2026-06-30',
    traders: 672,
  },
  {
    id: '9',
    question: 'Will Elon Musk tweets January 8 - January 10, 2026?',
    icon: '𝕏',
    category: 'Tech',
    yes_prob: 70,
    no_prob: 30,
    volume_24h: 45600,
    liquidity: 127345,
    end_date: '2026-01-10',
    traders: 456,
  },
  {
    id: '10',
    question: 'Presidential Election Winner 2028',
    icon: '🗳️',
    category: 'Politics',
    yes_prob: 28,
    no_prob: 72,
    volume_24h: 234500,
    liquidity: 1178921,
    end_date: '2028-11-08',
    traders: 2345,
    outcomes: [
      { name: 'JD Vance', prob: 28 },
      { name: 'Gavin Newsom', prob: 19 },
    ]
  },
  {
    id: '11',
    question: 'Who will Trump nominate as Fed Chair?',
    icon: '🏦',
    category: 'Finance',
    yes_prob: 42,
    no_prob: 58,
    volume_24h: 78900,
    liquidity: 329646,
    end_date: '2026-05-15',
    traders: 789,
    outcomes: [
      { name: 'Kevin Warsh', prob: 42 },
      { name: 'Kevin Hassett', prob: 37 },
    ]
  },
  {
    id: '12',
    question: 'Venezuela leader end of 2026?',
    icon: '🇻🇪',
    category: 'World',
    yes_prob: 52,
    no_prob: 48,
    volume_24h: 34200,
    liquidity: 102162,
    end_date: '2026-12-31',
    traders: 342,
    outcomes: [
      { name: 'Delcy Rodriguez', prob: 52 },
      { name: 'Maria Corina Machado', prob: 19 },
    ]
  },
  {
    id: '13',
    question: 'Will Portugal have a presidential election?',
    icon: '🇵🇹',
    category: 'World',
    yes_prob: 39,
    no_prob: 61,
    volume_24h: 23400,
    liquidity: 61975,
    end_date: '2026-03-31',
    traders: 234,
    outcomes: [
      { name: 'Luís Marques Mendes', prob: 39 },
      { name: 'António José Seguro', prob: 34 },
    ]
  },
  {
    id: '14',
    question: 'Logan Paul\'s Pikachu illustrator sale price',
    icon: '⚡',
    category: 'Entertainment',
    yes_prob: 98,
    no_prob: 2,
    volume_24h: 12300,
    liquidity: 33173,
    end_date: '2026-02-28',
    traders: 123,
    outcomes: [
      { name: '>$4m', prob: 98 },
      { name: '<$5m', prob: 94 },
    ]
  },
  {
    id: '15',
    question: 'GTA VI released before June 2026?',
    icon: '🎮',
    category: 'Tech',
    yes_prob: 15,
    no_prob: 85,
    volume_24h: 56700,
    liquidity: 216064,
    end_date: '2026-06-01',
    traders: 567,
  },
]

// Mock Whale Traders
export const mockWhaleTraders: WhaleTrader[] = [
  {
    address: '0x1a2b...3c4d',
    volume: 2450000,
    profit: 342000,
    trades: 1234,
    activity: 'Very Active',
    profit_ratio: 0.14,
    win_rate: 67,
  },
  {
    address: '0x5e6f...7g8h',
    volume: 1890000,
    profit: 198000,
    trades: 892,
    activity: 'Active',
    profit_ratio: 0.10,
    win_rate: 62,
  },
  {
    address: '0x9i0j...1k2l',
    volume: 1567000,
    profit: -45000,
    trades: 678,
    activity: 'Active',
    profit_ratio: -0.03,
    win_rate: 48,
  },
  {
    address: '0x3m4n...5o6p',
    volume: 1234000,
    profit: 156000,
    trades: 456,
    activity: 'Moderate',
    profit_ratio: 0.13,
    win_rate: 71,
  },
  {
    address: '0x7q8r...9s0t',
    volume: 987000,
    profit: 87000,
    trades: 342,
    activity: 'Moderate',
    profit_ratio: 0.09,
    win_rate: 59,
  },
  {
    address: '0x1u2v...3w4x',
    volume: 876000,
    profit: 234000,
    trades: 567,
    activity: 'Active',
    profit_ratio: 0.27,
    win_rate: 78,
  },
  {
    address: '0x5y6z...7a8b',
    volume: 765000,
    profit: 65000,
    trades: 234,
    activity: 'Low',
    profit_ratio: 0.08,
    win_rate: 56,
  },
  {
    address: '0x9c0d...1e2f',
    volume: 654000,
    profit: -23000,
    trades: 189,
    activity: 'Low',
    profit_ratio: -0.04,
    win_rate: 45,
  },
]
