import { marketCache } from './marketCache'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Helper function for retry logic
async function fetchWithRetry(url: string, options: RequestInit = {}, retries = 2): Promise<Response> {
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(url, { ...options, signal: AbortSignal.timeout(30000) })
      if (res.ok) return res

      // Don't retry on 4xx errors (client errors)
      if (res.status >= 400 && res.status < 500) {
        throw new Error(`Request failed with status ${res.status}`)
      }

      // Retry on 5xx errors
      if (i < retries) {
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)))
        continue
      }
      throw new Error(`Request failed with status ${res.status}`)
    } catch (err) {
      if (i === retries) throw err
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)))
    }
  }
  throw new Error('Max retries reached')
}

export const api = {
  // Query endpoints
  async executeQuery(question: string) {
    const res = await fetchWithRetry(`${API_BASE_URL}/api/query/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    return res.json();
  },

  async checkIntent(question: string) {
    const res = await fetch(`${API_BASE_URL}/api/query/intent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error('Intent check failed');
    return res.json();
  },

  // Markets endpoints
  async getFeaturedMarkets(limit: number = 100, platform: 'polymarket' | 'kalshi' = 'polymarket') {
    const cacheKey = `markets_${platform}_${limit}`

    // Check cache first
    const cached = marketCache.get<any[]>(cacheKey)
    if (cached) {
      console.log('Returning cached markets for', platform)
      return cached
    }

    // Fetch from API with retry
    const res = await fetchWithRetry(`${API_BASE_URL}/api/markets/${platform}/featured?limit=${limit}`);
    const data = await res.json();

    // Cache the result
    marketCache.set(cacheKey, data)
    return data;
  },

  async getMarkets(params: {
    platform?: 'polymarket' | 'kalshi';
    category?: string;
    min_volume?: number;
    sort_by?: string;
    limit?: number;
  }) {
    const platform = params.platform || 'polymarket';
    const queryParams = new URLSearchParams();
    if (params.category) queryParams.append('category', params.category);
    if (params.min_volume) queryParams.append('min_volume', params.min_volume.toString());
    if (params.sort_by) queryParams.append('sort_by', params.sort_by);
    if (params.limit) queryParams.append('limit', params.limit.toString());

    const res = await fetch(`${API_BASE_URL}/api/markets/${platform}/markets?${queryParams}`);
    if (!res.ok) throw new Error('Failed to fetch markets');
    return res.json();
  },

  async getWhales(limit: number = 20) {
    const res = await fetch(`${API_BASE_URL}/api/markets/polymarket/whales?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch whales');
    return res.json();
  },

  async getCategories(platform: 'polymarket' | 'kalshi' = 'polymarket') {
    const cacheKey = `categories_${platform}`

    // Check cache first
    const cached = marketCache.get<any[]>(cacheKey)
    if (cached) {
      console.log('Returning cached categories for', platform)
      return cached
    }

    const res = await fetchWithRetry(`${API_BASE_URL}/api/markets/${platform}/categories`);
    const data = await res.json();

    // Cache the result
    marketCache.set(cacheKey, data)
    return data;
  },

  async getMarketDetail(marketId: string) {
    const res = await fetch(`${API_BASE_URL}/api/markets/polymarket/market/${marketId}`);
    if (!res.ok) throw new Error('Failed to fetch market details');
    return res.json();
  },

  // Database endpoints
  async getCollections() {
    const res = await fetch(`${API_BASE_URL}/api/database/collections`);
    if (!res.ok) throw new Error('Failed to fetch collections');
    return res.json();
  },

  async browseCollection(collection: string, limit: number = 100, search?: string) {
    const params = new URLSearchParams({ limit: limit.toString() })
    if (search) params.append('search', search)

    const res = await fetchWithRetry(`${API_BASE_URL}/api/database/browse/${collection}?${params}`);
    return res.json();
  },

  async getStockOverview(ticker: string) {
    const res = await fetch(`${API_BASE_URL}/api/database/stock/${ticker}/overview`);
    if (!res.ok) throw new Error('Failed to fetch stock overview');
    return res.json();
  },
};
