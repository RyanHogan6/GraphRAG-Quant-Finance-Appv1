const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = {
  // Query endpoints
  async executeQuery(question: string) {
    const res = await fetch(`${API_BASE_URL}/api/query/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error('Query failed');
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
  async getMarkets(params: {
    category?: string;
    min_volume?: number;
    sort_by?: string;
    limit?: number;
  }) {
    const queryParams = new URLSearchParams();
    if (params.category) queryParams.append('category', params.category);
    if (params.min_volume) queryParams.append('min_volume', params.min_volume.toString());
    if (params.sort_by) queryParams.append('sort_by', params.sort_by);
    if (params.limit) queryParams.append('limit', params.limit.toString());

    const res = await fetch(`${API_BASE_URL}/api/markets/polymarket/markets?${queryParams}`);
    if (!res.ok) throw new Error('Failed to fetch markets');
    return res.json();
  },

  async getWhales(limit: number = 20) {
    const res = await fetch(`${API_BASE_URL}/api/markets/polymarket/whales?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch whales');
    return res.json();
  },

  async getCategories() {
    const res = await fetch(`${API_BASE_URL}/api/markets/polymarket/categories`);
    if (!res.ok) throw new Error('Failed to fetch categories');
    return res.json();
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

  async browseCollection(collection: string, limit: number = 50) {
    const res = await fetch(`${API_BASE_URL}/api/database/browse/${collection}?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to browse collection');
    return res.json();
  },

  async getStockOverview(ticker: string) {
    const res = await fetch(`${API_BASE_URL}/api/database/stock/${ticker}/overview`);
    if (!res.ok) throw new Error('Failed to fetch stock overview');
    return res.json();
  },
};
