import { marketCache } from './marketCache'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

function getWorkspaceSessionId(): string {
  if (typeof window === 'undefined') return ''
  let id = localStorage.getItem('karga_session_id')
  if (!id || id.length < 8) {
    id = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : `sess_${Date.now()}_${Math.random().toString(36).slice(2, 12)}`
    localStorage.setItem('karga_session_id', id)
  }
  return id
}

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

    // Fetch from API with retry - both platforms now have /featured endpoint
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

  // Research (Market Workup)
  async getResearchMarkets(platform: 'kalshi' | 'polymarket' = 'kalshi', category?: string, limit = 50) {
    const params = new URLSearchParams({ platform, limit: limit.toString() });
    if (category) params.append('category', category);
    const res = await fetchWithRetry(`${API_BASE_URL}/api/research/markets?${params}`);
    if (!res.ok) throw new Error('Failed to fetch research markets');
    return res.json();
  },
  async getMarketWorkup(marketId: string, platform: 'kalshi' | 'polymarket' = 'kalshi') {
    const res = await fetchWithRetry(`${API_BASE_URL}/api/research/market/${encodeURIComponent(marketId)}?platform=${platform}`);
    if (!res.ok) throw new Error(res.status === 404 ? 'Market not found' : 'Failed to load market workup');
    return res.json();
  },
  async getCongressionalTrades(ticker: string, days = 90) {
    const res = await fetchWithRetry(`${API_BASE_URL}/api/research/congressional/${encodeURIComponent(ticker)}?days=${days}`);
    if (!res.ok) return { ticker, trades: [] };
    return res.json();
  },
  async runResearchBacktest(params: {
    platform: 'kalshi' | 'polymarket';
    resolution_date: string;
    lookback_days?: number;
    market_id?: string;
    probability_series?: Array<{ date: string; probability: number }>;
    signals?: { macro?: boolean; options?: boolean; sec?: boolean; contracts?: boolean };
    theme?: string;
  }) {
    const res = await fetchWithRetry(`${API_BASE_URL}/api/research/backtest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Backtest failed');
    return res.json();
  },

  // Database endpoints
  async getCollections() {
    const res = await fetch(`${API_BASE_URL}/api/database/collections`);
    if (!res.ok) throw new Error('Failed to fetch collections');
    return res.json();
  },

  async browseCollection(collection: string, limit: number = 100, search?: string, offset: number = 0) {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString()
    })
    if (search) params.append('search', search)

    const res = await fetchWithRetry(`${API_BASE_URL}/api/database/browse/${collection}?${params}`);
    return res.json();
  },

  async getStockOverview(ticker: string) {
    const res = await fetch(`${API_BASE_URL}/api/database/stock/${ticker}/overview`);
    if (!res.ok) throw new Error('Failed to fetch stock overview');
    return res.json();
  },

  // Saved workspaces (session-scoped)
  async getWorkspaceHeaders(): Promise<{ id: string; name: string; type: string; question: string; created_at: number; updated_at: number }[]> {
    const res = await fetch(`${API_BASE_URL}/api/workspaces`, {
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to fetch workspaces');
    return res.json();
  },
  async createWorkspace(params: { name: string; type: 'nl' | 'builder'; question: string; forced_plan_aql?: string; watchlist?: string[] }) {
    const res = await fetch(`${API_BASE_URL}/api/workspaces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Session-Id': getWorkspaceSessionId() },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to save workspace');
    return res.json();
  },
  async getWorkspace(id: string) {
    const res = await fetch(`${API_BASE_URL}/api/workspaces/${id}`, {
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to load workspace');
    return res.json();
  },
  async deleteWorkspace(id: string) {
    const res = await fetch(`${API_BASE_URL}/api/workspaces/${id}`, {
      method: 'DELETE',
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to delete workspace');
  },
  async runWorkspace(id: string): Promise<{ results: any[]; analysis: string; follow_up_questions?: string[]; query_plan?: any; metadata?: any }> {
    const res = await fetch(`${API_BASE_URL}/api/workspaces/${id}/run`, {
      method: 'POST',
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to run workspace');
    return res.json();
  },

  // Alerts (session-scoped)
  async getAlerts(): Promise<{ id: string; name: string; workspace_id: string; created_at: number }[]> {
    const res = await fetch(`${API_BASE_URL}/api/alerts`, {
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to fetch alerts');
    return res.json();
  },
  async createAlert(params: { name: string; workspace_id: string }) {
    const res = await fetch(`${API_BASE_URL}/api/alerts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Session-Id': getWorkspaceSessionId() },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to create alert');
    return res.json();
  },
  async deleteAlert(id: string) {
    const res = await fetch(`${API_BASE_URL}/api/alerts/${id}`, {
      method: 'DELETE',
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to delete alert');
  },
  async getNotifications(unreadOnly?: boolean): Promise<{ id: string; alert_id: string; title: string; body: string; created_at: number; read: boolean }[]> {
    const q = unreadOnly ? '?unread_only=true' : '';
    const res = await fetch(`${API_BASE_URL}/api/alerts/notifications${q}`, {
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to fetch notifications');
    return res.json();
  },
  async markNotificationRead(id: string) {
    const res = await fetch(`${API_BASE_URL}/api/alerts/notifications/${id}/read`, {
      method: 'POST',
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to mark read');
  },
  async evaluateAlerts(): Promise<{ evaluated: number; notifications_created: number }> {
    const res = await fetch(`${API_BASE_URL}/api/alerts/evaluate`, {
      method: 'POST',
      headers: { 'X-Session-Id': getWorkspaceSessionId() },
    });
    if (!res.ok) throw new Error('Failed to evaluate alerts');
    return res.json();
  },
};
