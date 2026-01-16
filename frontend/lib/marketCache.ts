/**
 * Market data caching layer to avoid repeated API calls
 * Cache expires after 5 minutes
 */

interface CachedData<T> {
  data: T
  timestamp: number
}

const CACHE_DURATION = 5 * 60 * 1000 // 5 minutes

class MarketCache {
  private cache: Map<string, CachedData<any>> = new Map()

  get<T>(key: string): T | null {
    const cached = this.cache.get(key)
    if (!cached) return null

    const now = Date.now()
    if (now - cached.timestamp > CACHE_DURATION) {
      this.cache.delete(key)
      return null
    }

    return cached.data as T
  }

  set<T>(key: string, data: T): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    })
  }

  clear(): void {
    this.cache.clear()
  }

  has(key: string): boolean {
    const cached = this.cache.get(key)
    if (!cached) return false

    const now = Date.now()
    if (now - cached.timestamp > CACHE_DURATION) {
      this.cache.delete(key)
      return false
    }

    return true
  }
}

export const marketCache = new MarketCache()
