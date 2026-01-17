"""
In-memory query result cache with TTL
Speeds up repeated queries and follow-ups
"""
import time
import hashlib
import json
from typing import Optional, Dict, Any
from threading import Lock

class QueryCache:
    def __init__(self, ttl_seconds: int = 300):  # 5 minute default TTL
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.lock = Lock()

    def _generate_key(self, question: str, bind_vars: dict = None) -> str:
        """Generate cache key from question and bind variables"""
        # Normalize question (lowercase, strip whitespace)
        normalized = question.lower().strip()

        # Include bind_vars in key if present
        if bind_vars:
            key_data = f"{normalized}|{json.dumps(bind_vars, sort_keys=True)}"
        else:
            key_data = normalized

        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, question: str, bind_vars: dict = None) -> Optional[Dict[str, Any]]:
        """Get cached result if exists and not expired"""
        key = self._generate_key(question, bind_vars)

        with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check if expired
                if time.time() - entry['timestamp'] < self.ttl_seconds:
                    entry['hits'] += 1
                    print(f"[CACHE HIT] Question: {question[:50]}... (hits: {entry['hits']})")
                    return entry['data']
                else:
                    # Expired, remove
                    del self.cache[key]
                    print(f"[CACHE EXPIRED] Question: {question[:50]}...")

        return None

    def set(self, question: str, data: Dict[str, Any], bind_vars: dict = None):
        """Cache query result"""
        key = self._generate_key(question, bind_vars)

        with self.lock:
            self.cache[key] = {
                'data': data,
                'timestamp': time.time(),
                'hits': 0,
                'question': question
            }
            print(f"[CACHE SET] Question: {question[:50]}... (total cached: {len(self.cache)})")

    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            print("[CACHE CLEARED]")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_entries = len(self.cache)
            total_hits = sum(entry['hits'] for entry in self.cache.values())

            # Find most hit entry
            most_hit = None
            if self.cache:
                most_hit = max(self.cache.values(), key=lambda x: x['hits'])

            return {
                'total_entries': total_entries,
                'total_hits': total_hits,
                'ttl_seconds': self.ttl_seconds,
                'most_hit_question': most_hit['question'] if most_hit else None,
                'most_hit_count': most_hit['hits'] if most_hit else 0
            }

    def cleanup_expired(self):
        """Remove expired entries (run periodically)"""
        current_time = time.time()

        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time - entry['timestamp'] >= self.ttl_seconds
            ]

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                print(f"[CACHE CLEANUP] Removed {len(expired_keys)} expired entries")

# Global cache instance
query_cache = QueryCache(ttl_seconds=300)  # 5 minutes
