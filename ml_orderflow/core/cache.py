"""
Simple in-memory caching layer for LLM responses.

Provides time-based cache expiration and thread-safe operations.
"""

import time
import hashlib
import json
import threading
from typing import Any, Optional
from dataclasses import dataclass
from ml_orderflow.utils.initializer import logger_instance

logger = logger_instance.get_logger()


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time"""
    value: Any
    expires_at: float
    created_at: float
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > self.expires_at


class LRUCache:
    """
    Simple LRU cache with TTL support for LLM responses.
    
    Args:
        ttl_seconds: Time-to-live for cache entries
        max_size: Maximum number of entries to store
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 100):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # Track access order for LRU
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
        logger.info(f"LRU Cache initialized: TTL={ttl_seconds}s, max_size={max_size}")
    
    def _generate_key(self, data: Any) -> str:
        """
        Generate cache key from data.
        
        Args:
            data: Data to generate key from (will be JSON serialized)
        
        Returns:
            SHA256 hash of the data
        """
        try:
            # Sort keys for consistent hashing
            json_str = json.dumps(data, sort_keys=True)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to generate cache key: {e}")
            # Fallback to string representation
            return hashlib.sha256(str(data).encode()).hexdigest()
    
    def get(self, key_data: Any) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            key_data: Data to generate cache key from
        
        Returns:
            Cached value if found and not expired, None otherwise
        """
        key = self._generate_key(key_data)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                logger.debug(f"Cache miss for key: {key[:16]}...")
                return None
            
            if entry.is_expired():
                logger.debug(f"Cache entry expired for key: {key[:16]}...")
                del self._cache[key]
                self._access_order.remove(key)
                self._misses += 1
                return None
            
            # Update access order (move to end)
            self._access_order.remove(key)
            self._access_order.append(key)
            
            self._hits += 1
            age = time.time() - entry.created_at
            logger.debug(f"Cache hit for key: {key[:16]}... (age: {age:.1f}s)")
            return entry.value
    
    def set(self, key_data: Any, value: Any):
        """
        Store value in cache.
        
        Args:
            key_data: Data to generate cache key from
            value: Value to cache
        """
        key = self._generate_key(key_data)
        
        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:16]}...")
            
            # Create new entry
            now = time.time()
            entry = CacheEntry(
                value=value,
                expires_at=now + self.ttl_seconds,
                created_at=now
            )
            
            # Update cache
            if key in self._cache:
                self._access_order.remove(key)
            
            self._cache[key] = entry
            self._access_order.append(key)
            
            logger.debug(f"Cached value for key: {key[:16]}... (TTL: {self.ttl_seconds}s)")
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            logger.info(f"Cache cleared ({count} entries removed)")
    
    def cleanup_expired(self):
        """Remove all expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._access_order.remove(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
                "ttl_seconds": self.ttl_seconds
            }
