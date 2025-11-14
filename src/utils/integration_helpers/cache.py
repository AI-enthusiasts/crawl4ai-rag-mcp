"""Performance cache utilities for integration layer.

Provides high-performance caching with TTL-based expiration,
LRU eviction, and performance metrics tracking.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class PerformanceCache:
    """High-performance cache for validation results and search queries.

    Features:
    - TTL-based expiration
    - Size-based eviction (LRU)
    - Performance metrics
    - Async-safe operations
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize the performance cache.

        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_times: dict[str, float] = {}
        self._lock = asyncio.Lock()

        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    async def get(self, key: str) -> Any | None:
        """Get an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            item, expiry_time = self._cache[key]

            # Check if expired
            if time.time() > expiry_time:
                del self._cache[key]
                del self._access_times[key]
                self.misses += 1
                return None

            # Update access time for LRU
            self._access_times[key] = time.time()
            self.hits += 1
            return item

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override in seconds
        """
        async with self._lock:
            ttl = ttl or self.default_ttl
            expiry_time = time.time() + ttl

            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()

            self._cache[key] = (value, expiry_time)
            self._access_times[key] = time.time()

    async def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if not self._access_times:
            return

        # Find LRU item
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]

        # Remove from cache
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self.evictions += 1

    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }


def create_cache_key(*args: Any, **kwargs: Any) -> str:
    """Create a deterministic cache key from arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        MD5 hash of the arguments as cache key
    """
    # Create a string representation of all arguments
    key_parts = []

    # Add positional arguments
    for arg in args:
        if isinstance(arg, str | int | float | bool):
            key_parts.append(str(arg))
        else:
            key_parts.append(str(hash(str(arg))))

    # Add keyword arguments (sorted for consistency)
    for key, value in sorted(kwargs.items()):
        if isinstance(value, str | int | float | bool):
            key_parts.append(f"{key}={value}")
        else:
            key_parts.append(f"{key}={hash(str(value))}")

    # Create MD5 hash
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()
