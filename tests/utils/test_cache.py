"""
Tests for cache utility
"""
import pytest
import time
from utils.cache import Cache, cache_result


class TestCache:
    """Test Cache class"""

    def test_cache_initialization(self):
        """Test cache initializes correctly"""
        cache = Cache(max_size=100, ttl=300)
        assert cache.max_size == 100
        assert cache.ttl == 300
        assert len(cache.cache) == 0

    def test_cache_set_and_get(self):
        """Test cache set and get operations"""
        cache = Cache(max_size=10, ttl=60)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key2", {"data": "value2"})
        assert cache.get("key2") == {"data": "value2"}

    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = Cache()
        assert cache.get("nonexistent") is None

    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL"""
        cache = Cache(max_size=10, ttl=0.1)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_cache_max_size_lru_eviction(self):
        """Test LRU eviction when max size is reached"""
        cache = Cache(max_size=3, ttl=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_clear(self):
        """Test cache clear operation"""
        cache = Cache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache.cache) == 2
        cache.clear()
        assert len(cache.cache) == 0

    def test_cache_decorator(self):
        """Test cache_result decorator"""
        call_count = [0]

        @cache_result(ttl=60)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count[0] == 1

        # Second call with same argument (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count[0] == 1  # Not incremented

        # Call with different argument
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count[0] == 2

    def test_cache_decorator_with_multiple_args(self):
        """Test cache decorator with multiple arguments"""
        call_count = [0]

        @cache_result(ttl=60)
        def multi_arg_function(a, b, c=3):
            call_count[0] += 1
            return a + b + c

        result1 = multi_arg_function(1, 2)
        assert result1 == 6
        assert call_count[0] == 1

        result2 = multi_arg_function(1, 2)
        assert result2 == 6
        assert call_count[0] == 1

        result3 = multi_arg_function(1, 2, c=4)
        assert result3 == 7
        assert call_count[0] == 2

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation"""
        cache = Cache()

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        assert cache.hits == 2
        assert cache.misses == 1
        assert cache.hit_rate() == pytest.approx(0.666, 0.01)
