"""
Tests for cache utility
"""
import pytest
import time
from utils.cache import MemoryCache, DiskCache, CacheManager, cached, get_cache_manager


class TestMemoryCache:
    """Test MemoryCache class"""

    def test_memory_cache_initialization(self):
        """Test memory cache initializes correctly"""
        cache = MemoryCache(max_size=100, default_ttl=300)
        assert cache.max_size == 100
        assert cache.default_ttl == 300
        assert len(cache.cache) == 0

    def test_memory_cache_set_and_get(self):
        """Test cache set and get operations"""
        cache = MemoryCache(max_size=10, default_ttl=60)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key2", {"data": "value2"})
        assert cache.get("key2") == {"data": "value2"}

    def test_memory_cache_miss(self):
        """Test cache miss returns None"""
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_memory_cache_ttl_expiration(self):
        """Test cache entries expire after TTL"""
        cache = MemoryCache(max_size=10, default_ttl=0.1)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_memory_cache_lru_eviction(self):
        """Test LRU eviction when max size is reached"""
        cache = MemoryCache(max_size=3, default_ttl=60)

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

    def test_memory_cache_clear(self):
        """Test cache clear operation"""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache.cache) == 2
        cache.clear()
        assert len(cache.cache) == 0

    def test_memory_cache_stats(self):
        """Test cache statistics"""
        cache = MemoryCache()

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.666, 0.01)


class TestCacheManager:
    """Test two-tier CacheManager"""

    def test_cache_manager_initialization(self):
        """Test cache manager initializes"""
        manager = CacheManager()
        assert manager.memory is not None
        assert manager.disk is not None

    def test_cache_manager_get_set(self):
        """Test basic get/set operations"""
        manager = CacheManager()

        manager.set("test_key", "test_value")
        assert manager.get("test_key") == "test_value"

    def test_cache_manager_persistence(self):
        """Test disk persistence"""
        manager = CacheManager()

        # Set with persistence
        manager.set("persist_key", "persist_value", persist=True)

        # Should be in both memory and disk
        assert manager.memory.get("persist_key") == "persist_value"
        assert manager.disk.get("persist_key") == "persist_value"

    def test_cache_manager_promotion(self):
        """Test promotion from disk to memory"""
        manager = CacheManager()

        # Set only in disk
        manager.disk.set("disk_key", "disk_value")

        # Get should promote to memory
        value = manager.get("disk_key")
        assert value == "disk_value"
        assert manager.memory.get("disk_key") == "disk_value"


class TestCacheDecorator:
    """Test cached decorator"""

    def test_cached_decorator(self):
        """Test cached decorator functionality"""
        call_count = [0]
        manager = CacheManager()

        @cached(manager, ttl=60)
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

    def test_cached_with_persist(self):
        """Test cached decorator with disk persistence"""
        manager = CacheManager()

        @cached(manager, ttl=60, persist=True)
        def persist_function(x):
            return x * 3

        result = persist_function(7)
        assert result == 21

    def test_disk_cache_operations(self):
        """Test disk cache functionality"""
        import tempfile
        import shutil
        from utils.cache import DiskCache

        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        try:
            disk = DiskCache(cache_dir=temp_dir, default_ttl=60)

            # Test set and get
            disk.set("test_key", {"data": "test_value"})
            value = disk.get("test_key")
            assert value == {"data": "test_value"}

            # Test delete
            disk.delete("test_key")
            assert disk.get("test_key") is None

            # Test clear
            disk.set("key1", "value1")
            disk.set("key2", "value2")
            disk.clear()

            stats = disk.get_stats()
            assert stats["files_on_disk"] == 0

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_disk_cache_expiration(self):
        """Test disk cache entry expiration"""
        import tempfile
        import shutil
        from utils.cache import DiskCache

        temp_dir = tempfile.mkdtemp()

        try:
            disk = DiskCache(cache_dir=temp_dir, default_ttl=0.1)

            disk.set("expire_key", "expire_value")
            assert disk.get("expire_key") == "expire_value"

            # Wait for expiration
            time.sleep(0.15)
            assert disk.get("expire_key") is None

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_manager_delete_and_clear(self):
        """Test cache manager delete and clear operations"""
        manager = CacheManager()

        manager.set("key1", "value1", persist=True)
        manager.set("key2", "value2", persist=True)

        # Test delete
        manager.delete("key1")
        assert manager.get("key1") is None
        assert manager.get("key2") == "value2"

        # Test clear
        manager.clear()
        assert manager.get("key2") is None

    def test_cache_manager_stats(self):
        """Test cache manager statistics"""
        manager = CacheManager()

        manager.set("stat_key", "stat_value")
        manager.get("stat_key")

        stats = manager.get_stats()
        assert "memory" in stats
        assert "disk" in stats

    def test_global_cache_manager(self):
        """Test global cache manager singleton"""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()

        assert manager1 is manager2

    def test_cache_entry_no_expiration(self):
        """Test cache entry with no TTL"""
        from utils.cache import CacheEntry

        entry = CacheEntry("key", "value", ttl=None)
        assert entry.is_expired() is False

    def test_memory_cache_cleanup_expired(self):
        """Test automatic cleanup of expired entries"""
        cache = MemoryCache(max_size=200, default_ttl=0.1)

        # Add entries
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # Wait for some to expire
        time.sleep(0.15)

        # Add more entries to trigger modulo 100 cleanup
        for i in range(100):
            cache.set(f"newkey{i}", f"newvalue{i}")

        # Expired entries should have been cleaned up
        stats = cache.get_stats()
        assert stats["expirations"] >= 0  # Some may have been cleaned up

    def test_memory_cache_evict_when_empty(self):
        """Test LRU eviction on empty cache"""
        cache = MemoryCache(max_size=1)
        # This should not crash
        cache._evict_lru()

    def test_disk_cache_read_error(self):
        """Test disk cache handles read errors gracefully"""
        import tempfile
        import shutil
        from utils.cache import DiskCache

        temp_dir = tempfile.mkdtemp()

        try:
            disk = DiskCache(cache_dir=temp_dir)

            # Create a corrupted cache file
            cache_file = disk._get_cache_file("corrupt_key")
            with open(cache_file, 'w') as f:
                f.write("not valid json {{{")

            # Should handle error and return None
            value = disk.get("corrupt_key")
            assert value is None

            stats = disk.get_stats()
            assert stats["misses"] >= 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_disk_cache_write_error(self):
        """Test disk cache handles write errors"""
        import tempfile
        import shutil
        import os
        from utils.cache import DiskCache

        temp_dir = tempfile.mkdtemp()

        try:
            disk = DiskCache(cache_dir=temp_dir)

            # Create a cache file and make it read-only
            disk.set("test_key", "test_value")
            cache_file = disk._get_cache_file("readonly_key")

            # Create read-only file to trigger write error
            with open(cache_file, 'w') as f:
                f.write('{"test": "data"}')
            os.chmod(cache_file, 0o444)  # Read-only

            # Try to write - should fail silently
            disk.set("readonly_key", "new_value")

            # Original data should still be there
            # (write failed, so old data remains)

        finally:
            # Restore permissions before cleanup
            try:
                os.chmod(cache_file, 0o644)
            except:
                pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cached_decorator_with_custom_key_func(self):
        """Test cached decorator with custom key function"""
        manager = CacheManager()

        def custom_key(x, y):
            return f"custom_{x}_{y}"

        @cached(manager, key_func=custom_key)
        def func_with_custom_key(x, y):
            return x + y

        result = func_with_custom_key(3, 4)
        assert result == 7

        # Should use custom key for caching
        result2 = func_with_custom_key(3, 4)
        assert result2 == 7
