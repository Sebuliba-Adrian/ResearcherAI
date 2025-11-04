"""
Tests for circuit breaker utility
"""
import pytest
import time
from utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerConfig,
    CircuitState,
    get_circuit_breaker
)


class TestCircuitBreaker:
    """Test CircuitBreaker class"""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes correctly"""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=60)
        cb = CircuitBreaker("test", config)
        assert cb.config.failure_threshold == 3
        assert cb.config.timeout == 60
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_successful_call(self):
        """Test successful function call"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=1)
        cb = CircuitBreaker("test", config)

        @cb.protect
        def success_function():
            return "success"

        result = success_function()
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker tracks failures"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=1)
        cb = CircuitBreaker("test", config)

        @cb.protect
        def failing_function():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            failing_function()
        assert cb.failure_count == 1

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            failing_function()
        assert cb.failure_count == 2
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_opens_on_threshold(self):
        """Test circuit breaker opens after threshold"""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1)
        cb = CircuitBreaker("test", config)

        @cb.protect
        def failing_function():
            raise RuntimeError("Test error")

        # Fail until threshold
        for _ in range(3):
            with pytest.raises(RuntimeError):
                failing_function()

        assert cb.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            failing_function()

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker can recover"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1, success_threshold=1)
        cb = CircuitBreaker("test", config)

        call_count = [0]

        @cb.protect
        def sometimes_failing():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Initial failures")
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                sometimes_failing()

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.2)

        # Should try again (half-open) and succeed
        result = sometimes_failing()
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60)
        cb = CircuitBreaker("test", config)

        @cb.protect
        def failing_function():
            raise ValueError("Error")

        with pytest.raises(ValueError):
            failing_function()

        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_get_circuit_breaker_global(self):
        """Test getting circuit breaker from global manager"""
        cb = get_circuit_breaker("test_global")
        assert cb is not None
        assert cb.name == "test_global"

    def test_circuit_breaker_context_manager_failure(self):
        """Test circuit breaker context manager with failure"""
        cb = CircuitBreaker("test_ctx")

        try:
            with cb:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should have recorded failure
        stats = cb.get_stats()
        assert stats["failures"] == 1

    def test_circuit_breaker_context_manager_when_open(self):
        """Test context manager raises error when circuit is open"""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test_ctx_open", config)

        # Open the circuit
        try:
            with cb:
                raise ValueError("Error")
        except ValueError:
            pass

        # Should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            with cb:
                pass

    def test_circuit_breaker_half_open_allows_call(self):
        """Test half-open state allows calls through"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        cb = CircuitBreaker("test_half_open", config)

        # Open circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should allow call in half-open
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_get_state(self):
        """Test get_state returns complete state info"""
        cb = CircuitBreaker("test_state")

        cb.record_success()
        cb.record_failure()

        state = cb.get_state()
        assert "name" in state
        assert "state" in state
        assert "failure_count" in state
        assert "success_count" in state
        assert "failure_rate" in state

    def test_circuit_breaker_get_stats(self):
        """Test get_stats returns statistics"""
        cb = CircuitBreaker("test_stats")

        cb.record_success()
        cb.record_success()
        cb.record_failure()

        stats = cb.get_stats()
        assert stats["total_calls"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert "success_rate" in stats
        assert "failure_rate" in stats

    def test_circuit_breaker_manager_health_status(self):
        """Test CircuitBreakerManager health status"""
        from utils.circuit_breaker import CircuitBreakerManager, get_health_status

        manager = CircuitBreakerManager()

        # Create breakers in different states
        cb1 = manager.get_breaker("healthy")
        cb1.record_success()

        cb2_config = CircuitBreakerConfig(failure_threshold=1)
        cb2 = manager.get_breaker("unhealthy", cb2_config)
        cb2.record_failure()

        cb3_config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        cb3 = manager.get_breaker("degraded", cb3_config)
        cb3.record_failure()
        time.sleep(0.15)
        cb3.can_execute()  # Transition to half-open

        health = manager.get_health_status()
        assert "healthy" in health["healthy"]
        assert "unhealthy" in health["unhealthy"]
        assert "degraded" in health["degraded"]
        assert health["summary"]["total"] == 3

    def test_circuit_breaker_manager_get_all_stats(self):
        """Test getting stats for all circuit breakers"""
        from utils.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()
        cb1 = manager.get_breaker("breaker1")
        cb2 = manager.get_breaker("breaker2")

        cb1.record_success()
        cb2.record_failure()

        all_stats = manager.get_all_stats()
        assert "breaker1" in all_stats
        assert "breaker2" in all_stats

    def test_circuit_breaker_manager_reset_all(self):
        """Test resetting all circuit breakers"""
        from utils.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()
        cb1 = manager.get_breaker("reset1")
        cb2 = manager.get_breaker("reset2")

        cb1.record_failure()
        cb2.record_failure()

        manager.reset_all()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

    def test_get_health_status_global(self):
        """Test global health status function"""
        from utils.circuit_breaker import get_health_status

        health = get_health_status()
        assert "healthy" in health
        assert "degraded" in health
        assert "unhealthy" in health
        assert "summary" in health
