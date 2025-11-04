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
