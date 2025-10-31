"""
Circuit Breaker Pattern for Multi-Agent Systems
===============================================

Prevents cascade failures by isolating failing components.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail immediately
- HALF_OPEN: Testing if service recovered

Based on production resilience patterns.
"""

import logging
import time
from typing import Dict, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 60.0  # Seconds before trying half-open
    window_size: int = 10  # Rolling window for failure rate


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker to prevent cascade failures.

    Usage:
        breaker = CircuitBreaker("api_call")

        # Decorator pattern
        @breaker.protect
        def risky_operation():
            return call_external_api()

        # Context manager pattern
        with breaker:
            result = call_external_api()

        # Manual pattern
        if breaker.can_execute():
            try:
                result = call_external_api()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker
            config: Configuration (uses defaults if not provided)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()

        # Rolling window for tracking
        self.recent_calls: list = []  # (timestamp, success: bool)

        # Statistics
        self.stats = {
            "total_calls": 0,
            "successes": 0,
            "failures": 0,
            "rejections": 0,  # Calls rejected due to open circuit
            "state_changes": 0
        }

        logger.info(f"CircuitBreaker '{name}' initialized: {self.config}")

    # ========================================================================
    # CORE CIRCUIT BREAKER LOGIC
    # ========================================================================

    def can_execute(self) -> bool:
        """
        Check if a call can be executed.

        Returns:
            True if call should proceed, False if circuit is open
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # In half-open state, allow call to test recovery
            return True

        return False

    def record_success(self):
        """Record a successful execution"""
        self.stats["total_calls"] += 1
        self.stats["successes"] += 1

        self._add_to_window(success=True)

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(
                f"CircuitBreaker '{self.name}': Success in HALF_OPEN "
                f"({self.success_count}/{self.config.success_threshold})"
            )

            if self.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self):
        """Record a failed execution"""
        self.stats["total_calls"] += 1
        self.stats["failures"] += 1

        self._add_to_window(success=False)

        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.warning(
            f"CircuitBreaker '{self.name}': Failure recorded "
            f"({self.failure_count}/{self.config.failure_threshold})"
        )

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open immediately reopens circuit
            logger.warning(f"CircuitBreaker '{self.name}': Failed in HALF_OPEN, reopening")
            self._transition_to(CircuitState.OPEN)

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def record_rejection(self):
        """Record a rejected call (circuit was open)"""
        self.stats["rejections"] += 1

    # ========================================================================
    # DECORATOR AND CONTEXT MANAGER INTERFACES
    # ========================================================================

    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect a function with circuit breaker.

        Usage:
            @breaker.protect
            def my_function():
                return risky_operation()
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_execute():
                self.record_rejection()
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service is unavailable."
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        return wrapper

    def __enter__(self):
        """Context manager entry"""
        if not self.can_execute():
            self.record_rejection()
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False  # Don't suppress exceptions

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state"""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()
        self.stats["state_changes"] += 1

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"CircuitBreaker '{self.name}': CLOSED - Service recovered")

        elif new_state == CircuitState.OPEN:
            self.success_count = 0
            logger.error(
                f"CircuitBreaker '{self.name}': OPEN - Service failing, "
                f"rejecting calls for {self.config.timeout}s"
            )

        elif new_state == CircuitState.HALF_OPEN:
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"CircuitBreaker '{self.name}': HALF_OPEN - Testing recovery")

        logger.info(f"CircuitBreaker '{self.name}': {old_state} → {new_state}")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True

        time_since_last_failure = time.time() - self.last_failure_time
        return time_since_last_failure >= self.config.timeout

    def _add_to_window(self, success: bool):
        """Add call result to rolling window"""
        self.recent_calls.append((time.time(), success))

        # Keep only recent calls within window
        if len(self.recent_calls) > self.config.window_size:
            self.recent_calls = self.recent_calls[-self.config.window_size:]

    # ========================================================================
    # MONITORING AND STATISTICS
    # ========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_in_current_state": time.time() - self.last_state_change,
            "last_failure_time": self.last_failure_time,
            "failure_rate": self._calculate_failure_rate()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        total = self.stats["total_calls"]
        return {
            **self.stats,
            "success_rate": self.stats["successes"] / total if total > 0 else 0,
            "failure_rate": self.stats["failures"] / total if total > 0 else 0,
            "rejection_rate": self.stats["rejections"] / (total + self.stats["rejections"]) if total > 0 else 0,
            "current_state": self.state
        }

    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate in recent window"""
        if not self.recent_calls:
            return 0.0

        failures = sum(1 for _, success in self.recent_calls if not success)
        return failures / len(self.recent_calls)

    def reset(self):
        """Reset circuit breaker to initial state (for testing)"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.recent_calls = []
        logger.info(f"CircuitBreaker '{self.name}' reset to CLOSED state")


# ========================================================================
# CIRCUIT BREAKER MANAGER
# ========================================================================

class CircuitBreakerManager:
    """
    Centralized manager for all circuit breakers in the system.

    Usage:
        manager = CircuitBreakerManager()

        # Get or create circuit breaker
        breaker = manager.get_breaker("data_collector")

        # Check health of all breakers
        health = manager.get_health_status()
    """

    def __init__(self):
        """Initialize circuit breaker manager"""
        self.breakers: Dict[str, CircuitBreaker] = {}
        logger.info("CircuitBreakerManager initialized")

    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.

        Args:
            name: Circuit breaker identifier
            config: Configuration (uses defaults if not provided)

        Returns:
            CircuitBreaker instance
        """
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
            logger.info(f"Created new circuit breaker: {name}")

        return self.breakers[name]

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all circuit breakers.

        Returns:
            {
                "healthy": List of healthy breakers,
                "degraded": List of half-open breakers,
                "unhealthy": List of open breakers,
                "summary": Overall stats
            }
        """
        healthy = []
        degraded = []
        unhealthy = []

        for name, breaker in self.breakers.items():
            state = breaker.get_state()

            if state["state"] == CircuitState.CLOSED:
                healthy.append(name)
            elif state["state"] == CircuitState.HALF_OPEN:
                degraded.append(name)
            else:  # OPEN
                unhealthy.append(name)

        return {
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "summary": {
                "total": len(self.breakers),
                "healthy_count": len(healthy),
                "degraded_count": len(degraded),
                "unhealthy_count": len(unhealthy)
            }
        }

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all circuit breakers"""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers (for testing)"""
        for breaker in self.breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")


# ========================================================================
# GLOBAL CIRCUIT BREAKER MANAGER
# ========================================================================

# Singleton instance for application-wide use
_global_manager = CircuitBreakerManager()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get a circuit breaker from the global manager"""
    return _global_manager.get_breaker(name, config)


def get_health_status() -> Dict[str, Any]:
    """Get health status of all circuit breakers"""
    return _global_manager.get_health_status()
