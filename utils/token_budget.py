"""
Token Budget Management - The Hidden Cost Killer
================================================

Prevents token consumption spirals that can burn through $5k before anyone notices.

Three-level budgeting:
1. Per-task budgets: No single operation exceeds limits
2. Per-user budgets: Prevent individual users from consuming unfair resources
3. System-wide budgets: Circuit breaker for entire deployment
"""

import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)


# ========================================================================
# TOKEN PRICING (as of 2025)
# ========================================================================

TOKEN_PRICES = {
    # Gemini models (per 1M tokens)
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # OpenAI models (per 1M tokens)
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Claude models (per 1M tokens)
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


@dataclass
class TokenUsageRecord:
    """Record of token usage"""
    task_id: str
    user_id: Optional[str]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class TokenBudgetExceededError(Exception):
    """Raised when token budget is exceeded"""
    pass


class TokenBudgetManager:
    """
    Manages token budgets at three levels:
    - Per-task limits (prevent single expensive operation)
    - Per-user limits (fair resource allocation)
    - System-wide limits (circuit breaker for deployment)
    """

    def __init__(
        self,
        per_task_limit: int = 50000,  # 50K tokens per task
        per_user_limit: int = 500000,  # 500K tokens per user per hour
        system_wide_limit: int = 10000000,  # 10M tokens per hour
        reset_window_hours: float = 1.0
    ):
        """
        Initialize token budget manager.

        Args:
            per_task_limit: Maximum tokens for a single task
            per_user_limit: Maximum tokens per user per window
            system_wide_limit: Maximum system tokens per window
            reset_window_hours: Hours before usage resets
        """
        self.per_task_limit = per_task_limit
        self.per_user_limit = per_user_limit
        self.system_wide_limit = system_wide_limit
        self.reset_window = timedelta(hours=reset_window_hours)

        # Usage tracking
        self.task_usage: Dict[str, int] = {}
        self.user_usage: Dict[str, int] = {}
        self.system_usage: int = 0

        # Usage history
        self.usage_history: list[TokenUsageRecord] = []

        # Timestamps for window resets
        self.user_reset_times: Dict[str, float] = {}
        self.system_reset_time: float = time.time()

        # Thread safety
        self.lock = Lock()

        # Statistics
        self.stats = {
            "total_tasks": 0,
            "total_users": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "budget_violations": 0,
            "tasks_rejected": 0
        }

        logger.info(
            f"TokenBudgetManager initialized: "
            f"per_task={per_task_limit}, per_user={per_user_limit}, "
            f"system={system_wide_limit}, window={reset_window_hours}h"
        )

    # ========================================================================
    # BUDGET CHECKING
    # ========================================================================

    def can_execute(
        self,
        task_id: str,
        user_id: Optional[str],
        estimated_tokens: int
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a task can execute within budget constraints.

        Returns:
            (can_execute: bool, reason: Optional[str])
        """
        with self.lock:
            # Clean up expired windows
            self._cleanup_expired_usage()

            # Check per-task limit
            if estimated_tokens > self.per_task_limit:
                reason = (
                    f"Task would use {estimated_tokens} tokens, "
                    f"exceeds per-task limit of {self.per_task_limit}"
                )
                logger.warning(f"Budget check failed: {reason}")
                self.stats["tasks_rejected"] += 1
                return False, reason

            # Check per-user limit
            if user_id:
                current_user_usage = self.user_usage.get(user_id, 0)
                if current_user_usage + estimated_tokens > self.per_user_limit:
                    reason = (
                        f"User {user_id} would exceed limit: "
                        f"current={current_user_usage}, estimated={estimated_tokens}, "
                        f"limit={self.per_user_limit}"
                    )
                    logger.warning(f"Budget check failed: {reason}")
                    self.stats["budget_violations"] += 1
                    return False, reason

            # Check system-wide limit
            if self.system_usage + estimated_tokens > self.system_wide_limit:
                reason = (
                    f"System would exceed limit: "
                    f"current={self.system_usage}, estimated={estimated_tokens}, "
                    f"limit={self.system_wide_limit}"
                )
                logger.error(f"Budget check failed: {reason}")
                self.stats["budget_violations"] += 1
                return False, reason

            return True, None

    def record_usage(
        self,
        task_id: str,
        user_id: Optional[str],
        model: str,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Record actual token usage after task completion.

        Args:
            task_id: Task identifier
            user_id: User identifier (None for system tasks)
            model: Model used
            input_tokens: Input token count
            output_tokens: Output token count
        """
        with self.lock:
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            cost = self._calculate_cost(model, input_tokens, output_tokens)

            # Update tracking
            self.task_usage[task_id] = total_tokens

            if user_id:
                self.user_usage[user_id] = self.user_usage.get(user_id, 0) + total_tokens
                if user_id not in self.user_reset_times:
                    self.user_reset_times[user_id] = time.time()

            self.system_usage += total_tokens

            # Record in history
            record = TokenUsageRecord(
                task_id=task_id,
                user_id=user_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost
            )
            self.usage_history.append(record)

            # Update stats
            self.stats["total_tasks"] += 1
            self.stats["total_tokens"] += total_tokens
            self.stats["total_cost_usd"] += cost
            if user_id:
                self.stats["total_users"] = len(self.user_usage)

            logger.info(
                f"Token usage recorded: task={task_id}, user={user_id}, "
                f"tokens={total_tokens}, cost=${cost:.4f}"
            )

            # Keep history manageable
            if len(self.usage_history) > 10000:
                self.usage_history = self.usage_history[-5000:]

    # ========================================================================
    # COST CALCULATION
    # ========================================================================

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for token usage"""
        if model not in TOKEN_PRICES:
            logger.warning(f"Unknown model '{model}', using default pricing")
            # Default to Gemini 2.0 Flash pricing
            pricing = TOKEN_PRICES["gemini-2.0-flash"]
        else:
            pricing = TOKEN_PRICES[model]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def estimate_cost(self, model: str, estimated_tokens: int) -> float:
        """
        Estimate cost for a task (assumes 30% output, 70% input).

        Args:
            model: Model to use
            estimated_tokens: Estimated total tokens

        Returns:
            Estimated cost in USD
        """
        input_tokens = int(estimated_tokens * 0.7)
        output_tokens = int(estimated_tokens * 0.3)

        return self._calculate_cost(model, input_tokens, output_tokens)

    # ========================================================================
    # BUDGET MANAGEMENT
    # ========================================================================

    def _cleanup_expired_usage(self):
        """Remove usage records outside the reset window"""
        current_time = time.time()

        # Reset system usage if window expired
        if current_time - self.system_reset_time > self.reset_window.total_seconds():
            logger.info(f"Resetting system usage: {self.system_usage} tokens")
            self.system_usage = 0
            self.system_reset_time = current_time

        # Reset per-user usage if window expired
        users_to_reset = []
        for user_id, reset_time in self.user_reset_times.items():
            if current_time - reset_time > self.reset_window.total_seconds():
                users_to_reset.append(user_id)

        for user_id in users_to_reset:
            logger.info(f"Resetting user {user_id} usage: {self.user_usage.get(user_id, 0)} tokens")
            self.user_usage.pop(user_id, None)
            self.user_reset_times.pop(user_id, None)

    def get_remaining_budget(self, user_id: Optional[str] = None) -> Dict[str, int]:
        """
        Get remaining budget for user or system.

        Returns:
            {
                "system_remaining": int,
                "user_remaining": int (if user_id provided),
                "per_task_limit": int
            }
        """
        with self.lock:
            self._cleanup_expired_usage()

            result = {
                "system_remaining": self.system_wide_limit - self.system_usage,
                "per_task_limit": self.per_task_limit
            }

            if user_id:
                user_usage = self.user_usage.get(user_id, 0)
                result["user_remaining"] = self.per_user_limit - user_usage

            return result

    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================

    def get_stats(self) -> Dict:
        """Get usage statistics"""
        with self.lock:
            avg_tokens_per_task = (
                self.stats["total_tokens"] / self.stats["total_tasks"]
                if self.stats["total_tasks"] > 0
                else 0
            )

            avg_cost_per_task = (
                self.stats["total_cost_usd"] / self.stats["total_tasks"]
                if self.stats["total_tasks"] > 0
                else 0
            )

            return {
                **self.stats,
                "avg_tokens_per_task": avg_tokens_per_task,
                "avg_cost_per_task_usd": avg_cost_per_task,
                "current_system_usage": self.system_usage,
                "system_budget_used_pct": (self.system_usage / self.system_wide_limit) * 100,
                "active_users": len(self.user_usage)
            }

    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a specific user"""
        with self.lock:
            user_records = [r for r in self.usage_history if r.user_id == user_id]

            if not user_records:
                return {
                    "user_id": user_id,
                    "tasks": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "current_usage": 0
                }

            return {
                "user_id": user_id,
                "tasks": len(user_records),
                "total_tokens": sum(r.input_tokens + r.output_tokens for r in user_records),
                "total_cost_usd": sum(r.cost_usd for r in user_records),
                "current_usage": self.user_usage.get(user_id, 0),
                "budget_remaining": self.per_user_limit - self.user_usage.get(user_id, 0)
            }

    def get_recent_usage(self, limit: int = 100) -> list[Dict]:
        """Get recent usage records"""
        with self.lock:
            recent = self.usage_history[-limit:]
            return [
                {
                    "task_id": r.task_id,
                    "user_id": r.user_id,
                    "model": r.model,
                    "tokens": r.input_tokens + r.output_tokens,
                    "cost_usd": r.cost_usd,
                    "timestamp": datetime.fromtimestamp(r.timestamp).isoformat()
                }
                for r in recent
            ]


# ========================================================================
# GLOBAL TOKEN BUDGET MANAGER
# ========================================================================

# Singleton instance
_global_budget_manager: Optional[TokenBudgetManager] = None


def get_token_budget_manager() -> TokenBudgetManager:
    """Get the global token budget manager"""
    global _global_budget_manager
    if _global_budget_manager is None:
        _global_budget_manager = TokenBudgetManager()
    return _global_budget_manager


def check_budget(task_id: str, user_id: Optional[str], estimated_tokens: int) -> tuple[bool, Optional[str]]:
    """Convenience function to check budget"""
    manager = get_token_budget_manager()
    return manager.can_execute(task_id, user_id, estimated_tokens)


def record_tokens(
    task_id: str,
    user_id: Optional[str],
    model: str,
    input_tokens: int,
    output_tokens: int
):
    """Convenience function to record token usage"""
    manager = get_token_budget_manager()
    manager.record_usage(task_id, user_id, model, input_tokens, output_tokens)
