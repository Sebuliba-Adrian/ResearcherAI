"""
Tests for token budget utility
"""
import pytest
import time
from utils.token_budget import (
    TokenBudgetManager,
    TokenBudgetExceededError,
    get_token_budget_manager,
    check_budget,
    record_tokens
)


class TestTokenBudgetManager:
    """Test TokenBudgetManager class"""

    def test_budget_initialization(self):
        """Test budget initializes correctly"""
        manager = TokenBudgetManager(
            per_task_limit=1000,
            per_user_limit=10000,
            system_wide_limit=100000
        )
        assert manager.per_task_limit == 1000
        assert manager.per_user_limit == 10000
        assert manager.system_wide_limit == 100000
        assert manager.system_usage == 0

    def test_can_execute_within_budget(self):
        """Test checking if task can execute within budget"""
        manager = TokenBudgetManager(per_task_limit=1000)

        can_execute, reason = manager.can_execute("task1", "user1", 500)
        assert can_execute is True
        assert reason is None

    def test_exceed_per_task_limit(self):
        """Test exceeding per-task limit"""
        manager = TokenBudgetManager(per_task_limit=100)

        can_execute, reason = manager.can_execute("task1", "user1", 200)
        assert can_execute is False
        assert "per-task limit" in reason

    def test_exceed_per_user_limit(self):
        """Test exceeding per-user limit"""
        manager = TokenBudgetManager(per_user_limit=1000)

        # First task succeeds
        can_execute, _ = manager.can_execute("task1", "user1", 600)
        assert can_execute is True
        manager.record_usage("task1", "user1", "gemini-2.0-flash", 400, 200)

        # Second task would exceed user limit
        can_execute, reason = manager.can_execute("task2", "user1", 600)
        assert can_execute is False
        assert "user1" in reason

    def test_record_usage(self):
        """Test recording token usage"""
        manager = TokenBudgetManager()

        manager.record_usage("task1", "user1", "gemini-2.0-flash", 1000, 500)

        assert manager.task_usage["task1"] == 1500
        assert manager.user_usage["user1"] == 1500
        assert manager.system_usage == 1500

    def test_cost_calculation(self):
        """Test cost calculation"""
        manager = TokenBudgetManager()

        cost = manager.estimate_cost("gemini-2.0-flash", 1000)
        assert cost >= 0
        assert isinstance(cost, float)

    def test_get_remaining_budget(self):
        """Test getting remaining budget"""
        manager = TokenBudgetManager(
            per_task_limit=1000,
            per_user_limit=5000,
            system_wide_limit=10000
        )

        manager.record_usage("task1", "user1", "gemini-2.0-flash", 1000, 500)

        remaining = manager.get_remaining_budget("user1")
        assert remaining["system_remaining"] == 10000 - 1500
        assert remaining["user_remaining"] == 5000 - 1500
        assert remaining["per_task_limit"] == 1000

    def test_get_stats(self):
        """Test getting statistics"""
        manager = TokenBudgetManager()

        manager.record_usage("task1", "user1", "gemini-2.0-flash", 1000, 500)
        manager.record_usage("task2", "user1", "gpt-4", 800, 400)

        stats = manager.get_stats()
        assert stats["total_tasks"] == 2
        assert stats["total_tokens"] == 2700
        assert stats["total_cost_usd"] > 0

    def test_get_user_stats(self):
        """Test getting user-specific statistics"""
        manager = TokenBudgetManager()

        manager.record_usage("task1", "user1", "gemini-2.0-flash", 1000, 500)
        manager.record_usage("task2", "user2", "gpt-4", 800, 400)

        user_stats = manager.get_user_stats("user1")
        assert user_stats["tasks"] == 1
        assert user_stats["total_tokens"] == 1500

    def test_global_budget_manager(self):
        """Test global budget manager singleton"""
        manager = get_token_budget_manager()
        assert manager is not None
        assert isinstance(manager, TokenBudgetManager)

    def test_convenience_functions(self):
        """Test convenience functions"""
        # Reset global manager for clean test
        import utils.token_budget
        utils.token_budget._global_budget_manager = TokenBudgetManager()

        can_execute, _ = check_budget("test_task", "test_user", 1000)
        assert can_execute is True

        record_tokens("test_task", "test_user", "gemini-2.0-flash", 700, 300)

        # Should still be within budget
        can_execute, _ = check_budget("test_task2", "test_user", 1000)
        assert can_execute is True

    def test_exceed_system_wide_limit(self):
        """Test exceeding system-wide limit"""
        manager = TokenBudgetManager(system_wide_limit=1000)

        # Use up most of the system budget
        manager.record_usage("task1", "user1", "gemini-2.0-flash", 700, 200)

        # Next task would exceed system limit
        can_execute, reason = manager.can_execute("task2", "user2", 200)
        assert can_execute is False
        assert "System would exceed limit" in reason

    def test_budget_window_reset(self):
        """Test budget resets after window expires"""
        manager = TokenBudgetManager(
            per_user_limit=1000,
            reset_window_hours=0.00001  # Very short window (0.036 seconds)
        )

        # Use budget
        manager.record_usage("task1", "user1", "gemini-2.0-flash", 700, 300)
        assert manager.user_usage["user1"] == 1000

        # Wait for reset window to expire
        time.sleep(0.05)

        # Trigger cleanup
        can_execute, _ = manager.can_execute("task2", "user1", 100)

        # Budget should be reset, so new task should be allowed
        assert can_execute is True or "user1" not in manager.user_usage

    def test_unknown_model_pricing(self):
        """Test handling unknown model with default pricing"""
        manager = TokenBudgetManager()

        cost = manager.estimate_cost("unknown-model-xyz", 1000)
        # Should use default pricing (gemini-2.0-flash)
        assert cost > 0

    def test_get_recent_usage(self):
        """Test getting recent usage records"""
        manager = TokenBudgetManager()

        manager.record_usage("task1", "user1", "gemini-2.0-flash", 500, 200)
        manager.record_usage("task2", "user2", "gpt-4", 300, 100)

        recent = manager.get_recent_usage(limit=10)
        assert len(recent) == 2
        assert all("task_id" in r for r in recent)
        assert all("cost_usd" in r for r in recent)

    def test_budget_with_none_user(self):
        """Test budget management without user ID (system tasks)"""
        manager = TokenBudgetManager()

        can_execute, _ = manager.can_execute("system_task", None, 1000)
        assert can_execute is True

        manager.record_usage("system_task", None, "gemini-2.0-flash", 700, 300)

        # System usage should be tracked
        assert manager.system_usage == 1000

    def test_multiple_users_budgets(self):
        """Test independent budgets for different users"""
        manager = TokenBudgetManager(per_user_limit=500)

        # User 1
        manager.record_usage("task1", "user1", "gemini-2.0-flash", 300, 100)

        # User 2
        manager.record_usage("task2", "user2", "gemini-2.0-flash", 300, 100)

        # Both should be independent
        user1_stats = manager.get_user_stats("user1")
        user2_stats = manager.get_user_stats("user2")

        assert user1_stats["current_usage"] == 400
        assert user2_stats["current_usage"] == 400

    def test_budget_rejection_tracking(self):
        """Test tracking of budget rejections"""
        manager = TokenBudgetManager(per_task_limit=100)

        # Try to execute task that exceeds limit
        can_execute, _ = manager.can_execute("big_task", "user1", 200)
        assert can_execute is False

        stats = manager.get_stats()
        assert stats["tasks_rejected"] == 1

    def test_user_stats_for_nonexistent_user(self):
        """Test getting stats for user with no history"""
        manager = TokenBudgetManager()

        stats = manager.get_user_stats("nonexistent_user")
        assert stats["tasks"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_cost_usd"] == 0.0
