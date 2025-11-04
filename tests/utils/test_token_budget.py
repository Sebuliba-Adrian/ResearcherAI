"""
Tests for token budget utility
"""
import pytest
from utils.token_budget import TokenBudget, BudgetExceeded


class TestTokenBudget:
    """Test TokenBudget class"""

    def test_budget_initialization(self):
        """Test budget initializes correctly"""
        budget = TokenBudget(task_budget=1000, user_budget=10000, system_budget=100000)
        assert budget.task_budget == 1000
        assert budget.user_budget == 10000
        assert budget.system_budget == 100000
        assert budget.task_used == 0
        assert budget.user_used == 0
        assert budget.system_used == 0

    def test_budget_consume_tokens(self):
        """Test consuming tokens"""
        budget = TokenBudget(task_budget=1000)
        budget.consume(500)
        assert budget.task_used == 500
        assert budget.remaining() == 500

    def test_budget_exceed_task_budget(self):
        """Test exceeding task budget raises exception"""
        budget = TokenBudget(task_budget=100)
        budget.consume(90)

        with pytest.raises(BudgetExceeded):
            budget.consume(20)

    def test_budget_check_availability(self):
        """Test checking budget availability"""
        budget = TokenBudget(task_budget=1000)

        assert budget.can_consume(500) is True
        budget.consume(900)
        assert budget.can_consume(200) is False
        assert budget.can_consume(50) is True

    def test_budget_reset(self):
        """Test budget reset"""
        budget = TokenBudget(task_budget=1000)
        budget.consume(500)
        assert budget.task_used == 500

        budget.reset()
        assert budget.task_used == 0
        assert budget.remaining() == 1000

    def test_budget_percentage_used(self):
        """Test percentage calculation"""
        budget = TokenBudget(task_budget=1000)
        budget.consume(250)
        assert budget.percentage_used() == 25.0

        budget.consume(250)
        assert budget.percentage_used() == 50.0

    def test_budget_multi_level(self):
        """Test multi-level budget (task, user, system)"""
        budget = TokenBudget(
            task_budget=100,
            user_budget=500,
            system_budget=1000
        )

        budget.consume(50, level="task")
        assert budget.task_used == 50
        assert budget.user_used == 50
        assert budget.system_used == 50

        budget.consume(60, level="task")
        # Should fail at task level
        with pytest.raises(BudgetExceeded):
            pass
