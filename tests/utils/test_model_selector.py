"""
Tests for model selector utility
"""
import pytest
from utils.model_selector import ModelSelector, select_model_for_task


class TestModelSelector:
    """Test ModelSelector class"""

    def test_model_selector_initialization(self):
        """Test model selector initializes"""
        selector = ModelSelector()
        assert selector is not None

    def test_select_model_for_simple_task(self):
        """Test selecting model for simple task"""
        model = select_model_for_task(
            task_type="simple",
            estimated_tokens=100
        )
        # Should select cheap/fast model for simple tasks
        assert model in ["gemini-flash", "gpt-3.5-turbo"]

    def test_select_model_for_complex_task(self):
        """Test selecting model for complex task"""
        model = select_model_for_task(
            task_type="complex",
            estimated_tokens=5000
        )
        # Should select powerful model for complex tasks
        assert model in ["gemini-pro", "gpt-4", "claude-opus"]

    def test_select_model_by_token_count(self):
        """Test model selection based on token count"""
        # Small token count
        model1 = select_model_for_task(
            task_type="analysis",
            estimated_tokens=500
        )
        assert model1 is not None

        # Large token count
        model2 = select_model_for_task(
            task_type="analysis",
            estimated_tokens=50000
        )
        assert model2 is not None

    def test_model_cost_estimation(self):
        """Test model cost estimation"""
        selector = ModelSelector()
        cost = selector.estimate_cost("gemini-flash", input_tokens=1000, output_tokens=500)
        assert cost >= 0
        assert isinstance(cost, float)

    def test_model_selection_with_budget(self):
        """Test model selection respects budget"""
        selector = ModelSelector()
        model = selector.select_within_budget(
            task_type="analysis",
            max_cost=0.01,
            estimated_tokens=1000
        )
        assert model is not None

    def test_fallback_to_cheaper_model(self):
        """Test fallback when expensive model unavailable"""
        selector = ModelSelector()
        model = selector.select_with_fallback(
            preferred="claude-opus",
            task_type="simple"
        )
        # Should fallback to cheaper option for simple task
        assert model is not None
