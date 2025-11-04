"""
Tests for model selector utility
"""
import pytest
from utils.model_selector import (
    ModelSelector,
    ModelSpec,
    TaskRequirements,
    TaskComplexity,
    ModelTier,
    get_model_selector
)


class TestModelSelector:
    """Test ModelSelector class"""

    def test_model_selector_initialization(self):
        """Test model selector initializes"""
        selector = ModelSelector()
        assert selector is not None
        assert selector.default_model == "gemini-2.0-flash"

    def test_select_model_for_simple_task(self):
        """Test selecting model for simple task"""
        selector = ModelSelector()
        requirements = TaskRequirements(
            complexity=TaskComplexity.SIMPLE,
            min_quality_score=0.75
        )
        model = selector.select_model("simple_task", requirements, estimated_tokens=100)
        # Should select basic/standard tier model
        assert model.tier in [ModelTier.BASIC, ModelTier.STANDARD]

    def test_select_model_for_complex_task(self):
        """Test selecting model for complex task"""
        selector = ModelSelector()
        requirements = TaskRequirements(
            complexity=TaskComplexity.COMPLEX,
            min_quality_score=0.9
        )
        model = selector.select_model("complex_task", requirements, estimated_tokens=5000)
        # Should select advanced/premium tier model
        assert model.tier in [ModelTier.ADVANCED, ModelTier.PREMIUM]

    def test_model_cost_estimation(self):
        """Test model cost estimation"""
        selector = ModelSelector()
        from utils.model_selector import MODEL_REGISTRY
        model = MODEL_REGISTRY["gemini-2.0-flash"]
        cost = selector._estimate_cost(model, 1500)
        assert cost >= 0
        assert isinstance(cost, float)

    def test_predefined_task_requirements(self):
        """Test pre-defined task requirement helpers"""
        # Classification
        req = ModelSelector.for_classification()
        assert req.complexity == TaskComplexity.TRIVIAL
        assert req.min_quality_score == 0.75

        # Reasoning
        req = ModelSelector.for_reasoning()
        assert req.complexity == TaskComplexity.COMPLEX
        assert req.min_quality_score == 0.9

    def test_model_selection_stats(self):
        """Test model selection tracking"""
        selector = ModelSelector()
        requirements = ModelSelector.for_classification()

        selector.select_model("test_task", requirements)
        stats = selector.get_stats()

        assert stats["total_selections"] == 1
        assert "selections_by_model" in stats

    def test_get_global_selector(self):
        """Test global model selector"""
        selector = get_model_selector()
        assert selector is not None
        assert isinstance(selector, ModelSelector)
