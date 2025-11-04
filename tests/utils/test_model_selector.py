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

    def test_model_selection_with_no_candidates(self):
        """Test model selection when no models meet requirements"""
        selector = ModelSelector()

        # Impossible requirements
        requirements = TaskRequirements(
            complexity=TaskComplexity.EXPERT,
            min_quality_score=0.99,
            max_cost_per_1m=0.001  # Too cheap
        )

        model = selector.select_model("impossible_task", requirements)
        # Should fall back to default
        assert model.name == "gemini-2.0-flash"

    def test_model_selection_with_provider_filter(self):
        """Test model selection with allowed providers"""
        selector = ModelSelector(allowed_providers=["google"])

        requirements = ModelSelector.for_classification()
        model = selector.select_model("google_only_task", requirements)

        assert model.provider == "google"

    def test_model_selection_with_large_context(self):
        """Test model selection requiring large context window"""
        selector = ModelSelector()

        requirements = TaskRequirements(
            complexity=TaskComplexity.MODERATE,
            min_quality_score=0.8,
            requires_large_context=True
        )

        model = selector.select_model("large_context_task", requirements)
        assert model.context_window >= 64000

    def test_all_predefined_requirements(self):
        """Test all pre-defined task requirement methods"""
        # Extraction
        req = ModelSelector.for_extraction()
        assert req.complexity == TaskComplexity.SIMPLE

        # Summarization
        req = ModelSelector.for_summarization()
        assert req.complexity == TaskComplexity.SIMPLE

        # Graph extraction
        req = ModelSelector.for_graph_extraction()
        assert req.complexity == TaskComplexity.MODERATE

        # Quality check
        req = ModelSelector.for_quality_check()
        assert req.complexity == TaskComplexity.SIMPLE

        # Research
        req = ModelSelector.for_research()
        assert req.complexity == TaskComplexity.EXPERT

    def test_model_usage_breakdown(self):
        """Test model usage percentage breakdown"""
        selector = ModelSelector()

        requirements = ModelSelector.for_classification()
        selector.select_model("task1", requirements)
        selector.select_model("task2", requirements)

        breakdown = selector.get_model_usage_breakdown()
        assert isinstance(breakdown, dict)
        # Should have percentages
        for model, percentage in breakdown.items():
            assert 0 <= percentage <= 100

    def test_model_cost_savings_tracking(self):
        """Test cost savings tracking"""
        selector = ModelSelector(default_model="gpt-4")  # Expensive default

        requirements = ModelSelector.for_classification()
        selector.select_model("cheap_task", requirements)

        stats = selector.get_stats()
        # Should save money vs expensive default
        assert stats["cost_saved_usd"] >= 0

    def test_select_for_task_convenience_function(self):
        """Test convenience function for task selection"""
        from utils.model_selector import select_for_task

        requirements = ModelSelector.for_classification()
        model_name = select_for_task("test_task", requirements)

        assert isinstance(model_name, str)
        assert len(model_name) > 0
