"""
Tests for ReasoningAgent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from agents.reasoner_agent import ReasoningAgent


class TestReasoningAgentInitialization:
    """Test ReasoningAgent initialization"""

    @patch('agents.reasoner_agent.genai')
    def test_initialization_default_config(self, mock_genai):
        """Test reasoner initializes with default config"""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        vector_agent = Mock()

        agent = ReasoningAgent(graph_agent, vector_agent)

        assert agent.graph_agent == graph_agent
        assert agent.vector_agent == vector_agent
        assert agent.conversation_memory == 5
        assert agent.max_context_length == 4000
        assert agent.temperature == 0.7
        assert agent.max_tokens == 2048
        assert agent.conversation_history == []
        assert agent.model == mock_model

        # Verify genai.configure was called
        mock_genai.configure.assert_called_once()

        # Verify GenerativeModel was created with correct config
        mock_genai.GenerativeModel.assert_called_once_with(
            "gemini-2.0-flash",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2048
            }
        )

    @patch('agents.reasoner_agent.genai')
    def test_initialization_custom_config(self, mock_genai):
        """Test reasoner with custom configuration"""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        vector_agent = Mock()
        config = {
            "conversation_memory": 10,
            "max_context_length": 8000,
            "temperature": 0.9,
            "max_tokens": 4096,
            "model": "gemini-pro"
        }

        agent = ReasoningAgent(graph_agent, vector_agent, config)

        assert agent.conversation_memory == 10
        assert agent.max_context_length == 8000
        assert agent.temperature == 0.9
        assert agent.max_tokens == 4096

        # Verify custom model was used
        mock_genai.GenerativeModel.assert_called_once_with(
            "gemini-pro",
            generation_config={
                "temperature": 0.9,
                "max_output_tokens": 4096
            }
        )

    @patch('agents.reasoner_agent.genai')
    def test_initialization_missing_api_key(self, mock_genai):
        """Test initialization fails without GOOGLE_API_KEY"""
        import os

        # Temporarily remove the API key
        api_key = os.environ.pop("GOOGLE_API_KEY", None)

        try:
            graph_agent = Mock()
            vector_agent = Mock()

            with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable not set"):
                ReasoningAgent(graph_agent, vector_agent)
        finally:
            # Restore the API key
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key

    @patch('agents.reasoner_agent.genai')
    def test_initialization_empty_config(self, mock_genai):
        """Test reasoner handles empty config dict"""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        vector_agent = Mock()

        agent = ReasoningAgent(graph_agent, vector_agent, {})

        # Should use default values
        assert agent.conversation_memory == 5
        assert agent.max_context_length == 4000
        assert agent.temperature == 0.7
        assert agent.max_tokens == 2048


class TestSynthesizeAnswer:
    """Test synthesize_answer method"""

    @patch('agents.reasoner_agent.genai')
    def test_synthesize_answer_success(self, mock_genai):
        """Test successful answer synthesis"""
        # Setup mock model
        mock_response = Mock()
        mock_response.text = "This is a synthesized answer"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Setup mock agents
        graph_agent = Mock()
        graph_agent.query_graph.return_value = [
            {"nodes": ["Node1", "Node2"], "relationships": ["RELATED_TO"]}
        ]

        vector_agent = Mock()
        vector_agent.search.return_value = [
            {"title": "Paper 1", "score": 0.95, "source": "arXiv", "text": "Test content"}
        ]

        agent = ReasoningAgent(graph_agent, vector_agent)

        # Test synthesis
        answer = agent.synthesize_answer("What is machine learning?")

        assert answer == "This is a synthesized answer"
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["query"] == "What is machine learning?"
        assert agent.conversation_history[0]["answer"] == "This is a synthesized answer"
        assert agent.conversation_history[0]["graph_results"] == 1
        assert agent.conversation_history[0]["vector_results"] == 1

        # Verify agent methods were called correctly
        graph_agent.query_graph.assert_called_once_with("What is machine learning?", max_hops=2)
        vector_agent.search.assert_called_once_with("What is machine learning?", top_k=5)
        mock_model.generate_content.assert_called_once()

    @patch('agents.reasoner_agent.genai')
    def test_synthesize_answer_with_conversation_context(self, mock_genai):
        """Test synthesis uses conversation context"""
        mock_response = Mock()
        mock_response.text = "Answer with context"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        graph_agent.query_graph.return_value = []

        vector_agent = Mock()
        vector_agent.search.return_value = []

        agent = ReasoningAgent(graph_agent, vector_agent)

        # Add some conversation history
        agent.conversation_history = [
            {"query": "Tell me about AI", "answer": "AI is artificial intelligence", "graph_results": 0, "vector_results": 0}
        ]

        # Synthesize with context
        answer = agent.synthesize_answer("What else can you tell me?")

        assert answer == "Answer with context"
        assert len(agent.conversation_history) == 2

        # Verify the prompt included conversation context
        call_args = mock_model.generate_content.call_args[0][0]
        assert "PREVIOUS CONVERSATION:" in call_args

    @patch('agents.reasoner_agent.genai')
    def test_synthesize_answer_trims_history(self, mock_genai):
        """Test conversation history is trimmed to memory limit"""
        mock_response = Mock()
        mock_response.text = "Answer"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        graph_agent.query_graph.return_value = []

        vector_agent = Mock()
        vector_agent.search.return_value = []

        config = {"conversation_memory": 3}
        agent = ReasoningAgent(graph_agent, vector_agent, config)

        # Add 5 conversations (should trim to 3)
        for i in range(5):
            agent.synthesize_answer(f"Question {i}")

        assert len(agent.conversation_history) == 3
        # Should keep the last 3
        assert agent.conversation_history[0]["query"] == "Question 2"
        assert agent.conversation_history[1]["query"] == "Question 3"
        assert agent.conversation_history[2]["query"] == "Question 4"

    @patch('agents.reasoner_agent.genai')
    def test_synthesize_answer_handles_generation_error(self, mock_genai):
        """Test synthesis handles Gemini generation errors"""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        graph_agent.query_graph.return_value = []

        vector_agent = Mock()
        vector_agent.search.return_value = []

        agent = ReasoningAgent(graph_agent, vector_agent)

        answer = agent.synthesize_answer("Test query")

        assert "I encountered an error while generating the answer" in answer
        assert "API Error" in answer
        assert len(agent.conversation_history) == 1

    @patch('agents.reasoner_agent.genai')
    def test_synthesize_answer_empty_results(self, mock_genai):
        """Test synthesis with empty graph and vector results"""
        mock_response = Mock()
        mock_response.text = "No results answer"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        graph_agent.query_graph.return_value = []

        vector_agent = Mock()
        vector_agent.search.return_value = []

        agent = ReasoningAgent(graph_agent, vector_agent)

        answer = agent.synthesize_answer("Unknown topic")

        assert answer == "No results answer"
        assert agent.conversation_history[0]["graph_results"] == 0
        assert agent.conversation_history[0]["vector_results"] == 0


class TestConversationContextBuilding:
    """Test _build_conversation_context method"""

    @patch('agents.reasoner_agent.genai')
    def test_build_conversation_context_empty_history(self, mock_genai):
        """Test context building with empty history"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        context = agent._build_conversation_context()

        assert context == ""

    @patch('agents.reasoner_agent.genai')
    def test_build_conversation_context_single_turn(self, mock_genai):
        """Test context building with single conversation turn"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        agent.conversation_history = [
            {"query": "What is AI?", "answer": "AI is artificial intelligence technology", "graph_results": 1, "vector_results": 1}
        ]

        context = agent._build_conversation_context()

        assert "PREVIOUS CONVERSATION:" in context
        assert "Turn 1:" in context
        assert "What is AI?" in context
        assert "AI is artificial intelligence" in context

    @patch('agents.reasoner_agent.genai')
    def test_build_conversation_context_multiple_turns(self, mock_genai):
        """Test context building with multiple conversation turns"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        agent.conversation_history = [
            {"query": "Q1", "answer": "A1", "graph_results": 1, "vector_results": 1},
            {"query": "Q2", "answer": "A2", "graph_results": 1, "vector_results": 1},
            {"query": "Q3", "answer": "A3", "graph_results": 1, "vector_results": 1},
        ]

        context = agent._build_conversation_context()

        assert "PREVIOUS CONVERSATION:" in context
        assert "Turn 1:" in context
        assert "Turn 2:" in context
        assert "Turn 3:" in context

    @patch('agents.reasoner_agent.genai')
    def test_build_conversation_context_limits_to_last_three(self, mock_genai):
        """Test context only includes last 3 turns"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        agent.conversation_history = [
            {"query": "Q1", "answer": "A1", "graph_results": 1, "vector_results": 1},
            {"query": "Q2", "answer": "A2", "graph_results": 1, "vector_results": 1},
            {"query": "Q3", "answer": "A3", "graph_results": 1, "vector_results": 1},
            {"query": "Q4", "answer": "A4", "graph_results": 1, "vector_results": 1},
            {"query": "Q5", "answer": "A5", "graph_results": 1, "vector_results": 1},
        ]

        context = agent._build_conversation_context()

        # Should only include last 3 turns
        assert "Q3" in context
        assert "Q4" in context
        assert "Q5" in context
        assert "Q1" not in context
        assert "Q2" not in context

    @patch('agents.reasoner_agent.genai')
    def test_build_conversation_context_truncates_long_answers(self, mock_genai):
        """Test context truncates answers to 200 chars"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        long_answer = "A" * 500
        agent.conversation_history = [
            {"query": "Q1", "answer": long_answer, "graph_results": 1, "vector_results": 1}
        ]

        context = agent._build_conversation_context()

        # Should truncate to 200 chars plus "..."
        assert "A" * 200 in context
        assert context.count("...") >= 1


class TestFormattingMethods:
    """Test result formatting methods"""

    @patch('agents.reasoner_agent.genai')
    def test_format_graph_results_empty(self, mock_genai):
        """Test formatting empty graph results"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        formatted = agent._format_graph_results([])

        assert formatted == "No graph relationships found."

    @patch('agents.reasoner_agent.genai')
    def test_format_graph_results_single_path(self, mock_genai):
        """Test formatting single graph path"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        results = [
            {"nodes": ["Paper1", "Author1"], "relationships": ["AUTHORED_BY"]}
        ]

        formatted = agent._format_graph_results(results)

        assert "KNOWLEDGE GRAPH RELATIONSHIPS:" in formatted
        assert "Paper1" in formatted
        assert "Author1" in formatted
        assert "AUTHORED_BY" in formatted

    @patch('agents.reasoner_agent.genai')
    def test_format_graph_results_multiple_paths(self, mock_genai):
        """Test formatting multiple graph paths"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        results = [
            {"nodes": ["Paper1", "Author1"], "relationships": ["AUTHORED_BY"]},
            {"nodes": ["Paper1", "Concept1"], "relationships": ["ABOUT"]},
            {"nodes": ["Author1", "Institution1"], "relationships": ["AFFILIATED_WITH"]}
        ]

        formatted = agent._format_graph_results(results)

        assert "1." in formatted
        assert "2." in formatted
        assert "3." in formatted

    @patch('agents.reasoner_agent.genai')
    def test_format_graph_results_limits_to_10(self, mock_genai):
        """Test formatting limits to first 10 paths"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        results = [
            {"nodes": [f"Node{i}", f"Node{i+1}"], "relationships": ["REL"]}
            for i in range(15)
        ]

        formatted = agent._format_graph_results(results)

        # Should only include first 10
        assert "10." in formatted
        assert "11." not in formatted

    @patch('agents.reasoner_agent.genai')
    def test_format_graph_results_complex_path(self, mock_genai):
        """Test formatting complex graph path with multiple hops"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        results = [
            {
                "nodes": ["Paper1", "Author1", "Institution1", "Country1"],
                "relationships": ["AUTHORED_BY", "AFFILIATED_WITH", "LOCATED_IN"]
            }
        ]

        formatted = agent._format_graph_results(results)

        assert "Paper1 -AUTHORED_BY-> Author1" in formatted
        assert "Author1 -AFFILIATED_WITH-> Institution1" in formatted
        assert "Institution1 -LOCATED_IN-> Country1" in formatted

    @patch('agents.reasoner_agent.genai')
    def test_format_vector_results_empty(self, mock_genai):
        """Test formatting empty vector results"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        formatted = agent._format_vector_results([])

        assert formatted == "No relevant documents found."

    @patch('agents.reasoner_agent.genai')
    def test_format_vector_results_single_result(self, mock_genai):
        """Test formatting single vector result"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        results = [
            {
                "title": "Test Paper",
                "score": 0.95,
                "source": "arXiv",
                "text": "This is the paper content"
            }
        ]

        formatted = agent._format_vector_results(results)

        assert "RELEVANT RESEARCH:" in formatted
        assert "Test Paper" in formatted
        assert "0.950" in formatted
        assert "arXiv" in formatted
        assert "This is the paper content" in formatted

    @patch('agents.reasoner_agent.genai')
    def test_format_vector_results_multiple_results(self, mock_genai):
        """Test formatting multiple vector results"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        results = [
            {"title": "Paper1", "score": 0.95, "source": "arXiv", "text": "Content1"},
            {"title": "Paper2", "score": 0.85, "source": "Semantic Scholar", "text": "Content2"},
            {"title": "Paper3", "score": 0.75, "source": "PubMed", "text": "Content3"}
        ]

        formatted = agent._format_vector_results(results)

        assert "1." in formatted
        assert "2." in formatted
        assert "3." in formatted
        assert "Paper1" in formatted
        assert "Paper2" in formatted
        assert "Paper3" in formatted

    @patch('agents.reasoner_agent.genai')
    def test_format_vector_results_truncates_long_text(self, mock_genai):
        """Test formatting truncates long text to 300 chars"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        long_text = "A" * 500
        results = [
            {"title": "Paper", "score": 0.95, "source": "arXiv", "text": long_text}
        ]

        formatted = agent._format_vector_results(results)

        # Should truncate to 300 chars plus "..."
        assert "A" * 300 in formatted
        assert "..." in formatted

    @patch('agents.reasoner_agent.genai')
    def test_format_vector_results_handles_missing_fields(self, mock_genai):
        """Test formatting handles missing fields gracefully"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        results = [
            {"score": 0.95}  # Missing title, source, text
        ]

        formatted = agent._format_vector_results(results)

        assert "Unknown" in formatted


class TestPromptBuilding:
    """Test _build_prompt method"""

    @patch('agents.reasoner_agent.genai')
    def test_build_prompt_complete(self, mock_genai):
        """Test building complete prompt with all contexts"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        query = "What is deep learning?"
        conv_context = "Previous: AI discussion"
        graph_context = "Graph: Paper->Author"
        vector_context = "Vector: Research papers"

        prompt = agent._build_prompt(query, conv_context, graph_context, vector_context)

        assert "What is deep learning?" in prompt
        assert "Previous: AI discussion" in prompt
        assert "Graph: Paper->Author" in prompt
        assert "Vector: Research papers" in prompt
        assert "INSTRUCTIONS:" in prompt
        assert "ANSWER:" in prompt

    @patch('agents.reasoner_agent.genai')
    def test_build_prompt_truncates_if_too_long(self, mock_genai):
        """Test prompt truncation when exceeding max context length"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        config = {"max_context_length": 500}
        agent = ReasoningAgent(graph_agent, vector_agent, config)

        query = "Query"
        conv_context = "A" * 1000
        graph_context = "B" * 1000
        vector_context = "C" * 1000

        prompt = agent._build_prompt(query, conv_context, graph_context, vector_context)

        # Should be truncated
        assert len(prompt) <= 1000  # Should be reasonably short
        assert "Query" in prompt
        assert "ANSWER:" in prompt

    @patch('agents.reasoner_agent.genai')
    def test_build_prompt_empty_contexts(self, mock_genai):
        """Test building prompt with empty contexts"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        query = "Test query"

        prompt = agent._build_prompt(query, "", "", "")

        assert "Test query" in prompt
        assert "ANSWER:" in prompt


class TestConversationHistoryManagement:
    """Test conversation history management methods"""

    @patch('agents.reasoner_agent.genai')
    def test_clear_history(self, mock_genai):
        """Test clearing conversation history"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        agent.conversation_history = [
            {"query": "Q1", "answer": "A1", "graph_results": 1, "vector_results": 1},
            {"query": "Q2", "answer": "A2", "graph_results": 1, "vector_results": 1}
        ]

        agent.clear_history()

        assert agent.conversation_history == []

    @patch('agents.reasoner_agent.genai')
    def test_get_history(self, mock_genai):
        """Test getting conversation history"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        expected_history = [
            {"query": "Q1", "answer": "A1", "graph_results": 1, "vector_results": 1},
            {"query": "Q2", "answer": "A2", "graph_results": 1, "vector_results": 1}
        ]
        agent.conversation_history = expected_history

        history = agent.get_history()

        assert history == expected_history

    @patch('agents.reasoner_agent.genai')
    def test_get_history_empty(self, mock_genai):
        """Test getting empty history"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        history = agent.get_history()

        assert history == []


class TestStatisticsTracking:
    """Test statistics tracking methods"""

    @patch('agents.reasoner_agent.genai')
    def test_get_stats_empty_history(self, mock_genai):
        """Test getting stats with empty history"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        stats = agent.get_stats()

        assert stats["conversation_turns"] == 0
        assert stats["memory_limit"] == 5
        assert stats["total_graph_results"] == 0
        assert stats["total_vector_results"] == 0

    @patch('agents.reasoner_agent.genai')
    def test_get_stats_with_history(self, mock_genai):
        """Test getting stats with conversation history"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        agent.conversation_history = [
            {"query": "Q1", "answer": "A1", "graph_results": 3, "vector_results": 5},
            {"query": "Q2", "answer": "A2", "graph_results": 2, "vector_results": 4},
            {"query": "Q3", "answer": "A3", "graph_results": 1, "vector_results": 3}
        ]

        stats = agent.get_stats()

        assert stats["conversation_turns"] == 3
        assert stats["memory_limit"] == 5
        assert stats["total_graph_results"] == 6  # 3 + 2 + 1
        assert stats["total_vector_results"] == 12  # 5 + 4 + 3

    @patch('agents.reasoner_agent.genai')
    def test_get_stats_custom_memory_limit(self, mock_genai):
        """Test stats reflect custom memory limit"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        config = {"conversation_memory": 10}
        agent = ReasoningAgent(graph_agent, vector_agent, config)

        stats = agent.get_stats()

        assert stats["memory_limit"] == 10

    @patch('agents.reasoner_agent.genai')
    def test_get_stats_handles_missing_result_counts(self, mock_genai):
        """Test stats handles missing result counts in history"""
        mock_genai.GenerativeModel.return_value = Mock()

        graph_agent = Mock()
        vector_agent = Mock()
        agent = ReasoningAgent(graph_agent, vector_agent)

        agent.conversation_history = [
            {"query": "Q1", "answer": "A1"},  # Missing result counts
            {"query": "Q2", "answer": "A2", "graph_results": 2, "vector_results": 3}
        ]

        stats = agent.get_stats()

        assert stats["total_graph_results"] == 2
        assert stats["total_vector_results"] == 3


class TestIntegrationScenarios:
    """Test integration scenarios"""

    @patch('agents.reasoner_agent.genai')
    def test_multi_turn_conversation_flow(self, mock_genai):
        """Test multi-turn conversation maintains context"""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        # Setup different responses for each turn
        responses = [
            Mock(text="Answer 1"),
            Mock(text="Answer 2"),
            Mock(text="Answer 3")
        ]
        mock_model.generate_content.side_effect = responses

        graph_agent = Mock()
        graph_agent.query_graph.return_value = []

        vector_agent = Mock()
        vector_agent.search.return_value = []

        agent = ReasoningAgent(graph_agent, vector_agent)

        # Multi-turn conversation
        agent.synthesize_answer("What is AI?")
        agent.synthesize_answer("Tell me more")
        agent.synthesize_answer("Any examples?")

        assert len(agent.conversation_history) == 3
        assert agent.conversation_history[0]["answer"] == "Answer 1"
        assert agent.conversation_history[1]["answer"] == "Answer 2"
        assert agent.conversation_history[2]["answer"] == "Answer 3"

    @patch('agents.reasoner_agent.genai')
    def test_clear_and_restart_conversation(self, mock_genai):
        """Test clearing history and starting new conversation"""
        mock_response = Mock(text="New answer")
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        graph_agent.query_graph.return_value = []

        vector_agent = Mock()
        vector_agent.search.return_value = []

        agent = ReasoningAgent(graph_agent, vector_agent)

        # Have a conversation
        agent.synthesize_answer("Question 1")
        agent.synthesize_answer("Question 2")

        assert len(agent.conversation_history) == 2

        # Clear and restart
        agent.clear_history()
        agent.synthesize_answer("New question")

        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["query"] == "New question"

    @patch('agents.reasoner_agent.genai')
    def test_stats_update_throughout_conversation(self, mock_genai):
        """Test statistics update correctly throughout conversation"""
        mock_response = Mock(text="Answer")
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        graph_agent = Mock()
        graph_agent.query_graph.return_value = [{"nodes": ["A", "B"], "relationships": ["REL"]}]

        vector_agent = Mock()
        vector_agent.search.return_value = [{"title": "Paper", "score": 0.9, "source": "arXiv", "text": "Content"}]

        agent = ReasoningAgent(graph_agent, vector_agent)

        # Initial stats
        stats = agent.get_stats()
        assert stats["conversation_turns"] == 0
        assert stats["total_graph_results"] == 0
        assert stats["total_vector_results"] == 0

        # After first query
        agent.synthesize_answer("Query 1")
        stats = agent.get_stats()
        assert stats["conversation_turns"] == 1
        assert stats["total_graph_results"] == 1
        assert stats["total_vector_results"] == 1

        # After second query
        agent.synthesize_answer("Query 2")
        stats = agent.get_stats()
        assert stats["conversation_turns"] == 2
        assert stats["total_graph_results"] == 2
        assert stats["total_vector_results"] == 2
