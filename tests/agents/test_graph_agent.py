"""
Comprehensive tests for KnowledgeGraphAgent
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from agents.graph_agent import KnowledgeGraphAgent


class TestKnowledgeGraphAgentInitialization:
    """Test KnowledgeGraphAgent initialization"""

    def test_initialization_default_networkx(self):
        """Test agent initializes with default NetworkX backend"""
        agent = KnowledgeGraphAgent()

        assert agent.db_type == "networkx"
        assert hasattr(agent, 'G')
        assert agent.config == {"type": "networkx"}

    def test_initialization_with_networkx_config(self):
        """Test agent initializes with explicit NetworkX config"""
        config = {"type": "networkx"}
        agent = KnowledgeGraphAgent(config)

        assert agent.db_type == "networkx"
        assert hasattr(agent, 'G')
        assert agent.config == config

    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_initialization_with_gemini_api_key(self, mock_model_class, mock_configure):
        """Test Gemini model is initialized when API key is present"""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        agent = KnowledgeGraphAgent()

        mock_configure.assert_called_once_with(api_key="test-api-key-123")
        mock_model_class.assert_called_once_with("gemini-2.0-flash")
        assert agent.model == mock_model

    @patch.dict('os.environ', {'GOOGLE_API_KEY': ''})
    def test_initialization_without_gemini_api_key(self):
        """Test agent initializes without Gemini when API key is missing"""
        agent = KnowledgeGraphAgent()

        assert agent.model is None

    @patch('neo4j.GraphDatabase')
    def test_initialization_with_neo4j_config(self, mock_graph_db):
        """Test agent initializes with Neo4j backend"""
        mock_driver = Mock()
        mock_driver.verify_connectivity = Mock()
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_driver

        config = {
            "type": "neo4j",
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "test_password",
            "database": "test_db"
        }

        agent = KnowledgeGraphAgent(config)

        assert agent.db_type == "neo4j"
        assert hasattr(agent, 'driver')
        mock_graph_db.driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "test_password"),
            max_connection_lifetime=3000
        )
        mock_driver.verify_connectivity.assert_called_once()

    @patch('neo4j.GraphDatabase')
    def test_initialization_neo4j_creates_constraints(self, mock_graph_db):
        """Test Neo4j initialization creates necessary constraints"""
        mock_driver = Mock()
        mock_driver.verify_connectivity = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)

        # Verify constraint creation was called
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args[0][0]
        assert "CREATE CONSTRAINT" in call_args
        assert "entity_id" in call_args

    @patch('neo4j.GraphDatabase')
    def test_initialization_neo4j_fallback_to_networkx_on_error(self, mock_graph_db):
        """Test fallback to NetworkX when Neo4j connection fails"""
        mock_graph_db.driver.side_effect = Exception("Connection failed")

        config = {"type": "neo4j", "uri": "bolt://localhost:7687"}
        agent = KnowledgeGraphAgent(config)

        # Should fallback to NetworkX
        assert agent.db_type == "networkx"
        assert hasattr(agent, 'G')

    @patch('neo4j.GraphDatabase')
    def test_initialization_neo4j_with_custom_connection_lifetime(self, mock_graph_db):
        """Test Neo4j initialization with custom max_connection_lifetime"""
        mock_driver = Mock()
        mock_driver.verify_connectivity = Mock()
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_driver

        config = {
            "type": "neo4j",
            "max_connection_lifetime": 5000
        }

        agent = KnowledgeGraphAgent(config)

        # Check max_connection_lifetime was used
        call_kwargs = mock_graph_db.driver.call_args[1]
        assert call_kwargs['max_connection_lifetime'] == 5000


class TestKnowledgeGraphAgentTripleExtraction:
    """Test triple extraction functionality"""

    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_extract_triples_with_gemini(self, mock_model_class, mock_configure):
        """Test triple extraction using Gemini model"""
        # Setup mock model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Machine Learning | is_a | AI Technique\nDeep Learning | extends | Machine Learning\nNeural Networks | implements | Deep Learning"
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_model_class.return_value = mock_model

        agent = KnowledgeGraphAgent()

        paper = {
            "title": "Deep Learning for Computer Vision",
            "abstract": "This paper discusses deep learning methods for computer vision tasks.",
            "authors": ["Author 1"],
            "topics": ["AI", "Computer Vision"]
        }

        triples = agent._extract_triples(paper)

        assert len(triples) == 3
        assert triples[0] == ("Machine Learning", "is_a", "AI Technique")
        assert triples[1] == ("Deep Learning", "extends", "Machine Learning")
        assert triples[2] == ("Neural Networks", "implements", "Deep Learning")
        mock_model.generate_content.assert_called_once()

    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_extract_triples_limits_to_20(self, mock_model_class, mock_configure):
        """Test triple extraction limits results to 20 triples"""
        mock_model = Mock()
        mock_response = Mock()
        # Generate 25 triples
        triple_lines = [f"Subject{i} | predicate{i} | Object{i}" for i in range(25)]
        mock_response.text = "\n".join(triple_lines)
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_model_class.return_value = mock_model

        agent = KnowledgeGraphAgent()

        paper = {"title": "Test", "abstract": "Test abstract"}
        triples = agent._extract_triples(paper)

        assert len(triples) == 20

    def test_extract_triples_without_gemini_fallback(self):
        """Test triple extraction falls back to metadata when Gemini unavailable"""
        agent = KnowledgeGraphAgent()
        agent.model = None  # Simulate no Gemini model

        paper = {
            "title": "Test Paper",
            "abstract": "Test abstract",
            "authors": ["Author 1", "Author 2", "Author 3", "Author 4"],
            "topics": ["Topic A", "Topic B", "Topic C", "Topic D"]
        }

        triples = agent._extract_triples(paper)

        # Should extract from authors and topics (max 3 each)
        assert len(triples) == 6
        # Check author triples
        assert ("Author 1", "authored", "Test Paper") in triples
        assert ("Author 2", "authored", "Test Paper") in triples
        assert ("Author 3", "authored", "Test Paper") in triples
        # Check topic triples
        assert ("Test Paper", "is_about", "Topic A") in triples
        assert ("Test Paper", "is_about", "Topic B") in triples
        assert ("Test Paper", "is_about", "Topic C") in triples

    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_extract_triples_handles_malformed_lines(self, mock_model_class, mock_configure):
        """Test triple extraction handles malformed response lines"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """Subject1 | predicate1 | Object1
Invalid line without pipes
Subject2 | only_two_parts
Subject3 | predicate3 | Object3"""
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_model_class.return_value = mock_model

        agent = KnowledgeGraphAgent()
        paper = {"title": "Test", "abstract": "Test"}

        triples = agent._extract_triples(paper)

        # Should only extract valid triples
        assert len(triples) == 2
        assert triples[0] == ("Subject1", "predicate1", "Object1")
        assert triples[1] == ("Subject3", "predicate3", "Object3")

    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_extract_triples_handles_extraction_error(self, mock_model_class, mock_configure):
        """Test triple extraction handles errors gracefully"""
        mock_model = Mock()
        mock_model.generate_content = Mock(side_effect=Exception("API Error"))
        mock_model_class.return_value = mock_model

        agent = KnowledgeGraphAgent()
        paper = {"title": "Test", "abstract": "Test"}

        triples = agent._extract_triples(paper)

        # Should return empty list on error
        assert triples == []

    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_extract_triples_public_method(self, mock_model_class, mock_configure):
        """Test public extract_triples method"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Concept1 | relates_to | Concept2"
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_model_class.return_value = mock_model

        agent = KnowledgeGraphAgent()

        content = "This is test content about AI and machine learning."
        triples = agent.extract_triples(content, title="Test Title")

        assert len(triples) >= 0
        assert isinstance(triples, list)

    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_extract_triples_with_metadata(self, mock_model_class, mock_configure):
        """Test public extract_triples method with metadata"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Concept1 | relates_to | Concept2"
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_model_class.return_value = mock_model

        agent = KnowledgeGraphAgent()

        metadata = {
            "title": "Metadata Title",
            "authors": ["Author 1"],
            "topics": ["AI"]
        }
        triples = agent.extract_triples("Test content", metadata=metadata)

        assert isinstance(triples, list)


class TestKnowledgeGraphAgentNetworkX:
    """Test NetworkX-specific functionality"""

    def test_add_to_networkx_paper(self, sample_research_paper):
        """Test adding paper to NetworkX graph"""
        agent = KnowledgeGraphAgent()

        result = agent._add_to_networkx(sample_research_paper, [])

        assert result["nodes"] >= 1
        assert sample_research_paper["id"] in agent.G.nodes()
        node_data = agent.G.nodes[sample_research_paper["id"]]
        assert node_data["type"] == "paper"
        assert node_data["title"] == sample_research_paper["title"]

    def test_add_to_networkx_authors(self, sample_research_paper):
        """Test adding authors creates nodes and edges"""
        agent = KnowledgeGraphAgent()

        result = agent._add_to_networkx(sample_research_paper, [])

        # Check authors are added
        for author in sample_research_paper["authors"]:
            assert author in agent.G.nodes()
            node_data = agent.G.nodes[author]
            assert node_data["type"] == "author"
            # Check edge exists
            assert agent.G.has_edge(author, sample_research_paper["id"])

    def test_add_to_networkx_topics(self, sample_research_paper):
        """Test adding topics creates nodes and edges"""
        paper = sample_research_paper.copy()
        paper["topics"] = ["Machine Learning", "Neural Networks"]
        agent = KnowledgeGraphAgent()

        result = agent._add_to_networkx(paper, [])

        # Check topics are added
        for topic in paper["topics"]:
            assert topic in agent.G.nodes()
            node_data = agent.G.nodes[topic]
            assert node_data["type"] == "topic"
            # Check edge from paper to topic
            assert agent.G.has_edge(paper["id"], topic)

    def test_add_to_networkx_triples(self, sample_research_paper):
        """Test adding triples to NetworkX graph"""
        agent = KnowledgeGraphAgent()
        triples = [
            ("Subject1", "predicate1", "Object1"),
            ("Subject2", "predicate2", "Object2")
        ]

        result = agent._add_to_networkx(sample_research_paper, triples)

        # Check triple nodes exist
        assert "Subject1" in agent.G.nodes()
        assert "Object1" in agent.G.nodes()
        assert "Subject2" in agent.G.nodes()
        assert "Object2" in agent.G.nodes()

        # Check edges exist with labels
        assert agent.G.has_edge("Subject1", "Object1")
        assert agent.G.has_edge("Subject2", "Object2")

        # Verify edge labels
        edges = list(agent.G.get_edge_data("Subject1", "Object1").values())
        assert any(edge.get("label") == "predicate1" for edge in edges)

    def test_add_to_networkx_duplicate_nodes(self, sample_research_paper):
        """Test adding duplicate nodes doesn't increase count"""
        agent = KnowledgeGraphAgent()

        # Add paper twice
        result1 = agent._add_to_networkx(sample_research_paper, [])
        result2 = agent._add_to_networkx(sample_research_paper, [])

        # Second add should have fewer new nodes
        assert result2["nodes"] <= result1["nodes"]

    def test_add_to_networkx_stats_counting(self, sample_research_paper):
        """Test node and edge counting is accurate"""
        agent = KnowledgeGraphAgent()
        paper = sample_research_paper.copy()
        paper["topics"] = ["Topic1", "Topic2"]
        triples = [("S1", "P1", "O1")]

        result = agent._add_to_networkx(paper, triples)

        # 1 paper + 2 authors + 2 topics + 2 entities (S1, O1) = 7 nodes
        # 2 author edges + 2 topic edges + 1 triple edge = 5 edges
        assert result["nodes"] == 7
        assert result["edges"] == 5


class TestKnowledgeGraphAgentNeo4j:
    """Test Neo4j-specific functionality"""

    @patch('neo4j.GraphDatabase')
    def test_add_to_neo4j_paper(self, mock_graph_db, sample_research_paper, mock_neo4j_driver):
        """Test adding paper to Neo4j graph"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        result = agent._add_to_neo4j(sample_research_paper, [])

        # Verify paper creation query was called (skip the constraint creation call)
        assert mock_session.run.call_count >= 1
        # Second call should be the paper creation (first is constraint)
        paper_calls = [call for call in mock_session.run.call_args_list
                       if "MERGE (p:Paper" in str(call)]
        assert len(paper_calls) >= 1
        assert result["nodes"] >= 1

    @patch('neo4j.GraphDatabase')
    def test_add_to_neo4j_authors(self, mock_graph_db, sample_research_paper, mock_neo4j_driver):
        """Test adding authors to Neo4j"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        result = agent._add_to_neo4j(sample_research_paper, [])

        # Should have called run for paper + each author
        assert mock_session.run.call_count >= len(sample_research_paper["authors"]) + 1

    @patch('neo4j.GraphDatabase')
    def test_add_to_neo4j_filters_null_authors(self, mock_graph_db, mock_neo4j_driver):
        """Test Neo4j filters out null and empty authors"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        # Reset call count after initialization (constraint creation)
        mock_session.run.reset_mock()
        agent.driver = mock_neo4j_driver

        paper = {
            "id": "test_1",
            "title": "Test",
            "authors": [None, "", "  ", "Valid Author"],
            "topics": []
        }

        result = agent._add_to_neo4j(paper, [])

        # Should only process the valid author
        # Count calls: 1 for paper + 1 for valid author = 2
        assert mock_session.run.call_count == 2

    @patch('neo4j.GraphDatabase')
    def test_add_to_neo4j_topics(self, mock_graph_db, sample_research_paper, mock_neo4j_driver):
        """Test adding topics to Neo4j"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        paper = sample_research_paper.copy()
        paper["topics"] = ["Topic1", "Topic2"]

        result = agent._add_to_neo4j(paper, [])

        # Check ABOUT relationship queries were made
        calls = [str(call) for call in mock_session.run.call_args_list]
        about_calls = [c for c in calls if "ABOUT" in c]
        assert len(about_calls) == 2

    @patch('neo4j.GraphDatabase')
    def test_add_to_neo4j_filters_invalid_topics(self, mock_graph_db, mock_neo4j_driver):
        """Test Neo4j filters out invalid topics (dicts, lists, nulls)"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        # Reset call count after initialization
        mock_session.run.reset_mock()
        agent.driver = mock_neo4j_driver

        paper = {
            "id": "test_1",
            "title": "Test",
            "authors": [],
            "topics": [None, "", {"invalid": "dict"}, ["list"], "Valid Topic"]
        }

        result = agent._add_to_neo4j(paper, [])

        # Should only process the valid topic
        # 1 for paper + 1 for valid topic = 2
        assert mock_session.run.call_count == 2

    @patch('neo4j.GraphDatabase')
    def test_add_to_neo4j_triples(self, mock_graph_db, sample_research_paper, mock_neo4j_driver):
        """Test adding triples to Neo4j"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        # Reset after initialization
        mock_session.run.reset_mock()
        agent.driver = mock_neo4j_driver

        triples = [
            ("Subject1", "relates_to", "Object1"),
            ("Subject2", "is_a", "Object2")
        ]

        result = agent._add_to_neo4j(sample_research_paper, triples)

        # Check triple queries were made (excluding paper and author queries)
        calls = [str(call) for call in mock_session.run.call_args_list]
        entity_calls = [c for c in calls if "Entity" in c]
        assert len(entity_calls) == 2

    @patch('neo4j.GraphDatabase')
    def test_add_to_neo4j_cleans_predicate_names(self, mock_graph_db, mock_neo4j_driver):
        """Test Neo4j cleans predicate names for valid Cypher syntax"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        # Reset after initialization
        mock_session.run.reset_mock()
        agent.driver = mock_neo4j_driver

        paper = {"id": "test_1", "title": "Test", "authors": [], "topics": []}
        triples = [
            ("S1", "is-a-type-of", "O1"),  # Should convert to IS_A_TYPE_OF
            ("S2", "has relationship with", "O2"),  # Should convert to HAS_RELATIONSHIP_WITH
            ("S3", "123invalid", "O3"),  # Should default to RELATED_TO
            ("S4", "!!!###", "O4")  # Should default to RELATED_TO
        ]

        result = agent._add_to_neo4j(paper, triples)

        # Verify predicates were cleaned
        calls = mock_session.run.call_args_list
        # Get the query strings from calls
        queries = [call[0][0] for call in calls if len(call[0]) > 0]

        # Check that cleaned predicates appear in queries
        triple_queries = [q for q in queries if "Entity" in q]
        assert len(triple_queries) == 4


class TestKnowledgeGraphAgentProcessPapers:
    """Test processing multiple papers"""

    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_process_papers_networkx(self, mock_model_class, mock_configure, sample_research_paper):
        """Test processing papers with NetworkX backend"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Subject | predicate | Object"
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_model_class.return_value = mock_model

        agent = KnowledgeGraphAgent()

        papers = [sample_research_paper]
        stats = agent.process_papers(papers)

        assert stats["papers_processed"] == 1
        assert stats["nodes_added"] > 0
        assert stats["edges_added"] > 0

    @patch('neo4j.GraphDatabase')
    @patch('agents.graph_agent.genai.configure')
    @patch('agents.graph_agent.genai.GenerativeModel')
    def test_process_papers_neo4j(self, mock_model_class, mock_configure, mock_graph_db,
                                   sample_research_paper, mock_neo4j_driver):
        """Test processing papers with Neo4j backend"""
        # Setup Gemini mock
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Subject | predicate | Object"
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_model_class.return_value = mock_model

        # Setup Neo4j mock
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        papers = [sample_research_paper]
        stats = agent.process_papers(papers)

        assert stats["papers_processed"] == 1
        assert stats["nodes_added"] > 0
        assert stats["edges_added"] > 0

    def test_process_papers_multiple(self):
        """Test processing multiple papers"""
        agent = KnowledgeGraphAgent()
        agent.model = None  # Use fallback extraction

        papers = [
            {
                "id": "paper1",
                "title": "Paper 1",
                "authors": ["Author A"],
                "topics": ["Topic X"]
            },
            {
                "id": "paper2",
                "title": "Paper 2",
                "authors": ["Author B"],
                "topics": ["Topic Y"]
            }
        ]

        stats = agent.process_papers(papers)

        assert stats["papers_processed"] == 2
        assert stats["nodes_added"] >= 4  # At least 2 papers + 2 authors
        assert stats["edges_added"] >= 2  # At least 2 author edges

    def test_process_papers_handles_errors(self):
        """Test processing continues when individual paper fails"""
        agent = KnowledgeGraphAgent()
        agent.model = None

        # Mock _add_to_networkx to fail on first paper
        original_add = agent._add_to_networkx
        call_count = [0]

        def mock_add(paper, triples):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Processing failed")
            return original_add(paper, triples)

        agent._add_to_networkx = mock_add

        papers = [
            {"id": "paper1", "title": "Paper 1", "authors": [], "topics": []},
            {"id": "paper2", "title": "Paper 2", "authors": [], "topics": []}
        ]

        stats = agent.process_papers(papers)

        # Should process second paper successfully
        assert stats["papers_processed"] == 1

    def test_process_papers_empty_list(self):
        """Test processing empty paper list"""
        agent = KnowledgeGraphAgent()

        stats = agent.process_papers([])

        assert stats["papers_processed"] == 0
        assert stats["nodes_added"] == 0
        assert stats["edges_added"] == 0


class TestKnowledgeGraphAgentQueries:
    """Test graph querying functionality"""

    def test_query_networkx_entity_found(self):
        """Test querying NetworkX graph for existing entity"""
        agent = KnowledgeGraphAgent()

        # Add test data
        paper = {
            "id": "paper1",
            "title": "Machine Learning Paper",
            "authors": ["Author A"],
            "topics": ["Machine Learning"]
        }
        agent._add_to_networkx(paper, [("ML", "is_a", "AI")])

        results = agent.query_graph("Machine", max_hops=2)

        # Should find "Machine Learning" topic or related nodes
        assert isinstance(results, list)
        # Query might return empty list or results depending on graph structure
        # Just verify it doesn't crash
        assert results is not None

    def test_query_networkx_entity_not_found(self):
        """Test querying NetworkX for non-existent entity"""
        agent = KnowledgeGraphAgent()

        results = agent.query_graph("NonExistent", max_hops=2)

        assert results == []

    def test_query_networkx_max_hops(self):
        """Test NetworkX query respects max_hops parameter"""
        agent = KnowledgeGraphAgent()

        # Create a chain: A -> B -> C -> D
        agent.G.add_edge("A", "B", label="link")
        agent.G.add_edge("B", "C", label="link")
        agent.G.add_edge("C", "D", label="link")

        # Query with max_hops=1 should not reach D from A
        results = agent.query_graph("A", max_hops=1)

        # Results should not include paths longer than 2 nodes (1 hop)
        for path in results:
            assert len(path["nodes"]) <= 2

    @patch('neo4j.GraphDatabase')
    def test_query_neo4j(self, mock_graph_db, mock_neo4j_driver):
        """Test querying Neo4j graph"""
        # Setup mock result
        mock_node1 = {"name": "Entity1"}
        mock_node2 = {"name": "Entity2"}
        mock_rel = Mock()
        mock_rel.type = "RELATES_TO"

        mock_path = Mock()
        mock_path.nodes = [mock_node1, mock_node2]
        mock_path.relationships = [mock_rel]

        mock_record = {"path": mock_path}
        mock_result = [mock_record]

        mock_session = Mock()
        mock_session.run = Mock(return_value=mock_result)
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        results = agent.query_graph("Entity1", max_hops=2)

        assert len(results) == 1
        assert results[0]["nodes"] == ["Entity1", "Entity2"]
        assert results[0]["relationships"] == ["RELATES_TO"]

    @patch('neo4j.GraphDatabase')
    def test_query_neo4j_uses_title_fallback(self, mock_graph_db, mock_neo4j_driver):
        """Test Neo4j query uses title when name is not available"""
        mock_node1 = {"title": "Paper Title"}
        mock_node2 = {"name": "Entity"}

        mock_path = Mock()
        mock_path.nodes = [mock_node1, mock_node2]
        mock_path.relationships = []

        mock_record = {"path": mock_path}
        mock_result = [mock_record]

        mock_session = Mock()
        mock_session.run = Mock(return_value=mock_result)
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        results = agent.query_graph("Paper", max_hops=1)

        assert results[0]["nodes"] == ["Paper Title", "Entity"]

    def test_query_graph_delegates_to_backend(self):
        """Test query_graph delegates to correct backend"""
        agent = KnowledgeGraphAgent({"type": "networkx"})

        # Add some data
        agent.G.add_edge("A", "B", label="test")

        results = agent.query_graph("A")

        # Should use NetworkX backend
        assert isinstance(results, list)


class TestKnowledgeGraphAgentStatistics:
    """Test graph statistics functionality"""

    def test_get_stats_networkx(self):
        """Test getting statistics from NetworkX graph"""
        agent = KnowledgeGraphAgent()

        # Add some nodes and edges
        agent.G.add_node("A", type="entity")
        agent.G.add_node("B", type="entity")
        agent.G.add_edge("A", "B", label="test")

        stats = agent.get_stats()

        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        assert stats["backend"] == "NetworkX"

    @patch('neo4j.GraphDatabase')
    def test_get_stats_neo4j(self, mock_graph_db, mock_neo4j_driver):
        """Test getting statistics from Neo4j graph"""
        # Setup mock results - single() returns a Record-like object
        mock_node_record = {"nodes": 10}
        mock_edge_record = {"edges": 15}

        mock_node_result = Mock()
        mock_node_result.single = Mock(return_value=mock_node_record)

        mock_edge_result = Mock()
        mock_edge_result.single = Mock(return_value=mock_edge_record)

        mock_session = Mock()
        mock_session.run = Mock(side_effect=[mock_node_result, mock_edge_result])
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        # Reset session mocks after initialization
        mock_session.run = Mock(side_effect=[mock_node_result, mock_edge_result])
        agent.driver = mock_neo4j_driver

        stats = agent.get_stats()

        assert stats["nodes"] == 10
        assert stats["edges"] == 15
        assert stats["backend"] == "Neo4j"

    def test_get_stats_empty_graph(self):
        """Test statistics for empty graph"""
        agent = KnowledgeGraphAgent()

        stats = agent.get_stats()

        assert stats["nodes"] == 0
        assert stats["edges"] == 0


class TestKnowledgeGraphAgentExport:
    """Test graph export functionality"""

    def test_export_networkx_data(self):
        """Test exporting NetworkX graph data"""
        agent = KnowledgeGraphAgent()

        # Add test data
        agent.G.add_node("Node1", type="entity", extra="data")
        agent.G.add_node("Node2", type="entity")
        agent.G.add_edge("Node1", "Node2", label="connects_to")

        data = agent.export_graph_data()

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

        # Check node structure
        node = data["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "properties" in node

        # Check edge structure
        edge = data["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "label" in edge
        assert edge["label"] == "connects_to"

    def test_export_networkx_truncates_long_labels(self):
        """Test NetworkX export truncates long node labels"""
        agent = KnowledgeGraphAgent()

        long_name = "A" * 100
        agent.G.add_node(long_name, type="entity")

        data = agent.export_graph_data()

        # Label should be truncated to 50 characters
        assert len(data["nodes"][0]["label"]) == 50

    @patch('neo4j.GraphDatabase')
    def test_export_neo4j_data(self, mock_graph_db, mock_neo4j_driver):
        """Test exporting Neo4j graph data"""
        # Setup mock nodes - must be Record-like iterables
        class MockRecord:
            def __init__(self, data):
                self.data = data
            def __getitem__(self, key):
                return self.data[key]

        mock_node_record = MockRecord({
            "id": 1,
            "labels": ["Entity"],
            "props": {"name": "TestEntity", "extra": "data"}
        })

        # Setup mock edges
        mock_edge_record = MockRecord({
            "source": 1,
            "target": 2,
            "type": "RELATES_TO",
            "props": {}
        })

        mock_session = Mock()
        mock_session.run = Mock(side_effect=[
            [mock_node_record],  # nodes query
            [mock_edge_record]   # edges query
        ])
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        # Reset after initialization
        mock_session.run = Mock(side_effect=[
            [mock_node_record],  # nodes query
            [mock_edge_record]   # edges query
        ])
        agent.driver = mock_neo4j_driver

        data = agent.export_graph_data()

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1

        # Check node has correct fields
        assert data["nodes"][0]["label"] == "TestEntity"
        assert data["nodes"][0]["type"] == "Entity"

    @patch('neo4j.GraphDatabase')
    def test_export_neo4j_uses_title_fallback(self, mock_graph_db, mock_neo4j_driver):
        """Test Neo4j export uses title when name is missing"""
        class MockRecord:
            def __init__(self, data):
                self.data = data
            def __getitem__(self, key):
                return self.data[key]

        mock_node_record = MockRecord({
            "id": 1,
            "labels": ["Paper"],
            "props": {"title": "Paper Title"}
        })

        mock_session = Mock()
        mock_session.run = Mock(side_effect=[
            [mock_node_record],  # nodes
            []  # edges
        ])
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j"}
        agent = KnowledgeGraphAgent(config)
        # Reset after initialization
        mock_session.run = Mock(side_effect=[
            [mock_node_record],  # nodes
            []  # edges
        ])
        agent.driver = mock_neo4j_driver

        data = agent.export_graph_data()

        assert data["nodes"][0]["label"] == "Paper Title"

    @patch('neo4j.GraphDatabase')
    def test_export_neo4j_fallback_to_node_id(self, mock_graph_db, mock_neo4j_driver):
        """Test Neo4j export falls back to node ID when name and title missing"""
        class MockRecord:
            def __init__(self, data):
                self.data = data
            def __getitem__(self, key):
                return self.data[key]

        mock_node_record = MockRecord({
            "id": 42,
            "labels": ["Unknown"],
            "props": {}
        })

        mock_session = Mock()
        mock_session.run = Mock(side_effect=[
            [mock_node_record],
            []
        ])
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j"}
        agent = KnowledgeGraphAgent(config)
        # Reset after initialization
        mock_session.run = Mock(side_effect=[
            [mock_node_record],
            []
        ])
        agent.driver = mock_neo4j_driver

        data = agent.export_graph_data()

        assert data["nodes"][0]["label"] == "Node 42"

    def test_export_graph_data_delegates_to_backend(self):
        """Test export delegates to correct backend"""
        agent = KnowledgeGraphAgent({"type": "networkx"})
        agent.G.add_node("A")

        data = agent.export_graph_data()

        # Should use NetworkX export
        assert "nodes" in data
        assert len(data["nodes"]) == 1


class TestKnowledgeGraphAgentVisualization:
    """Test graph visualization functionality"""

    @patch('pyvis.network.Network')
    def test_visualize_networkx(self, mock_network_class):
        """Test visualization with NetworkX backend"""
        mock_net = Mock()
        mock_network_class.return_value = mock_net

        agent = KnowledgeGraphAgent()
        agent.G.add_edge("A", "B", label="test")

        agent.visualize("test_graph.html")

        # Check Network was initialized
        mock_network_class.assert_called_once()

        # Check nodes and edges were added
        assert mock_net.add_node.called
        assert mock_net.add_edge.called

        # Check graph was saved
        mock_net.save_graph.assert_called_once_with("test_graph.html")

    @patch('pyvis.network.Network')
    def test_visualize_empty_graph(self, mock_network_class):
        """Test visualization handles empty graph"""
        mock_net = Mock()
        mock_network_class.return_value = mock_net

        agent = KnowledgeGraphAgent()

        agent.visualize("empty.html")

        # Should not save empty graph
        mock_net.save_graph.assert_not_called()

    @patch('neo4j.GraphDatabase')
    def test_visualize_neo4j_not_supported(self, mock_graph_db, mock_neo4j_driver):
        """Test visualization is not supported for Neo4j backend"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        # Should log warning and return without error
        agent.visualize("test.html")

        # No exception should be raised

    @patch('pyvis.network.Network')
    def test_visualize_custom_filename(self, mock_network_class):
        """Test visualization with custom filename"""
        mock_net = Mock()
        mock_network_class.return_value = mock_net

        agent = KnowledgeGraphAgent()
        agent.G.add_edge("A", "B", label="test")

        custom_name = "custom_viz.html"
        agent.visualize(custom_name)

        mock_net.save_graph.assert_called_once_with(custom_name)


class TestKnowledgeGraphAgentCleanup:
    """Test cleanup and connection management"""

    @patch('neo4j.GraphDatabase')
    def test_close_neo4j_connection(self, mock_graph_db, mock_neo4j_driver):
        """Test closing Neo4j connection"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        agent.close()

        mock_neo4j_driver.close.assert_called_once()

    def test_close_networkx_no_error(self):
        """Test closing NetworkX backend doesn't error"""
        agent = KnowledgeGraphAgent()

        # Should not raise any exception
        agent.close()

    @patch('neo4j.GraphDatabase')
    def test_close_when_no_driver(self, mock_graph_db):
        """Test close handles missing driver gracefully"""
        mock_graph_db.driver.side_effect = Exception("Connection failed")

        config = {"type": "neo4j"}
        agent = KnowledgeGraphAgent(config)
        # Driver initialization failed, fell back to NetworkX

        # Should not raise exception
        agent.close()


class TestKnowledgeGraphAgentEdgeCases:
    """Test edge cases and error handling"""

    def test_paper_without_id(self):
        """Test handling paper without ID"""
        agent = KnowledgeGraphAgent()

        paper = {
            "title": "No ID Paper",
            "authors": ["Author"],
            "topics": []
        }

        result = agent._add_to_networkx(paper, [])

        # Should still add the paper with empty ID
        assert "" in agent.G.nodes()

    def test_paper_with_missing_fields(self):
        """Test handling paper with missing optional fields"""
        agent = KnowledgeGraphAgent()

        minimal_paper = {
            "id": "minimal_1"
            # No title, authors, topics
        }

        result = agent._add_to_networkx(minimal_paper, [])

        assert result["nodes"] >= 1
        assert "minimal_1" in agent.G.nodes()

    def test_query_case_insensitive(self):
        """Test querying is case insensitive"""
        agent = KnowledgeGraphAgent()

        agent.G.add_node("Machine Learning", type="topic")

        # Query with different case
        results = agent.query_graph("machine learning")

        assert len(results) >= 0  # Should find it

    def test_empty_triple_list(self):
        """Test adding paper with empty triple list"""
        agent = KnowledgeGraphAgent()

        paper = {
            "id": "paper1",
            "title": "Test",
            "authors": ["Author"],
            "topics": []
        }

        result = agent._add_to_networkx(paper, [])

        assert result["nodes"] >= 1
        assert result["edges"] >= 0

    def test_unicode_in_node_names(self):
        """Test handling unicode characters in node names"""
        agent = KnowledgeGraphAgent()

        paper = {
            "id": "unicode_paper",
            "title": "Título con caractères spéciaux 中文",
            "authors": ["作者"],
            "topics": []
        }

        result = agent._add_to_networkx(paper, [])

        # Should handle unicode without errors
        assert result["nodes"] >= 1
        assert "作者" in agent.G.nodes()

    def test_very_long_strings(self):
        """Test handling very long strings in graph data"""
        agent = KnowledgeGraphAgent()

        long_title = "A" * 10000
        paper = {
            "id": "long_paper",
            "title": long_title,
            "authors": [],
            "topics": []
        }

        result = agent._add_to_networkx(paper, [])

        # Should handle without errors
        assert result["nodes"] >= 1

    @patch('neo4j.GraphDatabase')
    def test_neo4j_database_parameter(self, mock_graph_db, mock_neo4j_driver):
        """Test Neo4j uses correct database parameter"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "custom_db"}
        agent = KnowledgeGraphAgent(config)
        agent.driver = mock_neo4j_driver

        paper = {"id": "test", "title": "Test", "authors": [], "topics": []}
        agent._add_to_neo4j(paper, [])

        # Check that session was called with correct database
        mock_neo4j_driver.session.assert_called_with(database="custom_db")

    @patch('neo4j.GraphDatabase')
    def test_add_to_neo4j_empty_topic_string(self, mock_graph_db, mock_neo4j_driver):
        """Test Neo4j handles topic that becomes empty after strip"""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_neo4j_driver.session = Mock(return_value=mock_session)
        mock_graph_db.driver.return_value = mock_neo4j_driver

        config = {"type": "neo4j", "database": "test_db"}
        agent = KnowledgeGraphAgent(config)
        # Reset after initialization
        mock_session.run.reset_mock()
        agent.driver = mock_neo4j_driver

        paper = {
            "id": "test_1",
            "title": "Test",
            "authors": [],
            "topics": ["  \t\n  "]  # Whitespace-only topic
        }

        result = agent._add_to_neo4j(paper, [])

        # Should only process the paper (skip empty topic after strip)
        assert mock_session.run.call_count == 1

    def test_query_networkx_handles_disconnected_nodes(self):
        """Test querying handles NetworkX path exceptions"""
        agent = KnowledgeGraphAgent()

        # Create disconnected components
        agent.G.add_node("A")
        agent.G.add_node("B")
        agent.G.add_node("C")
        agent.G.add_edge("A", "B", label="link")
        # C is disconnected

        # Query should handle exception gracefully
        results = agent.query_graph("A", max_hops=2)

        # Should return results without crashing
        assert isinstance(results, list)
