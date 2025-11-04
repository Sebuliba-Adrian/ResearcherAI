"""
Pytest fixtures for agent testing
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import os


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ["GOOGLE_API_KEY"] = "test-api-key-123"
    os.environ["USE_NEO4J"] = "false"
    os.environ["USE_QDRANT"] = "false"
    os.environ["USE_KAFKA"] = "false"
    yield
    # Cleanup
    pass


@pytest.fixture
def mock_gemini_model():
    """Mock Google Gemini model"""
    model = Mock()
    response = Mock()
    response.text = "This is a mock Gemini response"
    response.parts = [Mock(text="Mock response part")]
    model.generate_content = Mock(return_value=response)
    model.count_tokens = Mock(return_value=Mock(total_tokens=100))
    return model


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j database driver"""
    driver = Mock()
    session = Mock()
    result = Mock()
    result.single = Mock(return_value={"count": 10})
    result.data = Mock(return_value=[{"node": {"id": "1", "title": "Test"}}])
    session.run = Mock(return_value=result)
    session.close = Mock()
    driver.session = Mock(return_value=session)
    driver.close = Mock()
    return driver


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant vector database client"""
    client = Mock()
    client.get_collections = Mock(return_value=Mock(collections=[]))
    client.create_collection = Mock()
    client.upsert = Mock()
    client.search = Mock(return_value=[
        Mock(id="1", score=0.95, payload={"text": "Test result"})
    ])
    client.delete_collection = Mock()
    return client


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer"""
    producer = Mock()
    producer.send = Mock(return_value=Mock())
    producer.flush = Mock()
    producer.close = Mock()
    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer"""
    consumer = Mock()
    consumer.subscribe = Mock()
    consumer.poll = Mock(return_value={})
    consumer.close = Mock()
    return consumer


@pytest.fixture
def mock_arxiv_search():
    """Mock arXiv search results"""
    result = Mock()
    result.entry_id = "http://arxiv.org/abs/2301.12345"
    result.title = "Test Paper Title"
    result.summary = "This is a test paper abstract"
    result.authors = [Mock(name="Test Author")]
    result.published = Mock(year=2023, month=1, day=1)
    result.pdf_url = "http://arxiv.org/pdf/2301.12345"
    return [result]


@pytest.fixture
def mock_semantic_scholar_api():
    """Mock Semantic Scholar API"""
    api = Mock()
    api.get_paper = Mock(return_value={
        "paperId": "test123",
        "title": "Test Paper",
        "abstract": "Test abstract",
        "authors": [{"name": "Author"}],
        "year": 2023
    })
    api.search_paper = Mock(return_value=[{
        "paperId": "test123",
        "title": "Test Paper"
    }])
    return api


@pytest.fixture
def sample_research_paper():
    """Sample research paper data"""
    return {
        "id": "test_paper_123",
        "title": "Test Research Paper",
        "abstract": "This is a test abstract about machine learning",
        "authors": ["Test Author 1", "Test Author 2"],
        "year": 2023,
        "source": "arXiv",
        "url": "http://arxiv.org/abs/2301.12345"
    }


@pytest.fixture
def sample_query():
    """Sample research query"""
    return "machine learning transformers attention mechanism"


@pytest.fixture
def mock_embeddings():
    """Mock embedding vectors"""
    return [0.1] * 768  # 768-dimensional vector


@pytest.fixture
def mock_graph_data():
    """Mock knowledge graph data"""
    return {
        "nodes": [
            {"id": "1", "label": "Paper", "title": "Test Paper 1"},
            {"id": "2", "label": "Author", "name": "Test Author"},
            {"id": "3", "label": "Concept", "name": "Machine Learning"}
        ],
        "relationships": [
            {"source": "1", "target": "2", "type": "AUTHORED_BY"},
            {"source": "1", "target": "3", "type": "ABOUT"}
        ]
    }
