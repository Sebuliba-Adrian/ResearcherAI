"""
Pytest configuration and fixtures
"""
import os
import pytest
from unittest.mock import Mock, MagicMock

# Set test environment variables
os.environ.setdefault("GOOGLE_API_KEY", "test-api-key")
os.environ.setdefault("USE_NEO4J", "false")
os.environ.setdefault("USE_QDRANT", "false")
os.environ.setdefault("USE_KAFKA", "false")


@pytest.fixture
def mock_gemini_model():
    """Mock Gemini model"""
    model = Mock()
    model.generate_content = Mock(return_value=Mock(text="Mock response"))
    return model


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver"""
    driver = Mock()
    session = Mock()
    driver.session = Mock(return_value=session)
    return driver


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client"""
    client = Mock()
    client.search = Mock(return_value=[])
    client.upsert = Mock(return_value=True)
    return client


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer"""
    producer = Mock()
    producer.send = Mock(return_value=Mock(get=Mock(return_value=True)))
    return producer


@pytest.fixture
def sample_paper_data():
    """Sample paper data for testing"""
    return {
        "title": "Test Paper",
        "authors": ["Author 1", "Author 2"],
        "abstract": "This is a test abstract",
        "url": "https://arxiv.org/abs/1234.5678",
        "published": "2024-01-01"
    }


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "machine learning transformers"
