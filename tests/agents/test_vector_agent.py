"""
Comprehensive tests for VectorAgent - Vector Database and Semantic Search
Tests both FAISS (development) and Qdrant (production) backends
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np

from agents.vector_agent import VectorAgent


class TestVectorAgentInitialization:
    """Test VectorAgent initialization and configuration"""

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialization_default_faiss(self, mock_transformer, mock_genai):
        """Test default initialization with FAISS backend"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        agent = VectorAgent()

        assert agent.db_type == "faiss"
        assert agent.collection_name == "research_papers"
        assert agent.embedding_model_name == "all-MiniLM-L6-v2"
        assert agent.dimension == 384
        assert agent.embedding_model == mock_model
        assert hasattr(agent, 'index')
        assert hasattr(agent, 'chunks')
        assert hasattr(agent, 'chunk_texts')
        assert agent.chunks == []
        assert agent.chunk_texts == []

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialization_custom_config(self, mock_transformer, mock_genai):
        """Test initialization with custom configuration"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        config = {
            "type": "faiss",
            "collection_name": "test_collection",
            "embedding_model": "custom-model",
            "dimension": 512,
            "chunk_size": 500,
            "chunk_overlap": 100
        }

        agent = VectorAgent(config)

        assert agent.db_type == "faiss"
        assert agent.collection_name == "test_collection"
        assert agent.embedding_model_name == "custom-model"
        assert agent.dimension == 512
        assert agent.config["chunk_size"] == 500
        assert agent.config["chunk_overlap"] == 100

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialization_qdrant_backend(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test initialization with Qdrant backend"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        # Setup Qdrant mock
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.create_collection = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        config = {
            "type": "qdrant",
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_papers"
        }

        agent = VectorAgent(config)

        assert agent.db_type == "qdrant"
        assert hasattr(agent, 'client')
        mock_qdrant_client.assert_called_once()

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer', side_effect=ImportError)
    def test_initialization_without_sentence_transformers(self, mock_transformer, mock_genai):
        """Test initialization falls back when sentence-transformers not installed"""
        mock_genai.configure = Mock()
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        agent = VectorAgent()

        assert agent.embedding_model is None
        assert agent.model == mock_model

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialization_without_google_api_key(self, mock_transformer, mock_genai):
        """Test initialization without GOOGLE_API_KEY"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        with patch.dict('os.environ', {}, clear=True):
            agent = VectorAgent()

            assert agent.model is None


class TestVectorAgentQdrantBackend:
    """Test Qdrant-specific functionality"""

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_init_qdrant_creates_collection(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test Qdrant initialization creates collection if not exists"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        # Mock collection doesn't exist
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.create_collection = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        mock_client_instance.create_collection.assert_called_once()

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_init_qdrant_uses_existing_collection(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test Qdrant initialization uses existing collection"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        # Mock collection exists
        mock_client_instance = Mock()
        existing_collection = Mock()
        existing_collection.name = "research_papers"
        mock_client_instance.get_collections.return_value = Mock(collections=[existing_collection])
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        mock_client_instance.create_collection.assert_not_called()

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_init_qdrant_with_grpc_and_https(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test Qdrant initialization with gRPC and HTTPS options"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.return_value = mock_client_instance

        config = {
            "type": "qdrant",
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "prefer_grpc": "true",
            "https": "true",
            "api_key": "test-key"
        }
        agent = VectorAgent(config)

        mock_qdrant_client.assert_called_once_with(
            host="localhost",
            port=6333,
            grpc_port=6334,
            api_key="test-key",
            prefer_grpc=True,
            https=True
        )

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_init_qdrant_fallback_to_faiss_on_error(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test Qdrant falls back to FAISS on connection error"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_qdrant_client.side_effect = Exception("Connection failed")

        config = {"type": "qdrant", "host": "localhost", "port": 6333}

        with patch('faiss.IndexFlatL2'):
            agent = VectorAgent(config)

            assert agent.db_type == "faiss"


class TestVectorAgentFAISSBackend:
    """Test FAISS-specific functionality"""

    @patch('agents.vector_agent.genai')
    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_init_faiss_creates_index(self, mock_transformer, mock_faiss, mock_genai):
        """Test FAISS initialization creates index"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        mock_faiss.assert_called_once_with(384)
        assert agent.index == mock_index

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_init_faiss_without_library(self, mock_transformer, mock_genai):
        """Test FAISS initialization without FAISS library"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        with patch.dict('sys.modules', {'faiss': None}):
            agent = VectorAgent()

            assert agent.index is None
            assert agent.chunks == []
            assert agent.chunk_texts == []


class TestVectorAgentEmbeddings:
    """Test embedding generation"""

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_embedding_with_sentence_transformer(self, mock_transformer, mock_genai):
        """Test embedding generation with SentenceTransformer"""
        mock_model = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        mock_transformer.return_value = mock_model

        agent = VectorAgent()
        embedding = agent._get_embedding("test text")

        mock_model.encode.assert_called_once_with("test text", device='cpu')
        assert embedding == mock_embedding.tolist()

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer', side_effect=ImportError)
    def test_get_embedding_with_gemini_fallback(self, mock_transformer, mock_genai):
        """Test embedding generation falls back to Gemini"""
        mock_genai.configure = Mock()
        mock_genai.GenerativeModel = Mock()
        mock_genai.embed_content.return_value = {"embedding": [0.1, 0.2, 0.3]}

        agent = VectorAgent()
        embedding = agent._get_embedding("test text")

        mock_genai.embed_content.assert_called_once()
        assert embedding == [0.1, 0.2, 0.3]

    @patch('numpy.random.rand')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer', side_effect=ImportError)
    def test_get_embedding_random_fallback(self, mock_transformer, mock_genai, mock_rand):
        """Test embedding generation ultimate fallback to random"""
        mock_genai.configure = Mock()
        mock_genai.GenerativeModel = Mock()
        mock_genai.embed_content.side_effect = Exception("API error")
        mock_rand.return_value = np.array([0.5] * 384)

        agent = VectorAgent()
        embedding = agent._get_embedding("test text")

        mock_rand.assert_called_once_with(384)
        assert len(embedding) == 384


class TestVectorAgentTextChunking:
    """Test text chunking functionality"""

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_chunk_text_basic(self, mock_transformer, mock_genai):
        """Test basic text chunking"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        agent = VectorAgent()
        text = "word " * 100  # 100 words
        chunks = agent._chunk_text(text, chunk_size=40, overlap=10)

        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_chunk_text_with_overlap(self, mock_transformer, mock_genai):
        """Test text chunking with overlap"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        agent = VectorAgent()
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = agent._chunk_text(text, chunk_size=20, overlap=5)

        # Verify overlap exists (some words appear in multiple chunks)
        assert len(chunks) >= 2

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_chunk_text_short_text(self, mock_transformer, mock_genai):
        """Test chunking short text returns single chunk"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        agent = VectorAgent()
        text = "short text"
        chunks = agent._chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text


class TestVectorAgentDocumentProcessing:
    """Test document processing and storage"""

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_process_papers_faiss(self, mock_transformer, mock_genai, mock_faiss, sample_research_paper):
        """Test processing papers with FAISS backend"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 0
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        papers = [sample_research_paper]

        result = agent.process_papers(papers)

        assert result["documents_added"] == 1
        assert result["chunks_added"] > 0
        assert len(agent.chunks) > 0
        assert len(agent.chunk_texts) > 0

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_process_papers_qdrant(self, mock_transformer, mock_qdrant_client, mock_genai, sample_research_paper):
        """Test processing papers with Qdrant backend"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.get_collection.return_value = Mock(points_count=0)
        mock_client_instance.upsert = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)
        papers = [sample_research_paper]

        result = agent.process_papers(papers)

        assert result["documents_added"] == 1
        assert result["chunks_added"] > 0
        mock_client_instance.upsert.assert_called_once()

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_process_papers_multiple_papers(self, mock_transformer, mock_genai, mock_faiss):
        """Test processing multiple papers"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 0
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        papers = [
            {
                "id": "paper1",
                "title": "Paper 1",
                "abstract": "Abstract 1",
                "source": "arxiv",
                "url": "http://test1.com"
            },
            {
                "id": "paper2",
                "title": "Paper 2",
                "abstract": "Abstract 2",
                "source": "pubmed",
                "url": "http://test2.com"
            }
        ]

        result = agent.process_papers(papers)

        assert result["documents_added"] == 2
        assert result["chunks_added"] > 0

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_process_papers_with_custom_chunking(self, mock_transformer, mock_genai, mock_faiss):
        """Test processing papers with custom chunk size"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 0
        mock_faiss.return_value = mock_index

        config = {"chunk_size": 200, "chunk_overlap": 25}
        agent = VectorAgent(config)

        papers = [{
            "id": "paper1",
            "title": "Test",
            "abstract": " ".join(["word"] * 500),  # Long abstract
            "source": "test",
            "url": "http://test.com"
        }]

        result = agent.process_papers(papers)

        assert result["chunks_added"] > 1


class TestVectorAgentSearch:
    """Test semantic search functionality"""

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_search_faiss(self, mock_transformer, mock_genai, mock_faiss):
        """Test semantic search with FAISS"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),  # distances
            np.array([[0, 1, 2]])  # indices
        )
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        agent.chunks = [
            {"text": "chunk1", "paper_id": "1", "title": "Paper 1", "source": "arxiv", "url": "http://1.com"},
            {"text": "chunk2", "paper_id": "2", "title": "Paper 2", "source": "arxiv", "url": "http://2.com"},
            {"text": "chunk3", "paper_id": "3", "title": "Paper 3", "source": "arxiv", "url": "http://3.com"}
        ]

        results = agent.search("test query", top_k=3)

        assert len(results) == 3
        assert all("score" in r for r in results)
        assert all("text" in r for r in results)

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_search_qdrant(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test semantic search with Qdrant"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        # Mock search results
        mock_result1 = Mock()
        mock_result1.payload = {"text": "result1", "paper_id": "1", "title": "Paper 1"}
        mock_result1.score = 0.95

        mock_result2 = Mock()
        mock_result2.payload = {"text": "result2", "paper_id": "2", "title": "Paper 2"}
        mock_result2.score = 0.85

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.search.return_value = [mock_result1, mock_result2]
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        results = agent.search("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert results[1]["score"] == 0.85
        mock_client_instance.search.assert_called_once()

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_search_top_k_limit(self, mock_transformer, mock_genai, mock_faiss):
        """Test search respects top_k limit"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[0, 1]])
        )
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        agent.chunks = [
            {"text": "chunk1", "paper_id": "1", "title": "Paper 1", "source": "test", "url": "http://1.com"},
            {"text": "chunk2", "paper_id": "2", "title": "Paper 2", "source": "test", "url": "http://2.com"}
        ]

        results = agent.search("test query", top_k=2)

        assert len(results) <= 2

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_search_fallback_simple_text_search(self, mock_transformer, mock_genai):
        """Test search falls back to simple text search when FAISS unavailable"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        with patch.dict('sys.modules', {'faiss': None}):
            agent = VectorAgent()
            agent.chunks = [
                {"text": "machine learning algorithms", "paper_id": "1", "title": "ML Paper", "source": "test", "url": "http://1.com"},
                {"text": "deep neural networks", "paper_id": "2", "title": "DL Paper", "source": "test", "url": "http://2.com"},
                {"text": "quantum computing", "paper_id": "3", "title": "QC Paper", "source": "test", "url": "http://3.com"}
            ]

            results = agent.search("machine learning", top_k=2)

            assert len(results) > 0
            assert results[0]["text"] == "machine learning algorithms"
            assert "score" in results[0]


class TestVectorAgentStatistics:
    """Test statistics and metadata extraction"""

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_stats_faiss(self, mock_transformer, mock_genai, mock_faiss):
        """Test getting statistics for FAISS backend"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        agent.chunks = [{"text": "chunk1"}, {"text": "chunk2"}, {"text": "chunk3"}]

        stats = agent.get_stats()

        assert stats["chunks"] == 3
        assert stats["backend"] == "FAISS"
        assert stats["dimension"] == 384

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_stats_qdrant(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test getting statistics for Qdrant backend"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_collection_info = Mock()
        mock_collection_info.points_count = 42
        mock_client_instance.get_collection.return_value = mock_collection_info
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        stats = agent.get_stats()

        assert stats["chunks"] == 42
        assert stats["backend"] == "Qdrant"
        assert stats["dimension"] == 384

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_embedding_statistics_empty(self, mock_transformer, mock_genai, mock_faiss):
        """Test embedding statistics with empty database"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 0
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        stats = agent.get_embedding_statistics()

        assert stats["total_embeddings"] == 0
        assert stats["dimension"] == 384
        assert "error" in stats

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_embedding_statistics_with_data(self, mock_transformer, mock_genai, mock_faiss):
        """Test embedding statistics with data"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 3
        mock_index.reconstruct.side_effect = [
            np.array([0.1] * 384),
            np.array([0.2] * 384),
            np.array([0.3] * 384)
        ]
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        agent.chunks = [
            {"source": "arxiv"},
            {"source": "arxiv"},
            {"source": "pubmed"}
        ]

        stats = agent.get_embedding_statistics()

        assert stats["total_embeddings"] == 3
        assert stats["dimension"] == 384
        assert "mean_norm" in stats
        assert "std_norm" in stats
        assert stats["sources"]["arxiv"] == 2
        assert stats["sources"]["pubmed"] == 1
        assert stats["db_type"] == "faiss"

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_count_sources(self, mock_transformer, mock_genai, mock_faiss):
        """Test counting embeddings by source"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        metadata = [
            {"source": "arxiv"},
            {"source": "arxiv"},
            {"source": "pubmed"},
            {"source": "semanticscholar"},
            {"source": "arxiv"},
            {}  # No source
        ]

        sources = agent._count_sources(metadata)

        assert sources["arxiv"] == 3
        assert sources["pubmed"] == 1
        assert sources["semanticscholar"] == 1
        assert sources["Unknown"] == 1


class TestVectorAgentEmbeddingExtraction:
    """Test embedding extraction methods"""

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_faiss_embeddings(self, mock_transformer, mock_genai, mock_faiss):
        """Test extracting embeddings from FAISS"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 2
        mock_index.reconstruct.side_effect = [
            np.array([0.1] * 384),
            np.array([0.2] * 384)
        ]
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        agent.chunks = [
            {"text": "chunk1", "paper_id": "1"},
            {"text": "chunk2", "paper_id": "2"}
        ]

        embeddings, metadata = agent._get_faiss_embeddings()

        assert embeddings.shape == (2, 384)
        assert len(metadata) == 2
        assert metadata[0]["paper_id"] == "1"

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_faiss_embeddings_empty(self, mock_transformer, mock_genai, mock_faiss):
        """Test extracting embeddings from empty FAISS index"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 0
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        embeddings, metadata = agent._get_faiss_embeddings()

        assert len(embeddings) == 0
        assert len(metadata) == 0

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_qdrant_embeddings(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test extracting embeddings from Qdrant"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        # Mock Qdrant points
        mock_point1 = Mock()
        mock_point1.vector = [0.1] * 384
        mock_point1.payload = {"text": "chunk1", "paper_id": "1"}

        mock_point2 = Mock()
        mock_point2.vector = [0.2] * 384
        mock_point2.payload = {"text": "chunk2", "paper_id": "2"}

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.scroll.return_value = ([mock_point1, mock_point2], None)
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        embeddings, metadata = agent._get_qdrant_embeddings()

        assert embeddings.shape == (2, 384)
        assert len(metadata) == 2
        assert metadata[0]["paper_id"] == "1"
        mock_client_instance.scroll.assert_called_once()

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_qdrant_embeddings_empty_collection(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test Qdrant embedding extraction with empty collection"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.scroll.return_value = ([], None)  # Empty collection
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        embeddings, metadata = agent._get_qdrant_embeddings()

        assert len(embeddings) == 0
        assert len(metadata) == 0

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_qdrant_embeddings_error(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test Qdrant embedding extraction handles errors"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.scroll.side_effect = Exception("Connection error")
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        embeddings, metadata = agent._get_qdrant_embeddings()

        assert len(embeddings) == 0
        assert len(metadata) == 0


class TestVectorAgentFAISSPersistence:
    """Test FAISS index save/load functionality"""

    @patch('pathlib.Path.mkdir')
    @patch('faiss.write_index')
    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_save_faiss_index(self, mock_transformer, mock_genai, mock_faiss,
                             mock_write_index, mock_mkdir):
        """Test saving FAISS index to disk"""
        from unittest.mock import mock_open

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 2
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        agent.chunks = [{"text": "chunk1"}, {"text": "chunk2"}]
        agent.chunk_texts = ["chunk1", "chunk2"]

        with patch('builtins.open', mock_open()):
            with patch('json.dump'):
                result = agent.save_faiss_index("/tmp/test_index")

                assert result is True
                mock_write_index.assert_called_once()

    @patch('pathlib.Path.mkdir')
    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_save_faiss_index_empty(self, mock_transformer, mock_genai, mock_faiss, mock_mkdir):
        """Test saving empty FAISS index"""
        from unittest.mock import mock_open

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 0
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        with patch('builtins.open', mock_open()):
            with patch('json.dump'):
                result = agent.save_faiss_index("/tmp/test_index")

                # Should still return True but skip FAISS write
                assert result is True

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_save_faiss_index_wrong_backend(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test saving FAISS index with wrong backend returns False"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        result = agent.save_faiss_index("/tmp/test_index")

        assert result is False

    @patch('pathlib.Path.exists')
    @patch('faiss.read_index')
    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_load_faiss_index(self, mock_transformer, mock_genai, mock_faiss_init,
                             mock_read_index, mock_exists):
        """Test loading FAISS index from disk"""
        from unittest.mock import mock_open
        import json

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 2
        mock_faiss_init.return_value = Mock()
        mock_read_index.return_value = mock_index

        agent = VectorAgent()

        mock_metadata = {
            "chunks": [{"text": "chunk1"}, {"text": "chunk2"}],
            "chunk_texts": ["chunk1", "chunk2"],
            "dimension": 384
        }

        mock_exists.return_value = True

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_metadata))):
            with patch('json.load', return_value=mock_metadata):
                result = agent.load_faiss_index("/tmp/test_index")

                assert result is True
                assert agent.index == mock_index
                assert len(agent.chunks) == 2

    @patch('pathlib.Path.exists')
    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_load_faiss_index_no_file(self, mock_transformer, mock_genai, mock_faiss, mock_exists):
        """Test loading FAISS index when file doesn't exist"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        mock_exists.return_value = False

        result = agent.load_faiss_index("/tmp/nonexistent")

        assert result is False

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_load_faiss_index_wrong_backend(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test loading FAISS index with wrong backend returns False"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        result = agent.load_faiss_index("/tmp/test_index")

        assert result is False

    @patch('faiss.read_index')
    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_load_faiss_index_without_metadata(self, mock_transformer, mock_genai,
                                               mock_faiss_init, mock_read_index):
        """Test loading FAISS index when metadata file is missing"""
        from unittest.mock import mock_open, PropertyMock
        from pathlib import Path

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 2
        mock_faiss_init.return_value = Mock()
        mock_read_index.return_value = mock_index

        agent = VectorAgent()

        # Mock Path to return different exists() values for index vs metadata files
        with patch('pathlib.Path') as mock_path:
            def path_side_effect(path_str):
                mock_file = Mock(spec=Path)
                # Index file exists, metadata doesn't
                mock_file.exists.return_value = str(path_str).endswith('.index')
                mock_file.__str__ = Mock(return_value=str(path_str))
                mock_file.parent.mkdir = Mock()
                return mock_file

            mock_path.side_effect = path_side_effect

            with patch('builtins.open', mock_open()):
                result = agent.load_faiss_index("/tmp/test_index")

                assert result is True
                assert agent.index == mock_index
                assert agent.chunks == []
                assert agent.chunk_texts == []


class TestVectorAgentClose:
    """Test cleanup and connection closing"""

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_close_qdrant(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test closing Qdrant connection"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.close = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        agent.close()

        mock_client_instance.close.assert_called_once()

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_close_faiss(self, mock_transformer, mock_genai, mock_faiss):
        """Test closing FAISS (no-op)"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        # Should not raise an error
        agent.close()


class TestVectorAgentAddToBackends:
    """Test internal methods for adding to backends"""

    @patch('agents.vector_agent.genai')
    @patch('qdrant_client.QdrantClient')
    @patch('sentence_transformers.SentenceTransformer')
    def test_add_to_qdrant(self, mock_transformer, mock_qdrant_client, mock_genai):
        """Test adding chunks to Qdrant"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.get_collection.return_value = Mock(points_count=0)
        mock_client_instance.upsert = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        agent = VectorAgent(config)

        chunks = [
            {"text": "chunk1", "paper_id": "1", "title": "Paper 1", "source": "arxiv", "url": "http://1.com"},
            {"text": "chunk2", "paper_id": "2", "title": "Paper 2", "source": "arxiv", "url": "http://2.com"}
        ]

        result = agent._add_to_qdrant(chunks)

        assert result["chunks_added"] == 2
        mock_client_instance.upsert.assert_called_once()

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_add_to_faiss_with_index(self, mock_transformer, mock_genai, mock_faiss):
        """Test adding chunks to FAISS with index"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        chunks = [
            {"text": "chunk1", "paper_id": "1", "title": "Paper 1", "source": "arxiv", "url": "http://1.com"},
            {"text": "chunk2", "paper_id": "2", "title": "Paper 2", "source": "arxiv", "url": "http://2.com"}
        ]

        result = agent._add_to_faiss(chunks)

        assert result["chunks_added"] == 2
        mock_index.add.assert_called_once()
        assert len(agent.chunks) == 2
        assert len(agent.chunk_texts) == 2

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_add_to_faiss_single_chunk(self, mock_transformer, mock_genai, mock_faiss):
        """Test adding single chunk to FAISS (edge case for reshape)"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        chunks = [
            {"text": "single chunk", "paper_id": "1", "title": "Paper 1", "source": "arxiv", "url": "http://1.com"}
        ]

        result = agent._add_to_faiss(chunks)

        assert result["chunks_added"] == 1
        mock_index.add.assert_called_once()

    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_add_to_faiss_without_index(self, mock_transformer, mock_genai):
        """Test adding chunks to FAISS without index (in-memory only)"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        with patch.dict('sys.modules', {'faiss': None}):
            agent = VectorAgent()

            chunks = [
                {"text": "chunk1", "paper_id": "1", "title": "Paper 1", "source": "arxiv", "url": "http://1.com"}
            ]

            result = agent._add_to_faiss(chunks)

            assert result["chunks_added"] == 1
            assert len(agent.chunks) == 1
            assert len(agent.chunk_texts) == 1


class TestVectorAgentEdgeCases:
    """Test edge cases and error handling"""

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_process_empty_papers_list(self, mock_transformer, mock_genai, mock_faiss):
        """Test processing empty papers list"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        result = agent.process_papers([])

        assert result["documents_added"] == 0
        assert result["chunks_added"] == 0

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_search_empty_index(self, mock_transformer, mock_genai, mock_faiss):
        """Test search on empty index"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.search.return_value = (np.array([[]]), np.array([[]]))
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        results = agent.search("test query")

        assert len(results) == 0

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_process_papers_missing_fields(self, mock_transformer, mock_genai, mock_faiss):
        """Test processing papers with missing fields"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 0
        mock_faiss.return_value = mock_index

        agent = VectorAgent()

        # Paper with missing fields
        papers = [{"id": "1"}]  # Missing title, abstract, etc.

        result = agent.process_papers(papers)

        # Should handle gracefully
        assert result["documents_added"] == 1
        assert result["chunks_added"] >= 0

    @patch('faiss.IndexFlatL2')
    @patch('agents.vector_agent.genai')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_all_embeddings_and_metadata(self, mock_transformer, mock_genai, mock_faiss):
        """Test get_all_embeddings_and_metadata wrapper method"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.ntotal = 2
        mock_index.reconstruct.side_effect = [
            np.array([0.1] * 384),
            np.array([0.2] * 384)
        ]
        mock_faiss.return_value = mock_index

        agent = VectorAgent()
        agent.chunks = [{"text": "chunk1"}, {"text": "chunk2"}]

        embeddings, metadata = agent.get_all_embeddings_and_metadata()

        assert embeddings.shape == (2, 384)
        assert len(metadata) == 2
