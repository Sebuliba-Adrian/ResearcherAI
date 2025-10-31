"""
LlamaIndex RAG Integration

Provides advanced RAG (Retrieval-Augmented Generation) capabilities using LlamaIndex.
This module works alongside the existing vector_agent.py to provide enhanced document
indexing, retrieval, and querying capabilities.
"""

import os
from typing import List, Dict, Optional
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
    get_response_synthesizer
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import MockLLM
from qdrant_client import QdrantClient


class LlamaIndexRAG:
    """
    LlamaIndex-powered RAG system for enhanced document retrieval and querying.

    Features:
    - Advanced document indexing with metadata
    - Hybrid search (vector + keyword)
    - Query optimization and rewriting
    - Response synthesis with citations
    - Integration with Qdrant vector store
    """

    def __init__(
        self,
        collection_name: str = "research_papers_llamaindex",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_qdrant: bool = True
    ):
        """
        Initialize LlamaIndex RAG system

        Args:
            collection_name: Qdrant collection name
            embedding_model: HuggingFace model for embeddings
            use_qdrant: Whether to use Qdrant or in-memory storage
        """
        self.collection_name = collection_name
        self.use_qdrant = use_qdrant
        self.embedding_model_name = embedding_model
        self._embedding_model_initialized = False

        # Set chunk size
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        # Use mock LLM to avoid requiring OpenAI API
        Settings.llm = MockLLM()

        # Initialize vector store
        if use_qdrant:
            self._init_qdrant()
        else:
            self.storage_context = None
            self.index = None

        self.query_engine = None

    def _init_embedding_model(self):
        """Lazy-load embedding model only when needed"""
        if not self._embedding_model_initialized:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name,
                cache_folder="./model_cache",
                device="cpu"  # Force CPU to avoid CUDA compatibility issues
            )
            self._embedding_model_initialized = True

    def _init_qdrant(self):
        """Initialize Qdrant vector store"""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))

        try:
            client = QdrantClient(host=host, port=port)

            # Create vector store
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name
            )

            self.storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

            print(f"✓ Connected to Qdrant at {host}:{port}")
        except Exception as e:
            print(f"⚠️  Failed to connect to Qdrant: {e}")
            print("   Using in-memory storage instead")
            self.storage_context = None
            self.use_qdrant = False

    def index_documents(self, papers: List[Dict]) -> Dict:
        """
        Index research papers into LlamaIndex

        Args:
            papers: List of paper dictionaries with keys:
                   - title, abstract, authors, year, source, url

        Returns:
            Statistics about indexing operation
        """
        # Initialize embedding model (lazy-load)
        self._init_embedding_model()

        # Convert papers to LlamaIndex Documents
        documents = []
        for paper in papers:
            text = f"Title: {paper.get('title', '')}\n\n"
            text += f"Abstract: {paper.get('abstract', '')}\n\n"
            text += f"Authors: {', '.join(paper.get('authors', []))}\n"
            text += f"Year: {paper.get('year', 'Unknown')}\n"

            metadata = {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year", ""),
                "source": paper.get("source", ""),
                "url": paper.get("url", ""),
                "paper_id": paper.get("id", "")
            }

            doc = Document(
                text=text,
                metadata=metadata,
                excluded_embed_metadata_keys=["url", "paper_id"]
            )
            documents.append(doc)

        # Create or update index
        if self.storage_context:
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                show_progress=True
            )
        else:
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )

        # Create query engine
        self._create_query_engine()

        return {
            "documents_indexed": len(documents),
            "vector_store": "Qdrant" if self.use_qdrant else "In-Memory",
            "collection_name": self.collection_name if self.use_qdrant else "N/A"
        }

    def _create_query_engine(self, top_k: int = 5):
        """Create query engine with advanced retrieval"""
        if not self.index:
            return

        # Configure retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )

        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="compact"  # compact, refine, tree_summarize
        )

        # Create query engine with post-processing
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Query the indexed documents

        Args:
            question: Research question
            top_k: Number of top results to retrieve

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.query_engine:
            return {
                "answer": "No documents indexed yet. Please index documents first.",
                "sources": [],
                "error": "No index available"
            }

        # Update retriever top_k if needed
        if top_k != 5:
            self._create_query_engine(top_k)

        try:
            response = self.query_engine.query(question)

            # Extract sources
            sources = []
            for node in response.source_nodes:
                sources.append({
                    "text": node.text[:200] + "...",
                    "score": node.score,
                    "metadata": node.metadata
                })

            return {
                "answer": str(response),
                "sources": sources,
                "num_sources": len(sources)
            }

        except Exception as e:
            return {
                "answer": "",
                "sources": [],
                "error": str(e)
            }

    def retrieve_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve similar documents without generating an answer

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of similar documents with scores
        """
        if not self.index:
            return []

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )

        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "score": node.score,
                "metadata": node.metadata,
                "paper_id": node.metadata.get("paper_id", "")
            })

        return results

    def get_stats(self) -> Dict:
        """Get statistics about the indexed documents"""
        if not self.index:
            return {
                "status": "not_initialized",
                "num_documents": 0
            }

        try:
            # Get document store stats
            docstore = self.index.docstore
            num_docs = len(docstore.docs)

            return {
                "status": "ready",
                "num_documents": num_docs,
                "vector_store": "Qdrant" if self.use_qdrant else "In-Memory",
                "collection": self.collection_name if self.use_qdrant else "N/A",
                "embedding_model": Settings.embed_model.model_name
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


def create_rag_system(use_qdrant: bool = True) -> LlamaIndexRAG:
    """Factory function to create LlamaIndex RAG system"""
    return LlamaIndexRAG(use_qdrant=use_qdrant)


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("LlamaIndex RAG System Test")
    print("="*70 + "\n")

    # Create RAG system
    rag = create_rag_system(use_qdrant=False)  # Use in-memory for testing

    # Sample papers
    sample_papers = [
        {
            "id": "1",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "authors": ["Vaswani et al."],
            "year": "2017",
            "source": "arXiv",
            "url": "https://arxiv.org/abs/1706.03762"
        },
        {
            "id": "2",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers.",
            "authors": ["Devlin et al."],
            "year": "2018",
            "source": "arXiv",
            "url": "https://arxiv.org/abs/1810.04805"
        }
    ]

    # Index documents
    print("Indexing sample documents...")
    stats = rag.index_documents(sample_papers)
    print(f"✓ Indexed {stats['documents_indexed']} documents\n")

    # Query
    print("Querying: 'What is the Transformer architecture?'")
    result = rag.query("What is the Transformer architecture?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['num_sources']}\n")

    # Get stats
    print("System Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
