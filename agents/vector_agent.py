"""
VectorAgent - Vector Database and Semantic Search
================================================

Supports both Qdrant (production) and FAISS (development)
"""

import os
import logging
from typing import List, Dict, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)


class VectorAgent:
    """Vector database management and semantic search with dual backend support"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with config

        config = {
            "type": "qdrant" or "faiss",
            "host": "qdrant",  # for qdrant
            "port": 6333,
            "collection_name": "research_papers",
            "embedding_model": "all-MiniLM-L6-v2",
            "dimension": 384
        }
        """
        self.config = config or {"type": "faiss"}
        self.db_type = self.config.get("type", "faiss")
        self.collection_name = self.config.get("collection_name", "research_papers")
        self.embedding_model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.dimension = self.config.get("dimension", 384)

        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding model loaded: {self.embedding_model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed - using Gemini for embeddings")
            self.embedding_model = None

        # Initialize Gemini for search (optional)
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
            logger.info("Gemini configured for semantic search")
        else:
            self.model = None

        # Initialize backend
        if self.db_type == "qdrant":
            self._init_qdrant()
        else:
            self._init_faiss()

        logger.info(f"VectorAgent initialized with {self.db_type} backend")

    def _init_qdrant(self):
        """Initialize Qdrant connection"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            host = self.config.get("host", "localhost")
            port = self.config.get("port", 6333)

            self.client = QdrantClient(host=host, port=port)

            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            logger.info("Falling back to FAISS")
            self.db_type = "faiss"
            self._init_faiss()

    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            import faiss
            import numpy as np

            self.index = faiss.IndexFlatL2(self.dimension)
            self.chunks = []  # Store chunk metadata
            self.chunk_texts = []  # Store chunk texts
            logger.info(f"FAISS index initialized (dimension={self.dimension})")
        except ImportError:
            logger.warning("FAISS not installed - using in-memory search only")
            self.index = None
            self.chunks = []
            self.chunk_texts = []

    def process_papers(self, papers: List[Dict]) -> Dict:
        """Process papers and add to vector database"""
        logger.info(f"Processing {len(papers)} papers for vector search...")

        chunk_size = self.config.get("chunk_size", 400)
        chunk_overlap = self.config.get("chunk_overlap", 50)

        all_chunks = []

        for paper in papers:
            # Create chunks from paper content
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)

            for chunk_text in chunks:
                chunk_meta = {
                    "text": chunk_text,
                    "paper_id": paper.get("id", ""),
                    "title": paper.get("title", ""),
                    "source": paper.get("source", ""),
                    "url": paper.get("url", "")
                }
                all_chunks.append(chunk_meta)

        # Add to database
        if self.db_type == "qdrant":
            result = self._add_to_qdrant(all_chunks)
        else:
            result = self._add_to_faiss(all_chunks)

        logger.info(f"✅ Added {result['chunks_added']} chunks to vector database")
        return result

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.embedding_model:
            return self.embedding_model.encode(text).tolist()
        else:
            # Fallback: Use Gemini (not ideal for production)
            logger.warning("Using Gemini for embeddings (slow) - install sentence-transformers")
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                return result['embedding']
            except:
                # Ultimate fallback: random embedding (for testing only)
                import numpy as np
                return np.random.rand(self.dimension).tolist()

    def _add_to_qdrant(self, chunks: List[Dict]) -> Dict:
        """Add chunks to Qdrant"""
        from qdrant_client.models import PointStruct

        points = []
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk["text"])
            point = PointStruct(
                id=len(self.chunks) + i,
                vector=embedding,
                payload=chunk
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return {"chunks_added": len(chunks)}

    def _add_to_faiss(self, chunks: List[Dict]) -> Dict:
        """Add chunks to FAISS"""
        import numpy as np

        if self.index is None:
            # In-memory only
            self.chunks.extend(chunks)
            self.chunk_texts.extend([c["text"] for c in chunks])
            return {"chunks_added": len(chunks)}

        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            embedding = self._get_embedding(chunk["text"])
            embeddings.append(embedding)

        # Add to FAISS
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)

        # Store metadata
        self.chunks.extend(chunks)
        self.chunk_texts.extend([c["text"] for c in chunks])

        return {"chunks_added": len(chunks)}

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search for relevant chunks"""
        if self.db_type == "qdrant":
            return self._search_qdrant(query, top_k)
        else:
            return self._search_faiss(query, top_k)

    def _search_qdrant(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search Qdrant"""
        query_embedding = self._get_embedding(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        chunks = []
        for result in results:
            chunk = result.payload
            chunk["score"] = result.score
            chunks.append(chunk)

        return chunks

    def _search_faiss(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search FAISS"""
        import numpy as np

        if self.index is None:
            # Fallback: simple text search
            return self._simple_text_search(query, top_k)

        # Generate query embedding
        query_embedding = self._get_embedding(query)
        query_array = np.array([query_embedding]).astype('float32')

        # Search FAISS
        distances, indices = self.index.search(query_array, min(top_k, len(self.chunks)))

        # Return results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = 1 / (1 + dist)  # Convert distance to similarity score
                results.append(chunk)

        return results

    def _simple_text_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search (fallback)"""
        query_lower = query.lower()
        scores = []

        for i, chunk in enumerate(self.chunks):
            text_lower = chunk["text"].lower()
            # Simple scoring based on keyword matches
            score = sum(1 for word in query_lower.split() if word in text_lower)
            scores.append((score, i))

        # Sort by score
        scores.sort(reverse=True)

        # Return top_k results
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                chunk = self.chunks[idx].copy()
                chunk["score"] = score
                results.append(chunk)

        return results

    def get_stats(self) -> Dict:
        """Get vector database statistics"""
        if self.db_type == "qdrant":
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "chunks": collection_info.points_count,
                "backend": "Qdrant",
                "dimension": self.dimension
            }
        else:
            return {
                "chunks": len(self.chunks),
                "backend": "FAISS" if self.index else "In-Memory",
                "dimension": self.dimension
            }

    def close(self):
        """Close database connection"""
        if self.db_type == "qdrant" and hasattr(self, 'client'):
            self.client.close()
            logger.info("Qdrant connection closed")
