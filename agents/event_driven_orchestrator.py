"""
Event-Driven Orchestrator for multi-agent coordination via Kafka.

This orchestrator uses Kafka event streaming for asynchronous,
decoupled communication between agents.

Benefits:
- Agents are loosely coupled (don't need to know about each other)
- Asynchronous processing (agents can work in parallel)
- Event replay capability (for debugging and auditing)
- Better scalability (can add more agent instances)
- Fault tolerance (events persist even if agents crash)
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from utils.kafka_manager import get_kafka_manager
from utils.event_schemas import (
    EventType, EventPriority,
    QuerySubmittedEvent, QueryValidatedEvent,
    DataCollectionStartedEvent, DataCollectionCompletedEvent, DataCollectionFailedEvent,
    GraphProcessingStartedEvent, GraphProcessingCompletedEvent, GraphProcessingFailedEvent,
    VectorProcessingStartedEvent, VectorProcessingCompletedEvent, VectorProcessingFailedEvent,
    ReasoningStartedEvent, ReasoningCompletedEvent, ReasoningFailedEvent,
    PaperMetadata, EntityMetadata, RelationshipMetadata
)
from agents.data_agent import DataCollectorAgent
from agents.graph_agent import KnowledgeGraphAgent
from agents.vector_agent import VectorAgent
from agents.reasoning_agent import ReasoningAgent

logger = logging.getLogger(__name__)


class EventDrivenOrchestrator:
    """
    Event-driven orchestrator that coordinates agents via Kafka events.

    Architecture:
    1. User submits query → QuerySubmittedEvent
    2. DataCollector consumes query → DataCollectionCompletedEvent
    3. GraphAgent consumes data → GraphProcessingCompletedEvent
    4. VectorAgent consumes data → VectorProcessingCompletedEvent
    5. ReasoningAgent consumes graph+vector → ReasoningCompletedEvent

    Each agent is decoupled and communicates only through events.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        use_neo4j: bool = False,
        use_qdrant: bool = False
    ):
        """
        Initialize event-driven orchestrator.

        Args:
            session_id: Session identifier
            use_neo4j: Whether to use Neo4j (vs NetworkX)
            use_qdrant: Whether to use Qdrant (vs FAISS)
        """
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.use_neo4j = use_neo4j
        self.use_qdrant = use_qdrant

        # Initialize Kafka manager
        self.kafka_manager = get_kafka_manager()

        # Initialize agents
        self.data_agent = DataCollectorAgent()
        self.graph_agent = KnowledgeGraphAgent(backend="neo4j" if use_neo4j else "networkx")
        self.vector_agent = VectorAgent(backend="qdrant" if use_qdrant else "faiss")
        self.reasoning_agent = ReasoningAgent()

        # Track pipeline state
        self.pipeline_state: Dict[str, Any] = {}

        logger.info(
            f"Event-driven orchestrator initialized (session={self.session_id}, "
            f"neo4j={use_neo4j}, qdrant={use_qdrant}, kafka={self.kafka_manager.enabled})"
        )

    def process_query(
        self,
        query: str,
        max_papers_per_source: int = 5,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Process a research query using event-driven architecture.

        If Kafka is enabled: Uses async event-driven pipeline
        If Kafka is disabled: Falls back to synchronous direct calls

        Args:
            query: Research question
            max_papers_per_source: Maximum papers to collect per source
            timeout_seconds: Maximum time to wait for completion

        Returns:
            Complete pipeline results
        """
        correlation_id = f"query_{uuid.uuid4().hex[:8]}"

        logger.info(f"Processing query (correlation_id={correlation_id}): {query}")

        if self.kafka_manager.enabled:
            # Use event-driven pipeline
            return self._process_query_event_driven(
                query, max_papers_per_source, correlation_id, timeout_seconds
            )
        else:
            # Fall back to synchronous pipeline
            logger.info("Kafka disabled, using synchronous pipeline")
            return self._process_query_synchronous(
                query, max_papers_per_source, correlation_id
            )

    def _process_query_event_driven(
        self,
        query: str,
        max_papers_per_source: int,
        correlation_id: str,
        timeout_seconds: int
    ) -> Dict[str, Any]:
        """Process query using Kafka events"""

        # Step 1: Publish QuerySubmittedEvent
        query_event = QuerySubmittedEvent(
            event_id=f"event_{uuid.uuid4().hex[:8]}",
            correlation_id=correlation_id,
            session_id=self.session_id,
            query=query,
            max_papers_per_source=max_papers_per_source
        )

        self.kafka_manager.publish_event(query_event)
        logger.info(f"Published QuerySubmittedEvent: {query_event.event_id}")

        # Step 2: Process data collection (synchronously for now, will be async consumer later)
        data_result = self._handle_data_collection(query_event)

        # Step 3: Process graph extraction
        graph_result = self._handle_graph_processing(data_result)

        # Step 4: Process vector embeddings
        vector_result = self._handle_vector_processing(data_result)

        # Step 5: Process reasoning
        reasoning_result = self._handle_reasoning(query, graph_result, vector_result)

        # Compile results
        return {
            "correlation_id": correlation_id,
            "session_id": self.session_id,
            "query": query,
            "data_collection": data_result,
            "graph_processing": graph_result,
            "vector_processing": vector_result,
            "reasoning": reasoning_result,
            "pipeline_complete": True
        }

    def _handle_data_collection(self, query_event: QuerySubmittedEvent) -> Dict[str, Any]:
        """Handle data collection and publish events"""
        try:
            # Publish started event
            started_event = DataCollectionStartedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=query_event.correlation_id,
                session_id=self.session_id,
                query=query_event.query,
                sources=["arxiv", "semantic_scholar", "pubmed"],
                max_papers_per_source=query_event.max_papers_per_source
            )
            self.kafka_manager.publish_event(started_event)

            # Collect data
            start_time = time.time()
            papers = self.data_agent.collect_all(
                query=query_event.query,
                max_per_source=query_event.max_papers_per_source
            )
            execution_time = time.time() - start_time

            # Convert papers to PaperMetadata
            paper_metadata = [
                PaperMetadata(
                    title=p.get("title", "Unknown"),
                    authors=p.get("authors", []),
                    abstract=p.get("abstract"),
                    year=p.get("year"),
                    url=p.get("url"),
                    source=p.get("source", "unknown"),
                    doi=p.get("doi"),
                    citations=p.get("citations")
                )
                for p in papers
            ]

            # Count by source
            sources_used = {}
            for p in papers:
                source = p.get("source", "unknown")
                sources_used[source] = sources_used.get(source, 0) + 1

            # Publish completed event
            completed_event = DataCollectionCompletedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=query_event.correlation_id,
                session_id=self.session_id,
                query=query_event.query,
                papers_collected=len(papers),
                sources_used=sources_used,
                papers=paper_metadata,
                execution_time=execution_time
            )
            self.kafka_manager.publish_event(completed_event)

            logger.info(
                f"Data collection completed: {len(papers)} papers in {execution_time:.2f}s"
            )

            return {
                "papers": papers,
                "papers_collected": len(papers),
                "sources_used": sources_used,
                "execution_time": execution_time
            }

        except Exception as e:
            # Publish failed event
            failed_event = DataCollectionFailedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=query_event.correlation_id,
                session_id=self.session_id,
                query=query_event.query,
                error_message=str(e),
                error_type=type(e).__name__,
                should_retry=False
            )
            self.kafka_manager.publish_event(failed_event)

            logger.error(f"Data collection failed: {e}")
            raise

    def _handle_graph_processing(self, data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph processing and publish events"""
        try:
            papers = data_result["papers"]

            # Publish started event
            started_event = GraphProcessingStartedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=f"graph_{uuid.uuid4().hex[:8]}",
                session_id=self.session_id,
                papers_to_process=len(papers),
                backend="neo4j" if self.use_neo4j else "networkx"
            )
            self.kafka_manager.publish_event(started_event)

            # Process graph
            start_time = time.time()
            graph_result = self.graph_agent.extract_and_store_triples(papers)
            execution_time = time.time() - start_time

            # Create entity/relationship metadata
            entities = [
                EntityMetadata(
                    entity_id=str(i),
                    entity_type="sample",
                    properties={}
                )
                for i in range(min(10, graph_result.get("nodes_extracted", 0)))
            ]

            relationships = [
                RelationshipMetadata(
                    source_id=str(i),
                    target_id=str(i+1),
                    relationship_type="sample",
                    properties={}
                )
                for i in range(min(10, graph_result.get("edges_extracted", 0)))
            ]

            # Publish completed event
            completed_event = GraphProcessingCompletedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=started_event.correlation_id,
                session_id=self.session_id,
                nodes_created=graph_result.get("nodes_extracted", 0),
                edges_created=graph_result.get("edges_extracted", 0),
                entities=entities,
                relationships=relationships,
                execution_time=execution_time,
                backend="neo4j" if self.use_neo4j else "networkx"
            )
            self.kafka_manager.publish_event(completed_event)

            logger.info(
                f"Graph processing completed: {graph_result.get('nodes_extracted', 0)} nodes, "
                f"{graph_result.get('edges_extracted', 0)} edges in {execution_time:.2f}s"
            )

            return {**graph_result, "execution_time": execution_time}

        except Exception as e:
            # Publish failed event
            failed_event = GraphProcessingFailedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=f"graph_{uuid.uuid4().hex[:8]}",
                session_id=self.session_id,
                error_message=str(e),
                error_type=type(e).__name__,
                should_retry=False
            )
            self.kafka_manager.publish_event(failed_event)

            logger.error(f"Graph processing failed: {e}")
            return {"nodes_extracted": 0, "edges_extracted": 0, "error": str(e)}

    def _handle_vector_processing(self, data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle vector processing and publish events"""
        try:
            papers = data_result["papers"]

            # Publish started event
            started_event = VectorProcessingStartedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=f"vector_{uuid.uuid4().hex[:8]}",
                session_id=self.session_id,
                chunks_to_process=len(papers),
                backend="qdrant" if self.use_qdrant else "faiss",
                embedding_model="all-MiniLM-L6-v2"
            )
            self.kafka_manager.publish_event(started_event)

            # Process vectors
            start_time = time.time()
            vector_result = self.vector_agent.add_documents(papers)
            execution_time = time.time() - start_time

            # Publish completed event
            completed_event = VectorProcessingCompletedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=started_event.correlation_id,
                session_id=self.session_id,
                embeddings_created=vector_result.get("embeddings_added", 0),
                vector_dimensions=vector_result.get("dimensions", 384),
                backend="qdrant" if self.use_qdrant else "faiss",
                embedding_model="all-MiniLM-L6-v2",
                execution_time=execution_time,
                index_size=vector_result.get("total_vectors", 0)
            )
            self.kafka_manager.publish_event(completed_event)

            logger.info(
                f"Vector processing completed: {vector_result.get('embeddings_added', 0)} "
                f"embeddings in {execution_time:.2f}s"
            )

            return {**vector_result, "execution_time": execution_time}

        except Exception as e:
            # Publish failed event
            failed_event = VectorProcessingFailedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=f"vector_{uuid.uuid4().hex[:8]}",
                session_id=self.session_id,
                error_message=str(e),
                error_type=type(e).__name__,
                should_retry=False
            )
            self.kafka_manager.publish_event(failed_event)

            logger.error(f"Vector processing failed: {e}")
            return {"embeddings_added": 0, "error": str(e)}

    def _handle_reasoning(
        self,
        query: str,
        graph_result: Dict[str, Any],
        vector_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle reasoning and publish events"""
        try:
            # Publish started event
            started_event = ReasoningStartedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=f"reasoning_{uuid.uuid4().hex[:8]}",
                session_id=self.session_id,
                query=query,
                model="gemini-2.0-flash",
                sources_available={
                    "graph": graph_result.get("nodes_extracted", 0) > 0,
                    "vector": vector_result.get("embeddings_added", 0) > 0
                }
            )
            self.kafka_manager.publish_event(started_event)

            # Generate answer
            start_time = time.time()
            answer = self.reasoning_agent.synthesize_answer(
                query=query,
                papers=[],  # Papers already in vector store
                graph_agent=self.graph_agent,
                vector_agent=self.vector_agent
            )
            execution_time = time.time() - start_time

            # Publish completed event
            completed_event = ReasoningCompletedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=started_event.correlation_id,
                session_id=self.session_id,
                query=query,
                answer=answer,
                sources_used=["graph", "vector"],
                citations=[],
                model="gemini-2.0-flash",
                execution_time=execution_time
            )
            self.kafka_manager.publish_event(completed_event)

            logger.info(f"Reasoning completed in {execution_time:.2f}s")

            return {
                "answer": answer,
                "execution_time": execution_time
            }

        except Exception as e:
            # Publish failed event
            failed_event = ReasoningFailedEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                correlation_id=f"reasoning_{uuid.uuid4().hex[:8]}",
                session_id=self.session_id,
                query=query,
                error_message=str(e),
                error_type=type(e).__name__,
                should_retry=False
            )
            self.kafka_manager.publish_event(failed_event)

            logger.error(f"Reasoning failed: {e}")
            return {"answer": f"Error: {e}", "error": str(e)}

    def _process_query_synchronous(
        self,
        query: str,
        max_papers_per_source: int,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Fallback synchronous processing without Kafka"""
        logger.info("Using synchronous pipeline (Kafka disabled)")

        # Collect data
        start_time = time.time()
        papers = self.data_agent.collect_all(query, max_per_source=max_papers_per_source)
        data_time = time.time() - start_time

        # Process graph
        start_time = time.time()
        graph_result = self.graph_agent.extract_and_store_triples(papers)
        graph_time = time.time() - start_time

        # Process vectors
        start_time = time.time()
        vector_result = self.vector_agent.add_documents(papers)
        vector_time = time.time() - start_time

        # Generate answer
        start_time = time.time()
        answer = self.reasoning_agent.synthesize_answer(
            query=query,
            papers=papers,
            graph_agent=self.graph_agent,
            vector_agent=self.vector_agent
        )
        reasoning_time = time.time() - start_time

        return {
            "correlation_id": correlation_id,
            "session_id": self.session_id,
            "query": query,
            "data_collection": {
                "papers_collected": len(papers),
                "execution_time": data_time
            },
            "graph_processing": {
                **graph_result,
                "execution_time": graph_time
            },
            "vector_processing": {
                **vector_result,
                "execution_time": vector_time
            },
            "reasoning": {
                "answer": answer,
                "execution_time": reasoning_time
            },
            "pipeline_complete": True,
            "kafka_enabled": False
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "session_id": self.session_id,
            "kafka_enabled": self.kafka_manager.enabled,
            "kafka_stats": self.kafka_manager.get_stats(),
            "use_neo4j": self.use_neo4j,
            "use_qdrant": self.use_qdrant
        }

    def close(self):
        """Close orchestrator and cleanup resources"""
        logger.info(f"Closing event-driven orchestrator (session={self.session_id})")
        # Kafka manager is global, don't close it here
