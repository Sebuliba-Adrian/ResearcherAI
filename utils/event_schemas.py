"""
Event schemas for Kafka-based event-driven architecture.

These schemas define the structure of events flowing between agents,
ensuring type safety and validation across the entire pipeline.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Types of events in the system"""
    # Query events
    QUERY_SUBMITTED = "query.submitted"
    QUERY_VALIDATED = "query.validated"

    # Data collection events
    DATA_COLLECTION_STARTED = "data.collection.started"
    DATA_COLLECTION_COMPLETED = "data.collection.completed"
    DATA_COLLECTION_FAILED = "data.collection.failed"

    # Graph processing events
    GRAPH_PROCESSING_STARTED = "graph.processing.started"
    GRAPH_PROCESSING_COMPLETED = "graph.processing.completed"
    GRAPH_PROCESSING_FAILED = "graph.processing.failed"

    # Vector processing events
    VECTOR_PROCESSING_STARTED = "vector.processing.started"
    VECTOR_PROCESSING_COMPLETED = "vector.processing.completed"
    VECTOR_PROCESSING_FAILED = "vector.processing.failed"

    # Reasoning events
    REASONING_STARTED = "reasoning.started"
    REASONING_COMPLETED = "reasoning.completed"
    REASONING_FAILED = "reasoning.failed"

    # System events
    AGENT_HEALTH_CHECK = "agent.health.check"
    AGENT_ERROR = "agent.error"


class EventPriority(str, Enum):
    """Priority levels for events"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class BaseEvent(BaseModel):
    """Base event schema that all events inherit from"""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    correlation_id: str = Field(..., description="ID to correlate related events")
    session_id: Optional[str] = Field(None, description="Session identifier")
    priority: EventPriority = Field(default=EventPriority.NORMAL, description="Event priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Query Events
# ============================================================================

class QuerySubmittedEvent(BaseEvent):
    """Event emitted when a user submits a query"""
    event_type: Literal[EventType.QUERY_SUBMITTED] = EventType.QUERY_SUBMITTED
    query: str = Field(..., min_length=1, description="User's research query")
    user_id: Optional[str] = Field(None, description="User identifier")
    max_papers_per_source: int = Field(default=5, ge=1, le=100)
    sources: List[str] = Field(default_factory=list, description="Requested data sources")


class QueryValidatedEvent(BaseEvent):
    """Event emitted after query validation"""
    event_type: Literal[EventType.QUERY_VALIDATED] = EventType.QUERY_VALIDATED
    original_query: str = Field(..., description="Original query")
    validated_query: str = Field(..., description="Validated/cleaned query")
    is_valid: bool = Field(..., description="Whether query is valid")
    validation_errors: List[str] = Field(default_factory=list)


# ============================================================================
# Data Collection Events
# ============================================================================

class PaperMetadata(BaseModel):
    """Metadata for a collected paper"""
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    source: str
    doi: Optional[str] = None
    citations: Optional[int] = None


class DataCollectionStartedEvent(BaseEvent):
    """Event emitted when data collection starts"""
    event_type: Literal[EventType.DATA_COLLECTION_STARTED] = EventType.DATA_COLLECTION_STARTED
    query: str = Field(..., description="Query being processed")
    sources: List[str] = Field(..., description="Sources to collect from")
    max_papers_per_source: int = Field(..., ge=1)


class DataCollectionCompletedEvent(BaseEvent):
    """Event emitted when data collection completes"""
    event_type: Literal[EventType.DATA_COLLECTION_COMPLETED] = EventType.DATA_COLLECTION_COMPLETED
    query: str = Field(..., description="Query that was processed")
    papers_collected: int = Field(..., ge=0, description="Total papers collected")
    sources_used: Dict[str, int] = Field(..., description="Papers per source")
    papers: List[PaperMetadata] = Field(default_factory=list, description="Collected papers")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    token_usage: Optional[Dict[str, Any]] = Field(None, description="Token usage statistics")

    @validator('papers_collected')
    def validate_papers_count(cls, v, values):
        """Ensure papers_collected matches actual papers"""
        papers = values.get('papers', [])
        if v != len(papers):
            raise ValueError(f"papers_collected ({v}) doesn't match actual papers ({len(papers)})")
        return v


class DataCollectionFailedEvent(BaseEvent):
    """Event emitted when data collection fails"""
    event_type: Literal[EventType.DATA_COLLECTION_FAILED] = EventType.DATA_COLLECTION_FAILED
    query: str = Field(..., description="Query that failed")
    error_message: str = Field(..., description="Error description")
    error_type: str = Field(..., description="Error type/class")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    should_retry: bool = Field(default=False, description="Whether to retry")


# ============================================================================
# Graph Processing Events
# ============================================================================

class EntityMetadata(BaseModel):
    """Metadata for an extracted entity"""
    entity_id: str
    entity_type: str  # e.g., "author", "topic", "paper"
    properties: Dict[str, Any] = Field(default_factory=dict)


class RelationshipMetadata(BaseModel):
    """Metadata for an extracted relationship"""
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "authored", "is_about", "cites"
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphProcessingStartedEvent(BaseEvent):
    """Event emitted when graph processing starts"""
    event_type: Literal[EventType.GRAPH_PROCESSING_STARTED] = EventType.GRAPH_PROCESSING_STARTED
    papers_to_process: int = Field(..., ge=0, description="Number of papers to process")
    backend: str = Field(..., description="Graph backend (neo4j/networkx)")


class GraphProcessingCompletedEvent(BaseEvent):
    """Event emitted when graph processing completes"""
    event_type: Literal[EventType.GRAPH_PROCESSING_COMPLETED] = EventType.GRAPH_PROCESSING_COMPLETED
    nodes_created: int = Field(..., ge=0, description="Number of nodes created")
    edges_created: int = Field(..., ge=0, description="Number of edges created")
    entities: List[EntityMetadata] = Field(default_factory=list, description="Extracted entities")
    relationships: List[RelationshipMetadata] = Field(default_factory=list, description="Extracted relationships")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    backend: str = Field(..., description="Graph backend used")
    token_usage: Optional[Dict[str, Any]] = Field(None, description="Token usage statistics")


class GraphProcessingFailedEvent(BaseEvent):
    """Event emitted when graph processing fails"""
    event_type: Literal[EventType.GRAPH_PROCESSING_FAILED] = EventType.GRAPH_PROCESSING_FAILED
    error_message: str = Field(..., description="Error description")
    error_type: str = Field(..., description="Error type/class")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    papers_processed: int = Field(default=0, ge=0, description="Papers processed before failure")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    should_retry: bool = Field(default=False, description="Whether to retry")


# ============================================================================
# Vector Processing Events
# ============================================================================

class VectorProcessingStartedEvent(BaseEvent):
    """Event emitted when vector processing starts"""
    event_type: Literal[EventType.VECTOR_PROCESSING_STARTED] = EventType.VECTOR_PROCESSING_STARTED
    chunks_to_process: int = Field(..., ge=0, description="Number of text chunks to embed")
    backend: str = Field(..., description="Vector backend (qdrant/faiss)")
    embedding_model: str = Field(..., description="Embedding model name")


class VectorProcessingCompletedEvent(BaseEvent):
    """Event emitted when vector processing completes"""
    event_type: Literal[EventType.VECTOR_PROCESSING_COMPLETED] = EventType.VECTOR_PROCESSING_COMPLETED
    embeddings_created: int = Field(..., ge=0, description="Number of embeddings created")
    vector_dimensions: int = Field(..., ge=0, description="Dimension of each vector")
    backend: str = Field(..., description="Vector backend used")
    embedding_model: str = Field(..., description="Embedding model used")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    index_size: int = Field(..., ge=0, description="Total vectors in index")


class VectorProcessingFailedEvent(BaseEvent):
    """Event emitted when vector processing fails"""
    event_type: Literal[EventType.VECTOR_PROCESSING_FAILED] = EventType.VECTOR_PROCESSING_FAILED
    error_message: str = Field(..., description="Error description")
    error_type: str = Field(..., description="Error type/class")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    chunks_processed: int = Field(default=0, ge=0, description="Chunks processed before failure")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    should_retry: bool = Field(default=False, description="Whether to retry")


# ============================================================================
# Reasoning Events
# ============================================================================

class ReasoningStartedEvent(BaseEvent):
    """Event emitted when reasoning starts"""
    event_type: Literal[EventType.REASONING_STARTED] = EventType.REASONING_STARTED
    query: str = Field(..., description="Query to answer")
    model: str = Field(..., description="LLM model to use")
    sources_available: Dict[str, bool] = Field(..., description="Available data sources")


class ReasoningCompletedEvent(BaseEvent):
    """Event emitted when reasoning completes"""
    event_type: Literal[EventType.REASONING_COMPLETED] = EventType.REASONING_COMPLETED
    query: str = Field(..., description="Query that was answered")
    answer: str = Field(..., description="Generated answer")
    sources_used: List[str] = Field(default_factory=list, description="Sources used in answer")
    citations: List[str] = Field(default_factory=list, description="Citations included")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")
    model: str = Field(..., description="LLM model used")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    token_usage: Optional[Dict[str, Any]] = Field(None, description="Token usage statistics")


class ReasoningFailedEvent(BaseEvent):
    """Event emitted when reasoning fails"""
    event_type: Literal[EventType.REASONING_FAILED] = EventType.REASONING_FAILED
    query: str = Field(..., description="Query that failed")
    error_message: str = Field(..., description="Error description")
    error_type: str = Field(..., description="Error type/class")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    should_retry: bool = Field(default=False, description="Whether to retry")


# ============================================================================
# System Events
# ============================================================================

class AgentHealthCheckEvent(BaseEvent):
    """Event for agent health checks"""
    event_type: Literal[EventType.AGENT_HEALTH_CHECK] = EventType.AGENT_HEALTH_CHECK
    agent_name: str = Field(..., description="Name of the agent")
    status: str = Field(..., description="Health status")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Health metrics")


class AgentErrorEvent(BaseEvent):
    """Event emitted when an agent encounters an error"""
    event_type: Literal[EventType.AGENT_ERROR] = EventType.AGENT_ERROR
    agent_name: str = Field(..., description="Name of the agent")
    error_message: str = Field(..., description="Error description")
    error_type: str = Field(..., description="Error type/class")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    severity: str = Field(..., description="Error severity")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")


# ============================================================================
# Event Registry (for serialization/deserialization)
# ============================================================================

EVENT_TYPE_TO_CLASS = {
    EventType.QUERY_SUBMITTED: QuerySubmittedEvent,
    EventType.QUERY_VALIDATED: QueryValidatedEvent,
    EventType.DATA_COLLECTION_STARTED: DataCollectionStartedEvent,
    EventType.DATA_COLLECTION_COMPLETED: DataCollectionCompletedEvent,
    EventType.DATA_COLLECTION_FAILED: DataCollectionFailedEvent,
    EventType.GRAPH_PROCESSING_STARTED: GraphProcessingStartedEvent,
    EventType.GRAPH_PROCESSING_COMPLETED: GraphProcessingCompletedEvent,
    EventType.GRAPH_PROCESSING_FAILED: GraphProcessingFailedEvent,
    EventType.VECTOR_PROCESSING_STARTED: VectorProcessingStartedEvent,
    EventType.VECTOR_PROCESSING_COMPLETED: VectorProcessingCompletedEvent,
    EventType.VECTOR_PROCESSING_FAILED: VectorProcessingFailedEvent,
    EventType.REASONING_STARTED: ReasoningStartedEvent,
    EventType.REASONING_COMPLETED: ReasoningCompletedEvent,
    EventType.REASONING_FAILED: ReasoningFailedEvent,
    EventType.AGENT_HEALTH_CHECK: AgentHealthCheckEvent,
    EventType.AGENT_ERROR: AgentErrorEvent,
}


def deserialize_event(event_data: Dict[str, Any]) -> BaseEvent:
    """
    Deserialize event data into the appropriate event class.

    Args:
        event_data: Dictionary containing event data

    Returns:
        Deserialized event object

    Raises:
        ValueError: If event_type is unknown
    """
    event_type = EventType(event_data.get("event_type"))
    event_class = EVENT_TYPE_TO_CLASS.get(event_type)

    if event_class is None:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class(**event_data)
