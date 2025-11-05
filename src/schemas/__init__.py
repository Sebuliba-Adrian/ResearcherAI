"""Pydantic schemas for request/response validation."""
from src.schemas.collection import DataCollectionRequest, DataCollectionResponse
from src.schemas.common import (
    DataSource,
    ErrorResponse,
    PaperMetadata,
    ReasoningMode,
    StatusResponse,
    SummarizationMode,
)
from src.schemas.graph import (
    GraphNode,
    GraphQueryRequest,
    GraphQueryResponse,
    GraphRelationship,
    GraphStatsResponse,
)
from src.schemas.query import (
    QuestionRequest,
    QuestionResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.schemas.session import SessionCreate, SessionResponse, SessionStats, SessionUpdate
from src.schemas.summarization import (
    PaperSummary,
    SummarizationRequest,
    SummarizationResponse,
)

__all__ = [
    # Common
    "DataSource",
    "ReasoningMode",
    "SummarizationMode",
    "PaperMetadata",
    "StatusResponse",
    "ErrorResponse",
    # Collection
    "DataCollectionRequest",
    "DataCollectionResponse",
    # Query
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "QuestionRequest",
    "QuestionResponse",
    # Session
    "SessionCreate",
    "SessionUpdate",
    "SessionResponse",
    "SessionStats",
    # Summarization
    "SummarizationRequest",
    "SummarizationResponse",
    "PaperSummary",
    # Graph
    "GraphStatsResponse",
    "GraphQueryRequest",
    "GraphQueryResponse",
    "GraphNode",
    "GraphRelationship",
]
