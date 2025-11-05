"""
Schema-First Design with Pydantic
==================================

Strict validation for all agent inputs and outputs.
Prevents the "$100K comma bug" and other type-related failures.

Every agent input and output follows a strict schema. No exceptions.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


# ========================================================================
# ENUMS
# ========================================================================

class ReasoningMode(str, Enum):
    """Reasoning modes for the system"""
    QUICK = "quick"
    BALANCED = "balanced"
    DEEP = "deep"
    RESEARCH = "research"


class DataSource(str, Enum):
    """Data sources for collection"""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    ZENODO = "zenodo"
    WEBSEARCH = "websearch"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"


# ========================================================================
# DATA COLLECTOR SCHEMAS
# ========================================================================

class DataCollectionRequest(BaseModel):
    """Request for data collection"""
    query: str = Field(..., min_length=3, max_length=500)
    max_per_source: int = Field(default=10, ge=1, le=100)
    sources: Optional[List[DataSource]] = None

    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not just whitespace"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class PaperMetadata(BaseModel):
    """Schema for a single paper"""
    title: str = Field(..., min_length=5)
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(None, ge=1900, le=2100)
    source: str
    url: Optional[str] = None
    field: Optional[str] = None

    @validator('year')
    def validate_year(cls, v):
        """Ensure year is reasonable"""
        if v is not None and (v < 1900 or v > 2100):
            raise ValueError("Year must be between 1900 and 2100")
        return v


class DataCollectionResponse(BaseModel):
    """Response from data collection"""
    papers_collected: int = Field(..., ge=0)
    sources: Dict[str, int] = Field(default_factory=dict)
    papers: List[PaperMetadata] = Field(default_factory=list)
    execution_time: float = Field(..., ge=0)

    @validator('papers_collected')
    def validate_papers_count(cls, v, values):
        """Ensure papers_collected matches actual papers"""
        papers = values.get('papers', [])
        if v != len(papers):
            raise ValueError(
                f"papers_collected ({v}) doesn't match actual papers ({len(papers)})"
            )
        return v


# ========================================================================
# GRAPH PROCESSING SCHEMAS
# ========================================================================

class GraphProcessingRequest(BaseModel):
    """Request for graph processing"""
    papers: List[PaperMetadata]
    extract_entities: bool = True
    extract_relationships: bool = True


class GraphProcessingResponse(BaseModel):
    """Response from graph processing"""
    nodes: int = Field(..., ge=0)
    edges: int = Field(..., ge=0)
    triples_extracted: int = Field(default=0, ge=0)
    backend: str  # "neo4j" or "networkx"
    execution_time: float = Field(..., ge=0)

    @validator('edges')
    def validate_edges(cls, v, values):
        """Edges can't exceed maximum possible connections"""
        nodes = values.get('nodes', 0)
        max_edges = (nodes * (nodes - 1)) // 2
        if v > max_edges:
            raise ValueError(
                f"Edges ({v}) exceeds maximum possible ({max_edges}) for {nodes} nodes"
            )
        return v


# ========================================================================
# VECTOR PROCESSING SCHEMAS
# ========================================================================

class VectorProcessingRequest(BaseModel):
    """Request for vector processing"""
    papers: List[PaperMetadata]
    chunk_size: int = Field(default=400, ge=50, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class VectorProcessingResponse(BaseModel):
    """Response from vector processing"""
    embeddings_added: int = Field(..., ge=0)
    chunks_created: int = Field(..., ge=0)
    dimension: int = Field(..., ge=1)
    backend: str  # "qdrant" or "faiss"
    model: str
    execution_time: float = Field(..., ge=0)

    @validator('embeddings_added')
    def validate_embeddings(cls, v, values):
        """Embeddings should not exceed chunks"""
        chunks = values.get('chunks_created', 0)
        if v > chunks:
            raise ValueError(
                f"embeddings_added ({v}) exceeds chunks_created ({chunks})"
            )
        return v


# ========================================================================
# LLAMAINDEX SCHEMAS
# ========================================================================

class LlamaIndexIndexingRequest(BaseModel):
    """Request for LlamaIndex indexing"""
    papers: List[PaperMetadata]
    use_qdrant: bool = False


class LlamaIndexIndexingResponse(BaseModel):
    """Response from LlamaIndex indexing"""
    documents_indexed: int = Field(..., ge=0)
    nodes_parsed: int = Field(..., ge=0)
    vector_store: str  # "qdrant" or "in-memory"
    execution_time: float = Field(..., ge=0)


class LlamaIndexQueryRequest(BaseModel):
    """Request for LlamaIndex query"""
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class LlamaIndexQueryResponse(BaseModel):
    """Response from LlamaIndex query"""
    answer: str
    sources: List[str] = Field(default_factory=list)
    execution_time: float = Field(..., ge=0)


# ========================================================================
# REASONING SCHEMAS
# ========================================================================

class ReasoningRequest(BaseModel):
    """Request for reasoning"""
    question: str = Field(..., min_length=3, max_length=1000)
    mode: ReasoningMode = ReasoningMode.BALANCED
    use_graph: bool = True
    use_vector: bool = True
    use_llamaindex: bool = True


class ReasoningResponse(BaseModel):
    """Response from reasoning"""
    answer: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources_used: Dict[str, bool] = Field(default_factory=dict)
    conversation_turns: int = Field(default=0, ge=0)
    execution_time: float = Field(..., ge=0)

    @validator('answer')
    def validate_answer(cls, v):
        """Ensure answer is substantive"""
        if len(v.strip()) < 10:
            raise ValueError("Answer too short - must be at least 10 characters")

        # Check for common error patterns
        error_patterns = [
            "error occurred",
            "failed to",
            "unable to",
            "something went wrong"
        ]
        lower_answer = v.lower()
        for pattern in error_patterns:
            if pattern in lower_answer and len(v) < 100:
                raise ValueError(f"Answer contains error pattern: {pattern}")

        return v.strip()


# ========================================================================
# SELF-REFLECTION SCHEMAS
# ========================================================================

class SelfReflectionRequest(BaseModel):
    """Request for self-reflection"""
    question: str
    answer: str
    context: Dict[str, Any] = Field(default_factory=dict)


class SelfReflectionResponse(BaseModel):
    """Response from self-reflection"""
    quality_score: int = Field(..., ge=0, le=100)
    needs_correction: bool
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    execution_time: float = Field(..., ge=0)


# ========================================================================
# CRITIC REVIEW SCHEMAS
# ========================================================================

class CriticReviewRequest(BaseModel):
    """Request for critic review"""
    quality_score: int = Field(..., ge=0, le=100)
    answer_length: int = Field(..., ge=0)
    quality_threshold: int = Field(default=70, ge=0, le=100)
    min_length: int = Field(default=100, ge=0)


class CriticReviewResponse(BaseModel):
    """Response from critic review"""
    approved: bool
    quality_score: int
    answer_length: int
    suggestions: List[str] = Field(default_factory=list)
    execution_time: float = Field(default=0, ge=0)


# ========================================================================
# WORKFLOW STATE SCHEMA
# ========================================================================

class WorkflowState(BaseModel):
    """State for the entire workflow"""
    query: str
    max_per_source: int = 10
    current_step: str = "init"

    # Collected data
    papers: List[Dict] = Field(default_factory=list)

    # Processing results
    graph_data: Dict[str, Any] = Field(default_factory=dict)
    vector_data: Dict[str, Any] = Field(default_factory=dict)
    llamaindex_data: Dict[str, Any] = Field(default_factory=dict)
    reasoning_result: Dict[str, Any] = Field(default_factory=dict)
    reflection_feedback: Dict[str, Any] = Field(default_factory=dict)
    critic_feedback: Dict[str, Any] = Field(default_factory=dict)

    # Stage outputs for debugging
    stage_outputs: Dict[str, Dict] = Field(default_factory=dict)

    # Execution log
    messages: List[str] = Field(default_factory=list)

    # Airflow integration
    airflow_status: Optional[str] = None

    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True


# ========================================================================
# TOKEN BUDGET SCHEMAS
# ========================================================================

class TokenBudget(BaseModel):
    """Token budget constraints"""
    per_task_limit: int = Field(..., gt=0)
    per_user_limit: int = Field(..., gt=0)
    system_wide_limit: int = Field(..., gt=0)


class TokenUsage(BaseModel):
    """Token usage tracking"""
    task_id: str
    user_id: Optional[str] = None
    tokens_used: int = Field(..., ge=0)
    model: str
    cost_usd: float = Field(..., ge=0)
    timestamp: str

    @validator('tokens_used')
    def validate_tokens(cls, v):
        """Ensure tokens are reasonable"""
        if v > 1000000:  # 1M token sanity check
            raise ValueError(f"Token usage ({v}) exceeds sanity limit (1M)")
        return v


# ========================================================================
# API REQUEST/RESPONSE SCHEMAS
# ========================================================================

class CollectAPIRequest(BaseModel):
    """API request for /collect endpoint"""
    query: str = Field(..., min_length=3, max_length=500)
    sources: Optional[List[str]] = None
    max_results: int = Field(default=10, ge=1, le=100)


class AskAPIRequest(BaseModel):
    """API request for /ask endpoint"""
    question: str = Field(..., min_length=3, max_length=1000)
    reasoning_mode: str = Field(default="balanced")
    enable_critic: bool = True


class HealthCheckResponse(BaseModel):
    """API response for /health endpoint"""
    status: str  # "healthy" or "degraded"
    agents: Dict[str, str] = Field(default_factory=dict)
    circuit_breakers: Dict[str, str] = Field(default_factory=dict)
    timestamp: str


# ========================================================================
# ERROR SCHEMAS
# ========================================================================

class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    retry_allowed: bool = False


# ========================================================================
# VALIDATION HELPERS
# ========================================================================

def validate_and_parse(data: Dict, schema: type[BaseModel]) -> BaseModel:
    """
    Validate data against schema and return parsed object.

    Raises ValidationError if data doesn't match schema.

    Usage:
        validated = validate_and_parse(raw_data, DataCollectionResponse)
    """
    return schema(**data)


def safe_validate(data: Dict, schema: type[BaseModel]) -> tuple[bool, Optional[BaseModel], Optional[str]]:
    """
    Safely validate data, returning (success, parsed_obj, error_msg).

    Usage:
        success, validated, error = safe_validate(raw_data, DataCollectionResponse)
        if success:
            # Use validated
        else:
            # Handle error
    """
    try:
        parsed = schema(**data)
        return True, parsed, None
    except Exception as e:
        return False, None, str(e)
