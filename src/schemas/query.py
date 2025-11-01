"""
Schemas for query and search operations.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.schemas.common import PaperMetadata, ReasoningMode


class SearchRequest(BaseModel):
    """Request for semantic search."""

    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class SearchResult(BaseModel):
    """Single search result."""

    paper: PaperMetadata
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from search operation."""

    query: str
    results: List[SearchResult]
    total_results: int
    execution_time: float = Field(..., ge=0)


class QuestionRequest(BaseModel):
    """Request for question answering."""

    question: str = Field(..., min_length=10, max_length=1000)
    mode: ReasoningMode = ReasoningMode.BALANCED
    context: Optional[List[str]] = None
    max_turns: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Ensure question is not just whitespace."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return v.strip()


class QuestionResponse(BaseModel):
    """Response from question answering."""

    question: str
    answer: str
    reasoning_steps: Optional[List[str]] = None
    sources: List[PaperMetadata] = Field(default_factory=list)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    execution_time: float = Field(..., ge=0)
