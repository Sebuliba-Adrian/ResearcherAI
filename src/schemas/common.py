"""
Common schemas shared across the application.
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ReasoningMode(str, Enum):
    """Reasoning modes for the system."""

    QUICK = "quick"
    BALANCED = "balanced"
    DEEP = "deep"
    RESEARCH = "research"


class DataSource(str, Enum):
    """Data sources for collection."""

    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    ZENODO = "zenodo"
    WEBSEARCH = "websearch"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"


class SummarizationMode(str, Enum):
    """Summarization modes."""

    SIMPLE = "simple"
    DETAILED = "detailed"
    COMPARATIVE = "comparative"
    HIERARCHICAL = "hierarchical"


class PaperMetadata(BaseModel):
    """Schema for a single paper."""

    title: str = Field(..., min_length=5)
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(None, ge=1900, le=2100)
    source: str
    url: Optional[str] = None
    field: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        """Ensure year is reasonable."""
        if v is not None and (v < 1900 or v > 2100):
            raise ValueError("Year must be between 1900 and 2100")
        return v


class StatusResponse(BaseModel):
    """Generic status response."""

    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    status_code: int = 500
