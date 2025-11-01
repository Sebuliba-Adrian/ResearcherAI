"""
Schemas for summarization operations.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from src.schemas.common import PaperMetadata, SummarizationMode


class SummarizationRequest(BaseModel):
    """Request for paper summarization."""

    paper_ids: List[str] = Field(..., min_length=1, max_length=50)
    mode: SummarizationMode = SummarizationMode.SIMPLE
    focus_areas: Optional[List[str]] = None
    max_length: int = Field(default=500, ge=100, le=5000)


class PaperSummary(BaseModel):
    """Summary of a single paper."""

    paper: PaperMetadata
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    methodology: Optional[str] = None
    limitations: Optional[str] = None


class SummarizationResponse(BaseModel):
    """Response from summarization operation."""

    mode: SummarizationMode
    summaries: List[PaperSummary]
    comparative_analysis: Optional[str] = None
    execution_time: float = Field(..., ge=0)
