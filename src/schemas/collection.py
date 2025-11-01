"""
Schemas for data collection operations.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.schemas.common import DataSource, PaperMetadata


class DataCollectionRequest(BaseModel):
    """Request for data collection."""

    query: str = Field(..., min_length=3, max_length=500)
    max_per_source: int = Field(default=10, ge=1, le=100)
    sources: Optional[List[DataSource]] = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class DataCollectionResponse(BaseModel):
    """Response from data collection."""

    papers_collected: int = Field(..., ge=0)
    sources: Dict[str, int] = Field(default_factory=dict)
    papers: List[PaperMetadata] = Field(default_factory=list)
    execution_time: float = Field(..., ge=0)

    @field_validator("papers_collected")
    @classmethod
    def validate_papers_count(cls, v: int, info) -> int:
        """Ensure papers_collected matches actual papers."""
        papers = info.data.get("papers", [])
        if v != len(papers):
            raise ValueError(
                f"papers_collected ({v}) doesn't match actual papers ({len(papers)})"
            )
        return v
