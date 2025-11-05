"""
Schemas for session management.
"""
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """Request to create a new session."""

    description: Optional[str] = None
    research_topic: Optional[str] = Field(None, max_length=500)
    config: Optional[Dict[str, Any]] = None


class SessionUpdate(BaseModel):
    """Request to update a session."""

    description: Optional[str] = None
    research_topic: Optional[str] = Field(None, max_length=500)
    status: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Response with session information."""

    id: int
    session_id: str
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime
    description: Optional[str] = None
    research_topic: Optional[str] = None
    status: str
    config: Optional[Dict[str, Any]] = None
    papers_count: Optional[int] = None
    queries_count: Optional[int] = None

    class Config:
        from_attributes = True


class SessionStats(BaseModel):
    """Session statistics."""

    total_papers: int
    total_queries: int
    data_sources: Dict[str, int]
    graph_stats: Dict[str, Any]
    vector_stats: Dict[str, Any]
