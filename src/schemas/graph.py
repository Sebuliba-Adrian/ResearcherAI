"""
Schemas for knowledge graph operations.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GraphStatsResponse(BaseModel):
    """Knowledge graph statistics."""

    backend: str
    total_nodes: int
    total_relationships: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]


class GraphQueryRequest(BaseModel):
    """Request for graph query."""

    query: str = Field(..., min_length=3)
    params: Optional[Dict[str, Any]] = None
    limit: int = Field(default=100, ge=1, le=1000)


class GraphNode(BaseModel):
    """Single graph node."""

    id: str
    labels: List[str]
    properties: Dict[str, Any]


class GraphRelationship(BaseModel):
    """Single graph relationship."""

    id: str
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, Any]


class GraphQueryResponse(BaseModel):
    """Response from graph query."""

    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    execution_time: float = Field(..., ge=0)
