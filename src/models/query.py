"""
Query model for tracking user queries and responses.
"""
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from src.db.database import Base


class Query(Base):
    """User query model."""

    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)

    # Query content
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=True)

    # Query metadata
    query_type = Column(Text, nullable=True)  # search, summarize, reasoning, etc.
    context = Column(JSON, nullable=True)  # Additional context for the query

    # Performance metrics
    processing_time_ms = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)

    # Results
    results_count = Column(Integer, nullable=True)
    results = Column(JSON, nullable=True)  # Store query results

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    session = relationship("Session", back_populates="queries")

    def __repr__(self) -> str:
        return f"<Query(id={self.id}, query_text={self.query_text[:50]}, type={self.query_type})>"
