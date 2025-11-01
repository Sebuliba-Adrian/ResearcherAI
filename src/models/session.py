"""
Session model for tracking user research sessions.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship

from src.db.database import Base


class Session(Base):
    """Research session model."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)

    # Session metadata
    description = Column(Text, nullable=True)
    research_topic = Column(String(500), nullable=True)
    status = Column(String(50), default="active")  # active, archived, deleted

    # Configuration
    config = Column(JSON, nullable=True)  # Store session-specific configuration

    # Relationships
    papers = relationship("Paper", back_populates="session", cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, session_id={self.session_id}, topic={self.research_topic})>"
