"""
Paper model for storing research papers.
"""
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.db.database import Base


class Paper(Base):
    """Research paper model."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)

    # Paper identifiers
    paper_id = Column(String(255), unique=True, index=True, nullable=False)
    doi = Column(String(255), nullable=True, index=True)
    arxiv_id = Column(String(50), nullable=True, index=True)

    # Paper metadata
    title = Column(Text, nullable=False)
    authors = Column(JSON, nullable=True)  # List of author names
    abstract = Column(Text, nullable=True)
    publication_date = Column(DateTime, nullable=True)
    source = Column(String(100), nullable=True)  # arxiv, semantic_scholar, etc.
    url = Column(Text, nullable=True)

    # Additional metadata
    keywords = Column(JSON, nullable=True)  # List of keywords
    categories = Column(JSON, nullable=True)  # List of categories
    extra_metadata = Column(JSON, nullable=True)  # Additional flexible metadata

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    session = relationship("Session", back_populates="papers")

    def __repr__(self) -> str:
        return f"<Paper(id={self.id}, paper_id={self.paper_id}, title={self.title[:50]})>"
