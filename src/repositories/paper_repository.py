"""
Repository for paper data access.
"""
from typing import List, Optional

from sqlalchemy.orm import Session

from src.models.paper import Paper


class PaperRepository:
    """Repository for paper operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, paper_data: dict, session_id: int) -> Paper:
        """Create a new paper."""
        db_paper = Paper(
            session_id=session_id,
            paper_id=paper_data.get("paper_id"),
            doi=paper_data.get("doi"),
            arxiv_id=paper_data.get("arxiv_id"),
            title=paper_data.get("title"),
            authors=paper_data.get("authors", []),
            abstract=paper_data.get("abstract"),
            publication_date=paper_data.get("publication_date"),
            source=paper_data.get("source"),
            url=paper_data.get("url"),
            keywords=paper_data.get("keywords", []),
            categories=paper_data.get("categories", []),
            metadata=paper_data.get("metadata", {}),
        )
        self.db.add(db_paper)
        self.db.commit()
        self.db.refresh(db_paper)
        return db_paper

    def get_by_id(self, paper_id: int) -> Optional[Paper]:
        """Get paper by database ID."""
        return self.db.query(Paper).filter(Paper.id == paper_id).first()

    def get_by_paper_id(self, paper_id: str) -> Optional[Paper]:
        """Get paper by paper ID string."""
        return self.db.query(Paper).filter(Paper.paper_id == paper_id).first()

    def get_by_session(
        self, session_id: int, skip: int = 0, limit: int = 100
    ) -> List[Paper]:
        """Get all papers for a session."""
        return (
            self.db.query(Paper)
            .filter(Paper.session_id == session_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_source(
        self, session_id: int, source: str, skip: int = 0, limit: int = 100
    ) -> List[Paper]:
        """Get papers by source."""
        return (
            self.db.query(Paper)
            .filter(Paper.session_id == session_id, Paper.source == source)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def search_by_title(
        self, session_id: int, title_query: str, skip: int = 0, limit: int = 100
    ) -> List[Paper]:
        """Search papers by title."""
        return (
            self.db.query(Paper)
            .filter(Paper.session_id == session_id, Paper.title.ilike(f"%{title_query}%"))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count_by_session(self, session_id: int) -> int:
        """Count papers in a session."""
        return self.db.query(Paper).filter(Paper.session_id == session_id).count()

    def count_by_source(self, session_id: int, source: str) -> int:
        """Count papers from a specific source."""
        return (
            self.db.query(Paper)
            .filter(Paper.session_id == session_id, Paper.source == source)
            .count()
        )

    def delete(self, paper_id: int) -> bool:
        """Delete a paper."""
        db_paper = self.get_by_id(paper_id)
        if not db_paper:
            return False

        self.db.delete(db_paper)
        self.db.commit()
        return True
