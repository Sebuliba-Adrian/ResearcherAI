"""
Repository for query data access.
"""
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.models.query import Query


class QueryRepository:
    """Repository for query operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, query_data: dict, session_id: int) -> Query:
        """Create a new query."""
        db_query = Query(
            session_id=session_id,
            query_text=query_data.get("query_text"),
            response_text=query_data.get("response_text"),
            query_type=query_data.get("query_type"),
            context=query_data.get("context"),
            processing_time_ms=query_data.get("processing_time_ms"),
            tokens_used=query_data.get("tokens_used"),
            cost=query_data.get("cost"),
            results_count=query_data.get("results_count"),
            results=query_data.get("results"),
        )
        self.db.add(db_query)
        self.db.commit()
        self.db.refresh(db_query)
        return db_query

    def get_by_id(self, query_id: int) -> Optional[Query]:
        """Get query by ID."""
        return self.db.query(Query).filter(Query.id == query_id).first()

    def get_by_session(
        self, session_id: int, skip: int = 0, limit: int = 100
    ) -> List[Query]:
        """Get all queries for a session."""
        return (
            self.db.query(Query)
            .filter(Query.session_id == session_id)
            .order_by(desc(Query.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_type(
        self, session_id: int, query_type: str, skip: int = 0, limit: int = 100
    ) -> List[Query]:
        """Get queries by type."""
        return (
            self.db.query(Query)
            .filter(Query.session_id == session_id, Query.query_type == query_type)
            .order_by(desc(Query.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count_by_session(self, session_id: int) -> int:
        """Count queries in a session."""
        return self.db.query(Query).filter(Query.session_id == session_id).count()

    def get_total_cost(self, session_id: int) -> float:
        """Get total cost for a session."""
        from sqlalchemy import func

        result = (
            self.db.query(func.sum(Query.cost))
            .filter(Query.session_id == session_id)
            .scalar()
        )
        return result or 0.0

    def get_total_tokens(self, session_id: int) -> int:
        """Get total tokens used for a session."""
        from sqlalchemy import func

        result = (
            self.db.query(func.sum(Query.tokens_used))
            .filter(Query.session_id == session_id)
            .scalar()
        )
        return result or 0

    def delete(self, query_id: int) -> bool:
        """Delete a query."""
        db_query = self.get_by_id(query_id)
        if not db_query:
            return False

        self.db.delete(db_query)
        self.db.commit()
        return True
