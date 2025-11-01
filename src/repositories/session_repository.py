"""
Repository for session data access.
"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.models.session import Session as SessionModel
from src.schemas.session import SessionCreate, SessionUpdate


class SessionRepository:
    """Repository for session operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, session_data: SessionCreate) -> SessionModel:
        """Create a new session."""
        import uuid

        db_session = SessionModel(
            session_id=str(uuid.uuid4()),
            description=session_data.description,
            research_topic=session_data.research_topic,
            config=session_data.config,
            status="active",
        )
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        return db_session

    def get_by_id(self, session_id: int) -> Optional[SessionModel]:
        """Get session by database ID."""
        return self.db.query(SessionModel).filter(SessionModel.id == session_id).first()

    def get_by_session_id(self, session_id: str) -> Optional[SessionModel]:
        """Get session by session ID string."""
        return (
            self.db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
        )

    def get_all(
        self, skip: int = 0, limit: int = 100, status: Optional[str] = None
    ) -> List[SessionModel]:
        """Get all sessions with optional filtering."""
        query = self.db.query(SessionModel)

        if status:
            query = query.filter(SessionModel.status == status)

        return query.order_by(desc(SessionModel.updated_at)).offset(skip).limit(limit).all()

    def update(self, session_id: int, session_data: SessionUpdate) -> Optional[SessionModel]:
        """Update a session."""
        db_session = self.get_by_id(session_id)
        if not db_session:
            return None

        update_data = session_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_session, field, value)

        db_session.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(db_session)
        return db_session

    def update_last_accessed(self, session_id: int) -> Optional[SessionModel]:
        """Update the last accessed timestamp."""
        db_session = self.get_by_id(session_id)
        if not db_session:
            return None

        db_session.last_accessed = datetime.utcnow()
        self.db.commit()
        self.db.refresh(db_session)
        return db_session

    def delete(self, session_id: int) -> bool:
        """Delete a session."""
        db_session = self.get_by_id(session_id)
        if not db_session:
            return False

        self.db.delete(db_session)
        self.db.commit()
        return True

    def count(self, status: Optional[str] = None) -> int:
        """Count sessions with optional status filter."""
        query = self.db.query(SessionModel)
        if status:
            query = query.filter(SessionModel.status == status)
        return query.count()
