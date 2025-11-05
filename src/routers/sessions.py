"""
Session management endpoints.
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.db import get_db
from src.middleware import SessionNotFoundException
from src.repositories import PaperRepository, QueryRepository, SessionRepository
from src.schemas.session import SessionCreate, SessionResponse, SessionStats, SessionUpdate

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: SessionCreate, db: Session = Depends(get_db)
) -> SessionResponse:
    """
    Create a new research session.

    Args:
        session_data: Session creation data
        db: Database session

    Returns:
        Created session
    """
    repo = SessionRepository(db)
    db_session = repo.create(session_data)

    return SessionResponse(
        id=db_session.id,
        session_id=db_session.session_id,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at,
        last_accessed=db_session.last_accessed,
        description=db_session.description,
        research_topic=db_session.research_topic,
        status=db_session.status,
        config=db_session.config,
        papers_count=0,
        queries_count=0,
    )


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    skip: int = 0,
    limit: int = 100,
    status_filter: str = None,
    db: Session = Depends(get_db),
) -> List[SessionResponse]:
    """
    List all research sessions.

    Args:
        skip: Number of sessions to skip
        limit: Maximum number of sessions to return
        status_filter: Filter by session status
        db: Database session

    Returns:
        List of sessions
    """
    repo = SessionRepository(db)
    paper_repo = PaperRepository(db)
    query_repo = QueryRepository(db)

    sessions = repo.get_all(skip=skip, limit=limit, status=status_filter)

    return [
        SessionResponse(
            id=s.id,
            session_id=s.session_id,
            created_at=s.created_at,
            updated_at=s.updated_at,
            last_accessed=s.last_accessed,
            description=s.description,
            research_topic=s.research_topic,
            status=s.status,
            config=s.config,
            papers_count=paper_repo.count_by_session(s.id),
            queries_count=query_repo.count_by_session(s.id),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)) -> SessionResponse:
    """
    Get a specific session by ID.

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        Session details

    Raises:
        SessionNotFoundException: If session not found
    """
    repo = SessionRepository(db)
    paper_repo = PaperRepository(db)
    query_repo = QueryRepository(db)

    db_session = repo.get_by_session_id(session_id)
    if not db_session:
        raise SessionNotFoundException(session_id)

    # Update last accessed
    repo.update_last_accessed(db_session.id)

    return SessionResponse(
        id=db_session.id,
        session_id=db_session.session_id,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at,
        last_accessed=db_session.last_accessed,
        description=db_session.description,
        research_topic=db_session.research_topic,
        status=db_session.status,
        config=db_session.config,
        papers_count=paper_repo.count_by_session(db_session.id),
        queries_count=query_repo.count_by_session(db_session.id),
    )


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str, session_data: SessionUpdate, db: Session = Depends(get_db)
) -> SessionResponse:
    """
    Update a session.

    Args:
        session_id: Session ID
        session_data: Update data
        db: Database session

    Returns:
        Updated session

    Raises:
        SessionNotFoundException: If session not found
    """
    repo = SessionRepository(db)
    paper_repo = PaperRepository(db)
    query_repo = QueryRepository(db)

    # Get session by session_id first
    db_session = repo.get_by_session_id(session_id)
    if not db_session:
        raise SessionNotFoundException(session_id)

    # Update using database ID
    updated_session = repo.update(db_session.id, session_data)

    return SessionResponse(
        id=updated_session.id,
        session_id=updated_session.session_id,
        created_at=updated_session.created_at,
        updated_at=updated_session.updated_at,
        last_accessed=updated_session.last_accessed,
        description=updated_session.description,
        research_topic=updated_session.research_topic,
        status=updated_session.status,
        config=updated_session.config,
        papers_count=paper_repo.count_by_session(updated_session.id),
        queries_count=query_repo.count_by_session(updated_session.id),
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a session.

    Args:
        session_id: Session ID
        db: Database session

    Raises:
        SessionNotFoundException: If session not found
    """
    repo = SessionRepository(db)

    db_session = repo.get_by_session_id(session_id)
    if not db_session:
        raise SessionNotFoundException(session_id)

    repo.delete(db_session.id)


@router.get("/sessions/{session_id}/stats", response_model=SessionStats)
async def get_session_stats(session_id: str, db: Session = Depends(get_db)) -> SessionStats:
    """
    Get session statistics.

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        Session statistics

    Raises:
        SessionNotFoundException: If session not found
    """
    repo = SessionRepository(db)
    paper_repo = PaperRepository(db)
    query_repo = QueryRepository(db)

    db_session = repo.get_by_session_id(session_id)
    if not db_session:
        raise SessionNotFoundException(session_id)

    # Get paper stats by source
    papers = paper_repo.get_by_session(db_session.id, limit=10000)
    data_sources = {}
    for paper in papers:
        source = paper.source or "unknown"
        data_sources[source] = data_sources.get(source, 0) + 1

    return SessionStats(
        total_papers=paper_repo.count_by_session(db_session.id),
        total_queries=query_repo.count_by_session(db_session.id),
        data_sources=data_sources,
        graph_stats={"backend": "to_be_integrated"},
        vector_stats={"backend": "to_be_integrated"},
    )
