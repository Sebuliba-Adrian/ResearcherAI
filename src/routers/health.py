"""
Health check and system status endpoints.
"""
from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.config import settings
from src.db import get_db

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.

    Returns:
        System health status
    """
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "backends": {
            "graph": settings.graph_backend,
            "vector": settings.vector_backend,
            "kafka": "enabled" if settings.use_kafka else "disabled",
        },
    }


@router.get("/health/database")
async def database_health(db: Session = Depends(get_db)) -> Dict[str, str]:
    """
    Check database connectivity.

    Args:
        db: Database session

    Returns:
        Database health status
    """
    try:
        # Execute a simple query to check connectivity
        db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}


@router.get("/health/ready")
async def readiness_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.

    Args:
        db: Database session

    Returns:
        Readiness status
    """
    checks = {"database": False, "configuration": False}

    # Check database
    try:
        db.execute("SELECT 1")
        checks["database"] = True
    except Exception:
        pass

    # Check configuration
    try:
        if settings.google_api_key:
            checks["configuration"] = True
    except Exception:
        pass

    is_ready = all(checks.values())

    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
    }


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """
    Kubernetes liveness probe endpoint.

    Returns:
        Liveness status
    """
    return {"status": "alive"}
