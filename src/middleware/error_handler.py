"""
Global exception handlers for FastAPI.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse

import structlog

logger = structlog.get_logger(__name__)


class ResearcherAIException(Exception):
    """Base exception for ResearcherAI application."""

    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class SessionNotFoundException(ResearcherAIException):
    """Raised when a session is not found."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session not found: {session_id}",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"session_id": session_id},
        )


class PaperNotFoundException(ResearcherAIException):
    """Raised when a paper is not found."""

    def __init__(self, paper_id: str):
        super().__init__(
            message=f"Paper not found: {paper_id}",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"paper_id": paper_id},
        )


class ValidationException(ResearcherAIException):
    """Raised when validation fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message, status_code=status.HTTP_400_BAD_REQUEST, details=details
        )


class DataCollectionException(ResearcherAIException):
    """Raised when data collection fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


async def researcherai_exception_handler(
    request: Request, exc: ResearcherAIException
) -> JSONResponse:
    """
    Handle ResearcherAI custom exceptions.

    Args:
        request: FastAPI request
        exc: ResearcherAI exception

    Returns:
        JSON error response
    """
    logger.error(
        "application_error",
        error=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "path": str(request.url.path),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle generic exceptions.

    Args:
        request: FastAPI request
        exc: Generic exception

    Returns:
        JSON error response
    """
    logger.exception("unexpected_error", error=str(exc), path=request.url.path)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)},
            "path": str(request.url.path),
        },
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle Pydantic validation exceptions.

    Args:
        request: FastAPI request
        exc: Validation exception

    Returns:
        JSON error response
    """
    logger.warning("validation_error", error=str(exc), path=request.url.path)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": {"errors": str(exc)},
            "path": str(request.url.path),
        },
    )
