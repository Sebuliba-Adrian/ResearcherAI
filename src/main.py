"""
FastAPI application entry point for ResearcherAI.
"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.config import settings
from src.db import init_db
from src.middleware import (
    LoggingMiddleware,
    ResearcherAIException,
    generic_exception_handler,
    researcherai_exception_handler,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("application_starting", version=settings.app_version, environment=settings.environment)

    # Initialize database
    try:
        init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.error("database_initialization_failed", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-Grade Multi-Agent RAG System for Research Paper Analysis",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

# Add custom middleware
app.add_middleware(LoggingMiddleware)

# Add exception handlers
app.add_exception_handler(ResearcherAIException, researcherai_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Import and include routers
from src.routers import health, sessions  # noqa: E402

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", include_in_schema=False)
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers if not settings.api_reload else 1,
        log_level=settings.log_level.lower(),
    )
