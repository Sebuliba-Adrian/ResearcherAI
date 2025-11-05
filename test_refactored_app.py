"""
Simple test to verify the refactored application can start.
"""
import os
import sys

# Set minimal environment variables for testing
os.environ["GOOGLE_API_KEY"] = "test_key_for_validation"
os.environ["USE_NEO4J"] = "false"
os.environ["USE_QDRANT"] = "false"
os.environ["USE_KAFKA"] = "false"

try:
    print("=" * 60)
    print("Testing Refactored Application Structure")
    print("=" * 60)

    # Test 1: Import configuration
    print("\n[1/8] Testing configuration import...")
    from src.config import settings
    print(f"✓ Configuration loaded: {settings.app_name} v{settings.app_version}")
    print(f"  - Environment: {settings.environment}")
    print(f"  - Graph backend: {settings.graph_backend}")
    print(f"  - Vector backend: {settings.vector_backend}")

    # Test 2: Import database layer
    print("\n[2/8] Testing database layer import...")
    from src.db import Base, get_db, init_db
    print("✓ Database layer imported successfully")

    # Test 3: Import models
    print("\n[3/8] Testing SQLAlchemy models import...")
    from src.models import Paper, Query, Session
    print("✓ Models imported successfully")
    print(f"  - Session: {Session.__tablename__}")
    print(f"  - Paper: {Paper.__tablename__}")
    print(f"  - Query: {Query.__tablename__}")

    # Test 4: Import schemas
    print("\n[4/8] Testing Pydantic schemas import...")
    from src.schemas import (
        DataCollectionRequest,
        SessionCreate,
        SessionResponse,
        SearchRequest,
    )
    print("✓ Schemas imported successfully")

    # Test 5: Import repositories
    print("\n[5/8] Testing repositories import...")
    from src.repositories import PaperRepository, QueryRepository, SessionRepository
    print("✓ Repositories imported successfully")

    # Test 6: Import middleware
    print("\n[6/8] Testing middleware import...")
    from src.middleware import LoggingMiddleware, ResearcherAIException
    print("✓ Middleware imported successfully")

    # Test 7: Import routers
    print("\n[7/8] Testing routers import...")
    from src.routers import health, sessions
    print("✓ Routers imported successfully")

    # Test 8: Import FastAPI app
    print("\n[8/8] Testing FastAPI application import...")
    from src.main import app
    print("✓ FastAPI application created successfully")
    print(f"  - Title: {app.title}")
    print(f"  - Version: {app.version}")
    print(f"  - Routes: {len(app.routes)} registered")

    # Print route summary
    print("\n" + "=" * 60)
    print("Registered API Routes:")
    print("=" * 60)
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(sorted(route.methods))
            print(f"  {methods:20s} {route.path}")

    print("\n" + "=" * 60)
    print("✓ All Tests Passed!")
    print("=" * 60)
    print("\nThe refactored application structure is valid and ready for use.")
    print("\nTo start the application:")
    print("  python -m src.main")
    print("  OR")
    print("  uvicorn src.main:app --reload")

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
