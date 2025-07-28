"""
API Gateway - RESTful API for ResearcherAI
==========================================

FastAPI-based gateway with authentication, rate limiting, and monitoring
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import wraps

from fastapi import FastAPI, HTTPException, Depends, Header, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import yaml

# Import agents
from agents.orchestrator_agent import OrchestratorAgent
from agents.critic_agent import CriticAgent

# Import PDF parser
from utils.pdf_parser import parse_pdf_upload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

API_VERSION = "v1"
API_TITLE = "ResearcherAI API"
API_DESCRIPTION = """
🚀 **ResearcherAI Multi-Agent RAG System API**

Intelligent research paper collection, knowledge graph construction, and Q&A.

## Features
- 📚 Autonomous data collection from 7 sources
- 🕸️ Knowledge graph construction (Neo4j/NetworkX)
- 🔍 Semantic search (Qdrant/FAISS)
- 🤖 Multi-agent reasoning with conversation memory
- 🎯 Self-evaluation and quality assurance
- 💾 Session persistence

## Authentication
Use API key in header: `X-API-Key: your-api-key-here`
"""

# Load API keys from environment
VALID_API_KEYS = set(os.getenv("API_KEYS", "demo-key-123,test-key-456").split(","))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}

# ============================================================================
# Pydantic Models
# ============================================================================

class CollectRequest(BaseModel):
    """Request model for data collection"""
    query: str = Field(..., description="Research query", example="transformer neural networks")
    max_per_source: int = Field(10, description="Max papers per source", ge=1, le=50)
    session_name: Optional[str] = Field(None, description="Session name for persistence")

class AskRequest(BaseModel):
    """Request model for Q&A"""
    question: str = Field(..., description="Research question", example="What are transformers?")
    session_name: Optional[str] = Field("default", description="Session name")
    use_critic: bool = Field(True, description="Enable self-evaluation")

class SessionRequest(BaseModel):
    """Request model for session operations"""
    session_name: str = Field(..., description="Session name", example="my_research_session")

class SummarizePaperRequest(BaseModel):
    """Request model for paper summarization"""
    paper: Dict = Field(..., description="Paper metadata (title, abstract, authors, etc.)")
    style: str = Field("executive", description="Summary style: executive, technical, abstract, bullet_points, narrative")
    length: str = Field("medium", description="Summary length: brief, short, medium, detailed, comprehensive")

class SummarizeCollectionRequest(BaseModel):
    """Request model for collection summarization"""
    session_name: Optional[str] = Field("default", description="Session name to get papers from")
    papers: Optional[List[Dict]] = Field(None, description="Papers to summarize (uses session papers if not provided)")
    style: str = Field("executive", description="Summary style")
    focus: str = Field("research trends", description="What to focus on")

class SummarizeConversationRequest(BaseModel):
    """Request model for conversation summarization"""
    session_name: str = Field("default", description="Session name to get conversation from")
    style: str = Field("bullet_points", description="Summary style")

class ComparePapersRequest(BaseModel):
    """Request model for paper comparison"""
    papers: List[Dict] = Field(..., description="2-5 papers to compare")
    comparison_aspects: Optional[List[str]] = Field(None, description="What to compare (methodology, results, etc.)")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str = API_VERSION
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    agents: Dict[str, str] = {}

class CollectResponse(BaseModel):
    """Response model for data collection"""
    success: bool
    papers_collected: int
    graph_stats: Dict
    vector_stats: Dict
    critic_evaluation: Optional[Dict] = None
    session_name: str

class AskResponse(BaseModel):
    """Response model for Q&A"""
    success: bool
    question: str
    answer: str
    sources: List[Dict] = []
    graph_insights: Dict = {}
    critic_evaluation: Optional[Dict] = None
    session_name: str

class UploadPDFResponse(BaseModel):
    """Response model for PDF upload"""
    success: bool
    paper: Dict
    graph_stats: Dict
    vector_stats: Dict
    session_name: str
    message: str

# ============================================================================
# Authentication & Rate Limiting
# ============================================================================

security = HTTPBearer()

def verify_api_key(api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Verify API key"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

def rate_limit(api_key: str) -> None:
    """Simple rate limiting (use Redis in production)"""
    now = datetime.now()
    minute_key = now.strftime("%Y-%m-%d %H:%M")
    key = f"{api_key}:{minute_key}"

    if key not in rate_limit_storage:
        rate_limit_storage[key] = 0

    rate_limit_storage[key] += 1

    if rate_limit_storage[key] > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT_PER_MINUTE} requests per minute"
        )

    # Cleanup old keys (older than 5 minutes)
    old_keys = [k for k in rate_limit_storage if k.split(":")[1] < (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M")]
    for old_key in old_keys:
        del rate_limit_storage[old_key]

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=f"/{API_VERSION}/docs",
    redoc_url=f"/{API_VERSION}/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agents (initialized on startup)
orchestrator: Optional[OrchestratorAgent] = None
critic: Optional[CriticAgent] = None

# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global orchestrator, critic

    logger.info("🚀 Starting ResearcherAI API Gateway...")

    # Load configuration
    config = {}
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)

    # Initialize agents
    try:
        orchestrator = OrchestratorAgent(session_name="api_default", config=config)
        critic = CriticAgent(config=config.get("agents", {}).get("critic_agent", {}))
        logger.info("✅ All agents initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agents: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down ResearcherAI API Gateway...")
    # Add cleanup logic here if needed

# ============================================================================
# API Endpoints
# ============================================================================

@app.get(f"/{API_VERSION}/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint

    Returns system status and agent availability
    """
    return HealthResponse(
        status="healthy",
        agents={
            "orchestrator": "ready" if orchestrator else "not_initialized",
            "critic": "ready" if critic else "not_initialized",
            "data_collector": "ready",
            "knowledge_graph": "ready",
            "vector_search": "ready",
            "reasoning": "ready"
        }
    )

@app.post(f"/{API_VERSION}/collect", response_model=CollectResponse, tags=["Data Collection"])
async def collect_papers(
    request: CollectRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Collect research papers from multiple sources

    Automatically:
    - Searches 7 data sources (arXiv, Semantic Scholar, PubMed, etc.)
    - Builds knowledge graph
    - Creates vector embeddings
    - Evaluates collection quality (if critic enabled)

    **Rate Limit:** 10 requests per minute
    """
    rate_limit(api_key)

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Set session if provided
        if request.session_name:
            orchestrator.session_name = request.session_name

        # Collect papers
        logger.info(f"Collecting papers for query: {request.query}")
        result = orchestrator.collect_data(request.query, request.max_per_source)

        # Get papers for critic evaluation
        papers = orchestrator.data_collector.get_last_collection()

        # Evaluate with critic
        critic_eval = None
        if critic and papers:
            critic_eval = critic.evaluate_paper_collection(papers)
            logger.info(f"Critic evaluation: {critic_eval['overall_score']:.2f}")

        return CollectResponse(
            success=True,
            papers_collected=result.get("papers_collected", 0),
            graph_stats=result.get("graph_stats", {}),
            vector_stats=result.get("vector_stats", {}),
            critic_evaluation=critic_eval,
            session_name=orchestrator.session_name
        )

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"/{API_VERSION}/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(
    request: AskRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Ask a question about collected research papers

    Uses:
    - Knowledge graph for entity relationships
    - Vector search for semantic similarity
    - LLM reasoning with conversation memory
    - Self-evaluation for answer quality

    **Rate Limit:** 10 requests per minute
    """
    rate_limit(api_key)

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Set session if provided
        if request.session_name:
            orchestrator.session_name = request.session_name

        # Ask question
        logger.info(f"Processing question: {request.question}")
        result = orchestrator.ask_detailed(request.question)

        # Evaluate answer with critic
        critic_eval = None
        if request.use_critic and critic:
            context = {
                "papers_used": result.get("papers_used", []),
                "graph_data": result.get("graph_insights", {})
            }
            critic_eval = critic.evaluate_answer(
                request.question,
                result.get("answer", ""),
                context
            )
            logger.info(f"Answer evaluation: {critic_eval['overall_score']:.2f}")

        return AskResponse(
            success=True,
            question=request.question,
            answer=result.get("answer", ""),
            sources=result.get("papers_used", []),
            graph_insights=result.get("graph_insights", {}),
            critic_evaluation=critic_eval,
            session_name=orchestrator.session_name
        )

    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"/{API_VERSION}/sessions/{{session_name}}", tags=["Sessions"])
async def get_session_info(
    session_name: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get information about a specific session

    Returns:
    - Session metadata
    - Papers collected
    - Conversation history
    - Graph statistics
    """
    rate_limit(api_key)

    try:
        # Load session
        temp_orchestrator = OrchestratorAgent(session_name=session_name)

        return {
            "session_name": session_name,
            "metadata": temp_orchestrator.metadata,
            "statistics": temp_orchestrator.get_stats()
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")

@app.delete(f"/{API_VERSION}/sessions/{{session_name}}", tags=["Sessions"])
async def delete_session(
    session_name: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a session and all its data"""
    rate_limit(api_key)

    try:
        session_file = f"./volumes/sessions/{session_name}.pkl"
        if os.path.exists(session_file):
            os.remove(session_file)
            return {"success": True, "message": f"Session '{session_name}' deleted"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"/{API_VERSION}/critic/report", tags=["Quality Assurance"])
async def get_critic_report(api_key: str = Depends(verify_api_key)):
    """
    Get overall quality assurance report from CriticAgent

    Shows:
    - Total evaluations performed
    - Average quality scores
    - Common issues found
    - Recommendations for improvement
    """
    rate_limit(api_key)

    if not critic:
        raise HTTPException(status_code=503, detail="Critic agent not initialized")

    return critic.get_overall_quality_report()

@app.get(f"/{API_VERSION}/stats", tags=["System"])
async def get_system_stats(api_key: str = Depends(verify_api_key)):
    """
    Get overall system statistics

    Returns metrics for all components
    """
    rate_limit(api_key)

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    stats = orchestrator.get_stats()
    critic_report = critic.get_overall_quality_report() if critic else {}

    return {
        "system": stats,
        "quality_assurance": critic_report,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Summarization Endpoints
# ============================================================================

@app.post(f"/{API_VERSION}/summarize/paper", tags=["Summarization"])
async def summarize_paper(
    request: SummarizePaperRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Summarize a single research paper

    Supports multiple styles:
    - **executive**: Brief, actionable insights for decision-makers
    - **technical**: Detailed with technical terminology preserved
    - **abstract**: Academic abstract format
    - **bullet_points**: Key points as clear bullet list
    - **narrative**: Engaging story-like flow

    Supports multiple lengths:
    - **brief**: 2-3 sentences (50 words max)
    - **short**: 1 paragraph (100-150 words)
    - **medium**: 2-3 paragraphs (200-300 words)
    - **detailed**: 3-5 paragraphs (400-600 words)
    - **comprehensive**: Full analysis (800+ words)

    **Rate Limit:** 10 requests per minute
    """
    rate_limit(api_key)

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        summary = orchestrator.summarize_paper(
            request.paper,
            style=request.style,
            length=request.length
        )

        return {
            "success": True,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Paper summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"/{API_VERSION}/summarize/collection", tags=["Summarization"])
async def summarize_collection(
    request: SummarizeCollectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Summarize a collection of research papers

    Automatically identifies:
    - Key themes across papers
    - Research trends
    - Knowledge gaps
    - Top papers and why they're significant

    If papers not provided, uses papers from the specified session.

    **Rate Limit:** 10 requests per minute
    """
    rate_limit(api_key)

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Switch to specified session if needed
        if request.session_name and request.session_name != orchestrator.session_name:
            orchestrator.session_name = request.session_name
            orchestrator.load_session()

        summary = orchestrator.summarize_collection(
            papers=request.papers,
            style=request.style,
            focus=request.focus
        )

        if "error" in summary:
            raise HTTPException(status_code=400, detail=summary["error"])

        return {
            "success": True,
            "summary": summary,
            "session_name": orchestrator.session_name,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collection summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"/{API_VERSION}/summarize/conversation", tags=["Summarization"])
async def summarize_conversation(
    request: SummarizeConversationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Summarize a research conversation/session

    Returns:
    - Session summary (overview of the conversation)
    - Questions asked (list of all questions)
    - Key insights (main discoveries/learnings)
    - Topics covered (research areas discussed)

    **Rate Limit:** 10 requests per minute
    """
    rate_limit(api_key)

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Switch to specified session if needed
        if request.session_name != orchestrator.session_name:
            orchestrator.session_name = request.session_name
            orchestrator.load_session()

        summary = orchestrator.summarize_conversation(style=request.style)

        if "error" in summary:
            raise HTTPException(status_code=400, detail=summary["error"])

        return {
            "success": True,
            "summary": summary,
            "session_name": orchestrator.session_name,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"/{API_VERSION}/summarize/compare", tags=["Summarization"])
async def compare_papers(
    request: ComparePapersRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Compare multiple papers side-by-side

    Provides:
    - Comparison summary (how papers relate)
    - Similarities (common themes/methods)
    - Differences (contrasting approaches)
    - Strengths & weaknesses for each paper
    - Recommendation (which to read/use first)

    **Rate Limit:** 10 requests per minute
    """
    rate_limit(api_key)

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        comparison = orchestrator.compare_papers(
            request.papers,
            comparison_aspects=request.comparison_aspects
        )

        if "error" in comparison:
            raise HTTPException(status_code=400, detail=comparison["error"])

        return {
            "success": True,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PDF Upload Endpoint
# ============================================================================

@app.post(f"/{API_VERSION}/upload/pdf", response_model=UploadPDFResponse, tags=["Data Collection"])
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    session_name: Optional[str] = Header(None, alias="X-Session-Name"),
    api_key: str = Depends(verify_api_key)
):
    """
    Upload a local PDF research paper

    Automatically:
    - Extracts text from PDF using PyPDF2
    - Parses metadata (title, authors, abstract, keywords)
    - Inserts into knowledge graph (Neo4j)
    - Creates vector embeddings (Qdrant)
    - Makes paper searchable and queryable

    **File Requirements**:
    - Format: PDF only
    - Max size: 50MB
    - Content: Research papers with text (not scanned images)

    **Extracted Metadata**:
    - Title (from PDF metadata or first large text block)
    - Authors (from PDF metadata or heuristics)
    - Abstract (from "Abstract" section)
    - Keywords/Topics (from "Keywords" section or extracted)
    - Publication date (from PDF metadata or heuristics)

    **Rate Limit:** 10 requests per minute
    """
    rate_limit(api_key)

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    # Check file size (50MB limit)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    file_content = await file.read()

    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 50MB, got {len(file_content) / 1024 / 1024:.1f}MB"
        )

    try:
        logger.info(f"Processing PDF upload: {file.filename}")

        # Parse PDF
        paper = parse_pdf_upload(file_content, file.filename)
        logger.info(f"Extracted metadata - Title: {paper['title'][:50]}...")

        # Set session if provided
        if session_name:
            orchestrator.session_name = session_name

        # Insert into knowledge graph
        graph_stats = orchestrator.graph_agent.add_paper(paper)
        logger.info(f"Added to Neo4j: {graph_stats}")

        # Create vector embeddings
        vector_stats = orchestrator.vector_agent.add_documents([paper])
        logger.info(f"Added to Qdrant: {vector_stats}")

        # Update session metadata
        orchestrator.metadata["papers_collected"] = orchestrator.metadata.get("papers_collected", 0) + 1
        orchestrator.save_session()

        return UploadPDFResponse(
            success=True,
            paper={
                "id": paper["id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "source": paper["source"],
                "num_pages": paper.get("num_pages", 0)
            },
            graph_stats=graph_stats,
            vector_stats=vector_stats,
            session_name=orchestrator.session_name,
            message=f"Successfully uploaded and processed '{file.filename}'"
        )

    except ValueError as e:
        logger.error(f"PDF parsing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

    except Exception as e:
        logger.error(f"PDF upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_gateway:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
        log_level="info"
    )
