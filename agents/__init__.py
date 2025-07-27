"""
Multi-Agent Research System
============================

Production-ready multi-agent RAG system for autonomous research.

Agents:
- DataCollectorAgent: Autonomous data collection from 7 sources
- KnowledgeGraphAgent: Graph database management (Neo4j)
- VectorAgent: Vector search and embeddings (Qdrant)
- ReasoningAgent: Complex reasoning with conversation memory
- OrchestratorAgent: Multi-agent coordination and session management
- SchedulerAgent: Automated background data collection
"""

from .data_agent import DataCollectorAgent
from .graph_agent import KnowledgeGraphAgent
from .vector_agent import VectorAgent
from .reasoner_agent import ReasoningAgent
from .orchestrator_agent import OrchestratorAgent
from .scheduler_agent import SchedulerAgent

__all__ = [
    "DataCollectorAgent",
    "KnowledgeGraphAgent",
    "VectorAgent",
    "ReasoningAgent",
    "OrchestratorAgent",
    "SchedulerAgent",
]

__version__ = "2.0.0"
