"""
OrchestratorAgent - Multi-Agent Coordination & Session Management
================================================================

Coordinates all agents and manages research sessions with persistence
"""

import os
import pickle
import logging
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """Multi-agent coordination with session management"""

    def __init__(self, session_name: str = "default", config: Optional[Dict] = None):
        """
        Initialize orchestrator with all agents

        config = {
            "graph_db": {...},
            "vector_db": {...},
            "agents": {...},
            "llm": {...}
        }
        """
        self.session_name = session_name
        self.config = config or {}
        self.sessions_dir = Path("./volumes/sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing OrchestratorAgent for session '{session_name}'...")

        # Import agents
        from .data_agent import DataCollectorAgent
        from .graph_agent import KnowledgeGraphAgent
        from .vector_agent import VectorAgent
        from .reasoner_agent import ReasoningAgent

        # Initialize agents
        self.data_collector = DataCollectorAgent()

        graph_config = self.config.get("graph_db", {"type": "networkx"})
        self.graph_agent = KnowledgeGraphAgent(config=graph_config)

        vector_config = self.config.get("vector_db", {"type": "faiss"})
        self.vector_agent = VectorAgent(config=vector_config)

        agent_config = self.config.get("agents", {})
        self.reasoning_agent = ReasoningAgent(
            self.graph_agent,
            self.vector_agent,
            config=agent_config.get("reasoning_agent", {})
        )

        # Session metadata
        self.metadata = {
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "papers_collected": 0,
            "conversations": 0
        }

        # Try to load existing session
        self.load_session()

        logger.info("All agents initialized successfully")

    def collect_data(self, query: str, max_per_source: int = 10) -> Dict:
        """Autonomous data collection from all sources"""
        logger.info(f"Starting autonomous data collection: '{query}'")

        # Collect from all sources
        papers = self.data_collector.collect_all(query, max_per_source)

        if not papers:
            logger.warning("No papers collected")
            return {"papers_collected": 0}

        # Process with graph agent
        graph_stats = self.graph_agent.process_papers(papers)

        # Process with vector agent
        vector_stats = self.vector_agent.process_papers(papers)

        # Update metadata
        self.metadata["papers_collected"] += len(papers)
        self.metadata["last_modified"] = datetime.now().isoformat()

        # Auto-save session
        self.save_session()

        return {
            "papers_collected": len(papers),
            "graph_stats": graph_stats,
            "vector_stats": vector_stats,
            "sources": self.data_collector.get_stats()
        }

    def ask(self, query: str) -> str:
        """Ask a research question"""
        logger.info(f"Processing query: '{query}'")

        # Synthesize answer using reasoning agent
        answer = self.reasoning_agent.synthesize_answer(query)

        # Update metadata
        self.metadata["conversations"] += 1
        self.metadata["last_modified"] = datetime.now().isoformat()

        # Auto-save session
        self.save_session()

        return answer

    def save_session(self) -> bool:
        """Save current session state"""
        try:
            session_path = self.sessions_dir / f"{self.session_name}.pkl"

            # Build state
            state = {
                "session_name": self.session_name,
                "metadata": self.metadata,
                "conversation_history": self.reasoning_agent.get_history(),
                "data_collector_stats": self.data_collector.get_stats(),
                "timestamp": datetime.now().isoformat()
            }

            # Add backend-specific state
            if self.graph_agent.db_type == "networkx":
                state["graph_nodes"] = list(self.graph_agent.G.nodes(data=True))
                state["graph_edges"] = list(self.graph_agent.G.edges(data=True))

            if self.vector_agent.db_type == "faiss":
                state["vector_chunks"] = self.vector_agent.chunks
                # Note: FAISS index is not serializable with pickle
                # For production, use Qdrant which persists automatically

            # Save to file
            with open(session_path, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Session '{self.session_name}' saved to {session_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_session(self) -> bool:
        """Load existing session"""
        try:
            session_path = self.sessions_dir / f"{self.session_name}.pkl"

            if not session_path.exists():
                logger.info(f"No existing session '{self.session_name}', starting fresh")
                return False

            # Load state
            with open(session_path, "rb") as f:
                state = pickle.load(f)

            # Restore metadata
            self.metadata = state.get("metadata", self.metadata)

            # Restore conversation history
            self.reasoning_agent.conversation_history = state.get("conversation_history", [])

            # Restore data collector stats
            if "data_collector_stats" in state:
                self.data_collector.collection_stats = state["data_collector_stats"]

            # Restore graph (NetworkX only)
            if self.graph_agent.db_type == "networkx" and "graph_nodes" in state:
                self.graph_agent.G.add_nodes_from(state["graph_nodes"])
                self.graph_agent.G.add_edges_from(state["graph_edges"])

            # Restore vector chunks (FAISS only)
            if self.vector_agent.db_type == "faiss" and "vector_chunks" in state:
                self.vector_agent.chunks = state["vector_chunks"]
                self.vector_agent.chunk_texts = [c["text"] for c in self.vector_agent.chunks]

            logger.info(f"Session '{self.session_name}' loaded successfully")
            logger.info(f"  Papers: {self.metadata['papers_collected']}")
            logger.info(f"  Conversations: {len(self.reasoning_agent.conversation_history)}")

            if self.graph_agent.db_type == "networkx":
                logger.info(f"  Graph nodes: {len(self.graph_agent.G.nodes())}")

            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    def switch_session(self, new_session_name: str) -> bool:
        """Switch to a different session"""
        logger.info(f"Switching from '{self.session_name}' to '{new_session_name}'...")

        # Save current session
        self.save_session()

        # Update session name
        old_session = self.session_name
        self.session_name = new_session_name

        # Reset agents (keep config)
        from .graph_agent import KnowledgeGraphAgent
        from .vector_agent import VectorAgent
        from .reasoner_agent import ReasoningAgent

        graph_config = self.config.get("graph_db", {"type": "networkx"})
        self.graph_agent = KnowledgeGraphAgent(config=graph_config)

        vector_config = self.config.get("vector_db", {"type": "faiss"})
        self.vector_agent = VectorAgent(config=vector_config)

        agent_config = self.config.get("agents", {})
        self.reasoning_agent = ReasoningAgent(
            self.graph_agent,
            self.vector_agent,
            config=agent_config.get("reasoning_agent", {})
        )

        # Reset metadata
        self.metadata = {
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "papers_collected": 0,
            "conversations": 0
        }

        # Try to load new session
        success = self.load_session()

        logger.info(f"Switched to session '{new_session_name}'")
        return success

    def list_sessions(self) -> List[Dict]:
        """List all available sessions"""
        sessions = []

        for session_file in self.sessions_dir.glob("*.pkl"):
            try:
                with open(session_file, "rb") as f:
                    state = pickle.load(f)

                sessions.append({
                    "name": state.get("session_name", session_file.stem),
                    "created": state.get("metadata", {}).get("created", "Unknown"),
                    "last_modified": state.get("metadata", {}).get("last_modified", "Unknown"),
                    "papers_collected": state.get("metadata", {}).get("papers_collected", 0),
                    "conversations": len(state.get("conversation_history", []))
                })
            except Exception as e:
                logger.error(f"Failed to read session {session_file}: {e}")

        return sessions

    def get_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            "session": self.session_name,
            "metadata": self.metadata,
            "graph": self.graph_agent.get_stats(),
            "vector": self.vector_agent.get_stats(),
            "reasoning": self.reasoning_agent.get_stats(),
            "data_collector": self.data_collector.get_stats()
        }

        return stats

    def visualize_graph(self, filename: str = "knowledge_graph.html"):
        """Generate graph visualization"""
        self.graph_agent.visualize(filename)

    def close(self):
        """Clean up and close all connections"""
        logger.info("Closing orchestrator and all agents...")

        # Save session one last time
        self.save_session()

        # Close database connections
        self.graph_agent.close()
        self.vector_agent.close()

        logger.info("Orchestrator closed successfully")
