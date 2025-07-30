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
        from .summarization_agent import SummarizationAgent

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

        # Initialize summarization agent
        self.summarization_agent = SummarizationAgent(
            config=agent_config.get("summarization_agent", {})
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
        """Ask a research question (returns simple string answer)"""
        logger.info(f"Processing query: '{query}'")

        # Synthesize answer using reasoning agent
        answer = self.reasoning_agent.synthesize_answer(query)

        # Update metadata
        self.metadata["conversations"] += 1
        self.metadata["last_modified"] = datetime.now().isoformat()

        # Auto-save session
        self.save_session()

        return answer

    def ask_detailed(self, query: str) -> Dict:
        """Ask a research question (returns detailed result with sources and graph insights)"""
        logger.info(f"Processing detailed query: '{query}'")

        # Get graph results before generating answer
        graph_results = self.graph_agent.query_graph(query, max_hops=2)

        # Get vector search results
        vector_results = self.vector_agent.search(query, top_k=5)

        # Synthesize answer using reasoning agent
        answer = self.reasoning_agent.synthesize_answer(query)

        # Extract papers/sources from vector results
        papers_used = []
        for result in vector_results:
            paper_info = {
                "title": result.get("title", "Unknown"),
                "year": result.get("year", "N/A"),
                "source": result.get("source", "Unknown")
            }
            if paper_info not in papers_used:
                papers_used.append(paper_info)

        # Build graph insights
        graph_insights = {
            "related_concepts": [r.get("concept", "") for r in graph_results[:5] if r.get("concept")],
            "related_papers": len(graph_results),
            "nodes_found": len(graph_results)
        }

        # Update metadata
        self.metadata["conversations"] += 1
        self.metadata["last_modified"] = datetime.now().isoformat()

        # Auto-save session
        self.save_session()

        return {
            "answer": answer,
            "papers_used": papers_used,
            "graph_insights": graph_insights,
            "graph_results_count": len(graph_results),
            "vector_results_count": len(vector_results)
        }

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

            # Save FAISS index to separate file
            if self.vector_agent.db_type == "faiss":
                faiss_index_path = self.sessions_dir / f"{self.session_name}_faiss"
                self.vector_agent.save_faiss_index(str(faiss_index_path))
                state["faiss_index_saved"] = True
            # Qdrant persists automatically, no action needed

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

            # Restore FAISS index from separate file
            if self.vector_agent.db_type == "faiss" and state.get("faiss_index_saved"):
                faiss_index_path = self.sessions_dir / f"{self.session_name}_faiss"
                self.vector_agent.load_faiss_index(str(faiss_index_path))
            # Qdrant loads automatically from persistent storage

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

    # ========================================================================
    # SUMMARIZATION METHODS
    # ========================================================================

    def summarize_paper(self, paper: Dict, style: str = "executive", length: str = "medium") -> Dict:
        """
        Summarize a single research paper

        Args:
            paper: Paper metadata (title, abstract, authors, etc.)
            style: Summary style (executive, technical, abstract, bullet_points, narrative)
            length: Summary length (brief, short, medium, detailed, comprehensive)

        Returns:
            Summary dictionary with summary text, key points, style, length, word count
        """
        from .summarization_agent import SummaryStyle, SummaryLength

        # Convert string to enum
        style_enum = SummaryStyle(style) if isinstance(style, str) else style
        length_enum = SummaryLength(length) if isinstance(length, str) else length

        return self.summarization_agent.summarize_paper(paper, style_enum, length_enum)

    def summarize_collection(self, papers: Optional[List[Dict]] = None, style: str = "executive", focus: str = "research trends") -> Dict:
        """
        Summarize collection of papers

        Args:
            papers: List of papers (defaults to last collected papers)
            style: Summary style
            focus: What to focus on (trends, methods, findings, gaps)

        Returns:
            Collection summary with overall summary, key themes, trends, gaps, top papers
        """
        from .summarization_agent import SummaryStyle

        # Use last collected papers if not specified
        if papers is None:
            papers = self.data_collector.get_last_collection()

        if not papers:
            return {"error": "No papers to summarize. Collect papers first."}

        style_enum = SummaryStyle(style) if isinstance(style, str) else style
        return self.summarization_agent.summarize_collection(papers, style_enum, focus)

    def summarize_conversation(self, style: str = "bullet_points") -> Dict:
        """
        Summarize current research conversation/session

        Args:
            style: Summary style

        Returns:
            Conversation summary with session summary, questions, insights, topics
        """
        from .summarization_agent import SummaryStyle

        conversation_history = self.reasoning_agent.get_history()

        if not conversation_history:
            return {"error": "No conversation to summarize. Ask some questions first."}

        style_enum = SummaryStyle(style) if isinstance(style, str) else style
        return self.summarization_agent.summarize_conversation(conversation_history, style_enum)

    def compare_papers(self, papers: List[Dict], comparison_aspects: Optional[List[str]] = None) -> Dict:
        """
        Compare multiple papers side-by-side

        Args:
            papers: List of papers to compare (2-5 papers)
            comparison_aspects: What to compare (methodology, results, datasets, etc.)

        Returns:
            Comparison summary with similarities, differences, strengths/weaknesses, recommendation
        """
        if len(papers) < 2:
            return {"error": "Need at least 2 papers to compare"}

        return self.summarization_agent.compare_papers(papers, comparison_aspects)

    def close(self):
        """Clean up and close all connections"""
        logger.info("Closing orchestrator and all agents...")

        # Save session one last time
        self.save_session()

        # Close database connections
        self.graph_agent.close()
        self.vector_agent.close()

        logger.info("Orchestrator closed successfully")
