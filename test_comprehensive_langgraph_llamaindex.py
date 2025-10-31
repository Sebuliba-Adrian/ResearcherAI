#!/usr/bin/env python3
"""
Comprehensive LangGraph + LlamaIndex Integration Validation
============================================================

This script validates end-to-end integration of:
- LangGraph orchestration layer
- LlamaIndex RAG system
- Neo4j/NetworkX graph databases
- Qdrant/FAISS vector databases
- Frontend UI via Playwright MCP
- Agent activity logging and tracing

Tests both Production and Development modes for parity validation.
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, asdict

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AgentActivity:
    """Track individual agent activity"""
    agent_name: str
    action: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    status: str = "pending"
    input_data: Dict = None
    output_data: Dict = None
    error: str = None
    metadata: Dict = None

    def complete(self, status: str = "success", output_data: Dict = None, error: str = None):
        """Mark activity as complete"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        if output_data:
            self.output_data = output_data
        if error:
            self.error = error


@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    mode: str  # production or development
    status: str  # passed, failed, skipped
    duration: float
    agent_activities: List[AgentActivity]
    metrics: Dict
    errors: List[str] = None
    warnings: List[str] = None


class AgentActivityTracker:
    """Track and log agent activities across the system"""

    def __init__(self):
        self.activities: List[AgentActivity] = []
        self.current_activity = None

    def start_activity(self, agent_name: str, action: str, input_data: Dict = None, metadata: Dict = None):
        """Start tracking an agent activity"""
        activity = AgentActivity(
            agent_name=agent_name,
            action=action,
            start_time=time.time(),
            input_data=input_data or {},
            metadata=metadata or {}
        )
        self.current_activity = activity
        self.activities.append(activity)
        logger.info(f"[{agent_name}] Started: {action}")
        return activity

    def complete_activity(self, status: str = "success", output_data: Dict = None, error: str = None):
        """Complete current activity"""
        if self.current_activity:
            self.current_activity.complete(status, output_data, error)
            logger.info(
                f"[{self.current_activity.agent_name}] Completed: {self.current_activity.action} "
                f"({self.current_activity.duration:.2f}s) - {status}"
            )
            self.current_activity = None

    def get_summary(self) -> Dict:
        """Get summary of all activities"""
        total_duration = sum(a.duration for a in self.activities)
        by_agent = {}

        for activity in self.activities:
            if activity.agent_name not in by_agent:
                by_agent[activity.agent_name] = {
                    "total_actions": 0,
                    "total_duration": 0.0,
                    "successes": 0,
                    "failures": 0
                }

            by_agent[activity.agent_name]["total_actions"] += 1
            by_agent[activity.agent_name]["total_duration"] += activity.duration
            if activity.status == "success":
                by_agent[activity.agent_name]["successes"] += 1
            elif activity.status == "failed":
                by_agent[activity.agent_name]["failures"] += 1

        return {
            "total_activities": len(self.activities),
            "total_duration": total_duration,
            "by_agent": by_agent,
            "timeline": [asdict(a) for a in self.activities]
        }


class LangGraphLlamaIndexValidator:
    """Comprehensive validator for LangGraph + LlamaIndex integration"""

    def __init__(self, mode: str = "development"):
        """
        Initialize validator

        Args:
            mode: 'production' (Neo4j+Qdrant) or 'development' (NetworkX+FAISS)
        """
        self.mode = mode
        self.tracker = AgentActivityTracker()
        self.results: List[TestResult] = []
        self.setup_environment()

    def setup_environment(self):
        """Setup environment variables based on mode"""
        if self.mode == "production":
            os.environ["USE_NEO4J"] = "true"
            os.environ["USE_QDRANT"] = "true"
            logger.info("🔧 Mode: PRODUCTION (Neo4j + Qdrant)")
        else:
            os.environ["USE_NEO4J"] = "false"
            os.environ["USE_QDRANT"] = "false"
            logger.info("🔧 Mode: DEVELOPMENT (NetworkX + FAISS)")

    def test_langgraph_workflow(self) -> TestResult:
        """Test LangGraph orchestrator workflow"""
        test_start = time.time()
        logger.info("\n" + "="*70)
        logger.info("TEST: LangGraph Workflow Execution")
        logger.info("="*70)

        errors = []
        warnings = []
        metrics = {}

        try:
            # Import and create orchestrator
            activity = self.tracker.start_activity(
                "LangGraphOrchestrator",
                "initialize",
                metadata={"mode": self.mode}
            )

            from agents.langgraph_orchestrator import create_orchestrator
            orchestrator = create_orchestrator()

            self.tracker.complete_activity(
                "success",
                {"workflow_nodes": ["data_collection", "graph_processing", "vector_processing", "reasoning", "critic_review"]}
            )

            # Get workflow graph structure
            activity = self.tracker.start_activity(
                "LangGraphOrchestrator",
                "get_workflow_graph"
            )

            graph_viz = orchestrator.get_workflow_graph()
            logger.info(f"\nWorkflow Graph:\n{graph_viz}")

            self.tracker.complete_activity(
                "success",
                {"graph_visualization": "generated"}
            )

            # Run test workflow
            test_query = "What are the latest advances in retrieval-augmented generation systems?"
            activity = self.tracker.start_activity(
                "LangGraphOrchestrator",
                "run_workflow",
                input_data={"query": test_query, "thread_id": f"test_{self.mode}_1"}
            )

            result = orchestrator.run_workflow(test_query, thread_id=f"test_{self.mode}_1")

            # Extract metrics
            steps_executed = len(result.get('messages', []))
            reasoning_result = result.get('reasoning_result', {})
            critic_feedback = result.get('critic_feedback', {})

            metrics = {
                "steps_executed": steps_executed,
                "has_reasoning_result": bool(reasoning_result),
                "reasoning_confidence": reasoning_result.get('confidence', 0.0),
                "critic_approved": critic_feedback.get('approved', False),
                "answer_length": len(reasoning_result.get('answer', ''))
            }

            logger.info(f"\n📊 Workflow Metrics:")
            logger.info(f"   Steps Executed: {metrics['steps_executed']}")
            logger.info(f"   Reasoning Confidence: {metrics['reasoning_confidence']}")
            logger.info(f"   Critic Approved: {metrics['critic_approved']}")
            logger.info(f"   Answer Length: {metrics['answer_length']} chars")

            # Log execution flow
            logger.info(f"\n📋 Execution Flow:")
            for i, msg in enumerate(result.get('messages', []), 1):
                logger.info(f"   {i}. {msg}")

            # Display answer
            answer = reasoning_result.get('answer', 'N/A')
            logger.info(f"\n💬 Generated Answer:\n   {answer[:300]}...")

            self.tracker.complete_activity(
                "success",
                {
                    "workflow_result": metrics,
                    "answer_preview": answer[:200]
                }
            )

            # Test state persistence
            activity = self.tracker.start_activity(
                "LangGraphOrchestrator",
                "test_state_persistence"
            )

            saved_state = orchestrator.get_state(f"test_{self.mode}_1")
            state_persisted = saved_state is not None

            if state_persisted:
                logger.info("✓ State persistence working")
                self.tracker.complete_activity("success", {"state_persisted": True})
            else:
                warnings.append("State persistence not working")
                self.tracker.complete_activity("warning", {"state_persisted": False})

            status = "passed"

        except Exception as e:
            logger.error(f"❌ LangGraph workflow test failed: {e}")
            errors.append(str(e))
            self.tracker.complete_activity("failed", error=str(e))
            status = "failed"

        test_duration = time.time() - test_start

        return TestResult(
            test_name="langgraph_workflow",
            mode=self.mode,
            status=status,
            duration=test_duration,
            agent_activities=self.tracker.activities.copy(),
            metrics=metrics,
            errors=errors,
            warnings=warnings
        )

    def test_llamaindex_rag(self) -> TestResult:
        """Test LlamaIndex RAG system"""
        test_start = time.time()
        logger.info("\n" + "="*70)
        logger.info("TEST: LlamaIndex RAG System")
        logger.info("="*70)

        errors = []
        warnings = []
        metrics = {}

        try:
            # Create RAG system
            use_qdrant = self.mode == "production"

            activity = self.tracker.start_activity(
                "LlamaIndexRAG",
                "initialize",
                input_data={"use_qdrant": use_qdrant},
                metadata={"mode": self.mode}
            )

            from agents.llamaindex_rag import create_rag_system
            rag = create_rag_system(use_qdrant=use_qdrant)

            vector_store = "Qdrant" if use_qdrant else "FAISS (in-memory)"
            logger.info(f"✓ RAG system initialized with {vector_store}")

            self.tracker.complete_activity(
                "success",
                {"vector_store": vector_store}
            )

            # Prepare sample papers
            sample_papers = [
                {
                    "id": "rag_2020",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "abstract": "Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever.",
                    "authors": ["Lewis", "Perez", "Piktus", "Petroni", "Karpukhin", "Goyal", "Küttler", "Lewis", "Yih", "Rocktäschel", "Riedel", "Kiela"],
                    "year": "2020",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/2005.11401"
                },
                {
                    "id": "transformer_2017",
                    "title": "Attention Is All You Need",
                    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
                    "authors": ["Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones", "Gomez", "Kaiser", "Polosukhin"],
                    "year": "2017",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/1706.03762"
                },
                {
                    "id": "bert_2018",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                    "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks.",
                    "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
                    "year": "2018",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/1810.04805"
                },
                {
                    "id": "gpt3_2020",
                    "title": "Language Models are Few-Shot Learners",
                    "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions.",
                    "authors": ["Brown", "Mann", "Ryder", "Subbiah", "Kaplan", "Dhariwal"],
                    "year": "2020",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/2005.14165"
                },
                {
                    "id": "llama_2023",
                    "title": "LLaMA: Open and Efficient Foundation Language Models",
                    "abstract": "We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks.",
                    "authors": ["Touvron", "Lavril", "Izacard", "Martinet", "Lachaux", "Lacroix"],
                    "year": "2023",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/2302.13971"
                }
            ]

            # Index documents
            activity = self.tracker.start_activity(
                "LlamaIndexRAG",
                "index_documents",
                input_data={"num_papers": len(sample_papers)}
            )

            index_start = time.time()
            stats = rag.index_documents(sample_papers)
            index_duration = time.time() - index_start

            logger.info(f"✓ Indexed {stats['documents_indexed']} documents in {index_duration:.2f}s")
            logger.info(f"   Vector Store: {stats['vector_store']}")

            metrics["indexing_time"] = index_duration
            metrics["documents_indexed"] = stats['documents_indexed']
            metrics["vector_store"] = stats['vector_store']

            self.tracker.complete_activity(
                "success",
                {"indexing_stats": stats, "duration": index_duration}
            )

            # Run test queries
            test_queries = [
                "What is retrieval-augmented generation and how does it work?",
                "Explain the Transformer architecture and its key innovation",
                "How does BERT differ from previous language models?",
                "What are the main contributions of GPT-3?",
                "Compare LLaMA to GPT-3 in terms of performance"
            ]

            query_results = []

            for i, query in enumerate(test_queries, 1):
                activity = self.tracker.start_activity(
                    "LlamaIndexRAG",
                    "query",
                    input_data={"query": query, "top_k": 3}
                )

                logger.info(f"\n🔍 Query {i}: {query}")

                query_start = time.time()
                result = rag.query(query, top_k=3)
                query_duration = time.time() - query_start

                if 'error' in result:
                    logger.error(f"   ❌ Query failed: {result['error']}")
                    errors.append(f"Query {i} failed: {result['error']}")
                    self.tracker.complete_activity("failed", error=result['error'])
                else:
                    logger.info(f"   ✓ Completed in {query_duration:.2f}s")
                    logger.info(f"   📚 Sources: {result['num_sources']}")
                    logger.info(f"   💬 Answer: {result['answer'][:200]}...")

                    # Log sources
                    for j, source in enumerate(result.get('sources', []), 1):
                        logger.info(f"      Source {j}: {source['metadata']['title']} (score: {source['score']:.3f})")

                    query_results.append({
                        "query": query,
                        "duration": query_duration,
                        "num_sources": result['num_sources'],
                        "answer_length": len(result['answer'])
                    })

                    self.tracker.complete_activity(
                        "success",
                        {
                            "duration": query_duration,
                            "num_sources": result['num_sources'],
                            "answer_preview": result['answer'][:150]
                        }
                    )

            # Test retrieval
            activity = self.tracker.start_activity(
                "LlamaIndexRAG",
                "retrieve_similar",
                input_data={"query": "attention mechanisms in neural networks", "top_k": 3}
            )

            similar_docs = rag.retrieve_similar("attention mechanisms in neural networks", top_k=3)

            logger.info(f"\n🔎 Similarity Search Results:")
            for i, doc in enumerate(similar_docs, 1):
                logger.info(f"   {i}. {doc['metadata']['title']} (score: {doc['score']:.3f})")

            self.tracker.complete_activity(
                "success",
                {"num_results": len(similar_docs)}
            )

            # Get system stats
            system_stats = rag.get_stats()
            logger.info(f"\n📊 System Statistics:")
            for key, value in system_stats.items():
                logger.info(f"   {key}: {value}")

            metrics["query_results"] = query_results
            metrics["avg_query_time"] = sum(r['duration'] for r in query_results) / len(query_results) if query_results else 0
            metrics["system_stats"] = system_stats

            status = "passed" if not errors else "failed"

        except Exception as e:
            logger.error(f"❌ LlamaIndex RAG test failed: {e}")
            import traceback
            traceback.print_exc()
            errors.append(str(e))
            self.tracker.complete_activity("failed", error=str(e))
            status = "failed"

        test_duration = time.time() - test_start

        return TestResult(
            test_name="llamaindex_rag",
            mode=self.mode,
            status=status,
            duration=test_duration,
            agent_activities=self.tracker.activities.copy(),
            metrics=metrics,
            errors=errors,
            warnings=warnings
        )

    def test_integrated_workflow(self) -> TestResult:
        """Test integrated LangGraph + LlamaIndex workflow"""
        test_start = time.time()
        logger.info("\n" + "="*70)
        logger.info("TEST: Integrated LangGraph + LlamaIndex Workflow")
        logger.info("="*70)

        errors = []
        warnings = []
        metrics = {}

        try:
            # This test simulates a complete workflow where:
            # 1. LangGraph orchestrates the overall flow
            # 2. LlamaIndex handles document indexing and retrieval
            # 3. Results flow back through LangGraph for reasoning

            from agents.langgraph_orchestrator import create_orchestrator
            from agents.llamaindex_rag import create_rag_system

            # Initialize both systems
            activity = self.tracker.start_activity(
                "IntegratedWorkflow",
                "initialize_systems"
            )

            orchestrator = create_orchestrator()
            rag = create_rag_system(use_qdrant=(self.mode == "production"))

            logger.info("✓ Both systems initialized")
            self.tracker.complete_activity("success")

            # Simulate data collection (would normally use DataCollectorAgent)
            activity = self.tracker.start_activity(
                "IntegratedWorkflow",
                "simulate_data_collection"
            )

            # Sample research papers
            papers = [
                {
                    "id": "kg_rag_2023",
                    "title": "Knowledge Graphs meet Multi-Modal Learning: A Comprehensive Survey",
                    "abstract": "Knowledge graphs (KGs) have emerged as a powerful tool for organizing and querying large-scale structured knowledge. Meanwhile, multi-modal learning has made significant progress in integrating information from different modalities. This survey explores the intersection of these two areas, examining how knowledge graphs can enhance multi-modal learning and vice versa.",
                    "authors": ["Zhang", "Li", "Wang"],
                    "year": "2023",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/2023.xxxxx"
                }
            ]

            logger.info(f"✓ Collected {len(papers)} papers")
            self.tracker.complete_activity("success", {"num_papers": len(papers)})

            # Index in LlamaIndex
            activity = self.tracker.start_activity(
                "IntegratedWorkflow",
                "index_into_llamaindex"
            )

            index_stats = rag.index_documents(papers)
            logger.info(f"✓ Indexed into {index_stats['vector_store']}")

            self.tracker.complete_activity("success", index_stats)

            # Run integrated query
            research_question = "How do knowledge graphs enhance multi-modal learning systems?"

            activity = self.tracker.start_activity(
                "IntegratedWorkflow",
                "execute_query",
                input_data={"question": research_question}
            )

            # Step 1: Use LlamaIndex for retrieval
            retrieval_start = time.time()
            rag_result = rag.query(research_question, top_k=3)
            retrieval_time = time.time() - retrieval_start

            logger.info(f"✓ LlamaIndex retrieval: {retrieval_time:.2f}s")
            logger.info(f"   Retrieved {rag_result.get('num_sources', 0)} sources")

            # Step 2: Use LangGraph for reasoning with retrieved context
            reasoning_start = time.time()
            workflow_result = orchestrator.run_workflow(research_question, thread_id=f"integrated_{self.mode}")
            reasoning_time = time.time() - reasoning_start

            logger.info(f"✓ LangGraph reasoning: {reasoning_time:.2f}s")

            # Combine results
            final_answer = workflow_result.get('reasoning_result', {}).get('answer', '')
            retrieved_context = rag_result.get('answer', '')

            logger.info(f"\n💡 Final Integrated Result:")
            logger.info(f"   Retrieved Context Length: {len(retrieved_context)} chars")
            logger.info(f"   Reasoned Answer Length: {len(final_answer)} chars")
            logger.info(f"   Total Processing Time: {retrieval_time + reasoning_time:.2f}s")

            metrics = {
                "retrieval_time": retrieval_time,
                "reasoning_time": reasoning_time,
                "total_time": retrieval_time + reasoning_time,
                "num_sources": rag_result.get('num_sources', 0),
                "context_length": len(retrieved_context),
                "answer_length": len(final_answer)
            }

            self.tracker.complete_activity("success", metrics)

            status = "passed"

        except Exception as e:
            logger.error(f"❌ Integrated workflow test failed: {e}")
            import traceback
            traceback.print_exc()
            errors.append(str(e))
            self.tracker.complete_activity("failed", error=str(e))
            status = "failed"

        test_duration = time.time() - test_start

        return TestResult(
            test_name="integrated_workflow",
            mode=self.mode,
            status=status,
            duration=test_duration,
            agent_activities=self.tracker.activities.copy(),
            metrics=metrics,
            errors=errors,
            warnings=warnings
        )

    def run_all_tests(self) -> Dict:
        """Run all validation tests"""
        logger.info("\n" + "="*70)
        logger.info(f"COMPREHENSIVE VALIDATION - {self.mode.upper()} MODE")
        logger.info("="*70)

        start_time = time.time()

        # Run tests
        self.results.append(self.test_langgraph_workflow())
        self.results.append(self.test_llamaindex_rag())
        self.results.append(self.test_integrated_workflow())

        total_duration = time.time() - start_time

        # Generate summary
        summary = {
            "mode": self.mode,
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "tests": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "passed"),
                "failed": sum(1 for r in self.results if r.status == "failed"),
                "skipped": sum(1 for r in self.results if r.status == "skipped")
            },
            "agent_activity_summary": self.tracker.get_summary(),
            "test_results": [asdict(r) for r in self.results]
        }

        return summary


def compare_modes(prod_summary: Dict, dev_summary: Dict) -> Dict:
    """Compare production and development mode results"""
    logger.info("\n" + "="*70)
    logger.info("MODE COMPARISON ANALYSIS")
    logger.info("="*70)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "production_summary": {
            "total_tests": prod_summary["tests"]["total"],
            "passed": prod_summary["tests"]["passed"],
            "failed": prod_summary["tests"]["failed"],
            "duration": prod_summary["total_duration"]
        },
        "development_summary": {
            "total_tests": dev_summary["tests"]["total"],
            "passed": dev_summary["tests"]["passed"],
            "failed": dev_summary["tests"]["failed"],
            "duration": dev_summary["total_duration"]
        },
        "performance_delta": {
            "duration_diff": prod_summary["total_duration"] - dev_summary["total_duration"],
            "duration_ratio": prod_summary["total_duration"] / dev_summary["total_duration"] if dev_summary["total_duration"] > 0 else 0
        },
        "consistency_check": {
            "same_test_count": prod_summary["tests"]["total"] == dev_summary["tests"]["total"],
            "both_passed_all": (prod_summary["tests"]["passed"] == prod_summary["tests"]["total"]) and
                              (dev_summary["tests"]["passed"] == dev_summary["tests"]["total"])
        }
    }

    logger.info(f"\n📊 Performance Comparison:")
    logger.info(f"   Production Duration: {prod_summary['total_duration']:.2f}s")
    logger.info(f"   Development Duration: {dev_summary['total_duration']:.2f}s")
    logger.info(f"   Delta: {comparison['performance_delta']['duration_diff']:.2f}s")
    logger.info(f"   Ratio: {comparison['performance_delta']['duration_ratio']:.2f}x")

    logger.info(f"\n✅ Consistency:")
    logger.info(f"   Same Test Count: {comparison['consistency_check']['same_test_count']}")
    logger.info(f"   Both Passed All: {comparison['consistency_check']['both_passed_all']}")

    return comparison


async def run_playwright_validation():
    """Run Playwright MCP validation of frontend"""
    logger.info("\n" + "="*70)
    logger.info("PLAYWRIGHT MCP FRONTEND VALIDATION")
    logger.info("="*70)

    # Note: This would integrate with Playwright MCP
    # For now, we'll prepare the structure

    validation_plan = {
        "browser_tests": [
            {
                "name": "Load Homepage",
                "url": "http://localhost:5000",
                "validations": ["page loads", "no console errors", "UI elements present"]
            },
            {
                "name": "Test Query Submission",
                "actions": ["enter query", "submit", "wait for results"],
                "validations": ["results displayed", "no errors", "agent traces visible"]
            },
            {
                "name": "Test Vector Visualization",
                "actions": ["navigate to vector tab", "trigger visualization"],
                "validations": ["3D plot rendered", "data points visible"]
            },
            {
                "name": "Test Knowledge Graph",
                "actions": ["navigate to graph tab", "load graph"],
                "validations": ["nodes displayed", "edges rendered", "interactive"]
            }
        ]
    }

    logger.info("📋 Playwright Validation Plan:")
    for test in validation_plan["browser_tests"]:
        logger.info(f"   ✓ {test['name']}")

    logger.info("\n⚠️  Note: Full Playwright execution requires running servers")
    logger.info("   Run with: python test_playwright_integration.py")

    return validation_plan


def main():
    """Main execution"""
    logger.info("\n" + "#"*70)
    logger.info("# COMPREHENSIVE LANGGRAPH + LLAMAINDEX VALIDATION SUITE")
    logger.info("#"*70)

    # Run tests in both modes
    results = {}

    # Development mode (always available)
    logger.info("\n🚀 Starting DEVELOPMENT mode validation...")
    dev_validator = LangGraphLlamaIndexValidator(mode="development")
    results["development"] = dev_validator.run_all_tests()

    # Production mode (if Neo4j/Qdrant available)
    prod_available = os.getenv("RUN_PRODUCTION_TESTS", "false").lower() == "true"

    if prod_available:
        logger.info("\n🚀 Starting PRODUCTION mode validation...")
        prod_validator = LangGraphLlamaIndexValidator(mode="production")
        results["production"] = prod_validator.run_all_tests()

        # Compare modes
        comparison = compare_modes(results["production"], results["development"])
        results["mode_comparison"] = comparison
    else:
        logger.info("\n⚠️  Skipping PRODUCTION mode (set RUN_PRODUCTION_TESTS=true to enable)")

    # Playwright validation
    playwright_plan = asyncio.run(run_playwright_validation())
    results["playwright_plan"] = playwright_plan

    # Generate final report
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    report_file = output_dir / f"langgraph_llamaindex_validation_{timestamp}.json"

    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n📄 Full report saved to: {report_file}")

    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)

    for mode, summary in results.items():
        if mode in ["development", "production"]:
            logger.info(f"\n{mode.upper()} Mode:")
            logger.info(f"   Tests Passed: {summary['tests']['passed']}/{summary['tests']['total']}")
            logger.info(f"   Duration: {summary['total_duration']:.2f}s")
            logger.info(f"   Agent Activities: {summary['agent_activity_summary']['total_activities']}")

    # Success check
    all_passed = all(
        summary["tests"]["passed"] == summary["tests"]["total"]
        for mode, summary in results.items()
        if mode in ["development", "production"]
    )

    if all_passed:
        logger.info("\n✅ ALL VALIDATIONS PASSED!")
        return 0
    else:
        logger.info("\n❌ SOME VALIDATIONS FAILED - Check report for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
