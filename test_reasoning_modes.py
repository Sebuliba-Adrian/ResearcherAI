"""
Test All Three Reasoning Modes End-to-End
=========================================

Tests the three core reasoning capabilities:
1. 🧩 Graph Reasoning (Cypher queries for relationships)
2. 🔎 Semantic Retrieval (embeddings + similarity search)
3. ⚙️ Hybrid Mode (Semantic → Graph integration)
"""

import os
import sys
import logging
from typing import Dict, List
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator_agent import OrchestratorAgent
from agents.graph_agent import KnowledgeGraphAgent
from agents.vector_agent import VectorAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 100)
    print(f"{title}")
    print("=" * 100)
    print()


def test_graph_reasoning(graph_agent: KnowledgeGraphAgent):
    """
    🧩 TEST 1: Graph Reasoning (Cypher Queries)

    Purpose: Trace relationships like authorships, citations, collaborations
    Method: Use Cypher queries to traverse the graph
    """
    print_section("🧩 TEST 1: GRAPH REASONING (Relationship Traversal)")

    print("📝 What is Graph Reasoning?")
    print("   - Uses Cypher queries to find connected nodes")
    print("   - Works best for: authorships, citations, institutions, collaborations")
    print("   - Example: 'Who authored papers about transformers?' or 'Which datasets are cited together?'")
    print()

    test_queries = [
        ("transformer", 2, "Find papers and authors related to 'transformer'"),
        ("attention mechanism", 2, "Find concepts connected to 'attention mechanism'"),
        ("neural networks", 1, "Find immediate neighbors of 'neural networks'")
    ]

    results_found = 0

    for entity, max_hops, description in test_queries:
        print(f"🔍 Query: {description}")
        print(f"   Entity: '{entity}', Max Hops: {max_hops}")

        try:
            results = graph_agent.query_graph(entity, max_hops=max_hops)

            if results:
                print(f"   ✅ Found {len(results)} paths in graph")

                # Show sample result
                if len(results) > 0:
                    sample = results[0]
                    print(f"   📊 Sample Path:")
                    print(f"      Nodes: {sample.get('nodes', [])[: 3]}")
                    print(f"      Relationships: {sample.get('relationships', [])[:3]}")

                results_found += len(results)
            else:
                print(f"   ⚠️  No paths found (graph may be empty or entity not present)")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()

    print(f"📊 Graph Reasoning Summary:")
    print(f"   Total paths found: {results_found}")
    print(f"   Status: {'✅ WORKING' if results_found > 0 else '⚠️  NO DATA (graph needs papers)'}")
    print()

    return results_found > 0


def test_semantic_retrieval(vector_agent: VectorAgent):
    """
    🔎 TEST 2: Semantic Retrieval (Embeddings + Similarity)

    Purpose: Find meaning - things about a concept, even if words differ
    Method: Use embeddings and cosine similarity search
    """
    print_section("🔎 TEST 2: SEMANTIC RETRIEVAL (Meaning-Based Search)")

    print("📝 What is Semantic Retrieval?")
    print("   - Uses embeddings (384-dim vectors) to find similar content")
    print("   - Works best for: 'papers like this one', 'topics about X', conceptual similarity")
    print("   - Example: 'Papers about attention' finds content even if it uses 'self-attention' or 'multi-head attention'")
    print()

    test_queries = [
        ("attention mechanism in transformers", 5, "Find papers about attention mechanisms"),
        ("machine learning neural networks", 5, "Find papers about ML and neural networks"),
        ("natural language processing", 3, "Find NLP-related papers")
    ]

    results_found = 0
    total_score = 0.0

    for query, top_k, description in test_queries:
        print(f"🔍 Query: {description}")
        print(f"   Search: '{query}', Top-K: {top_k}")

        try:
            results = vector_agent.search(query, top_k=top_k)

            if results:
                print(f"   ✅ Found {len(results)} similar documents")

                # Show top result
                if len(results) > 0:
                    top = results[0]
                    score = top.get('score', 0.0)
                    title = top.get('title', 'Unknown')[:60]

                    print(f"   🏆 Top Result:")
                    print(f"      Score: {score:.4f}")
                    print(f"      Title: {title}...")

                    total_score += score

                results_found += len(results)
            else:
                print(f"   ⚠️  No results found (vector database may be empty)")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()

    avg_score = total_score / len(test_queries) if test_queries else 0.0

    print(f"📊 Semantic Retrieval Summary:")
    print(f"   Total documents found: {results_found}")
    print(f"   Average relevance score: {avg_score:.4f}")
    print(f"   Status: {'✅ WORKING' if results_found > 0 else '⚠️  NO DATA (vector DB needs papers)'}")
    print()

    return results_found > 0


def test_hybrid_mode(orchestrator: OrchestratorAgent):
    """
    ⚙️ TEST 3: Hybrid Mode (Semantic → Graph Integration)

    Purpose: Find meaning AND reason over structure
    Method: Semantic search to get candidates → Graph query to trace relationships
    """
    print_section("⚙️ TEST 3: HYBRID MODE (Semantic Search → Graph Reasoning)")

    print("📝 What is Hybrid Mode?")
    print("   - Step 1: Semantic search finds relevant papers/concepts")
    print("   - Step 2: Graph queries trace relationships among them")
    print("   - Works best for: collaboration discovery, thematic trends, dataset linkage")
    print("   - Example: 'Find authors who work on similar topics' or 'Which papers cite each other?'")
    print()

    test_queries = [
        ("What are transformers used for in machine learning?",
         "Integration of semantic understanding + structural relationships"),
        ("How do attention mechanisms work?",
         "Combines conceptual search with relationship tracing"),
        ("Who are the key researchers in neural networks?",
         "Uses vectors to find relevant papers, then graphs to find authors")
    ]

    results_found = 0

    for query, description in test_queries:
        print(f"🔍 Query: {description}")
        print(f"   Question: '{query}'")

        try:
            # Use ask_detailed which implements hybrid mode
            result = orchestrator.ask_detailed(query)

            answer = result.get('answer', '')
            papers_used = result.get('papers_used', [])
            graph_insights = result.get('graph_insights', {})

            print(f"   ✅ Hybrid query successful")
            print()
            print(f"   📝 Answer ({len(answer)} chars):")
            print(f"      {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print()
            print(f"   📚 Semantic Component:")
            print(f"      Papers retrieved: {len(papers_used)}")
            if papers_used:
                print(f"      Sample: {papers_used[0].get('title', 'Unknown')[:50]}...")
            print()
            print(f"   🕸️  Graph Component:")
            print(f"      Related concepts: {len(graph_insights.get('related_concepts', []))}")
            print(f"      Related papers: {graph_insights.get('related_papers', 0)}")
            print(f"      Nodes found: {graph_insights.get('nodes_found', 0)}")

            if papers_used or graph_insights.get('nodes_found', 0) > 0:
                results_found += 1

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

        print()

    print(f"📊 Hybrid Mode Summary:")
    print(f"   Successful hybrid queries: {results_found}/{len(test_queries)}")
    print(f"   Status: {'✅ WORKING' if results_found > 0 else '⚠️  NEEDS DATA'}")
    print()

    return results_found > 0


def verify_data_exists(orchestrator: OrchestratorAgent) -> Dict[str, bool]:
    """Check if data exists in both Neo4j and Qdrant"""
    print_section("🔍 PRE-FLIGHT CHECK: Verify Data Exists")

    status = {
        "graph_has_data": False,
        "vector_has_data": False
    }

    # Check graph
    try:
        test_results = orchestrator.graph_agent.query_graph("test", max_hops=1)
        if test_results:
            status["graph_has_data"] = True
            print(f"✅ Graph Database: {len(test_results)} paths found")
        else:
            print(f"⚠️  Graph Database: Empty (no papers inserted yet)")
    except Exception as e:
        print(f"❌ Graph Database: Error - {e}")

    # Check vector
    try:
        test_results = orchestrator.vector_agent.search("test", top_k=1)
        if test_results:
            status["vector_has_data"] = True
            print(f"✅ Vector Database: {len(test_results)} documents found")
        else:
            print(f"⚠️  Vector Database: Empty (no papers inserted yet)")
    except Exception as e:
        print(f"❌ Vector Database: Error - {e}")

    print()

    if not (status["graph_has_data"] and status["vector_has_data"]):
        print("⚠️  WARNING: Databases are empty!")
        print("   Please insert some papers first:")
        print("   1. Via API: POST /v1/collect?query='transformers'")
        print("   2. Via PDF: POST /v1/upload/pdf")
        print("   3. Via ETL: python verify_etl_pipeline.py")
        print()

    return status


def main():
    """Run all reasoning mode tests"""
    print("\n")
    print("╔" + "═" * 98 + "╗")
    print("║" + " " * 25 + "REASONING MODES END-TO-END TEST" + " " * 42 + "║")
    print("╚" + "═" * 98 + "╝")
    print()
    print("Testing three core reasoning capabilities:")
    print("  1. 🧩 Graph Reasoning - Trace relationships with Cypher")
    print("  2. 🔎 Semantic Retrieval - Find meaning with embeddings")
    print("  3. ⚙️ Hybrid Mode - Combine both for powerful queries")
    print()

    # Load configuration
    config = {}
    config_file = "config/settings.yaml" if os.path.exists("config/settings.yaml") else "config/config.yaml"
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = yaml.safe_load(f)

    # Initialize orchestrator
    try:
        print("🚀 Initializing ResearcherAI orchestrator...")
        orchestrator = OrchestratorAgent(session_name="reasoning_test", config=config)
        print("✅ Orchestrator initialized")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1

    # Verify data exists
    status = verify_data_exists(orchestrator)

    # Run tests
    results = {}

    # Test 1: Graph Reasoning
    results["graph"] = test_graph_reasoning(orchestrator.graph_agent)

    # Test 2: Semantic Retrieval
    results["semantic"] = test_semantic_retrieval(orchestrator.vector_agent)

    # Test 3: Hybrid Mode
    results["hybrid"] = test_hybrid_mode(orchestrator)

    # Final Summary
    print_section("📊 FINAL SUMMARY: ALL THREE REASONING MODES")

    modes = [
        ("🧩 Graph Reasoning", results["graph"], "Cypher queries for relationship traversal"),
        ("🔎 Semantic Retrieval", results["semantic"], "Embedding-based similarity search"),
        ("⚙️ Hybrid Mode", results["hybrid"], "Semantic → Graph integration")
    ]

    all_working = all(results.values())

    for name, working, description in modes:
        status_icon = "✅" if working else "⚠️"
        status_text = "WORKING" if working else "NEEDS DATA"
        print(f"{status_icon} {name}: {status_text}")
        print(f"   {description}")
        print()

    print("=" * 100)

    if all_working:
        print("✅ SUCCESS: All three reasoning modes are working perfectly!")
        print()
        print("Your ResearcherAI system can:")
        print("  • Trace relationships through the knowledge graph")
        print("  • Find semantically similar content via embeddings")
        print("  • Combine both for powerful hybrid queries")
        return 0
    else:
        print("⚠️  PARTIAL SUCCESS: Some modes need data")
        print()
        print("Next steps:")
        if not results["graph"]:
            print("  • Graph Database: Insert papers to enable relationship queries")
        if not results["semantic"]:
            print("  • Vector Database: Insert papers to enable semantic search")
        print()
        print("Run: python verify_etl_pipeline.py")
        print("Or:  curl -X POST http://localhost:8000/v1/collect -d '{\"query\":\"transformers\"}'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
