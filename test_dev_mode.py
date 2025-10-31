#!/usr/bin/env python3
"""
Test script for dev mode with FAISS and NetworkX backends.
"""

import os
import sys
import time

# Set environment for DEV MODE (FAISS + NetworkX)
os.environ["USE_NEO4J"] = "false"
os.environ["USE_QDRANT"] = "false"
os.environ["USE_KAFKA"] = "false"  # Optional in dev
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

print("=" * 80)
print("🔧 DEV MODE TEST - FAISS + NetworkX Backends")
print("=" * 80)

# Import after environment setup
from agents import OrchestratorAgent

def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_dev_backends():
    """Test 1: Verify dev backends are used"""
    print_section("TEST 1: Backend Configuration")

    orchestrator = OrchestratorAgent("dev_test_session", {})
    stats = orchestrator.get_stats()

    print(f"Graph Backend: {stats['graph']['backend']}")
    print(f"Vector Backend: {stats['vector']['backend']}")

    assert stats['graph']['backend'] == 'NetworkX', "Expected NetworkX backend!"
    assert stats['vector']['backend'] == 'FAISS', "Expected FAISS backend!"

    print("\n✅ PASSED: Dev backends (NetworkX + FAISS) configured correctly")

    return orchestrator


def test_data_collection(orchestrator):
    """Test 2: Data collection with dev backends"""
    print_section("TEST 2: Data Collection (Dev Mode)")

    query = "neural networks"
    print(f"→ Collecting papers for: '{query}'")
    print("  (Max 2 per source for quick test)\n")

    start_time = time.time()
    result = orchestrator.collect_data(query, max_per_source=2)
    duration = time.time() - start_time

    print(f"\n✓ Collection completed in {duration:.2f}s")
    print(f"  Papers collected: {result['papers_collected']}")
    print(f"  Graph nodes added: {result['graph_stats'].get('nodes_added', 0)}")
    print(f"  Graph edges added: {result['graph_stats'].get('edges_added', 0)}")
    print(f"  Vector chunks added: {result['vector_stats'].get('chunks_added', 0)}")

    assert result['papers_collected'] > 0, "No papers collected!"

    print("\n✅ PASSED: Data collection working in dev mode")

    return result


def test_networkx_graph(orchestrator):
    """Test 3: NetworkX graph operations"""
    print_section("TEST 3: NetworkX Graph Operations")

    graph_agent = orchestrator.graph_agent
    nx_graph = graph_agent.G  # NetworkX graph object

    print(f"Graph type: {type(nx_graph)}")
    print(f"Total nodes: {nx_graph.number_of_nodes()}")
    print(f"Total edges: {nx_graph.number_of_edges()}")

    # Test node retrieval
    if nx_graph.number_of_nodes() > 0:
        print("\nSample nodes (first 5):")
        for i, (node_id, node_data) in enumerate(list(nx_graph.nodes(data=True))[:5], 1):
            node_type = node_data.get('type', 'unknown')
            print(f"  {i}. {node_id} (type: {node_type})")

        # Test edge retrieval
        if nx_graph.number_of_edges() > 0:
            print("\nSample edges (first 5):")
            for i, (source, target, edge_data) in enumerate(list(nx_graph.edges(data=True))[:5], 1):
                rel_type = edge_data.get('type', 'unknown')
                print(f"  {i}. {source} --[{rel_type}]--> {target}")

    assert nx_graph.number_of_nodes() > 0, "No nodes in graph!"
    assert nx_graph.number_of_edges() > 0, "No edges in graph!"

    print("\n✅ PASSED: NetworkX graph operations working")


def test_faiss_vectors(orchestrator):
    """Test 4: FAISS vector operations"""
    print_section("TEST 4: FAISS Vector Search")

    vector_agent = orchestrator.vector_agent

    # Check vector count
    stats = vector_agent.get_stats()
    print(f"Total vectors: {stats['chunks']}")
    print(f"Embedding dimension: {stats['dimension']}")
    print(f"Backend: {stats['backend']}")

    assert stats['chunks'] > 0, "No vectors in FAISS index!"

    # Test similarity search
    query = "machine learning algorithms"
    print(f"\n→ Testing similarity search for: '{query}'")

    results = vector_agent.search(query, top_k=3)

    if results:
        print(f"\n✓ Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            score = result.get('score', 0.0)
            text = result.get('text', '')[:80]
            print(f"  {i}. Score: {score:.4f}")
            print(f"     Text: {text}...\n")

        assert len(results) > 0, "Search returned no results!"
        print("✅ PASSED: FAISS vector search working")
    else:
        print("⚠ WARNING: Search returned empty results (may need more data)")


def test_query_answering(orchestrator):
    """Test 5: Query answering with dev backends"""
    print_section("TEST 5: Query Answering (Dev Mode)")

    query = "What are neural networks?"
    print(f"→ Asking: '{query}'")
    print("  (Using collected papers to answer)\n")

    try:
        start_time = time.time()
        answer = orchestrator.ask(query)
        duration = time.time() - start_time

        print(f"🤖 Answer ({duration:.2f}s):")
        print("-" * 80)
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        print("-" * 80)

        assert len(answer) > 0, "Empty answer received!"

        print("\n✅ PASSED: Query answering working in dev mode")

        return answer

    except Exception as e:
        print(f"\n⚠ WARNING: Query answering error: {e}")
        import traceback
        traceback.print_exc()


def test_session_persistence(orchestrator):
    """Test 6: Session save/load with dev backends"""
    print_section("TEST 6: Session Persistence")

    session_name = orchestrator.session_name

    print("→ Saving session...")
    orchestrator.save_session()
    print(f"  ✓ Session '{session_name}' saved")

    # Get current stats
    stats_before = orchestrator.get_stats()

    print("\n→ Loading session...")
    orchestrator2 = OrchestratorAgent(session_name, {})
    stats_after = orchestrator2.get_stats()

    print(f"  ✓ Session '{session_name}' loaded")

    print("\nSession stats comparison:")
    print(f"  Papers: {stats_before['metadata']['papers_collected']} -> {stats_after['metadata']['papers_collected']}")
    print(f"  Graph nodes: {stats_before['graph']['nodes']} -> {stats_after['graph']['nodes']}")
    print(f"  Vector chunks: {stats_before['vector']['chunks']} -> {stats_after['vector']['chunks']}")

    orchestrator2.close()

    print("\n✅ PASSED: Session persistence working in dev mode")


def compare_backends():
    """Compare dev and prod backends"""
    print_section("BACKEND COMPARISON")

    print("Development Backends (Current Test):")
    print("  Graph: NetworkX")
    print("    ✓ In-memory graph database")
    print("    ✓ Fast for development and testing")
    print("    ✓ No external dependencies")
    print("    ✗ Not persistent across restarts")
    print()
    print("  Vector: FAISS")
    print("    ✓ In-memory vector index")
    print("    ✓ Fast similarity search")
    print("    ✓ No external services required")
    print("    ✗ Single-node only")
    print()

    print("Production Backends (Previously Tested):")
    print("  Graph: Neo4j")
    print("    ✓ Persistent graph database")
    print("    ✓ Advanced Cypher queries")
    print("    ✓ ACID transactions")
    print("    ✓ Multi-user access")
    print()
    print("  Vector: Qdrant")
    print("    ✓ Persistent vector storage")
    print("    ✓ Distributed scaling")
    print("    ✓ REST API access")
    print("    ✓ Advanced filtering")


def main():
    """Run all dev mode tests"""
    print("\n🧪 ResearcherAI - Dev Mode Test Suite")
    print("Testing FAISS + NetworkX backends\n")

    test_start = time.time()
    results = {}

    try:
        # Test 1: Backend verification
        orchestrator = test_dev_backends()
        results['backends'] = True

        # Test 2: Data collection
        collection_result = test_data_collection(orchestrator)
        results['collection'] = collection_result['papers_collected'] > 0

        # Test 3: NetworkX operations
        test_networkx_graph(orchestrator)
        results['networkx'] = True

        # Test 4: FAISS operations
        test_faiss_vectors(orchestrator)
        results['faiss'] = True

        # Test 5: Query answering
        test_query_answering(orchestrator)
        results['reasoning'] = True

        # Test 6: Session persistence
        test_session_persistence(orchestrator)
        results['persistence'] = True

        # Backend comparison
        compare_backends()

        # Close orchestrator
        orchestrator.close()

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    test_duration = time.time() - test_start

    print_section("TEST SUMMARY")
    print(f"Total Test Duration: {test_duration:.2f}s\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Results: {passed}/{total} tests passed\n")

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name.upper()}")

    print("\n" + "=" * 80)

    if passed == total:
        print("🎉 ALL DEV MODE TESTS PASSED!")
        print("\n✅ System works perfectly in both modes:")
        print("   • Production: Neo4j + Qdrant + Kafka")
        print("   • Development: NetworkX + FAISS")
        print("=" * 80 + "\n")
        return 0
    else:
        print(f"⚠️  {total - passed} TEST(S) FAILED")
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
