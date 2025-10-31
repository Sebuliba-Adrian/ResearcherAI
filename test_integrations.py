#!/usr/bin/env python3
"""
Test script for LangGraph and LlamaIndex integrations

This script verifies that the new integrations work correctly
and don't break existing functionality.
"""

import sys
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """Test that all imports work"""
    print("="*70)
    print("TEST 1: Import Test")
    print("="*70 + "\n")

    try:
        # Test LangGraph import
        from agents.langgraph_orchestrator import create_orchestrator
        print("✓ LangGraph orchestrator import successful")

        # Test LlamaIndex import
        from agents.llamaindex_rag import create_rag_system
        print("✓ LlamaIndex RAG import successful")

        # Test existing agents still work
        from agents.orchestrator_agent import OrchestratorAgent
        print("✓ Existing OrchestratorAgent import successful")

        from agents.vector_agent import VectorAgent
        print("✓ Existing VectorAgent import successful")

        print("\n✅ All imports successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Import test failed: {e}\n")
        return False


def test_langgraph():
    """Test LangGraph orchestrator"""
    print("="*70)
    print("TEST 2: LangGraph Orchestrator Test")
    print("="*70 + "\n")

    try:
        from agents.langgraph_orchestrator import create_orchestrator

        orchestrator = create_orchestrator()
        print("✓ LangGraph orchestrator created")

        # Test workflow graph generation
        graph = orchestrator.get_workflow_graph()
        print("✓ Workflow graph generated")
        print("\nWorkflow Structure:")
        print(graph)

        print("\n✅ LangGraph test successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ LangGraph test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_llamaindex():
    """Test LlamaIndex RAG system"""
    print("="*70)
    print("TEST 3: LlamaIndex RAG Test")
    print("="*70 + "\n")

    try:
        from agents.llamaindex_rag import create_rag_system

        # Create RAG system (in-memory for testing)
        rag = create_rag_system(use_qdrant=False)
        print("✓ LlamaIndex RAG system created")

        # Test with sample data
        sample_papers = [
            {
                "id": "test1",
                "title": "Test Paper on RAG Systems",
                "abstract": "This paper explores RAG systems for question answering.",
                "authors": ["Test Author"],
                "year": "2024",
                "source": "test",
                "url": "https://example.com"
            }
        ]

        stats = rag.index_documents(sample_papers)
        print(f"✓ Indexed {stats['documents_indexed']} documents")

        # Test query
        result = rag.query("What is RAG?")
        print(f"✓ Query executed (found {result['num_sources']} sources)")

        # Get stats
        system_stats = rag.get_stats()
        print(f"✓ System stats: {system_stats['status']}")

        print("\n✅ LlamaIndex test successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ LlamaIndex test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_existing_functionality():
    """Test that existing agents still work"""
    print("="*70)
    print("TEST 4: Existing Functionality Test")
    print("="*70 + "\n")

    try:
        # Test VectorAgent initialization
        from agents.vector_agent import VectorAgent
        vector_agent = VectorAgent(db_type="faiss")
        print("✓ VectorAgent (FAISS) initializes correctly")

        # Test basic stats
        stats = vector_agent.get_stats()
        print(f"✓ VectorAgent stats: {stats.get('total_embeddings', 0)} embeddings")

        print("\n✅ Existing functionality test successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Existing functionality test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_api_gateway():
    """Test that API gateway still works"""
    print("="*70)
    print("TEST 5: API Gateway Compatibility Test")
    print("="*70 + "\n")

    try:
        # Import main API module
        import api_gateway
        print("✓ API gateway imports successfully")

        print("\n✅ API gateway compatibility test successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ API gateway test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ResearcherAI - Integration Test Suite")
    print("Testing LangGraph & LlamaIndex Integrations")
    print("="*70 + "\n")

    results = {
        "imports": test_imports(),
        "langgraph": test_langgraph(),
        "llamaindex": test_llamaindex(),
        "existing": test_existing_functionality(),
        "api_gateway": test_api_gateway()
    }

    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():20s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Integrations are working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
