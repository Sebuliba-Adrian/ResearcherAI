#!/usr/bin/env python3
"""
Test script for LangGraph and LlamaIndex integrations
Verifies both new components work correctly with existing infrastructure
"""

import os
import sys
import time
from pathlib import Path

# Ensure we can import from agents directory
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required packages can be imported"""
    print("=" * 70)
    print("TEST 1: Package Imports")
    print("=" * 70)

    try:
        import langgraph
        version = getattr(langgraph, '__version__', 'unknown')
        print(f"✓ LangGraph version: {version}")
    except ImportError as e:
        print(f"✗ LangGraph import failed: {e}")
        return False

    try:
        import llama_index
        version = getattr(llama_index, '__version__', 'unknown')
        print(f"✓ LlamaIndex version: {version}")
    except ImportError as e:
        print(f"✗ LlamaIndex import failed: {e}")
        return False

    try:
        from agents.langgraph_orchestrator import LangGraphOrchestrator, create_orchestrator
        print("✓ LangGraphOrchestrator imported successfully")
    except ImportError as e:
        print(f"✗ LangGraphOrchestrator import failed: {e}")
        return False

    try:
        from agents.llamaindex_rag import LlamaIndexRAG, create_rag_system
        print("✓ LlamaIndexRAG imported successfully")
    except ImportError as e:
        print(f"✗ LlamaIndexRAG import failed: {e}")
        return False

    print()
    return True


def test_langgraph_orchestrator():
    """Test LangGraph orchestrator functionality"""
    print("=" * 70)
    print("TEST 2: LangGraph Orchestrator")
    print("=" * 70)

    try:
        from agents.langgraph_orchestrator import create_orchestrator

        # Create orchestrator
        print("Creating LangGraph orchestrator...")
        orchestrator = create_orchestrator()
        print("✓ Orchestrator created successfully")

        # Get workflow graph
        print("\nWorkflow Graph Structure:")
        graph_viz = orchestrator.get_workflow_graph()
        print(graph_viz)

        # Run a simple test workflow
        print("\nRunning test workflow...")
        query = "What are the latest advances in RAG systems?"
        result = orchestrator.run_workflow(query, thread_id="test_thread_1")

        print(f"✓ Workflow completed")
        print(f"  Steps executed: {len(result.get('messages', []))}")

        # Display execution flow
        print("\n  Execution steps:")
        for i, msg in enumerate(result.get('messages', []), 1):
            print(f"    {i}. {msg}")

        # Check reasoning result
        reasoning = result.get('reasoning_result', {})
        if reasoning:
            answer = reasoning.get('answer', 'N/A')
            print(f"\n  Generated answer: {answer[:150]}...")
            print(f"  Confidence: {reasoning.get('confidence', 0)}")

        # Test state persistence
        print("\nTesting state persistence...")
        saved_state = orchestrator.get_state("test_thread_1")
        if saved_state:
            print("✓ State saved and retrieved successfully")

        print()
        return True

    except Exception as e:
        print(f"✗ LangGraph test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_llamaindex_rag():
    """Test LlamaIndex RAG functionality"""
    print("=" * 70)
    print("TEST 3: LlamaIndex RAG System")
    print("=" * 70)

    try:
        from agents.llamaindex_rag import create_rag_system

        # Create RAG system (in-memory for testing)
        print("Creating LlamaIndex RAG system (in-memory mode)...")
        rag = create_rag_system(use_qdrant=False)
        print("✓ RAG system created successfully")

        # Sample papers for testing
        sample_papers = [
            {
                "id": "arxiv_2017_attention",
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                "authors": ["Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones", "Gomez", "Kaiser", "Polosukhin"],
                "year": "2017",
                "source": "arXiv",
                "url": "https://arxiv.org/abs/1706.03762"
            },
            {
                "id": "arxiv_2018_bert",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
                "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
                "year": "2018",
                "source": "arXiv",
                "url": "https://arxiv.org/abs/1810.04805"
            },
            {
                "id": "arxiv_2023_rag",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "abstract": "Large pre-trained language models have been shown to store factual knowledge in their parameters. However, their ability to access and precisely manipulate knowledge is still limited. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG).",
                "authors": ["Lewis", "Perez", "Piktus", "Petroni", "Karpukhin", "Goyal"],
                "year": "2020",
                "source": "arXiv",
                "url": "https://arxiv.org/abs/2005.11401"
            }
        ]

        # Index documents
        print(f"\nIndexing {len(sample_papers)} sample papers...")
        start_time = time.time()
        stats = rag.index_documents(sample_papers)
        index_time = time.time() - start_time

        print(f"✓ Indexed {stats['documents_indexed']} documents in {index_time:.2f}s")
        print(f"  Vector store: {stats['vector_store']}")

        # Test queries
        test_queries = [
            "What is the Transformer architecture?",
            "How does BERT differ from previous models?",
            "What is retrieval-augmented generation?"
        ]

        print(f"\nRunning {len(test_queries)} test queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n  Query {i}: {query}")
            start_time = time.time()
            result = rag.query(query, top_k=3)
            query_time = time.time() - start_time

            if 'error' in result:
                print(f"    ✗ Query failed: {result['error']}")
            else:
                print(f"    ✓ Query completed in {query_time:.2f}s")
                print(f"    Sources: {result['num_sources']}")
                print(f"    Answer preview: {result['answer'][:150]}...")

        # Test retrieval
        print("\nTesting document retrieval...")
        similar_docs = rag.retrieve_similar("attention mechanisms in transformers", top_k=2)
        print(f"✓ Retrieved {len(similar_docs)} similar documents")
        for i, doc in enumerate(similar_docs, 1):
            print(f"  {i}. {doc['metadata']['title']} (score: {doc['score']:.3f})")

        # Get system stats
        print("\nSystem Statistics:")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print()
        return True

    except Exception as e:
        print(f"✗ LlamaIndex test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_integration_compatibility():
    """Test compatibility with existing ResearcherAI infrastructure"""
    print("=" * 70)
    print("TEST 4: Integration Compatibility")
    print("=" * 70)

    try:
        # Check if existing modules still work
        print("Checking existing module compatibility...")

        # Test imports of core modules
        modules_to_check = [
            "agents.data_agent",
            "agents.graph_agent",
            "agents.vector_agent",
            "agents.reasoner_agent",
            "agents.orchestrator_agent"
        ]

        for module_name in modules_to_check:
            try:
                __import__(module_name)
                print(f"  ✓ {module_name}")
            except ImportError as e:
                print(f"  ✗ {module_name}: {e}")
                return False

        print("\n✓ All core modules remain functional")
        print()
        return True

    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("LangGraph & LlamaIndex Integration Test Suite")
    print("=" * 70 + "\n")

    # Track results
    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['langgraph'] = test_langgraph_orchestrator()
    results['llamaindex'] = test_llamaindex_rag()
    results['compatibility'] = test_integration_compatibility()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.upper()}: {status}")

    print()
    print(f"Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! LangGraph and LlamaIndex integrations are working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
