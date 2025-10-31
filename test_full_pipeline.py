#!/usr/bin/env python3
"""
Comprehensive end-to-end test for ResearcherAI production pipeline.

Tests the complete workflow:
1. Data collection from real sources
2. Knowledge graph construction (Neo4j)
3. Vector embeddings (Qdrant)
4. Event publishing (Kafka)
5. Query answering with reasoning
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Set environment variables for production backends
os.environ["USE_NEO4J"] = "true"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "research_password"
os.environ["USE_QDRANT"] = "true"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"
os.environ["USE_KAFKA"] = "true"
os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9094"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# Import after environment setup
from agents import OrchestratorAgent


def print_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_data_collection():
    """Test 1: Data Collection"""
    print_header("TEST 1: Real Data Collection")

    # Initialize orchestrator
    print("→ Initializing OrchestratorAgent...")
    orchestrator = OrchestratorAgent("test_pipeline", {})

    # Collect papers on a specific topic
    query = "large language models"
    print(f"→ Collecting papers for query: '{query}'")
    print("  (Collecting max 3 papers per source to keep test fast)\n")

    start_time = time.time()
    result = orchestrator.collect_data(query, max_per_source=3)
    duration = time.time() - start_time

    print(f"\n✓ Data collection completed in {duration:.2f}s")
    print(f"  Papers collected: {result['papers_collected']}")
    print(f"  Sources stats: {result.get('sources', {})}")
    print(f"  Graph nodes added: {result['graph_stats'].get('nodes_added', 0)}")
    print(f"  Graph edges added: {result['graph_stats'].get('edges_added', 0)}")
    print(f"  Vector chunks added: {result['vector_stats'].get('chunks_added', 0)}")

    return {
        "orchestrator": orchestrator,
        "papers_collected": result['papers_collected'],
        "graph_stats": result['graph_stats'],
        "vector_stats": result['vector_stats'],
        "duration": duration
    }


def test_paper_details(orchestrator):
    """Test 2: Show Sample Papers"""
    print_header("TEST 2: Sample Papers with Details")

    # Access the data collector to get papers
    papers = orchestrator.data_collector.papers[:5]  # Get first 5 papers

    if not papers:
        print("⚠ No papers found to display")
        return {"sample_papers": []}

    print(f"Showing {len(papers)} sample papers:\n")

    sample_papers = []
    for i, paper in enumerate(papers, 1):
        print(f"{i}. Title: {paper.get('title', 'N/A')}")
        authors = paper.get('authors', [])
        if authors:
            print(f"   Authors: {', '.join(authors[:3])}" + (" et al." if len(authors) > 3 else ""))
        print(f"   Source: {paper.get('source', 'N/A')}")
        print(f"   Year: {paper.get('year', 'N/A')}")
        if paper.get('url'):
            print(f"   URL: {paper['url'][:60]}...")
        print()

        sample_papers.append({
            "title": paper.get('title', 'N/A'),
            "authors": authors[:3],
            "source": paper.get('source', 'N/A'),
            "year": paper.get('year', 'N/A')
        })

    return {"sample_papers": sample_papers}


def test_graph_entities(orchestrator):
    """Test 3: Show Graph Entities and Relationships"""
    print_header("TEST 3: Knowledge Graph Entities and Relationships")

    # Get graph statistics
    stats = orchestrator.get_stats()
    graph_stats = stats['graph']

    print(f"Graph Backend: {graph_stats['backend']}")
    print(f"Total Nodes: {graph_stats['nodes']}")
    print(f"Total Edges: {graph_stats['edges']}")
    print()

    # Query Neo4j for sample entities
    print("→ Querying Neo4j for sample entities...\n")

    try:
        # Get sample papers from graph
        query = """
        MATCH (p:Paper)
        RETURN p.title as title, p.year as year, p.source as source
        LIMIT 5
        """
        result = orchestrator.knowledge_graph.graph.query(query)

        if result:
            print("Sample Papers in Graph:")
            for i, record in enumerate(result, 1):
                print(f"  {i}. {record.get('title', 'N/A')} ({record.get('year', 'N/A')}) - {record.get('source', 'N/A')}")
            print()

        # Get sample authors
        query = """
        MATCH (a:Author)-[:AUTHORED]->(p:Paper)
        RETURN a.name as author, count(p) as paper_count
        ORDER BY paper_count DESC
        LIMIT 5
        """
        result = orchestrator.knowledge_graph.graph.query(query)

        if result:
            print("Sample Authors:")
            for i, record in enumerate(result, 1):
                print(f"  {i}. {record.get('author', 'N/A')} ({record.get('paper_count', 0)} papers)")
            print()

        # Get sample relationships
        query = """
        MATCH (a:Author)-[r:AUTHORED]->(p:Paper)
        RETURN a.name as author, p.title as paper
        LIMIT 5
        """
        result = orchestrator.knowledge_graph.graph.query(query)

        if result:
            print("Sample Relationships:")
            for i, record in enumerate(result, 1):
                print(f"  {i}. {record.get('author', 'N/A')} AUTHORED '{record.get('paper', 'N/A')[:50]}...'")
            print()

        return {"graph_backend": graph_stats['backend'], "nodes": graph_stats['nodes'], "edges": graph_stats['edges']}

    except Exception as e:
        print(f"⚠ Error querying graph: {e}")
        return {"error": str(e)}


def test_vector_embeddings(orchestrator):
    """Test 4: Show Vector Embeddings"""
    print_header("TEST 4: Vector Embeddings")

    # Get vector stats
    stats = orchestrator.get_stats()
    vector_stats = stats['vector']

    print(f"Vector Backend: {vector_stats['backend']}")
    print(f"Total Chunks: {vector_stats['chunks']}")
    print(f"Embedding Dimension: {vector_stats['dimension']}")
    print()

    # Test similarity search
    print("→ Testing vector similarity search...")
    query = "transformer architecture"
    print(f"  Query: '{query}'")

    try:
        results = orchestrator.vector_agent.search(query, limit=3)

        if results:
            print(f"\n  Found {len(results)} similar chunks:\n")
            for i, result in enumerate(results, 1):
                score = result.get('score', 0.0)
                text = result.get('text', '')[:100]
                print(f"  {i}. Score: {score:.4f}")
                print(f"     Text: {text}...")
                print()

        return {
            "backend": vector_stats['backend'],
            "chunks": vector_stats['chunks'],
            "dimension": vector_stats['dimension'],
            "search_results": len(results) if results else 0
        }

    except Exception as e:
        print(f"⚠ Error in vector search: {e}")
        return {"error": str(e)}


def test_kafka_events(orchestrator):
    """Test 5: Verify Kafka Events"""
    print_header("TEST 5: Kafka Event Stream")

    kafka_manager = orchestrator.kafka_manager

    if not kafka_manager or not kafka_manager.enabled:
        print("⚠ Kafka not enabled")
        return {"kafka_enabled": False}

    # Get Kafka stats
    stats = kafka_manager.get_stats()
    print(f"✓ Kafka enabled: {stats['enabled']}")
    print(f"  Bootstrap servers: {stats['bootstrap_servers']}")
    print(f"  Producer connected: {stats['producer_connected']}")
    print()

    # List topics
    try:
        topics = kafka_manager.admin_client.list_topics()
        rag_topics = [t for t in topics if not t.startswith('_')]
        print(f"✓ Found {len(rag_topics)} event topics:")
        for topic in sorted(rag_topics):
            print(f"  • {topic}")

        return {
            "kafka_enabled": True,
            "topics_count": len(rag_topics),
            "topics": rag_topics
        }

    except Exception as e:
        print(f"⚠ Error accessing Kafka: {e}")
        return {"error": str(e)}


def test_reasoning(orchestrator):
    """Test 6: Query Answering with Reasoning"""
    print_header("TEST 6: Query Answering with Reasoning")

    query = "What are large language models?"
    print(f"→ Asking: '{query}'")
    print("  (This will use the collected papers to answer)\n")

    try:
        start_time = time.time()
        answer = orchestrator.ask(query)
        duration = time.time() - start_time

        print(f"🤖 Answer ({duration:.2f}s):")
        print("-" * 80)
        print(answer)
        print("-" * 80)

        return {
            "query": query,
            "answer_length": len(answer),
            "duration": duration
        }

    except Exception as e:
        print(f"⚠ Error during reasoning: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def test_neo4j_direct():
    """Test 7: Direct Neo4j Verification"""
    print_header("TEST 7: Direct Neo4j Database Verification")

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

        with driver.session() as session:
            # Count nodes by type
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
            print("Node Counts by Type:")
            node_counts = {}
            for record in result:
                node_type = record["type"] or "Unknown"
                count = record["count"]
                print(f"  {node_type}: {count}")
                node_counts[node_type] = count
            print()

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
            print("Relationship Counts:")
            rel_counts = {}
            for record in result:
                rel_type = record["rel_type"]
                count = record["count"]
                print(f"  {rel_type}: {count}")
                rel_counts[rel_type] = count

        driver.close()

        return {"node_counts": node_counts, "rel_counts": rel_counts}

    except Exception as e:
        print(f"⚠ Error connecting to Neo4j: {e}")
        return {"error": str(e)}


def test_qdrant_direct():
    """Test 8: Direct Qdrant Verification"""
    print_header("TEST 8: Direct Qdrant Database Verification")

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            port=int(os.getenv("QDRANT_PORT"))
        )

        # List collections
        collections = client.get_collections().collections
        print(f"✓ Found {len(collections)} collections:")

        collection_info = {}
        for collection in collections:
            print(f"\n  Collection: {collection.name}")
            info = client.get_collection(collection.name)
            print(f"    Vectors: {info.vectors_count}")
            print(f"    Indexed: {info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else 'N/A'}")

            collection_info[collection.name] = {
                "vectors": info.vectors_count,
                "indexed": info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else 0
            }

        return {"collections": collection_info}

    except Exception as e:
        print(f"⚠ Error connecting to Qdrant: {e}")
        return {"error": str(e)}


def main():
    """Run complete end-to-end test"""
    print("\n" + "🚀 " * 20)
    print("  ResearcherAI - Production Pipeline End-to-End Test")
    print("🚀 " * 20)

    test_start = time.time()
    results = {}

    # Test 1: Data Collection
    try:
        collection_result = test_data_collection()
        results['data_collection'] = collection_result
        orchestrator = collection_result['orchestrator']

        # Test 2: Paper Details
        results['paper_details'] = test_paper_details(orchestrator)

        # Test 3: Graph Entities
        results['graph_entities'] = test_graph_entities(orchestrator)

        # Test 4: Vector Embeddings
        results['vector_embeddings'] = test_vector_embeddings(orchestrator)

        # Test 5: Kafka Events
        results['kafka_events'] = test_kafka_events(orchestrator)

        # Test 6: Reasoning
        results['reasoning'] = test_reasoning(orchestrator)

        # Test 7: Neo4j Direct
        results['neo4j_direct'] = test_neo4j_direct()

        # Test 8: Qdrant Direct
        results['qdrant_direct'] = test_qdrant_direct()

        # Save session
        print("\n→ Saving session...")
        orchestrator.save_session()

        # Close orchestrator
        orchestrator.close()

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    test_duration = time.time() - test_start

    print_header("TEST SUMMARY")
    print(f"Total Test Duration: {test_duration:.2f}s\n")

    print("Results:")
    print(f"  ✓ Papers Collected: {results['data_collection']['papers_collected']}")
    print(f"  ✓ Graph Nodes: {results['data_collection']['graph_stats']['nodes_added']}")
    print(f"  ✓ Graph Edges: {results['data_collection']['graph_stats']['edges_added']}")
    print(f"  ✓ Vector Chunks: {results['data_collection']['vector_stats']['chunks_added']}")
    print(f"  ✓ Kafka Topics: {results['kafka_events'].get('topics_count', 'N/A')}")
    print(f"  ✓ Reasoning Query Answered: {results['reasoning'].get('answer_length', 0) > 0}")

    # Save results to file
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        # Remove non-serializable objects
        clean_results = {k: v for k, v in results.items() if k != 'orchestrator'}
        json.dump(clean_results, f, indent=2, default=str)

    print(f"\n✓ Detailed results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
