#!/usr/bin/env python3
"""
Simple verification script to show collected data from databases.
"""

import os
import sys

# Set environment variables (use Docker network hostnames)
os.environ["NEO4J_URI"] = "bolt://neo4j:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "research_password"
os.environ["QDRANT_HOST"] = "qdrant"
os.environ["QDRANT_PORT"] = "6333"
os.environ["KAFKA_BOOTSTRAP"] = "kafka:9092"

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def verify_neo4j():
    """Verify Neo4j database contains data"""
    print_header("Neo4j Database - Sample Papers")

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

        with driver.session() as session:
            # Get sample papers
            result = session.run("""
                MATCH (p:Paper)
                RETURN p.title as title, p.year as year, p.source as source
                ORDER BY p.title
                LIMIT 10
            """)

            papers = list(result)
            print(f"Found {len(papers)} papers in Neo4j:\n")

            for i, record in enumerate(papers, 1):
                print(f"{i}. {record['title']}")
                print(f"   Year: {record.get('year', 'N/A')} | Source: {record.get('source', 'N/A')}\n")

            # Get authors
            print_header("Neo4j Database - Sample Authors")

            result = session.run("""
                MATCH (a:Author)-[:AUTHORED]->(p:Paper)
                RETURN a.name as author, count(p) as paper_count
                ORDER BY paper_count DESC, a.name
                LIMIT 10
            """)

            authors = list(result)
            print(f"Found {len(authors)} authors:\n")

            for i, record in enumerate(authors, 1):
                print(f"{i}. {record['author']} ({record['paper_count']} papers)")

            # Get relationships
            print_header("Neo4j Database - Sample Relationships")

            result = session.run("""
                MATCH (a:Author)-[r:AUTHORED]->(p:Paper)
                RETURN a.name as author, p.title as paper
                LIMIT 10
            """)

            rels = list(result)
            for i, record in enumerate(rels, 1):
                print(f"{i}. {record['author']} AUTHORED:")
                print(f"   '{record['paper'][:70]}...'\n")

            # Get node counts
            print_header("Neo4j Database - Statistics")

            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(n) as count
                ORDER BY count DESC
            """)

            print("Node Counts:")
            total_nodes = 0
            for record in result:
                node_type = record["type"] or "Unknown"
                count = record["count"]
                print(f"  {node_type}: {count}")
                total_nodes += count

            result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
            total_edges = list(result)[0]["total"]

            print(f"\nTotal: {total_nodes} nodes, {total_edges} edges")

        driver.close()
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_qdrant():
    """Verify Qdrant database contains vectors"""
    print_header("Qdrant Database - Vector Collections")

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            port=int(os.getenv("QDRANT_PORT"))
        )

        collections = client.get_collections().collections

        print(f"Found {len(collections)} collections:\n")

        for collection in collections:
            info = client.get_collection(collection.name)
            print(f"Collection: {collection.name}")
            print(f"  Vectors: {info.vectors_count}")
            print(f"  Status: {info.status}")
            if hasattr(info.config, 'params'):
                print(f"  Dimension: {info.config.params.vectors.size}")
            print()

            # Test search
            if info.vectors_count > 0:
                print(f"  Testing search in '{collection.name}'...")
                try:
                    # Use a generic query embedding (all zeros as placeholder)
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    query_vector = model.encode("large language models")

                    results = client.search(
                        collection_name=collection.name,
                        query_vector=query_vector.tolist(),
                        limit=3
                    )

                    print(f"  Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        text = result.payload.get('text', '')[:80] if result.payload else 'No text'
                        print(f"    {i}. Score: {result.score:.4f} | {text}...")
                    print()

                except Exception as e:
                    print(f"  ⚠ Search error: {e}\n")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_kafka():
    """Verify Kafka topics"""
    print_header("Kafka - Event Topics")

    try:
        from kafka import KafkaAdminClient

        admin_client = KafkaAdminClient(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "kafka:9092"),
            client_id='verifier'
        )

        topics = admin_client.list_topics()
        rag_topics = sorted([t for t in topics if not t.startswith('_')])

        print(f"Found {len(rag_topics)} RAG event topics:\n")

        for topic in rag_topics:
            print(f"  • {topic}")

        admin_client.close()
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "🔍 " * 20)
    print("  ResearcherAI - Data Verification")
    print("🔍 " * 20)

    results = {}

    # Verify Neo4j
    results['neo4j'] = verify_neo4j()

    # Verify Qdrant
    results['qdrant'] = verify_qdrant()

    # Verify Kafka
    results['kafka'] = verify_kafka()

    # Summary
    print_header("VERIFICATION SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Results: {passed}/{total} systems verified\n")

    for system, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {system.upper()}")

    print("\n" + "=" * 80)

    if passed == total:
        print("🎉 ALL VERIFICATIONS PASSED!")
        return 0
    else:
        print(f"⚠️  {total - passed} VERIFICATION(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
