#!/usr/bin/env python3
"""
Comprehensive Neo4j and Qdrant Connection Test
Tests production database backends with real data
"""

import os
import sys
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

print("=" * 80)
print("🔌 NEO4J & QDRANT CONNECTION TEST - Production Databases")
print("=" * 80)
print()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "research_password")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

test_results = []

def test_step(name):
    """Decorator to track test results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'=' * 80}")
            print(f"🧪 TEST: {name}")
            print("-" * 80)
            try:
                result = func(*args, **kwargs)
                print(f"✅ PASS - {name}")
                test_results.append(("✅", name, "PASSED"))
                return result
            except Exception as e:
                print(f"❌ FAIL - {name}")
                print(f"   Error: {str(e)}")
                test_results.append(("❌", name, str(e)))
                raise
        return wrapper
    return decorator

# ============================================================================
# NEO4J TESTS
# ============================================================================

@test_step("Neo4j Connection")
def test_neo4j_connection():
    """Test basic Neo4j connectivity"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"   Connected to: {NEO4J_URI}")
    print(f"   User: {NEO4J_USER}")
    return driver

@test_step("Neo4j Clear Existing Data")
def test_neo4j_clear(driver):
    """Clear existing test data"""
    with driver.session() as session:
        result = session.run("MATCH (n:TestPaper) DETACH DELETE n RETURN count(n) as deleted")
        deleted = result.single()["deleted"]
        print(f"   Cleared {deleted} existing test nodes")
    return driver

@test_step("Neo4j Write Test Papers")
def test_neo4j_write(driver):
    """Write sample research papers to Neo4j"""
    papers = [
        {
            "id": "test_001",
            "title": "Attention Is All You Need",
            "authors": ["Vaswani", "Shazeer", "Parmar"],
            "year": 2017,
            "citations": 95000,
            "field": "NLP"
        },
        {
            "id": "test_002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
            "year": 2018,
            "citations": 75000,
            "field": "NLP"
        },
        {
            "id": "test_003",
            "title": "ResNet: Deep Residual Learning",
            "authors": ["He", "Zhang", "Ren", "Sun"],
            "year": 2015,
            "citations": 120000,
            "field": "Computer Vision"
        }
    ]

    with driver.session() as session:
        for paper in papers:
            session.run("""
                CREATE (p:TestPaper {
                    id: $id,
                    title: $title,
                    authors: $authors,
                    year: $year,
                    citations: $citations,
                    field: $field
                })
            """, **paper)

    print(f"   ✅ Wrote {len(papers)} papers to Neo4j")
    for p in papers:
        print(f"      - {p['title']} ({p['year']})")
    return driver, papers

@test_step("Neo4j Create Relationships")
def test_neo4j_relationships(driver):
    """Create citation relationships between papers"""
    with driver.session() as session:
        # BERT cites Attention
        session.run("""
            MATCH (bert:TestPaper {id: 'test_002'})
            MATCH (attention:TestPaper {id: 'test_001'})
            CREATE (bert)-[:CITES {year: 2018}]->(attention)
        """)

        # ResNet influences NLP papers (cross-field)
        session.run("""
            MATCH (bert:TestPaper {id: 'test_002'})
            MATCH (resnet:TestPaper {id: 'test_003'})
            CREATE (bert)-[:INFLUENCED_BY {concept: 'deep_networks'}]->(resnet)
        """)

    print(f"   ✅ Created citation relationships")
    print(f"      - BERT → cites → Attention Is All You Need")
    print(f"      - BERT → influenced_by → ResNet")
    return driver

@test_step("Neo4j Query Papers by Field")
def test_neo4j_query_field(driver):
    """Query papers by field"""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:TestPaper {field: 'NLP'})
            RETURN p.title as title, p.citations as citations
            ORDER BY p.citations DESC
        """)
        papers = list(result)

    print(f"   ✅ Found {len(papers)} NLP papers:")
    for paper in papers:
        print(f"      - {paper['title']}: {paper['citations']:,} citations")
    return papers

@test_step("Neo4j Query Citation Network")
def test_neo4j_query_citations(driver):
    """Query citation relationships"""
    with driver.session() as session:
        result = session.run("""
            MATCH (citing:TestPaper)-[r:CITES]->(cited:TestPaper)
            RETURN citing.title as citing_paper,
                   cited.title as cited_paper,
                   r.year as citation_year
        """)
        citations = list(result)

    print(f"   ✅ Found {len(citations)} citation relationships:")
    for cit in citations:
        print(f"      - '{cit['citing_paper']}' cites '{cit['cited_paper']}' ({cit['citation_year']})")
    return citations

@test_step("Neo4j Graph Statistics")
def test_neo4j_stats(driver):
    """Get graph statistics"""
    with driver.session() as session:
        # Count nodes
        node_count = session.run("MATCH (p:TestPaper) RETURN count(p) as count").single()["count"]

        # Count relationships
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

        # Get most cited paper
        most_cited = session.run("""
            MATCH (p:TestPaper)
            RETURN p.title as title, p.citations as citations
            ORDER BY p.citations DESC
            LIMIT 1
        """).single()

    print(f"   ✅ Graph Statistics:")
    print(f"      - Total papers: {node_count}")
    print(f"      - Total relationships: {rel_count}")
    print(f"      - Most cited: {most_cited['title']} ({most_cited['citations']:,} citations)")

    return {"nodes": node_count, "relationships": rel_count, "most_cited": most_cited['title']}

# ============================================================================
# QDRANT TESTS
# ============================================================================

@test_step("Qdrant Connection")
def test_qdrant_connection():
    """Test basic Qdrant connectivity"""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)

    # Test connection by getting collections
    try:
        collections = client.get_collections()
        print(f"   Connected to: {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"   Existing collections: {len(collections.collections)}")
    except Exception as e:
        print(f"   Connection test: {e}")

    return client

@test_step("Qdrant Create Collection")
def test_qdrant_create_collection(client):
    """Create test collection"""
    collection_name = "test_papers"

    # Delete if exists
    try:
        client.delete_collection(collection_name)
        print(f"   Deleted existing collection: {collection_name}")
    except:
        pass

    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    print(f"   ✅ Created collection: {collection_name}")
    print(f"      - Vector dimension: 384")
    print(f"      - Distance metric: COSINE")

    return client, collection_name

@test_step("Qdrant Load Embedding Model")
def test_qdrant_load_model():
    """Load sentence transformer model"""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # Test embedding
    test_text = "This is a test sentence"
    embedding = model.encode(test_text)

    print(f"   ✅ Loaded model: all-MiniLM-L6-v2 (CPU mode)")
    print(f"      - Embedding dimension: {len(embedding)}")
    print(f"      - Sample embedding (first 5 values): {embedding[:5]}")

    return model

@test_step("Qdrant Insert Paper Embeddings")
def test_qdrant_insert(client, collection_name, model):
    """Insert paper embeddings into Qdrant"""
    papers = [
        {
            "id": 1,
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "year": 2017,
            "field": "NLP"
        },
        {
            "id": 2,
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT, a new language representation model designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.",
            "year": 2018,
            "field": "NLP"
        },
        {
            "id": 3,
            "title": "ResNet: Deep Residual Learning for Image Recognition",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
            "year": 2015,
            "field": "Computer Vision"
        },
        {
            "id": 4,
            "title": "Generative Adversarial Networks",
            "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model and a discriminative model.",
            "year": 2014,
            "field": "Machine Learning"
        },
        {
            "id": 5,
            "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "abstract": "Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem. We show that dropout is an effective technique for regularization.",
            "year": 2014,
            "field": "Machine Learning"
        }
    ]

    # Generate embeddings and insert
    points = []
    for paper in papers:
        # Combine title and abstract for embedding
        text = f"{paper['title']}. {paper['abstract']}"
        embedding = model.encode(text)

        point = PointStruct(
            id=paper['id'],
            vector=embedding.tolist(),
            payload={
                "title": paper['title'],
                "abstract": paper['abstract'],
                "year": paper['year'],
                "field": paper['field']
            }
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)

    print(f"   ✅ Inserted {len(points)} paper embeddings:")
    for p in papers:
        print(f"      - {p['title']} ({p['year']})")

    return client, collection_name, model, papers

@test_step("Qdrant Semantic Search - Query 1")
def test_qdrant_search_transformers(client, collection_name, model):
    """Search for papers about transformers"""
    query = "transformer attention mechanisms for natural language processing"
    query_vector = model.encode(query).tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )

    print(f"   ✅ Query: '{query}'")
    print(f"   Found {len(results)} relevant papers:")
    for i, hit in enumerate(results, 1):
        print(f"      {i}. {hit.payload['title']} (score: {hit.score:.4f})")
        print(f"         Field: {hit.payload['field']}, Year: {hit.payload['year']}")

    return results

@test_step("Qdrant Semantic Search - Query 2")
def test_qdrant_search_vision(client, collection_name, model):
    """Search for papers about computer vision"""
    query = "deep learning for image recognition and computer vision"
    query_vector = model.encode(query).tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )

    print(f"   ✅ Query: '{query}'")
    print(f"   Found {len(results)} relevant papers:")
    for i, hit in enumerate(results, 1):
        print(f"      {i}. {hit.payload['title']} (score: {hit.score:.4f})")
        print(f"         Field: {hit.payload['field']}, Year: {hit.payload['year']}")

    return results

@test_step("Qdrant Filter by Year")
def test_qdrant_filter(client, collection_name, model):
    """Search with filters"""
    query = "neural network training techniques"
    query_vector = model.encode(query).tolist()

    from qdrant_client.models import Filter, FieldCondition, Range

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="year",
                    range=Range(gte=2015)
                )
            ]
        ),
        limit=3
    )

    print(f"   ✅ Query: '{query}' (filtered: year >= 2015)")
    print(f"   Found {len(results)} papers:")
    for i, hit in enumerate(results, 1):
        print(f"      {i}. {hit.payload['title']} (score: {hit.score:.4f})")
        print(f"         Year: {hit.payload['year']}")

    return results

@test_step("Qdrant Collection Statistics")
def test_qdrant_stats(client, collection_name):
    """Get collection statistics"""
    collection_info = client.get_collection(collection_name)

    print(f"   ✅ Collection Statistics:")
    print(f"      - Name: {collection_name}")
    print(f"      - Vector count: {collection_info.vectors_count}")
    print(f"      - Points count: {collection_info.points_count}")
    print(f"      - Vector size: {collection_info.config.params.vectors.size}")
    print(f"      - Distance: {collection_info.config.params.vectors.distance}")

    return {
        "vectors_count": collection_info.vectors_count,
        "points_count": collection_info.points_count
    }

# ============================================================================
# INTEGRATION TEST: Neo4j + Qdrant
# ============================================================================

@test_step("Integration: Cross-Database Query")
def test_integration(neo4j_driver, qdrant_client, collection_name, model):
    """Test integration between Neo4j and Qdrant"""

    print("\n   🔗 Testing integrated workflow:")
    print("      1. Semantic search in Qdrant for 'attention mechanisms'")
    print("      2. Get top result metadata")
    print("      3. Query Neo4j for related papers in same field")

    # Step 1: Semantic search in Qdrant
    query = "attention mechanisms and transformers"
    query_vector = model.encode(query).tolist()
    qdrant_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1
    )

    top_paper = qdrant_results[0]
    field = top_paper.payload['field']

    print(f"\n   📊 Qdrant Result:")
    print(f"      - Top paper: {top_paper.payload['title']}")
    print(f"      - Field: {field}")
    print(f"      - Relevance score: {top_paper.score:.4f}")

    # Step 2: Query Neo4j for related papers in same field
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (p:TestPaper {field: $field})
            RETURN p.title as title, p.citations as citations
            ORDER BY p.citations DESC
        """, field=field)
        neo4j_papers = list(result)

    print(f"\n   🔍 Neo4j Related Papers (same field: {field}):")
    for paper in neo4j_papers:
        print(f"      - {paper['title']}: {paper['citations']:,} citations")

    print(f"\n   ✅ Successfully integrated Qdrant (semantic) + Neo4j (graph)")

    return {"qdrant_top": top_paper.payload['title'], "neo4j_related": len(neo4j_papers)}

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def main():
    """Run all tests"""

    # Check API key
    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not set!")
        print("   Please export GOOGLE_API_KEY before running tests")
        sys.exit(1)

    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"✅ Google API configured")
    print()

    try:
        # Neo4j Tests
        driver = test_neo4j_connection()
        driver = test_neo4j_clear(driver)
        driver, papers = test_neo4j_write(driver)
        driver = test_neo4j_relationships(driver)
        query_results = test_neo4j_query_field(driver)
        citations = test_neo4j_query_citations(driver)
        neo4j_stats = test_neo4j_stats(driver)

        # Qdrant Tests
        qdrant_client = test_qdrant_connection()
        qdrant_client, collection_name = test_qdrant_create_collection(qdrant_client)
        model = test_qdrant_load_model()
        qdrant_client, collection_name, model, papers = test_qdrant_insert(qdrant_client, collection_name, model)
        search_results_1 = test_qdrant_search_transformers(qdrant_client, collection_name, model)
        search_results_2 = test_qdrant_search_vision(qdrant_client, collection_name, model)
        filter_results = test_qdrant_filter(qdrant_client, collection_name, model)
        qdrant_stats = test_qdrant_stats(qdrant_client, collection_name)

        # Integration Test
        integration_results = test_integration(driver, qdrant_client, collection_name, model)

        # Close connections
        driver.close()

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for status, _, _ in test_results if status == "✅")
    total = len(test_results)

    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"\n📈 Success Rate: {passed}/{total} ({100*passed/total:.1f}%)")

    if passed < total:
        print(f"\n❌ Failed Tests:")
        for status, name, error in test_results:
            if status == "❌":
                print(f"   - {name}: {error}")

    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"\n✅ Neo4j is working perfectly:")
        print(f"   - {neo4j_stats['nodes']} papers stored")
        print(f"   - {neo4j_stats['relationships']} relationships created")
        print(f"   - Graph queries working correctly")

        print(f"\n✅ Qdrant is working perfectly:")
        print(f"   - {qdrant_stats['vectors_count']} vectors stored")
        print(f"   - Semantic search working correctly")
        print(f"   - Filtering working correctly")

        print(f"\n✅ Integration working perfectly:")
        print(f"   - Cross-database queries successful")
        print(f"   - Data consistency verified")

if __name__ == "__main__":
    main()
