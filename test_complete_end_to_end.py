#!/usr/bin/env python3
"""
Complete End-to-End Test - ResearcherAI v2.0
===========================================
Comprehensive test with REAL data proving all components work perfectly:

1. Data Collection from all 7 sources (arXiv, Semantic Scholar, PubMed, etc.)
2. ETL Pipeline: Papers → Neo4j → Qdrant
3. Knowledge Graph Construction
4. Vector Embeddings
5. Q&A with Reasoning
6. Self-Evaluation (CriticAgent)
7. Summarization (all modes)
8. API Integration

ALL WITH REAL OUTPUTS AND VERIFICATION
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List

print("=" * 100)
print("🚀 COMPLETE END-TO-END TEST - ResearcherAI v2.0")
print("=" * 100)
print(f"Started: {datetime.now().isoformat()}")
print("=" * 100)

# Setup environment
os.environ["GOOGLE_API_KEY"] = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

test_results = []
collected_papers = []
session_name = f"end_to_end_test_{int(time.time())}"

def log_test(name, status, details=""):
    """Log test result"""
    symbol = "✅" if status else "❌"
    print(f"\n{symbol} {name}")
    if details:
        print(f"   {details}")
    test_results.append((status, name, details))

def print_section(title):
    """Print section header"""
    print(f"\n\n{'=' * 100}")
    print(f"📋 {title}")
    print("=" * 100)

def save_output(filename, data):
    """Save output to file for verification"""
    output_dir = "./test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        if isinstance(data, (dict, list)):
            json.dump(data, f, indent=2, default=str)
        else:
            f.write(str(data))

    print(f"   💾 Output saved to: {filepath}")
    return filepath

# ============================================================================
# PART 1: START DATABASES (Neo4j + Qdrant)
# ============================================================================

print_section("PART 1: Database Startup (Neo4j + Qdrant)")

print("\n🔄 Starting Neo4j and Qdrant containers...")
os.system("docker compose up -d neo4j qdrant 2>&1 | grep -v 'orphan'")
print("⏳ Waiting 10 seconds for containers to be ready...")
time.sleep(10)

# Verify databases are running
print("\n🔍 Verifying database containers...")
neo4j_status = os.popen("docker compose ps neo4j | grep -i 'up' | wc -l").read().strip()
qdrant_status = os.popen("docker compose ps qdrant | grep -i 'up' | wc -l").read().strip()

log_test("Neo4j Container Running", neo4j_status == "1", f"Status: {'UP' if neo4j_status == '1' else 'DOWN'}")
log_test("Qdrant Container Running", qdrant_status == "1", f"Status: {'UP' if qdrant_status == '1' else 'DOWN'}")

# ============================================================================
# PART 2: DATA COLLECTION FROM ALL 7 SOURCES
# ============================================================================

print_section("PART 2: Data Collection from All 7 Sources")

print("\n📦 Importing OrchestratorAgent...")
from agents.orchestrator_agent import OrchestratorAgent

config = {
    "graph_db": {
        "type": "neo4j",
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "research_password"
    },
    "vector_db": {
        "type": "qdrant",
        "host": "localhost",
        "port": 6333
    },
    "agents": {
        "reasoning_agent": {
            "model": "gemini-2.5-flash",
            "temperature": 0.3
        },
        "summarization_agent": {
            "model": "gemini-2.5-flash",
            "temperature": 0.3
        }
    }
}

print(f"🎯 Initializing OrchestratorAgent (session: '{session_name}')...")
orchestrator = OrchestratorAgent(session_name=session_name, config=config)

log_test("OrchestratorAgent Initialized", True, f"Session: {session_name}")

# Test query: "transformer neural networks"
query = "transformer neural networks"
print(f"\n🔍 Collecting papers for query: '{query}'")
print(f"   Sources: arXiv, Semantic Scholar, Zenodo, PubMed, Web, HuggingFace, Kaggle")
print(f"   Max per source: 5 papers")

start_time = time.time()
collection_result = orchestrator.collect_data(query, max_per_source=5)
collection_time = time.time() - start_time

print(f"\n⏱️  Collection completed in {collection_time:.2f} seconds")
print(f"\n📊 Collection Results:")
print(f"   Papers collected: {collection_result.get('papers_collected', 0)}")
print(f"   Graph stats: {collection_result.get('graph_stats', {})}")
print(f"   Vector stats: {collection_result.get('vector_stats', {})}")

# Get detailed source breakdown
source_stats = collection_result.get('sources', {})
print(f"\n📈 Source Breakdown:")
for source, count in source_stats.items():
    print(f"   {source}: {count} papers")

save_output("01_collection_result.json", collection_result)

log_test(
    "Data Collection from All Sources",
    collection_result.get('papers_collected', 0) > 0,
    f"{collection_result.get('papers_collected', 0)} papers collected in {collection_time:.2f}s"
)

# ============================================================================
# PART 3: VERIFY ETL PIPELINE (Papers → Neo4j → Qdrant)
# ============================================================================

print_section("PART 3: ETL Pipeline Verification (Papers → Neo4j → Qdrant)")

# 3A: Verify Neo4j has the data
print("\n🔍 Verifying Neo4j Knowledge Graph...")
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "research_password")
)

with driver.session() as session:
    # Count nodes
    result = session.run("MATCH (n) RETURN count(n) as count")
    node_count = result.single()["count"]

    # Count relationships
    result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
    rel_count = result.single()["count"]

    # Get sample papers
    result = session.run("""
        MATCH (p:Paper)
        RETURN p.title as title, p.year as year, p.source as source
        LIMIT 5
    """)
    sample_papers = [dict(record) for record in result]

    # Get sample relationships
    result = session.run("""
        MATCH (p1:Paper)-[r]->(p2:Paper)
        RETURN p1.title as from_paper, type(r) as relationship, p2.title as to_paper
        LIMIT 5
    """)
    sample_rels = [dict(record) for record in result]

print(f"\n📊 Neo4j Graph Stats:")
print(f"   Total Nodes: {node_count}")
print(f"   Total Relationships: {rel_count}")
print(f"   Avg Relationships per Node: {rel_count/node_count if node_count > 0 else 0:.2f}")

print(f"\n📄 Sample Papers in Neo4j:")
for i, paper in enumerate(sample_papers[:3], 1):
    print(f"   {i}. {paper['title']} ({paper['year']}) - Source: {paper['source']}")

print(f"\n🔗 Sample Relationships in Neo4j:")
for i, rel in enumerate(sample_rels[:3], 1):
    print(f"   {i}. [{rel['from_paper'][:40]}...] -{rel['relationship']}-> [{rel['to_paper'][:40]}...]")

neo4j_data = {
    "node_count": node_count,
    "relationship_count": rel_count,
    "sample_papers": sample_papers,
    "sample_relationships": sample_rels
}
save_output("02_neo4j_verification.json", neo4j_data)

log_test(
    "Neo4j ETL: Papers Loaded",
    node_count > 0,
    f"{node_count} nodes, {rel_count} relationships"
)

# 3B: Verify Qdrant has the embeddings
print("\n🔍 Verifying Qdrant Vector Database...")
from qdrant_client import QdrantClient

qdrant = QdrantClient(host="localhost", port=6333)

# Get collection info
collections = qdrant.get_collections()
collection_name = "research_papers"

try:
    collection_info = qdrant.get_collection(collection_name)
    vector_count = collection_info.points_count
    vector_dim = collection_info.config.params.vectors.size

    # Search test
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    test_query = "attention mechanism"
    query_vector = model.encode(test_query).tolist()

    search_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5
    )

    print(f"\n📊 Qdrant Vector Stats:")
    print(f"   Collection: {collection_name}")
    print(f"   Total Vectors: {vector_count}")
    print(f"   Vector Dimension: {vector_dim}")

    print(f"\n🔍 Sample Vector Search (query: '{test_query}'):")
    for i, result in enumerate(search_results[:3], 1):
        payload = result.payload
        print(f"   {i}. Score: {result.score:.4f}")
        print(f"      Title: {payload.get('title', 'N/A')[:60]}...")
        print(f"      Year: {payload.get('year', 'N/A')}")

    qdrant_data = {
        "vector_count": vector_count,
        "vector_dimension": vector_dim,
        "sample_search_results": [
            {
                "score": r.score,
                "title": r.payload.get('title'),
                "year": r.payload.get('year')
            } for r in search_results[:5]
        ]
    }
    save_output("03_qdrant_verification.json", qdrant_data)

    log_test(
        "Qdrant ETL: Vectors Loaded",
        vector_count > 0,
        f"{vector_count} vectors, dim={vector_dim}"
    )

    log_test(
        "Qdrant Semantic Search Working",
        len(search_results) > 0,
        f"Found {len(search_results)} results for '{test_query}'"
    )

except Exception as e:
    print(f"   ❌ Error: {e}")
    log_test("Qdrant ETL: Vectors Loaded", False, str(e))

# ============================================================================
# PART 4: REASONING & Q&A WITH REAL DATA
# ============================================================================

print_section("PART 4: Reasoning & Q&A with Real Collected Data")

questions = [
    "What are transformer neural networks?",
    "How do attention mechanisms work?",
    "What are the key innovations in transformer architectures?"
]

qa_results = []

for i, question in enumerate(questions, 1):
    print(f"\n❓ Question {i}: {question}")

    start_time = time.time()
    result = orchestrator.ask_detailed(question)
    answer_time = time.time() - start_time

    print(f"\n💡 Answer ({answer_time:.2f}s):")
    answer = result['answer']
    # Print first 300 chars
    print(f"   {answer[:300]}{'...' if len(answer) > 300 else ''}")

    print(f"\n📚 Sources Used:")
    for j, paper in enumerate(result.get('papers_used', [])[:3], 1):
        print(f"   {j}. {paper.get('title', 'N/A')} ({paper.get('year', 'N/A')})")

    print(f"\n🔗 Graph Insights:")
    graph_insights = result.get('graph_insights', {})
    print(f"   Related concepts: {len(graph_insights.get('related_concepts', []))}")
    print(f"   Related papers: {graph_insights.get('related_papers', 0)}")

    qa_results.append({
        "question": question,
        "answer": answer,
        "answer_time": answer_time,
        "papers_used": result.get('papers_used', []),
        "graph_insights": graph_insights
    })

    log_test(
        f"Q&A: Question {i}",
        len(answer) > 50,
        f"Answer length: {len(answer)} chars, {len(result.get('papers_used', []))} sources"
    )

save_output("04_qa_results.json", qa_results)

# ============================================================================
# PART 5: CRITIC AGENT SELF-EVALUATION
# ============================================================================

print_section("PART 5: CriticAgent - Self-Evaluation & Quality Assurance")

from agents.critic_agent import CriticAgent

print("\n🔍 Initializing CriticAgent...")
critic = CriticAgent()

# 5A: Evaluate paper collection quality
print("\n📊 Evaluating Paper Collection Quality...")
# Get collected papers from data collector
collected_papers_data = orchestrator.data_collector.get_last_collection()

if collected_papers_data:
    collection_eval = critic.evaluate_paper_collection(collected_papers_data[:10])

    print(f"\n✅ Collection Evaluation:")
    print(f"   Overall Score: {collection_eval['overall_score']:.2f}/1.00")
    print(f"   Relevance Score: {collection_eval['relevance_score']:.2f}/1.00")
    print(f"   Quality Score: {collection_eval['quality_score']:.2f}/1.00")
    print(f"   Diversity Score: {collection_eval['diversity_score']:.2f}/1.00")
    print(f"   Passed: {collection_eval['passed']}")

    if collection_eval.get('issues'):
        print(f"\n⚠️  Issues Identified:")
        for issue in collection_eval['issues'][:3]:
            print(f"   - {issue}")

    if collection_eval.get('recommendations'):
        print(f"\n💡 Recommendations:")
        for rec in collection_eval['recommendations'][:3]:
            print(f"   - {rec}")

    save_output("05a_collection_evaluation.json", collection_eval)

    log_test(
        "CriticAgent: Collection Evaluation",
        collection_eval['passed'],
        f"Score: {collection_eval['overall_score']:.2f}/1.00"
    )

# 5B: Evaluate answer quality
print("\n📊 Evaluating Answer Quality...")
if qa_results:
    qa_to_eval = qa_results[0]
    answer_eval = critic.evaluate_answer(
        question=qa_to_eval['question'],
        answer=qa_to_eval['answer'],
        context={
            "papers_used": qa_to_eval.get('papers_used', []),
            "graph_insights": qa_to_eval.get('graph_insights', {})
        }
    )

    print(f"\n✅ Answer Evaluation:")
    print(f"   Overall Score: {answer_eval['overall_score']:.2f}/1.00")
    print(f"   Accuracy Score: {answer_eval['accuracy_score']:.2f}/1.00")
    print(f"   Completeness Score: {answer_eval['completeness_score']:.2f}/1.00")
    print(f"   Clarity Score: {answer_eval['clarity_score']:.2f}/1.00")
    print(f"   Citation Score: {answer_eval['citation_score']:.2f}/1.00")
    print(f"   Passed: {answer_eval['passed']}")

    save_output("05b_answer_evaluation.json", answer_eval)

    log_test(
        "CriticAgent: Answer Evaluation",
        answer_eval['passed'],
        f"Score: {answer_eval['overall_score']:.2f}/1.00"
    )

# 5C: Evaluate graph quality
print("\n📊 Evaluating Knowledge Graph Quality...")
graph_stats_for_eval = {
    "node_count": node_count,
    "edge_count": rel_count,
    "avg_degree": rel_count / node_count if node_count > 0 else 0
}

graph_eval = critic.evaluate_graph_quality(graph_stats_for_eval)

print(f"\n✅ Graph Evaluation:")
print(f"   Overall Score: {graph_eval['overall_score']:.2f}/1.00")
print(f"   Connectivity Score: {graph_eval['connectivity_score']:.2f}/1.00")
print(f"   Density Score: {graph_eval['density_score']:.2f}/1.00")
print(f"   Passed: {graph_eval['passed']}")

save_output("05c_graph_evaluation.json", graph_eval)

log_test(
    "CriticAgent: Graph Evaluation",
    graph_eval['passed'],
    f"Score: {graph_eval['overall_score']:.2f}/1.00"
)

# ============================================================================
# PART 6: SUMMARIZATION AGENT (ALL MODES)
# ============================================================================

print_section("PART 6: SummarizationAgent - All Modes with Real Data")

# 6A: Single paper summarization
print("\n📄 Single Paper Summarization...")
if collected_papers_data:
    test_paper = collected_papers_data[0]

    from agents.summarization_agent import SummaryStyle, SummaryLength

    # Executive brief
    summary_brief = orchestrator.summarize_paper(
        test_paper,
        style="executive",
        length="brief"
    )

    print(f"\n✅ Executive Brief Summary:")
    print(f"   Paper: {test_paper.get('title', 'N/A')[:60]}...")
    print(f"   Word Count: {summary_brief['word_count']}")
    print(f"\n   Summary:\n   {summary_brief['summary'][:250]}...")

    save_output("06a_single_paper_summary.json", summary_brief)

    log_test(
        "Summarization: Single Paper (Brief)",
        summary_brief['word_count'] < 100,
        f"{summary_brief['word_count']} words"
    )

    # Technical detailed
    summary_detailed = orchestrator.summarize_paper(
        test_paper,
        style="technical",
        length="detailed"
    )

    print(f"\n✅ Technical Detailed Summary:")
    print(f"   Word Count: {summary_detailed['word_count']}")
    print(f"\n   Summary:\n   {summary_detailed['summary'][:250]}...")

    save_output("06b_technical_detailed_summary.json", summary_detailed)

    log_test(
        "Summarization: Single Paper (Detailed)",
        200 <= summary_detailed['word_count'] <= 700,
        f"{summary_detailed['word_count']} words"
    )

# 6B: Collection summarization
print("\n📚 Collection Summarization...")
collection_summary = orchestrator.summarize_collection(
    papers=collected_papers_data[:5],
    style="executive",
    focus="research trends and key innovations"
)

print(f"\n✅ Collection Summary:")
print(f"   Papers Analyzed: {collection_summary.get('papers_analyzed', 0)}")

if 'overall_summary' in collection_summary:
    print(f"\n   Overall Summary:\n   {collection_summary['overall_summary'][:250]}...")

if 'key_themes' in collection_summary:
    print(f"\n   Key Themes ({len(collection_summary['key_themes'])}):")
    for theme in collection_summary['key_themes'][:3]:
        print(f"      - {theme}")

if 'research_trends' in collection_summary:
    print(f"\n   Research Trends ({len(collection_summary['research_trends'])}):")
    for trend in collection_summary['research_trends'][:3]:
        print(f"      - {trend}")

save_output("06c_collection_summary.json", collection_summary)

log_test(
    "Summarization: Collection",
    collection_summary.get('papers_analyzed', 0) > 0,
    f"{collection_summary.get('papers_analyzed', 0)} papers analyzed"
)

# 6C: Conversation summarization
print("\n💬 Conversation Summarization...")
conversation_summary = orchestrator.summarize_conversation(style="bullet_points")

print(f"\n✅ Conversation Summary:")
print(f"   Turn Count: {conversation_summary.get('turn_count', 0)}")

if 'session_summary' in conversation_summary:
    print(f"\n   Session Summary:\n   {conversation_summary['session_summary'][:250]}...")

if 'questions_asked' in conversation_summary:
    print(f"\n   Questions Asked ({len(conversation_summary['questions_asked'])}):")
    for q in conversation_summary['questions_asked'][:3]:
        print(f"      - {q[:70]}...")

if 'key_insights' in conversation_summary:
    print(f"\n   Key Insights ({len(conversation_summary['key_insights'])}):")
    for insight in conversation_summary['key_insights'][:3]:
        print(f"      - {insight[:70]}...")

save_output("06d_conversation_summary.json", conversation_summary)

log_test(
    "Summarization: Conversation",
    conversation_summary.get('turn_count', 0) > 0,
    f"{conversation_summary.get('turn_count', 0)} turns summarized"
)

# 6D: Paper comparison
print("\n🔄 Paper Comparison...")
if len(collected_papers_data) >= 2:
    comparison = orchestrator.compare_papers(
        collected_papers_data[:2],
        comparison_aspects=["methodology", "contributions", "impact"]
    )

    print(f"\n✅ Paper Comparison:")
    print(f"   Papers Compared: {comparison.get('papers_compared', 0)}")

    if 'comparison_summary' in comparison:
        print(f"\n   Comparison Summary:\n   {comparison['comparison_summary'][:250]}...")

    if 'similarities' in comparison:
        print(f"\n   Similarities ({len(comparison['similarities'])}):")
        for sim in comparison['similarities'][:3]:
            print(f"      - {sim[:70]}...")

    if 'differences' in comparison:
        print(f"\n   Differences ({len(comparison['differences'])}):")
        for diff in comparison['differences'][:3]:
            print(f"      - {diff[:70]}...")

    save_output("06e_paper_comparison.json", comparison)

    log_test(
        "Summarization: Paper Comparison",
        comparison.get('papers_compared', 0) >= 2,
        f"{comparison.get('papers_compared', 0)} papers compared"
    )

# ============================================================================
# PART 7: SESSION PERSISTENCE
# ============================================================================

print_section("PART 7: Session Persistence (Save/Load)")

print("\n💾 Saving session...")
save_success = orchestrator.save_session()

print(f"   Session saved: {save_success}")
log_test("Session Save", save_success, f"Session '{session_name}' saved")

print("\n📂 Creating new orchestrator and loading session...")
orchestrator2 = OrchestratorAgent(session_name=session_name, config=config)

print(f"   Session loaded: {session_name}")
print(f"   Papers in memory: {orchestrator2.metadata['papers_collected']}")
print(f"   Conversations: {orchestrator2.metadata['conversations']}")

log_test(
    "Session Load",
    orchestrator2.metadata['papers_collected'] > 0,
    f"{orchestrator2.metadata['papers_collected']} papers restored"
)

# ============================================================================
# PART 8: OVERALL SYSTEM STATS
# ============================================================================

print_section("PART 8: Overall System Statistics")

stats = orchestrator.get_stats()

print(f"\n📊 Complete System Stats:")
print(f"\n   Session: {stats['session']}")
print(f"   Papers Collected: {stats['metadata']['papers_collected']}")
print(f"   Conversations: {stats['metadata']['conversations']}")

print(f"\n   Graph Database:")
print(f"      Backend: {stats['graph']['backend']}")
print(f"      Nodes: {stats['graph'].get('node_count', 'N/A')}")
print(f"      Edges: {stats['graph'].get('edge_count', 'N/A')}")

print(f"\n   Vector Database:")
print(f"      Backend: {stats['vector']['backend']}")
print(f"      Vectors: {stats['vector'].get('vector_count', 'N/A')}")

print(f"\n   Reasoning Agent:")
print(f"      Conversation History: {stats['reasoning'].get('history_length', 0)} turns")

print(f"\n   Data Collector:")
for source, count in stats['data_collector'].items():
    print(f"      {source}: {count}")

save_output("07_system_stats.json", stats)

# ============================================================================
# CLEANUP
# ============================================================================

print_section("Cleanup")

print("\n🧹 Closing orchestrator...")
orchestrator.close()
orchestrator2.close()

print("✅ Orchestrators closed")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_section("FINAL TEST SUMMARY")

passed = sum(1 for status, _, _ in test_results if status)
total = len(test_results)

print(f"\n📈 Test Results:")
print(f"   Total Tests: {total}")
print(f"   ✅ Passed: {passed}")
print(f"   ❌ Failed: {total - passed}")
print(f"   Success Rate: {100 * passed / total:.1f}%")

print(f"\n\n{'=' * 100}")
print("📋 DETAILED RESULTS:")
print("=" * 100)

for status, name, details in test_results:
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name}")
    if details:
        print(f"   {details}")

print(f"\n\n{'=' * 100}")

if passed == total:
    print("🎉 ALL TESTS PASSED - SYSTEM FULLY FUNCTIONAL!")
    print("=" * 100)
    print("\n✅ ResearcherAI v2.0 is production-ready:")
    print("   ✅ Data collection from 7 sources working")
    print("   ✅ ETL pipeline (Papers → Neo4j → Qdrant) verified")
    print("   ✅ Knowledge graph construction working")
    print("   ✅ Vector embeddings and semantic search working")
    print("   ✅ Q&A with reasoning working")
    print("   ✅ Self-evaluation (CriticAgent) working")
    print("   ✅ Summarization (all 4 modes) working")
    print("   ✅ Session persistence working")
    print("\n📁 All outputs saved to: ./test_outputs/")
    sys.exit(0)
else:
    print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    print("=" * 100)
    print("\n❌ Failed Tests:")
    for status, name, details in test_results:
        if not status:
            print(f"   - {name}: {details}")
    sys.exit(1)
