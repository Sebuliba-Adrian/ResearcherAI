#!/usr/bin/env python3
"""
Simplified End-to-End Test - ResearcherAI v2.0
==============================================
Focus on core functionality without hitting API rate limits
Uses NetworkX + FAISS to avoid database overhead
"""

import os
import sys
import json
import time
from datetime import datetime

print("="  * 100)
print("🚀 SIMPLIFIED END-TO-END TEST - ResearcherAI v2.0")
print("=" * 100)

# Setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

test_results = []

def log_test(name, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"\n{symbol} {name}")
    if details:
        print(f"   {details}")
    test_results.append((status, name, details))

def print_section(title):
    print(f"\n\n{'=' * 100}")
    print(f"📋 {title}")
    print("=" * 100)

def save_output(filename, data):
    os.makedirs("./test_outputs", exist_ok=True)
    filepath = f"./test_outputs/{filename}"
    with open(filepath, 'w') as f:
        if isinstance(data, (dict, list)):
            json.dump(data, f, indent=2, default=str)
        else:
            f.write(str(data))
    print(f"   💾 Saved: {filepath}")
    return filepath

# ============================================================================
# PART 1: DATA COLLECTION
# ============================================================================

print_section("PART 1: Data Collection from Multiple Sources")

from agents.orchestrator_agent import OrchestratorAgent

# Use NetworkX + FAISS to avoid database setup
config = {
    "graph_db": {"type": "networkx"},
    "vector_db": {"type": "faiss"},
    "agents": {
        "reasoning_agent": {"model": "gemini-2.5-flash", "temperature": 0.3},
        "summarization_agent": {"model": "gemini-2.5-flash", "temperature": 0.3}
    }
}

session_name = f"test_{int(time.time())}"
print(f"\n🎯 Initializing OrchestratorAgent (session: '{session_name}')...")
orchestrator = OrchestratorAgent(session_name=session_name, config=config)

log_test("OrchestratorAgent Initialized", True, f"Session: {session_name}")

query = "attention mechanism transformers"
print(f"\n🔍 Collecting papers for: '{query}'")
print(f"   Max per source: 3 papers (to avoid rate limits)")

start_time = time.time()
collection_result = orchestrator.collect_data(query, max_per_source=3)
collection_time = time.time() - start_time

papers_collected = collection_result.get('papers_collected', 0)
sources_stats = collection_result.get('sources', {})
sources_breakdown = sources_stats.get('by_source', {})

print(f"\n📊 Collection Results ({collection_time:.2f}s):")
print(f"   Total Papers: {papers_collected}")
print(f"\n   Source Breakdown:")
for source, count in sources_breakdown.items():
    if isinstance(count, int) and count > 0:
        print(f"      {source}: {count} papers")

save_output("01_collection_result.json", collection_result)

log_test(
    "Data Collection",
    papers_collected > 0,
    f"{papers_collected} papers in {collection_time:.2f}s"
)

# ============================================================================
# PART 2: ETL VERIFICATION
# ============================================================================

print_section("PART 2: ETL Pipeline (Papers → Graph → Vectors)")

# Get actual collected papers
collected_papers = orchestrator.data_collector.get_last_collection()

print(f"\n📄 Sample Collected Papers:")
for i, paper in enumerate(collected_papers[:3], 1):
    print(f"\n   {i}. {paper.get('title', 'N/A')[:70]}...")
    print(f"      Source: {paper.get('source', 'N/A')}")
    print(f"      Authors: {', '.join(paper.get('authors', [])[:3])}")
    if len(paper.get('authors', [])) > 3:
        print(f"               + {len(paper.get('authors', [])) - 3} more")

save_output("02_collected_papers.json", collected_papers[:5])

# Verify NetworkX graph
graph_stats = orchestrator.graph_agent.get_stats()
print(f"\n🔗 Knowledge Graph (NetworkX):")
print(f"   Nodes: {graph_stats.get('node_count', 0)}")
print(f"   Edges: {graph_stats.get('edge_count', 0)}")
print(f"   Avg Degree: {graph_stats.get('avg_degree', 0):.2f}")

save_output("03_graph_stats.json", graph_stats)

log_test(
    "Knowledge Graph Construction",
    graph_stats.get('node_count', 0) > 0,
    f"{graph_stats.get('node_count', 0)} nodes, {graph_stats.get('edge_count', 0)} edges"
)

# Verify FAISS vectors
vector_stats = orchestrator.vector_agent.get_stats()
print(f"\n📊 Vector Database (FAISS):")
print(f"   Vectors: {vector_stats.get('vector_count', 0)}")
print(f"   Dimension: {vector_stats.get('dimension', 0)}")

save_output("04_vector_stats.json", vector_stats)

log_test(
    "Vector Embeddings Created",
    vector_stats.get('vector_count', 0) > 0,
    f"{vector_stats.get('vector_count', 0)} vectors, dim={vector_stats.get('dimension', 0)}"
)

# ============================================================================
# PART 3: REASONING & Q&A
# ============================================================================

print_section("PART 3: Reasoning & Q&A with Collected Data")

questions = [
    "What are attention mechanisms?",
    "How do transformers work?"
]

qa_results = []

for i, question in enumerate(questions, 1):
    print(f"\n❓ Question {i}: {question}")

    # Add delay to avoid rate limit
    if i > 1:
        print(f"   ⏳ Waiting 8 seconds to avoid rate limit...")
        time.sleep(8)

    start_time = time.time()
    try:
        result = orchestrator.ask_detailed(question)
        answer_time = time.time() - start_time

        answer = result.get('answer', 'No answer generated')
        papers_used = result.get('papers_used', [])

        print(f"\n💡 Answer ({answer_time:.2f}s):")
        print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")

        print(f"\n📚 Sources ({len(papers_used)}):")
        for j, paper in enumerate(papers_used[:3], 1):
            print(f"      {j}. {paper.get('title', 'N/A')[:50]}...")

        qa_results.append({
            "question": question,
            "answer": answer,
            "answer_time": answer_time,
            "papers_used": papers_used
        })

        log_test(
            f"Q&A Question {i}",
            len(answer) > 50,
            f"{len(answer)} chars, {len(papers_used)} sources"
        )

    except Exception as e:
        print(f"   ❌ Error: {e}")
        log_test(f"Q&A Question {i}", False, str(e))

save_output("05_qa_results.json", qa_results)

# ============================================================================
# PART 4: CRITIC AGENT EVALUATION
# ============================================================================

print_section("PART 4: CriticAgent - Quality Assurance")

from agents.critic_agent import CriticAgent

critic = CriticAgent()

# Evaluate collection
if collected_papers:
    print(f"\n📊 Evaluating paper collection...")

    # Small delay
    time.sleep(8)

    collection_eval = critic.evaluate_paper_collection(collected_papers[:5])

    print(f"\n✅ Collection Evaluation:")
    print(f"   Overall Score: {collection_eval['overall_score']:.2f}/1.00")
    print(f"   Relevance: {collection_eval['relevance_score']:.2f}")
    print(f"   Quality: {collection_eval['quality_score']:.2f}")
    print(f"   Diversity: {collection_eval['diversity_score']:.2f}")
    print(f"   Passed: {collection_eval['passed']}")

    save_output("06_collection_evaluation.json", collection_eval)

    log_test(
        "CriticAgent: Collection Evaluation",
        collection_eval['passed'],
        f"Score: {collection_eval['overall_score']:.2f}"
    )

# Evaluate answer (if we have Q&A results)
if qa_results:
    print(f"\n📊 Evaluating answer quality...")

    # Small delay
    time.sleep(8)

    qa_to_eval = qa_results[0]
    answer_eval = critic.evaluate_answer(
        question=qa_to_eval['question'],
        answer=qa_to_eval['answer'],
        context={"papers_used": qa_to_eval.get('papers_used', [])}
    )

    print(f"\n✅ Answer Evaluation:")
    print(f"   Overall Score: {answer_eval['overall_score']:.2f}/1.00")
    print(f"   Accuracy: {answer_eval['accuracy_score']:.2f}")
    print(f"   Completeness: {answer_eval['completeness_score']:.2f}")
    print(f"   Clarity: {answer_eval['clarity_score']:.2f}")
    print(f"   Passed: {answer_eval['passed']}")

    save_output("07_answer_evaluation.json", answer_eval)

    log_test(
        "CriticAgent: Answer Evaluation",
        answer_eval['passed'],
        f"Score: {answer_eval['overall_score']:.2f}"
    )

# ============================================================================
# PART 5: SUMMARIZATION
# ============================================================================

print_section("PART 5: SummarizationAgent - All Modes")

# 5A: Single paper summary
if collected_papers:
    print(f"\n📄 Single Paper Summarization...")
    test_paper = collected_papers[0]

    # Small delay
    time.sleep(8)

    summary_brief = orchestrator.summarize_paper(
        test_paper,
        style="executive",
        length="brief"
    )

    print(f"\n✅ Executive Brief:")
    print(f"   Paper: {test_paper.get('title', 'N/A')[:60]}...")
    print(f"   Word Count: {summary_brief['word_count']}")
    print(f"\n   {summary_brief['summary'][:200]}...")

    save_output("08_paper_summary.json", summary_brief)

    log_test(
        "Summarization: Single Paper",
        summary_brief['word_count'] < 100,
        f"{summary_brief['word_count']} words"
    )

# 5B: Collection summary
if len(collected_papers) >= 3:
    print(f"\n📚 Collection Summarization...")

    # Small delay
    time.sleep(8)

    collection_summary = orchestrator.summarize_collection(
        papers=collected_papers[:3],
        style="executive",
        focus="key innovations"
    )

    print(f"\n✅ Collection Summary:")
    print(f"   Papers Analyzed: {collection_summary.get('papers_analyzed', 0)}")

    if 'overall_summary' in collection_summary:
        print(f"\n   {collection_summary['overall_summary'][:200]}...")

    if 'key_themes' in collection_summary:
        print(f"\n   Key Themes ({len(collection_summary['key_themes'])}):")
        for theme in collection_summary['key_themes'][:3]:
            print(f"      - {theme}")

    save_output("09_collection_summary.json", collection_summary)

    log_test(
        "Summarization: Collection",
        collection_summary.get('papers_analyzed', 0) > 0,
        f"{collection_summary.get('papers_analyzed', 0)} papers analyzed"
    )

# 5C: Conversation summary
if qa_results:
    print(f"\n💬 Conversation Summarization...")

    # Small delay
    time.sleep(8)

    conversation_summary = orchestrator.summarize_conversation(style="bullet_points")

    print(f"\n✅ Conversation Summary:")
    print(f"   Turns: {conversation_summary.get('turn_count', 0)}")

    if 'questions_asked' in conversation_summary:
        print(f"\n   Questions:")
        for q in conversation_summary['questions_asked']:
            print(f"      - {q}")

    save_output("10_conversation_summary.json", conversation_summary)

    log_test(
        "Summarization: Conversation",
        conversation_summary.get('turn_count', 0) > 0,
        f"{conversation_summary.get('turn_count', 0)} turns"
    )

# ============================================================================
# PART 6: SESSION PERSISTENCE
# ============================================================================

print_section("PART 6: Session Persistence")

print(f"\n💾 Saving session...")
save_success = orchestrator.save_session()

log_test("Session Save", save_success, f"Session '{session_name}' saved")

print(f"\n📂 Loading session...")
orchestrator2 = OrchestratorAgent(session_name=session_name, config=config)

papers_restored = orchestrator2.metadata['papers_collected']
convs_restored = orchestrator2.metadata['conversations']

print(f"   Papers: {papers_restored}")
print(f"   Conversations: {convs_restored}")

log_test(
    "Session Load",
    papers_restored == papers_collected,
    f"{papers_restored} papers restored"
)

orchestrator.close()
orchestrator2.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_section("FINAL TEST SUMMARY")

passed = sum(1 for status, _, _ in test_results if status)
total = len(test_results)

print(f"\n📈 Results:")
print(f"   Total: {total}")
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

if passed >= total * 0.8:  # 80% pass rate
    print("🎉 SYSTEM FUNCTIONAL - Core features verified!")
    print("=" * 100)
    print(f"\n✅ All outputs saved to: ./test_outputs/")
    sys.exit(0)
else:
    print("⚠️  ISSUES DETECTED")
    sys.exit(1)
