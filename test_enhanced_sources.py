#!/usr/bin/env python3
"""
Test Enhanced System: 5 Data Sources + ETL Pipeline
===================================================
Tests:
1. arXiv
2. Semantic Scholar
3. Zenodo
4. PubMed
5. Web Search
6. ETL Pipeline (Extract-Transform-Load-Validate)
"""

import sys
import time
from datetime import datetime

print("="*80)
print("🧪 TESTING ENHANCED SYSTEM: 5 DATA SOURCES + ETL PIPELINE")
print("="*80)

# Import enhanced system
from multi_agent_rag_enhanced import DataCollectorAgent, ETLPipeline

# Create agent
agent = DataCollectorAgent()

results = {
    "arxiv": {"tested": False, "working": False, "count": 0, "time": 0},
    "semantic_scholar": {"tested": False, "working": False, "count": 0, "time": 0},
    "zenodo": {"tested": False, "working": False, "count": 0, "time": 0},
    "pubmed": {"tested": False, "working": False, "count": 0, "time": 0},
    "websearch": {"tested": False, "working": False, "count": 0, "time": 0},
    "etl_pipeline": {"tested": False, "working": False, "stats": {}}
}

# ============================================================================
# TEST 1: arXiv
# ============================================================================
print("\n\n" + "="*80)
print("📋 TEST 1: arXiv")
print("="*80)

try:
    start = time.time()
    papers = agent.fetch_arxiv(category="cs.AI", days=7, max_results=5)
    elapsed = time.time() - start

    results["arxiv"]["tested"] = True
    results["arxiv"]["count"] = len(papers)
    results["arxiv"]["time"] = elapsed

    if papers:
        results["arxiv"]["working"] = True
        print(f"\n✅ arXiv WORKING")
        print(f"   Papers fetched: {len(papers)}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"\n   Sample Paper:")
        print(f"   - Title: {papers[0]['title'][:70]}...")
        print(f"   - Authors: {', '.join(papers[0]['authors'][:2])}")
        print(f"   - Source: {papers[0]['source']}")
    else:
        print(f"\n⚠️  arXiv returned 0 papers (may be rate limited)")

except Exception as e:
    print(f"\n❌ arXiv FAILED: {e}")
    results["arxiv"]["tested"] = True

time.sleep(2)  # Rate limiting

# ============================================================================
# TEST 2: Semantic Scholar
# ============================================================================
print("\n\n" + "="*80)
print("📋 TEST 2: Semantic Scholar")
print("="*80)

try:
    start = time.time()
    papers = agent.fetch_semantic_scholar(query="machine learning", max_results=5)
    elapsed = time.time() - start

    results["semantic_scholar"]["tested"] = True
    results["semantic_scholar"]["count"] = len(papers)
    results["semantic_scholar"]["time"] = elapsed

    if papers:
        results["semantic_scholar"]["working"] = True
        print(f"\n✅ Semantic Scholar WORKING")
        print(f"   Papers fetched: {len(papers)}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"\n   Sample Paper:")
        print(f"   - Title: {papers[0]['title'][:70]}...")
        print(f"   - Authors: {', '.join(papers[0]['authors'][:2])}")
        print(f"   - Citations: {papers[0].get('citation_count', 0)}")
        print(f"   - Source: {papers[0]['source']}")
    else:
        print(f"\n⚠️  Semantic Scholar returned 0 papers")

except Exception as e:
    print(f"\n❌ Semantic Scholar FAILED: {e}")
    results["semantic_scholar"]["tested"] = True

time.sleep(2)

# ============================================================================
# TEST 3: Zenodo
# ============================================================================
print("\n\n" + "="*80)
print("📋 TEST 3: Zenodo")
print("="*80)

try:
    start = time.time()
    papers = agent.fetch_zenodo(query="artificial intelligence", max_results=5)
    elapsed = time.time() - start

    results["zenodo"]["tested"] = True
    results["zenodo"]["count"] = len(papers)
    results["zenodo"]["time"] = elapsed

    if papers:
        results["zenodo"]["working"] = True
        print(f"\n✅ Zenodo WORKING")
        print(f"   Papers fetched: {len(papers)}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"\n   Sample Paper:")
        print(f"   - Title: {papers[0]['title'][:70]}...")
        print(f"   - Authors: {', '.join(papers[0]['authors'][:2])}")
        print(f"   - DOI: {papers[0].get('doi', 'N/A')}")
        print(f"   - Source: {papers[0]['source']}")
    else:
        print(f"\n⚠️  Zenodo returned 0 papers")

except Exception as e:
    print(f"\n❌ Zenodo FAILED: {e}")
    results["zenodo"]["tested"] = True

time.sleep(2)

# ============================================================================
# TEST 4: PubMed
# ============================================================================
print("\n\n" + "="*80)
print("📋 TEST 4: PubMed")
print("="*80)

try:
    start = time.time()
    papers = agent.fetch_pubmed(query="artificial intelligence", max_results=5)
    elapsed = time.time() - start

    results["pubmed"]["tested"] = True
    results["pubmed"]["count"] = len(papers)
    results["pubmed"]["time"] = elapsed

    if papers:
        results["pubmed"]["working"] = True
        print(f"\n✅ PubMed WORKING")
        print(f"   Papers fetched: {len(papers)}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"\n   Sample Paper:")
        print(f"   - Title: {papers[0]['title'][:70]}...")
        print(f"   - Authors: {', '.join(papers[0]['authors'][:2]) if papers[0]['authors'] else 'N/A'}")
        print(f"   - Source: {papers[0]['source']}")
    else:
        print(f"\n⚠️  PubMed returned 0 papers")

except Exception as e:
    print(f"\n❌ PubMed FAILED: {e}")
    results["pubmed"]["tested"] = True

time.sleep(2)

# ============================================================================
# TEST 5: Web Search
# ============================================================================
print("\n\n" + "="*80)
print("📋 TEST 5: Web Search (DuckDuckGo)")
print("="*80)

try:
    start = time.time()
    papers = agent.fetch_web(query="latest AI research 2025", max_results=3)
    elapsed = time.time() - start

    results["websearch"]["tested"] = True
    results["websearch"]["count"] = len(papers)
    results["websearch"]["time"] = elapsed

    if papers:
        results["websearch"]["working"] = True
        print(f"\n✅ Web Search WORKING")
        print(f"   Results fetched: {len(papers)}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"\n   Sample Result:")
        print(f"   - Title: {papers[0]['title'][:70]}...")
        print(f"   - URL: {papers[0]['url'][:60]}...")
        print(f"   - Source: {papers[0]['source']}")
    else:
        print(f"\n⚠️  Web Search returned 0 results")

except Exception as e:
    print(f"\n❌ Web Search FAILED: {e}")
    results["websearch"]["tested"] = True

# ============================================================================
# TEST 6: ETL Pipeline
# ============================================================================
print("\n\n" + "="*80)
print("📋 TEST 6: ETL Pipeline (Extract-Transform-Load-Validate)")
print("="*80)

try:
    etl = ETLPipeline()

    # Create sample raw data
    sample_data = [{
        "id": "test_1",
        "title": "Machine Learning for Beginners: A Comprehensive Guide",
        "abstract": "This paper provides a comprehensive introduction to machine learning concepts, algorithms, and applications. We cover supervised learning, unsupervised learning, and reinforcement learning with practical examples.",
        "authors": ["John Doe", "Jane Smith"],
        "topics": ["ML", "AI"],
        "source": "Test",
        "url": "http://test.com/paper1",
        "published": "2025-10-25"
    }]

    print("\n[1] EXTRACT Stage")
    extracted = etl.extract("test_source", lambda: sample_data)

    print("\n[2] TRANSFORM Stage")
    transformed = etl.transform(extracted, "test_source")

    print("\n[3] VALIDATE Stage")
    valid, invalid = etl.validate(transformed)

    print("\n[4] LOAD Stage")
    loaded = etl.load(valid, target="test_output")

    # Get stats
    stats = etl.get_stats()

    results["etl_pipeline"]["tested"] = True
    results["etl_pipeline"]["stats"] = stats

    if loaded and len(valid) > 0:
        results["etl_pipeline"]["working"] = True
        print(f"\n✅ ETL Pipeline WORKING")
        print(f"\n   ETL Statistics:")
        print(f"   - Extracted: {stats['extraction']['success']}")
        print(f"   - Valid: {stats['transformation']['valid']}")
        print(f"   - Invalid: {stats['transformation']['invalid']}")
        print(f"   - Success Rate: {stats['success_rate']:.1f}%")
    else:
        print(f"\n⚠️  ETL Pipeline issues detected")

except Exception as e:
    print(f"\n❌ ETL Pipeline FAILED: {e}")
    results["etl_pipeline"]["tested"] = True
    import traceback
    traceback.print_exc()

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n\n" + "="*80)
print("🏆 FINAL TEST REPORT")
print("="*80)

print("\n📊 DATA SOURCES:")
sources = ["arxiv", "semantic_scholar", "zenodo", "pubmed", "websearch"]
working_count = 0
total_papers = 0

for source in sources:
    r = results[source]
    status = "✅ WORKING" if r["working"] else "❌ NOT WORKING"
    tested = "✓" if r["tested"] else "✗"

    print(f"\n{source.upper().replace('_', ' ')}")
    print(f"   Tested: {tested}")
    print(f"   Status: {status}")
    print(f"   Papers: {r['count']}")
    print(f"   Time: {r['time']:.2f}s")

    if r["working"]:
        working_count += 1
        total_papers += r["count"]

print(f"\n📈 ETL PIPELINE:")
etl_result = results["etl_pipeline"]
etl_status = "✅ WORKING" if etl_result["working"] else "❌ NOT WORKING"
print(f"   Status: {etl_status}")
if etl_result.get("stats"):
    print(f"   Success Rate: {etl_result['stats']['success_rate']:.1f}%")

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"Data Sources Working: {working_count}/5")
print(f"Total Papers Collected: {total_papers}")
print(f"ETL Pipeline: {'✅ Working' if etl_result['working'] else '❌ Not Working'}")

if working_count >= 3 and etl_result["working"]:
    print("\n🎉 SYSTEM READY FOR PRODUCTION")
    print("   At least 3 data sources working + ETL pipeline functional")
elif working_count >= 1:
    print("\n✅ SYSTEM PARTIALLY FUNCTIONAL")
    print(f"   {working_count} data source(s) working")
else:
    print("\n⚠️  SYSTEM NEEDS ATTENTION")
    print("   No data sources working")

print("="*80)
print("✅ Test complete!")
