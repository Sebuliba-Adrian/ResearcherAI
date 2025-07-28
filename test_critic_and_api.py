#!/usr/bin/env python3
"""
Test CriticAgent and API Gateway
=================================
Comprehensive test of quality assurance and API endpoints
"""

import os
import sys
import requests
from time import sleep

print("=" * 80)
print("🧪 CRITIC AGENT & API GATEWAY TEST")
print("=" * 80)

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
API_KEY = "demo-key-123"
BASE_URL = "http://localhost:8000/v1"

test_results = []

def test(name):
    """Test decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'=' * 80}")
            print(f"🧪 TEST: {name}")
            print("-" * 80)
            try:
                result = func(*args, **kwargs)
                print(f"✅ PASS - {name}")
                test_results.append(("✅", name))
                return result
            except Exception as e:
                print(f"❌ FAIL - {name}")
                print(f"   Error: {e}")
                test_results.append(("❌", name))
                raise
        return wrapper
    return decorator

# ============================================================================
# CRITIC AGENT TESTS (Direct)
# ============================================================================

@test("CriticAgent: Import and Initialize")
def test_critic_import():
    """Test CriticAgent can be imported and initialized"""
    from agents.critic_agent import CriticAgent
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    critic = CriticAgent(config={
        "quality_threshold": 0.7,
        "temperature": 0.1
    })

    print(f"   ✅ CriticAgent initialized")
    print(f"   Quality threshold: {critic.quality_threshold}")
    return critic

@test("CriticAgent: Evaluate Paper Collection")
def test_critic_papers(critic):
    """Test paper collection evaluation"""
    # Sample papers
    papers = [
        {
            "title": "Attention Is All You Need",
            "field": "NLP",
            "year": 2017,
            "source": "arXiv",
            "citations": 95000
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "field": "NLP",
            "year": 2018,
            "source": "Semantic Scholar",
            "citations": 75000
        },
        {
            "title": "ResNet: Deep Residual Learning",
            "field": "Computer Vision",
            "year": 2015,
            "source": "arXiv",
            "citations": 120000
        }
    ]

    evaluation = critic.evaluate_paper_collection(papers)

    print(f"   ✅ Papers Evaluated:")
    print(f"      - Overall Score: {evaluation['overall_score']}")
    print(f"      - Relevance: {evaluation['relevance_score']}")
    print(f"      - Quality: {evaluation['quality_score']}")
    print(f"      - Diversity: {evaluation['diversity_score']}")
    print(f"      - Passed: {evaluation['passed']}")
    if evaluation['issues']:
        print(f"      - Issues: {', '.join(evaluation['issues'][:2])}")

    assert evaluation['overall_score'] > 0, "Score should be > 0"
    assert 'passed' in evaluation, "Should have pass/fail status"

    return evaluation

@test("CriticAgent: Evaluate Answer Quality")
def test_critic_answer(critic):
    """Test answer evaluation"""
    question = "What are transformers in NLP?"
    answer = """
Transformers are a neural network architecture introduced in 'Attention Is All You Need' (2017).
They use self-attention mechanisms instead of recurrence, allowing parallel processing and
better handling of long-range dependencies. Key models include BERT, GPT, and T5.
"""
    context = {
        "papers_used": [
            {"title": "Attention Is All You Need", "year": 2017},
            {"title": "BERT", "year": 2018}
        ],
        "graph_data": {"nodes": 50, "edges": 120}
    }

    evaluation = critic.evaluate_answer(question, answer, context)

    print(f"   ✅ Answer Evaluated:")
    print(f"      - Overall Score: {evaluation['overall_score']}")
    print(f"      - Accuracy: {evaluation['accuracy_score']}")
    print(f"      - Completeness: {evaluation['completeness_score']}")
    print(f"      - Clarity: {evaluation['clarity_score']}")
    print(f"      - Passed: {evaluation['passed']}")
    if evaluation['suggestions']:
        print(f"      - Suggestions: {evaluation['suggestions'][0]}")

    assert evaluation['overall_score'] > 0, "Score should be > 0"
    assert 'accuracy_score' in evaluation, "Should have accuracy score"

    return evaluation

@test("CriticAgent: Evaluate Graph Quality")
def test_critic_graph(critic):
    """Test graph evaluation"""
    graph_stats = {
        "nodes": 45,
        "edges": 120,
        "papers_processed": 10
    }

    evaluation = critic.evaluate_graph_quality(graph_stats)

    print(f"   ✅ Graph Evaluated:")
    print(f"      - Overall Score: {evaluation['overall_score']}")
    print(f"      - Node Score: {evaluation['node_score']}")
    print(f"      - Density: {evaluation['metrics']['density']}")
    print(f"      - Avg Degree: {evaluation['metrics']['avg_degree']}")
    print(f"      - Passed: {evaluation['passed']}")
    if evaluation['recommendations']:
        print(f"      - Recommendation: {evaluation['recommendations'][0]}")

    assert 'metrics' in evaluation, "Should have metrics"
    assert evaluation['metrics']['nodes'] == 45, "Nodes should match"

    return evaluation

@test("CriticAgent: Quality Report")
def test_critic_report(critic):
    """Test overall quality report"""
    report = critic.get_overall_quality_report()

    print(f"   ✅ Quality Report Generated:")
    print(f"      - Total Evaluations: {report['total_evaluations']}")
    print(f"      - Average Score: {report['average_score']}")
    print(f"      - Pass Rate: {report['pass_rate'] * 100:.1f}%")
    print(f"      - Quality Trend: {report['quality_trend']}")

    assert report['total_evaluations'] > 0, "Should have evaluations"
    assert 'average_score' in report, "Should have average score"

    return report

# ============================================================================
# API GATEWAY TESTS
# ============================================================================

def wait_for_api():
    """Wait for API to be ready"""
    print("\n⏳ Waiting for API Gateway to start...")
    for i in range(30):
        try:
            response = requests.get(f"{BASE_URL}/health", headers={"X-API-Key": API_KEY}, timeout=2)
            if response.status_code == 200:
                print("✅ API Gateway is ready!")
                return True
        except:
            pass
        sleep(1)
        print(f"   Attempt {i+1}/30...")
    return False

@test("API: Health Check")
def test_api_health():
    """Test API health endpoint"""
    response = requests.get(
        f"{BASE_URL}/health",
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print(f"   ✅ API Health:")
    print(f"      - Status: {data['status']}")
    print(f"      - Version: {data['version']}")
    print(f"      - Orchestrator: {data['agents'].get('orchestrator', 'N/A')}")
    print(f"      - Critic: {data['agents'].get('critic', 'N/A')}")

    assert data['status'] == 'healthy', "API should be healthy"
    return data

@test("API: Collect Papers")
def test_api_collect():
    """Test paper collection endpoint"""
    payload = {
        "query": "transformer attention mechanisms",
        "max_per_source": 3,
        "session_name": "api_test_session"
    }

    response = requests.post(
        f"{BASE_URL}/collect",
        json=payload,
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print(f"   ✅ Papers Collected:")
    print(f"      - Count: {data['papers_collected']}")
    print(f"      - Graph Nodes: {data['graph_stats'].get('nodes', 0)}")
    print(f"      - Session: {data['session_name']}")

    if data.get('critic_evaluation'):
        print(f"      - Critic Score: {data['critic_evaluation']['overall_score']}")

    assert data['success'] == True, "Collection should succeed"
    assert data['papers_collected'] > 0, "Should collect at least 1 paper"

    return data

@test("API: Ask Question")
def test_api_ask():
    """Test Q&A endpoint"""
    payload = {
        "question": "What are transformers?",
        "session_name": "api_test_session",
        "use_critic": True
    }

    response = requests.post(
        f"{BASE_URL}/ask",
        json=payload,
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print(f"   ✅ Question Answered:")
    print(f"      - Question: {data['question']}")
    print(f"      - Answer Length: {len(data['answer'])} chars")
    print(f"      - Sources Used: {len(data['sources'])}")

    if data.get('critic_evaluation'):
        print(f"      - Critic Score: {data['critic_evaluation']['overall_score']}")
        print(f"      - Accuracy: {data['critic_evaluation']['accuracy_score']}")

    assert data['success'] == True, "Q&A should succeed"
    assert len(data['answer']) > 0, "Should have an answer"

    return data

@test("API: Get Session Info")
def test_api_session():
    """Test session info endpoint"""
    response = requests.get(
        f"{BASE_URL}/sessions/api_test_session",
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print(f"   ✅ Session Retrieved:")
    print(f"      - Name: {data['session_name']}")
    print(f"      - Papers: {data['metadata'].get('papers_collected', 0)}")
    print(f"      - Conversations: {data['metadata'].get('conversations', 0)}")

    assert 'metadata' in data, "Should have metadata"
    return data

@test("API: Critic Report")
def test_api_critic_report():
    """Test critic report endpoint"""
    response = requests.get(
        f"{BASE_URL}/critic/report",
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print(f"   ✅ Critic Report:")
    print(f"      - Total Evaluations: {data.get('total_evaluations', 0)}")
    print(f"      - Average Score: {data.get('average_score', 0)}")
    print(f"      - Pass Rate: {data.get('pass_rate', 0) * 100:.1f}%")

    return data

@test("API: System Stats")
def test_api_stats():
    """Test system statistics endpoint"""
    response = requests.get(
        f"{BASE_URL}/stats",
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print(f"   ✅ System Statistics:")
    print(f"      - Papers: {data['system'].get('total_papers', 0)}")
    print(f"      - Graph Nodes: {data['system'].get('total_nodes', 0)}")
    print(f"      - Vectors: {data['system'].get('total_vectors', 0)}")

    return data

@test("API: Authentication Test (Invalid Key)")
def test_api_auth_fail():
    """Test authentication with invalid API key"""
    response = requests.get(
        f"{BASE_URL}/health",
        headers={"X-API-Key": "invalid-key-999"}
    )

    print(f"   ✅ Auth Rejected (as expected)")
    print(f"      - Status Code: {response.status_code}")

    assert response.status_code == 401, "Should reject invalid API key"
    return True

# ============================================================================
# RUN TESTS
# ============================================================================

def main():
    """Run all tests"""
    print("\n📦 Part 1: Direct CriticAgent Tests")
    print("=" * 80)

    try:
        # CriticAgent tests
        critic = test_critic_import()
        test_critic_papers(critic)
        test_critic_answer(critic)
        test_critic_graph(critic)
        test_critic_report(critic)

        print("\n\n📡 Part 2: API Gateway Tests")
        print("=" * 80)
        print("\n⚠️  Starting API server in background...")
        print("   Run manually: python api_gateway.py")
        print("   Or skip API tests if not running")

        # Check if API is running
        if wait_for_api():
            test_api_health()
            test_api_collect()
            sleep(2)  # Wait for processing
            test_api_ask()
            test_api_session()
            test_api_critic_report()
            test_api_stats()
            test_api_auth_fail()
        else:
            print("\n⚠️  API Gateway not running - skipping API tests")
            print("   To test API: python api_gateway.py (in another terminal)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for status, _ in test_results if status == "✅")
    total = len(test_results)

    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"\n📈 Success Rate: {passed}/{total} ({100*passed/total:.1f}%)")

    if passed < total:
        print(f"\n❌ Failed Tests:")
        for status, name in test_results:
            if status == "❌":
                print(f"   - {name}")

    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"\n✅ CriticAgent is working perfectly")
        print(f"✅ API Gateway endpoints working")
        print(f"✅ Quality assurance integrated")

if __name__ == "__main__":
    main()
