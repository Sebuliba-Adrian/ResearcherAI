#!/usr/bin/env python3
"""
Automated Test Script for Complete Multi-Agent System
Tests all features: conversation memory, sessions, persistence
"""

import os
import sys
import time
import pickle
from datetime import datetime

# Add test sample knowledge for quick testing
TEST_KNOWLEDGE = """
Machine Learning and AI Research
================================

GPT-3 was developed by OpenAI in 2020. It contains 175 billion parameters and uses transformer architecture. OpenAI is an AI research company founded by Sam Altman, Elon Musk, and others.

BERT (Bidirectional Encoder Representations from Transformers) was developed by Google in 2018. BERT revolutionized natural language processing by using bidirectional training. Jacob Devlin led the BERT project at Google.

Claude is an AI assistant created by Anthropic. Anthropic was founded by former OpenAI members including Dario Amodei and Daniela Amodei. Claude uses Constitutional AI for safety.

PyTorch is a machine learning framework developed by Meta. It is widely used for deep learning research. PyTorch supports dynamic computation graphs.

TensorFlow is a machine learning framework created by Google Brain team. TensorFlow 2.0 introduced eager execution. Jeff Dean oversees Google's AI efforts.
"""

def create_test_document():
    """Create test knowledge document"""
    test_file = "test_knowledge_ml.txt"
    with open(test_file, "w") as f:
        f.write(TEST_KNOWLEDGE)
    return test_file

def simulate_user_input(orchestrator, inputs):
    """Simulate user inputs programmatically"""
    results = []

    for user_input in inputs:
        print(f"\n👤 You: {user_input}")

        if user_input.lower() in ["exit", "quit", "q"]:
            orchestrator.save_session()
            print("\n👋 Goodbye!")
            break

        if user_input.lower() == "sessions":
            from multi_agent_rag_complete import list_sessions
            sessions = list_sessions()
            if sessions:
                print(f"\n📂 Available Sessions ({len(sessions)}):")
                for s in sessions:
                    marker = "→" if s["name"] == orchestrator.session_name else " "
                    print(f"\n{marker} {s['name']}")
                    print(f"   Papers: {s['papers_collected']}")
                    print(f"   Conversations: {s['conversations']}")
            results.append({"command": "sessions", "success": True, "count": len(sessions)})
            continue

        if user_input.lower().startswith("switch "):
            new_session = user_input[7:].strip()
            orchestrator.switch_session(new_session)
            results.append({"command": "switch", "session": new_session, "success": True})
            continue

        if user_input.lower() == "memory":
            history = orchestrator.reasoning_agent.conversation_history
            print(f"\n💾 Conversation History ({len(history)} turns):")
            for i, turn in enumerate(history, 1):
                print(f"\n{i}. Q: {turn['query']}")
                print(f"   A: {turn['answer'][:100]}...")
            results.append({"command": "memory", "turns": len(history)})
            continue

        if user_input.lower() == "stats":
            print(f"\n📊 Session Statistics:")
            print(f"   Session: {orchestrator.session_name}")
            print(f"   Graph nodes: {len(orchestrator.graph_agent.G.nodes())}")
            print(f"   Graph edges: {len(orchestrator.graph_agent.G.edges())}")
            print(f"   Text chunks: {len(orchestrator.vector_agent.chunks)}")
            print(f"   Conversations: {len(orchestrator.reasoning_agent.conversation_history)}")
            results.append({"command": "stats", "success": True})
            continue

        # Process as research question
        answer = orchestrator.reasoning_agent.synthesize_answer(user_input)
        print(f"\n🤖 Agent:\n{answer}\n")
        results.append({"query": user_input, "answer": answer, "success": True})
        time.sleep(1)  # Rate limiting

    return results

def run_comprehensive_test():
    """Run comprehensive feature test"""
    print("="*70)
    print("🧪 COMPREHENSIVE SYSTEM TEST")
    print("="*70)

    test_results = {
        "initialization": False,
        "data_loading": False,
        "conversation_memory": False,
        "session_creation": False,
        "session_switching": False,
        "persistence": False,
        "errors": []
    }

    try:
        # Test 1: Import and Initialize
        print("\n📋 Test 1: System Initialization")
        from multi_agent_rag_complete import OrchestratorAgent, list_sessions

        orchestrator = OrchestratorAgent("test_session_1")
        test_results["initialization"] = True
        print("   ✅ System initialized successfully")

        # Test 2: Load Sample Data
        print("\n📋 Test 2: Data Loading")
        test_file = create_test_document()

        # Manually add test papers
        test_paper = {
            "id": "test_paper_1",
            "title": "Machine Learning Fundamentals",
            "abstract": TEST_KNOWLEDGE[:500],
            "authors": ["Test Author"],
            "topics": ["Machine Learning", "AI"],
            "source": "Test",
            "url": "http://test.com",
            "published": datetime.now().isoformat(),
            "text": TEST_KNOWLEDGE
        }

        orchestrator.graph_agent.process_papers([test_paper])
        orchestrator.vector_agent.process_papers([test_paper])
        test_results["data_loading"] = True
        print("   ✅ Test data loaded successfully")

        # Test 3: Conversation Memory
        print("\n📋 Test 3: Conversation Memory")

        queries = [
            "Who created Claude?",
            "Tell me more about them",  # Should reference Anthropic
            "What else did they do?"    # Should maintain context
        ]

        results = simulate_user_input(orchestrator, queries)

        if len(orchestrator.reasoning_agent.conversation_history) == 3:
            test_results["conversation_memory"] = True
            print("   ✅ Conversation memory working (3 turns recorded)")
        else:
            test_results["errors"].append(f"Expected 3 turns, got {len(orchestrator.reasoning_agent.conversation_history)}")

        # Test 4: Session Creation
        print("\n📋 Test 4: Session Creation & Switching")

        # Save current session
        orchestrator.save_session()

        # Create new session
        orchestrator.switch_session("test_session_2")
        test_results["session_creation"] = True
        print("   ✅ New session created")

        # Verify new session is empty
        if len(orchestrator.reasoning_agent.conversation_history) == 0:
            print("   ✅ New session has empty history")
        else:
            test_results["errors"].append("New session should have empty history")

        # Test 5: Session Switching
        print("\n📋 Test 5: Session Switching Back")

        orchestrator.switch_session("test_session_1")

        # Verify old session restored
        if len(orchestrator.reasoning_agent.conversation_history) == 3:
            test_results["session_switching"] = True
            print("   ✅ Session switching preserves history")
        else:
            test_results["errors"].append(f"Expected 3 turns after switch, got {len(orchestrator.reasoning_agent.conversation_history)}")

        # Test 6: Persistence
        print("\n📋 Test 6: Persistence Across Restarts")

        # Simulate restart by creating new orchestrator
        del orchestrator
        orchestrator_new = OrchestratorAgent("test_session_1")

        if len(orchestrator_new.reasoning_agent.conversation_history) == 3:
            test_results["persistence"] = True
            print("   ✅ Session persisted across restart")
        else:
            test_results["errors"].append(f"Persistence failed: expected 3 turns, got {len(orchestrator_new.reasoning_agent.conversation_history)}")

        # Cleanup
        print("\n📋 Cleanup: Removing test sessions")
        import os
        for session in ["test_session_1", "test_session_2"]:
            path = f"research_sessions/{session}.pkl"
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(test_file):
            os.remove(test_file)

    except Exception as e:
        test_results["errors"].append(str(e))
        import traceback
        traceback.print_exc()

    # Print Results
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)

    tests = [
        ("System Initialization", test_results["initialization"]),
        ("Data Loading", test_results["data_loading"]),
        ("Conversation Memory", test_results["conversation_memory"]),
        ("Session Creation", test_results["session_creation"]),
        ("Session Switching", test_results["session_switching"]),
        ("Persistence", test_results["persistence"])
    ]

    passed = sum(1 for _, result in tests if result)
    total = len(tests)

    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    if test_results["errors"]:
        print("\n⚠️  Errors:")
        for error in test_results["errors"]:
            print(f"   - {error}")

    print(f"\n🏆 Score: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - System is fully functional!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - see errors above")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
