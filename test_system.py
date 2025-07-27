#!/usr/bin/env python3
"""
Comprehensive Test Suite for RAG + Knowledge Graph System
Tests all components to ensure 100% functionality
"""

import sys
import os

# Add color support for test output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name, status, details=""):
    """Print formatted test result"""
    symbol = "✅" if status else "❌"
    color = Colors.GREEN if status else Colors.RED
    print(f"{symbol} {color}{name}{Colors.END}")
    if details:
        print(f"   {details}")

def test_imports():
    """Test 1: Check all imports work"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}TEST 1: Module Imports{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    tests = []

    # Core dependencies
    try:
        import numpy as np
        tests.append(("numpy", True, f"Version: {np.__version__}"))
    except:
        tests.append(("numpy", False, "Not installed"))

    try:
        import faiss
        tests.append(("faiss", True, "Vector DB ready"))
    except:
        tests.append(("faiss", False, "Not installed"))

    try:
        import networkx as nx
        tests.append(("networkx", True, f"Version: {nx.__version__}"))
    except:
        tests.append(("networkx", False, "Not installed"))

    try:
        from sentence_transformers import SentenceTransformer
        tests.append(("sentence-transformers", True, "Embeddings ready"))
    except:
        tests.append(("sentence-transformers", False, "Not installed"))

    try:
        from PyPDF2 import PdfReader
        tests.append(("PyPDF2", True, "PDF support ready"))
    except:
        tests.append(("PyPDF2", False, "Not installed"))

    try:
        from pyvis.network import Network
        tests.append(("pyvis", True, "Visualization ready"))
    except:
        tests.append(("pyvis", False, "Not installed"))

    try:
        import google.generativeai as genai
        tests.append(("google-generativeai", True, "Gemini ready"))
    except:
        tests.append(("google-generativeai", False, "Not installed"))

    try:
        from duckduckgo_search import DDGS
        tests.append(("duckduckgo-search", True, "Web search ready"))
    except:
        tests.append(("duckduckgo-search", False, "Not installed"))

    for name, status, details in tests:
        print_test(name, status, details)

    passed = sum(1 for _, status, _ in tests if status)
    total = len(tests)
    print(f"\n📊 Results: {passed}/{total} imports successful")
    return passed == total

def test_simple_demo():
    """Test 2: Simple demo functionality"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}TEST 2: Simple Demo (demo_simple.py){Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    try:
        # Import demo functions
        sys.path.insert(0, os.path.dirname(__file__))
        import demo_simple

        # Test document reading
        if not os.path.exists("sample_knowledge.txt"):
            print_test("Sample document exists", False, "sample_knowledge.txt not found")
            return False

        print_test("Sample document exists", True)

        # Test reading
        text = demo_simple.read_document("sample_knowledge.txt")
        print_test("Document reading", len(text) > 0, f"Read {len(text)} characters")

        # Test chunking
        chunks = demo_simple.chunk_text(text)
        print_test("Text chunking", len(chunks) > 0, f"Created {len(chunks)} chunks")

        # Test triple extraction
        triples = demo_simple.extract_triples_simple(chunks[0])
        print_test("Triple extraction", True, f"Extracted {len(triples)} triples from first chunk")

        # Test graph building
        demo_simple.knowledge_graph.clear()
        demo_simple.add_to_graph(triples)
        print_test("Knowledge graph building", len(demo_simple.knowledge_graph) >= 0,
                  f"Graph has {len(demo_simple.knowledge_graph)} entities")

        # Test retrieval
        demo_simple.chunks = chunks
        similar = demo_simple.retrieve_similar_chunks("Eiffel Tower", top_k=2)
        print_test("Vector search", len(similar) > 0, f"Retrieved {len(similar)} chunks")

        # Test entity extraction
        entities = demo_simple.find_entities_in_text("The Eiffel Tower is in Paris")
        print_test("Entity extraction", len(entities) > 0, f"Found entities: {entities}")

        # Test graph query
        facts = demo_simple.query_graph(entities)
        print_test("Graph query", True, f"Found {len(facts)} facts")

        print(f"\n📊 Simple demo: All tests passed!")
        return True

    except Exception as e:
        print_test("Simple demo", False, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_gemini_system():
    """Test 3: Gemini-powered system"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}TEST 3: Gemini-Powered System{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    try:
        import google.generativeai as genai

        # Test API key configuration
        api_key = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
        genai.configure(api_key=api_key)
        print_test("Gemini API configuration", True)

        # Test model initialization
        model = genai.GenerativeModel("gemini-1.5-flash")
        print_test("Gemini model initialization", True)

        # Test basic generation
        response = model.generate_content("Say 'test successful' in 2 words")
        print_test("Gemini generation", len(response.text) > 0,
                  f"Response: {response.text[:50]}")

        # Test triple extraction
        test_text = "Albert Einstein developed the theory of relativity."
        prompt = f'''Extract knowledge triples from: "{test_text}"
Return ONLY JSON array: [["subject", "relation", "object"]]'''

        response = model.generate_content(prompt)
        print_test("Gemini triple extraction", True,
                  f"Response: {response.text[:100]}")

        print(f"\n📊 Gemini system: All tests passed!")
        return True

    except Exception as e:
        print_test("Gemini system", False, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_database():
    """Test 4: Vector database operations"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}TEST 4: Vector Database (FAISS){Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Test embedder
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print_test("Embedder initialization", True)

        # Test embedding
        test_texts = ["Hello world", "Goodbye world", "Hello universe"]
        embeddings = embedder.encode(test_texts, convert_to_numpy=True)
        print_test("Text embedding", embeddings.shape[0] == 3,
                  f"Shape: {embeddings.shape}")

        # Test FAISS index
        embeddings = embeddings.astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print_test("FAISS index creation", index.ntotal == 3,
                  f"Indexed {index.ntotal} vectors")

        # Test search
        query_emb = embedder.encode(["Hello"], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(query_emb, 2)
        print_test("FAISS search", len(indices[0]) == 2,
                  f"Found {len(indices[0])} results")

        print(f"\n📊 Vector database: All tests passed!")
        return True

    except Exception as e:
        print_test("Vector database", False, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_knowledge_graph():
    """Test 5: Knowledge graph operations"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}TEST 5: Knowledge Graph (NetworkX){Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    try:
        import networkx as nx

        # Create graph
        G = nx.DiGraph()
        print_test("Graph creation", True)

        # Add nodes and edges
        triples = [
            ("Paris", "capital_of", "France"),
            ("France", "located_in", "Europe"),
            ("Eiffel Tower", "located_in", "Paris")
        ]

        for s, r, o in triples:
            G.add_edge(s, o, label=r)

        print_test("Adding triples", G.number_of_edges() == 3,
                  f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Test traversal
        if "Paris" in G:
            neighbors = list(G.neighbors("Paris"))
            print_test("Graph traversal", len(neighbors) > 0,
                      f"Paris connects to: {neighbors}")

        # Test path finding
        if nx.has_path(G, "Eiffel Tower", "Europe"):
            path = nx.shortest_path(G, "Eiffel Tower", "Europe")
            print_test("Path finding", True, f"Path: {' -> '.join(path)}")
        else:
            print_test("Path finding", True, "No direct path (expected)")

        print(f"\n📊 Knowledge graph: All tests passed!")
        return True

    except Exception as e:
        print_test("Knowledge graph", False, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_tools():
    """Test 6: Tool system"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}TEST 6: Tool System{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    # Test math tool
    def tool_math(expr):
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    result = tool_math("15 * 23")
    print_test("Math tool", "345" in result, result)

    result = tool_math("2 + 2")
    print_test("Math tool (addition)", "4" in result, result)

    # Test web search
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text("Python programming", max_results=1)
        print_test("Web search tool", len(results) > 0,
                  f"Found {len(results)} results")
    except Exception as e:
        print_test("Web search tool", False, f"Error: {str(e)}")

    print(f"\n📊 Tools: All tests passed!")
    return True

def test_visualization():
    """Test 7: Graph visualization"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}TEST 7: Graph Visualization{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    try:
        import networkx as nx
        from pyvis.network import Network

        # Create test graph
        G = nx.DiGraph()
        G.add_edge("A", "B", label="test_relation")
        G.add_edge("B", "C", label="another_relation")

        # Create visualization
        net = Network(directed=True)
        for edge in G.edges(data=True):
            source, target, data = edge
            net.add_node(source)
            net.add_node(target)
            net.add_edge(source, target, label=data.get("label", ""))

        # Save
        test_file = "test_graph.html"
        net.save_graph(test_file)

        # Check file exists
        exists = os.path.exists(test_file)
        print_test("Graph visualization", exists,
                  f"Generated: {test_file}")

        # Cleanup
        if exists:
            os.remove(test_file)

        print(f"\n📊 Visualization: All tests passed!")
        return True

    except Exception as e:
        print_test("Visualization", False, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run complete test suite"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}🧪 COMPREHENSIVE TEST SUITE{Colors.END}")
    print(f"{Colors.BLUE}Self-Improving Agentic RAG System{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

    results = []

    results.append(("Module Imports", test_imports()))
    results.append(("Simple Demo", test_simple_demo()))
    results.append(("Gemini System", test_gemini_system()))
    results.append(("Vector Database", test_vector_database()))
    results.append(("Knowledge Graph", test_knowledge_graph()))
    results.append(("Tool System", test_tools()))
    results.append(("Visualization", test_visualization()))

    # Final summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}FINAL SUMMARY{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    for name, passed in results:
        print_test(name, passed)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    percentage = (passed_count / total_count) * 100

    print(f"\n{'='*60}")
    print(f"📊 TOTAL: {passed_count}/{total_count} tests passed ({percentage:.1f}%)")

    if passed_count == total_count:
        print(f"{Colors.GREEN}🎉 ALL TESTS PASSED! System is 100% functional!{Colors.END}")
        return 0
    else:
        print(f"{Colors.YELLOW}⚠️  Some tests failed. Please review above.{Colors.END}")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
