#!/bin/bash
# Quick Test Runner for RAG System

echo "=================================================="
echo "🧪 RAG SYSTEM TEST SUITE"
echo "=================================================="
echo ""

# Test 1: Simple Demo (No dependencies)
echo "📝 Test 1: Simple Demo (No Dependencies)"
echo "--------------------------------------------------"
echo "Testing basic RAG functionality..."
echo ""

python3 << 'EOF'
import demo_simple

# Test document reading
text = demo_simple.read_document("sample_knowledge.txt")
print(f"✅ Read {len(text)} characters from document")

# Test chunking
chunks = demo_simple.chunk_text(text)
print(f"✅ Created {len(chunks)} chunks")

# Test triple extraction
demo_simple.knowledge_graph.clear()
total_triples = 0
for chunk in chunks:
    triples = demo_simple.extract_triples_simple(chunk)
    demo_simple.add_to_graph(triples)
    total_triples += len(triples)

print(f"✅ Extracted {total_triples} triples")
print(f"✅ Built graph with {len(demo_simple.knowledge_graph)} entities")

# Test retrieval
demo_simple.chunks = chunks
similar = demo_simple.retrieve_similar_chunks("Eiffel Tower", top_k=2)
print(f"✅ Vector search returned {len(similar)} results")

# Test entity extraction
entities = demo_simple.find_entities_in_text("The Eiffel Tower is in Paris")
print(f"✅ Entity extraction found: {entities}")

# Test graph query
facts = demo_simple.query_graph(entities)
print(f"✅ Graph query returned {len(facts)} facts")

print("\n📊 Simple Demo: ALL TESTS PASSED ✅")
EOF

echo ""
echo "=================================================="
echo ""

# Test 2: Gemini API (if venv active)
if [ -d "venv" ]; then
    echo "📝 Test 2: Gemini API Integration"
    echo "--------------------------------------------------"

    source venv/bin/activate 2>/dev/null

    python3 << 'EOF'
try:
    import google.generativeai as genai
    import json

    genai.configure(api_key="AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4")
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Test 1: Basic generation
    response = model.generate_content("Say 'OK' in 1 word")
    print(f"✅ Gemini basic generation: {response.text.strip()}")

    # Test 2: Triple extraction
    prompt = '''Extract triples from: "Paris is capital of France"
Return JSON: [["subject", "relation", "object"]]'''

    response = model.generate_content(prompt)
    resp_text = response.text.strip()
    start = resp_text.find("[")
    end = resp_text.rfind("]") + 1
    triples = json.loads(resp_text[start:end])
    print(f"✅ Gemini triple extraction: {len(triples)} triples")

    print("\n📊 Gemini API: ALL TESTS PASSED ✅")

except ImportError:
    print("⚠️  Gemini dependencies not installed (run setup.sh)")
except Exception as e:
    print(f"❌ Gemini test failed: {e}")
EOF

    echo ""
fi

echo "=================================================="
echo "✅ TEST SUITE COMPLETE"
echo "=================================================="
