#!/usr/bin/env python3
"""
WORKING DEMO - All Components Except SentenceTransformers
Shows: FAISS + NetworkX + Gemini + Self-Learning
Uses simple embeddings temporarily until sentence-transformers installs
"""

import json
import re
import os
import numpy as np
import faiss
import networkx as nx
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS
from pyvis.network import Network
import google.generativeai as genai
import hashlib

# Configuration
GOOGLE_API_KEY = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
GEMINI_MODEL = "gemini-2.5-flash"

# Global state
G = nx.DiGraph()
chunks = []
index = None
chunk_embeddings = []

print("🚀 Initializing All 4 Components...")
print("="*60)

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)
print("✅ 1/4: Gemini AI initialized")

# NetworkX ready
print("✅ 2/4: NetworkX Knowledge Graph ready")

# FAISS ready
print("✅ 3/4: FAISS Vector Database ready")

print("✅ 4/4: Self-Learning enabled")
print("="*60)
print()

# ===========================================================
# Simple Embedder (temporary until sentence-transformers installs)
# ===========================================================

def simple_embed(text, dim=128):
    """Simple embedding using hash + normalization"""
    # Use hash to create consistent vectors
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()

    # Convert to vector
    vec = np.frombuffer(hash_bytes[:dim*4], dtype=np.float32)[:dim].copy()

    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec

# ===========================================================
# Document Processing
# ===========================================================

def read_document(path):
    """Read document"""
    if path.endswith(".pdf"):
        text = ""
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif path.endswith(".txt"):
        with open(path, "r") as f:
            return f.read()
    raise ValueError("Use .pdf or .txt")

def chunk_text(text, size=400):
    """Split into chunks"""
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks_list = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < size:
            current += sent + " "
        else:
            if current.strip():
                chunks_list.append(current.strip())
            current = sent + " "

    if current.strip():
        chunks_list.append(current.strip())

    return chunks_list

# ===========================================================
# Component 1: FAISS Vector Database
# ===========================================================

def build_vector_db(chunks_list):
    """Build FAISS index"""
    global index, chunk_embeddings

    print(f"\n📦 Building Vector Database...")
    embeddings = np.array([simple_embed(chunk) for chunk in chunks_list]).astype("float32")
    chunk_embeddings = embeddings

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"✅ Indexed {index.ntotal} vectors")

def retrieve_similar(query, k=3):
    """Search vector DB"""
    if index is None:
        return []

    q_emb = simple_embed(query).astype("float32").reshape(1, -1)
    _, idxs = index.search(q_emb, min(k, len(chunks)))
    return [chunks[i] for i in idxs[0] if i < len(chunks)]

# ===========================================================
# Component 2: Knowledge Graph (NetworkX)
# ===========================================================

def extract_triples_gemini(text):
    """Extract triples with Gemini"""
    prompt = f"""Extract knowledge triples from this text.
Return ONLY a JSON array: [["subject", "relation", "object"], ...]

Text: {text[:800]}

JSON:"""

    try:
        response = gemini_model.generate_content(prompt)
        resp_text = response.text.strip()

        start = resp_text.find("[")
        end = resp_text.rfind("]") + 1

        if start != -1 and end > start:
            json_str = resp_text[start:end]
            triples = json.loads(json_str)

            valid = []
            for t in triples:
                if isinstance(t, list) and len(t) == 3:
                    s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                    if s and r and o:
                        valid.append((s, r, o))

            return valid
    except Exception as e:
        print(f"⚠️  Triple extraction error: {e}")

    return []

def add_triples_to_graph(triples):
    """Add to graph"""
    for s, r, o in triples:
        G.add_edge(s, o, label=r)

def query_knowledge_graph(entities):
    """Query graph"""
    facts = []
    for entity in entities:
        if entity in G.nodes():
            for neighbor in G.neighbors(entity):
                rel = G[entity][neighbor].get("label", "related_to")
                facts.append(f"{entity} [{rel}] {neighbor}")
    return list(set(facts))

# ===========================================================
# Component 3: Gemini Reasoning
# ===========================================================

def extract_entities_gemini(text):
    """Extract entities with Gemini"""
    prompt = f"""Extract named entities from: "{text}"
Return ONLY JSON array: ["Entity1", "Entity2"]

JSON:"""

    try:
        response = gemini_model.generate_content(prompt)
        resp_text = response.text.strip()

        start = resp_text.find("[")
        end = resp_text.rfind("]") + 1

        if start != -1 and end > start:
            entities = json.loads(resp_text[start:end])
            return [str(e).strip() for e in entities if e]
    except:
        pass

    # Fallback
    words = text.split()
    return list(set([w for w in words if w and w[0].isupper() and len(w) > 2]))

def answer_query_gemini(query):
    """Answer using all components"""
    # 1. Vector search
    vector_results = retrieve_similar(query)

    # 2. Graph query
    entities = extract_entities_gemini(query)
    graph_facts = query_knowledge_graph(entities)

    # 3. Build context
    context_parts = []

    if vector_results:
        context_parts.append("=== Vector Database Results ===")
        context_parts.extend(vector_results)

    if graph_facts:
        context_parts.append("\n=== Knowledge Graph Facts ===")
        context_parts.extend(graph_facts)

    context = "\n".join(context_parts)

    # 4. Gemini reasoning
    prompt = f"""You are a helpful AI assistant.
Answer based ONLY on the provided context.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error: {str(e)}"

    return answer

# ===========================================================
# Component 4: Self-Learning
# ===========================================================

def learn_from_text(new_text):
    """Add new knowledge"""
    global chunks, index

    if not new_text or len(new_text) < 10:
        return

    print(f"\n🧠 Learning: {new_text[:80]}...")

    # Add to chunks
    chunks.append(new_text)

    # Update vector DB
    new_emb = simple_embed(new_text).astype("float32").reshape(1, -1)
    index.add(new_emb)

    # Extract and add triples
    triples = extract_triples_gemini(new_text)
    if triples:
        add_triples_to_graph(triples)
        print(f"   Added {len(triples)} new triples to graph")

# ===========================================================
# Visualization
# ===========================================================

def visualize_graph(filename="knowledge_graph.html"):
    """Generate interactive graph"""
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.barnes_hut(gravity=-8000, central_gravity=0.3)

    for edge in G.edges(data=True):
        source, target, data = edge
        label = data.get("label", "")
        net.add_node(source, label=source, title=source, color="#4CAF50")
        net.add_node(target, label=target, title=target, color="#2196F3")
        net.add_edge(source, target, label=label, title=label, arrows="to")

    net.save_graph(filename)
    print(f"\n📊 Graph saved: {filename}")
    print(f"   Open in browser: file://{os.path.abspath(filename)}")

# ===========================================================
# Build System
# ===========================================================

def build_system(doc_path):
    """Build complete RAG + KG system"""
    global chunks

    print(f"\n📘 Reading: {doc_path}")
    text = read_document(doc_path)

    print("✂️  Chunking...")
    chunks = chunk_text(text)
    print(f"   Created {len(chunks)} chunks")

    # Build vector DB
    build_vector_db(chunks)

    # Build knowledge graph
    print(f"\n🕸️  Building Knowledge Graph with Gemini...")
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}...", end='\r')
        triples = extract_triples_gemini(chunk)
        add_triples_to_graph(triples)

    print(f"\n\n✅ System Ready!")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Graph Entities: {len(G.nodes())}")
    print(f"   - Graph Relationships: {len(G.edges())}")

    # Generate visualization
    visualize_graph()

# ===========================================================
# Interactive Loop
# ===========================================================

def run_interactive(doc_path):
    """Run interactive session"""
    print("\n" + "="*60)
    print("🤖 Self-Improving Agentic RAG (Working Demo)")
    print("="*60)
    print()

    build_system(doc_path)

    print("\n💡 Commands:")
    print("   - Ask any question")
    print("   - 'graph' - visualize")
    print("   - 'stats' - show statistics")
    print("   - 'exit' - quit")
    print()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "graph":
                visualize_graph()
                continue

            if user_input.lower() == "stats":
                print(f"\n📊 Statistics:")
                print(f"   - Chunks: {len(chunks)}")
                print(f"   - Entities: {len(G.nodes())}")
                print(f"   - Relationships: {len(G.edges())}")
                print(f"   - Vector DB size: {index.ntotal if index else 0}")
                continue

            # Answer query
            print(f"\n🔍 Processing...")
            answer = answer_query_gemini(user_input)
            print(f"\n🤖 Agent:\n{answer}")

            # Self-learn
            learn_from_text(f"Q: {user_input}\nA: {answer}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":
    import sys

    doc_path = sys.argv[1] if len(sys.argv) > 1 else "sample_knowledge.txt"
    run_interactive(doc_path)
