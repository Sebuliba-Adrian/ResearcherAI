                                                                                                                                                                                                                                    #!/usr/bin/env python3
"""
FIXED RAG System - Uses Gemini for Better Retrieval
Solves the "Jennifer" problem by using Gemini to find relevant chunks
"""

import json
import re
import os
import numpy as np
import faiss
import networkx as nx
from PyPDF2 import PdfReader
from pyvis.network import Network
import google.generativeai as genai

# Configuration
GOOGLE_API_KEY = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
GEMINI_MODEL = "gemini-2.5-flash"

# Global state
G = nx.DiGraph()
chunks = []

print("🚀 Initializing Enhanced RAG System...")
print("="*60)

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)
print("✅ Gemini AI initialized")
print("✅ NetworkX Knowledge Graph ready")
print("="*60)
print()

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
# FIXED: Gemini-Based Retrieval
# ===========================================================

def retrieve_with_gemini(query, chunks_list):
    """Use Gemini to intelligently find relevant chunks"""

    # Create numbered chunks
    chunks_text = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk}"
        for i, chunk in enumerate(chunks_list)
    ])

    prompt = f"""You are a document search system. Find which chunks contain information to answer this query.

CHUNKS:
{chunks_text}

QUERY: "{query}"

Task:
1. Identify ALL chunks that contain relevant information
2. Return ONLY a JSON array of chunk numbers, e.g., [1, 3, 5]
3. If no chunks are relevant, return []

JSON:"""

    try:
        response = gemini_model.generate_content(prompt)
        resp_text = response.text.strip()

        # Extract JSON
        start = resp_text.find("[")
        end = resp_text.rfind("]") + 1

        if start != -1 and end > start:
            json_str = resp_text[start:end]
            chunk_numbers = json.loads(json_str)

            # Get relevant chunks
            relevant = []
            for num in chunk_numbers:
                if isinstance(num, int) and 1 <= num <= len(chunks_list):
                    relevant.append(chunks_list[num - 1])

            return relevant

    except Exception as e:
        print(f"⚠️  Retrieval error: {e}")

    # Fallback: return all chunks
    return chunks_list

# ===========================================================
# Knowledge Graph
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
        pass

    return []

def add_triples_to_graph(triples):
    """Add to graph"""
    for s, r, o in triples:
        G.add_edge(s, o, label=r)

def query_knowledge_graph(entities):
    """Query graph"""
    facts = []
    for entity in entities:
        # Check exact match and partial match
        for node in G.nodes():
            if entity.lower() in node.lower() or node.lower() in entity.lower():
                # Outgoing edges
                for neighbor in G.neighbors(node):
                    rel = G[node][neighbor].get("label", "related_to")
                    facts.append(f"{node} [{rel}] {neighbor}")

                # Incoming edges
                for pred in G.predecessors(node):
                    rel = G[pred][node].get("label", "related_to")
                    facts.append(f"{pred} [{rel}] {node}")

    return list(set(facts))

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

# ===========================================================
# Answer Generation
# ===========================================================

def answer_query(query):
    """Answer using enhanced retrieval"""

    print(f"\n🔍 Searching for relevant information...")

    # 1. Use Gemini to find relevant chunks (FIXED!)
    relevant_chunks = retrieve_with_gemini(query, chunks)

    print(f"✅ Found {len(relevant_chunks)} relevant chunks")

    # 2. Query knowledge graph
    entities = extract_entities_gemini(query)
    graph_facts = query_knowledge_graph(entities)

    # 3. Build context
    context_parts = []

    if relevant_chunks:
        context_parts.append("=== Retrieved Information ===")
        for i, chunk in enumerate(relevant_chunks[:3], 1):
            context_parts.append(f"{i}. {chunk}")

    if graph_facts:
        context_parts.append("\n=== Knowledge Graph Facts ===")
        context_parts.extend(graph_facts[:5])

    context = "\n".join(context_parts)

    # Show context to user
    print(f"\n📄 Context being used:")
    print(context[:500] + "..." if len(context) > 500 else context)
    print()

    # 4. Generate answer with Gemini
    prompt = f"""You are a helpful AI assistant.
Answer based ONLY on the provided context.
Be specific and cite the information.

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
# Build System
# ===========================================================

def build_system(doc_path):
    """Build RAG + KG system"""
    global chunks

    print(f"\n📘 Reading: {doc_path}")
    text = read_document(doc_path)

    print("✂️  Chunking...")
    chunks = chunk_text(text)
    print(f"   Created {len(chunks)} chunks\n")

    # Show chunks for transparency
    print("📦 Chunks created:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   {i}. {chunk[:80]}...")

    print(f"\n🕸️  Building Knowledge Graph with Gemini...")
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}...", end='\r')
        triples = extract_triples_gemini(chunk)
        add_triples_to_graph(triples)

    print(f"\n\n✅ System Ready!")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Graph Entities: {len(G.nodes())}")
    print(f"   - Graph Relationships: {len(G.edges())}")

# ===========================================================
# Interactive Loop
# ===========================================================

def run_interactive(doc_path):
    """Run interactive session"""
    print("\n" + "="*60)
    print("🤖 Fixed RAG System (Gemini-Powered Retrieval)")
    print("="*60)
    print()

    build_system(doc_path)

    print("\n💡 Commands:")
    print("   - Ask any question")
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

            if user_input.lower() == "stats":
                print(f"\n📊 Statistics:")
                print(f"   - Chunks: {len(chunks)}")
                print(f"   - Entities: {len(G.nodes())}")
                print(f"   - Relationships: {len(G.edges())}")
                continue

            # Answer query
            answer = answer_query(user_input)
            print(f"\n🤖 Agent:\n{answer}")

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
