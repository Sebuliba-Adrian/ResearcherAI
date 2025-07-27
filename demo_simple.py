#!/usr/bin/env python3
"""
Simplified Demo of RAG + Knowledge Graph
Works with minimal dependencies - demonstrates core concepts
"""

import re
import json

# Simple in-memory knowledge structures
chunks = []
knowledge_graph = {}  # {entity: [(relation, target), ...]}

# ===========================================================
# 1️⃣ Document Processing
# ===========================================================

def read_document(path: str) -> str:
    """Read text file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text: str, size: int = 300) -> list:
    """Split text into chunks"""
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
# 2️⃣ Simple Triple Extraction (Rule-based)
# ===========================================================

def extract_triples_simple(text: str) -> list:
    """
    Simple pattern-based triple extraction
    Patterns:
    - "X is Y" -> (X, is, Y)
    - "X was Yed by Z" -> (X, Yed_by, Z)
    - "X is located in Y" -> (X, located_in, Y)
    """
    triples = []

    # Pattern 1: X is Y / X is the Y of Z
    pattern1 = r'([A-Z][a-zA-Z\s]+?)\s+is\s+(?:the\s+)?([a-z]+)\s+(?:of\s+)?([A-Z][a-zA-Z\s]+)'
    for match in re.finditer(pattern1, text):
        subj = match.group(1).strip()
        rel = match.group(2).strip()
        obj = match.group(3).strip() if match.group(3) else match.group(2).strip()
        triples.append((subj, rel, obj))

    # Pattern 2: X was Yed by Z
    pattern2 = r'([A-Z][a-zA-Z\s]+?)\s+was\s+([a-z]+)\s+by\s+([A-Z][a-zA-Z\s]+)'
    for match in re.finditer(pattern2, text):
        subj = match.group(1).strip()
        rel = match.group(2).strip() + "_by"
        obj = match.group(3).strip()
        triples.append((subj, rel, obj))

    # Pattern 3: X located in Y
    pattern3 = r'([A-Z][a-zA-Z\s]+?)\s+(?:is\s+)?located\s+in\s+([A-Z][a-zA-Z\s]+)'
    for match in re.finditer(pattern3, text):
        subj = match.group(1).strip()
        obj = match.group(2).strip()
        triples.append((subj, "located_in", obj))

    # Pattern 4: X, created/developed by Y
    pattern4 = r'([A-Z][a-zA-Z\s-]+),?\s+(?:created|developed)\s+by\s+([A-Z][a-zA-Z\s]+)'
    for match in re.finditer(pattern4, text):
        subj = match.group(1).strip()
        obj = match.group(2).strip()
        triples.append((subj, "created_by", obj))

    return triples

def add_to_graph(triples: list):
    """Add triples to knowledge graph"""
    for subj, rel, obj in triples:
        if subj not in knowledge_graph:
            knowledge_graph[subj] = []
        knowledge_graph[subj].append((rel, obj))

# ===========================================================
# 3️⃣ Simple Vector Search (TF-IDF style)
# ===========================================================

def simple_similarity(query: str, chunk: str) -> float:
    """Calculate simple word overlap similarity"""
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())

    if not query_words or not chunk_words:
        return 0.0

    overlap = query_words.intersection(chunk_words)
    return len(overlap) / len(query_words.union(chunk_words))

def retrieve_similar_chunks(query: str, top_k: int = 3) -> list:
    """Retrieve most similar chunks"""
    scored = [(chunk, simple_similarity(query, chunk)) for chunk in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored[:top_k] if score > 0]

# ===========================================================
# 4️⃣ Graph Query
# ===========================================================

def find_entities_in_text(text: str) -> list:
    """Find capitalized words (simple entity detection)"""
    words = text.split()
    entities = []
    for word in words:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', word)
        if cleaned and cleaned[0].isupper() and len(cleaned) > 2:
            entities.append(cleaned)
    return list(set(entities))

def query_graph(entities: list) -> list:
    """Find facts about entities in graph"""
    facts = []
    for entity in entities:
        if entity in knowledge_graph:
            for rel, target in knowledge_graph[entity]:
                facts.append(f"{entity} [{rel}] {target}")

        # Also check if entity appears as object
        for subj, relations in knowledge_graph.items():
            for rel, obj in relations:
                if entity.lower() in obj.lower():
                    facts.append(f"{subj} [{rel}] {obj}")

    return list(set(facts))

# ===========================================================
# 5️⃣ Build System from Document
# ===========================================================

def build_system(doc_path: str):
    """Build RAG + KG from document"""
    global chunks

    print(f"\n📘 Reading: {doc_path}")
    text = read_document(doc_path)

    print("✂️  Chunking...")
    chunks = chunk_text(text)
    print(f"   Created {len(chunks)} chunks")

    print("🕸️  Building knowledge graph...")
    for chunk in chunks:
        triples = extract_triples_simple(chunk)
        add_to_graph(triples)

    print(f"✅ System ready!")
    print(f"   - {len(chunks)} chunks")
    print(f"   - {len(knowledge_graph)} entities")
    print(f"   - {sum(len(v) for v in knowledge_graph.values())} relationships")

# ===========================================================
# 6️⃣ Answer Query
# ===========================================================

def answer_query(query: str) -> str:
    """Answer using RAG + KG"""
    print(f"\n🔍 Processing query: {query}")

    # Retrieve similar chunks
    similar = retrieve_similar_chunks(query)

    # Find entities and query graph
    entities = find_entities_in_text(query)
    graph_facts = query_graph(entities)

    # Build response
    print("\n" + "="*60)

    if similar:
        print("📄 Retrieved Information:")
        for i, chunk in enumerate(similar, 1):
            print(f"{i}. {chunk[:200]}...")

    if graph_facts:
        print("\n🕸️  Knowledge Graph Facts:")
        for fact in graph_facts[:5]:  # Show top 5
            print(f"  • {fact}")

    print("="*60)

    # Simple answer generation
    if similar:
        return f"\nBased on the knowledge base:\n{similar[0]}"
    else:
        return "\nNo relevant information found."

# ===========================================================
# 7️⃣ Visualize Graph (Text-based)
# ===========================================================

def show_graph_stats():
    """Show graph statistics"""
    print("\n📊 Knowledge Graph Statistics:")
    print(f"   Total entities: {len(knowledge_graph)}")
    print(f"   Total relationships: {sum(len(v) for v in knowledge_graph.values())}")

    print("\n🔝 Top Entities (by connections):")
    sorted_entities = sorted(
        knowledge_graph.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    for entity, relations in sorted_entities[:10]:
        print(f"   • {entity}: {len(relations)} connections")

    print("\n🔗 Sample Relationships:")
    count = 0
    for entity, relations in knowledge_graph.items():
        for rel, obj in relations[:2]:
            print(f"   {entity} --[{rel}]--> {obj}")
            count += 1
            if count >= 15:
                break
        if count >= 15:
            break

# ===========================================================
# 8️⃣ Interactive Loop
# ===========================================================

def run_interactive():
    """Run interactive session"""
    print("\n" + "="*60)
    print("🤖 Simple RAG + Knowledge Graph Demo")
    print("="*60)

    doc_path = "sample_knowledge.txt"
    build_system(doc_path)

    print("\n💡 Commands:")
    print("   - Ask any question")
    print("   - 'graph' - show graph statistics")
    print("   - 'exit' - quit")

    while True:
        try:
            query = input("\n👤 You: ").strip()

            if not query:
                continue

            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break

            if query.lower() == 'graph':
                show_graph_stats()
                continue

            answer = answer_query(query)
            print(f"\n🤖 Answer: {answer}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":
    run_interactive()
