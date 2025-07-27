#!/usr/bin/env python3
"""
Self-Improving Agentic RAG System with Knowledge Graph Visualization
Combines vector search (FAISS) + knowledge graph (NetworkX) + LLM reasoning
Features:
- Automatic triple extraction from documents
- Self-learning from conversations and tool use
- Live knowledge graph visualization
- Tool integration (math, web search)
"""

import json
import re
import os
import numpy as np
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS
from pyvis.network import Network
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# 1️⃣ Configuration & Setup
# ===========================================================

class Config:
    """Central configuration"""
    CHUNK_SIZE = 400
    VECTOR_TOP_K = 3
    MAX_NEW_TOKENS = 200
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gpt2"  # Lightweight for testing; replace with better model
    GRAPH_HTML = "knowledge_graph.html"
    SPACY_MODEL = "en_core_web_sm"

# Global state
embedder = None
llm = None
nlp = None
G = nx.DiGraph()
chunks = []
index = None

def initialize_models():
    """Initialize all ML models"""
    global embedder, llm, nlp

    print("🔄 Loading models...")
    embedder = SentenceTransformer(Config.EMBEDDING_MODEL)

    # Using GPT-2 for demo; replace with Mistral/Llama for production
    llm = pipeline(
        "text-generation",
        model=Config.LLM_MODEL,
        max_length=512,
        truncation=True
    )

    # Try to load spacy, provide fallback
    try:
        import spacy
        nlp = spacy.load(Config.SPACY_MODEL)
    except:
        print("⚠️  Spacy model not found. Run: python -m spacy download en_core_web_sm")
        nlp = None

    print("✅ Models loaded successfully!")

# ===========================================================
# 2️⃣ Tool System
# ===========================================================

def tool_math(expr: str) -> str:
    """Safe math evaluation tool"""
    try:
        # Only allow basic math operations
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return f"✅ Math result: {result}"
    except Exception as e:
        return f"❌ Math error: {str(e)}"

def tool_search_web(query: str) -> str:
    """Web search tool using DuckDuckGo"""
    try:
        results = DDGS().text(query, max_results=3)
        formatted = "\n".join([
            f"{i+1}. {r['title']}\n   {r['href']}\n   {r['body'][:100]}..."
            for i, r in enumerate(results)
        ])
        return f"🌐 Web search results:\n{formatted}"
    except Exception as e:
        return f"❌ Web search error: {str(e)}"

TOOLS = {
    "math": tool_math,
    "search": tool_search_web
}

# ===========================================================
# 3️⃣ Document Processing
# ===========================================================

def read_document(path: str) -> str:
    """Read PDF or TXT file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".pdf"):
        text = ""
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Use .pdf or .txt")

def chunk_text(text: str, size: int = None) -> list:
    """Split text into semantic chunks"""
    if size is None:
        size = Config.CHUNK_SIZE

    # Clean and split into sentences
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
# 4️⃣ Triple Extraction (LLM-based)
# ===========================================================

def extract_triples_llm(text: str) -> list:
    """Extract subject-relation-object triples using LLM"""
    prompt = f"""Extract knowledge triples from this text.
Return ONLY a JSON list of [subject, relation, object] triples.

Text: {text[:500]}

JSON:"""

    try:
        response = llm(prompt, max_new_tokens=Config.MAX_NEW_TOKENS)[0]["generated_text"]

        # Try to extract JSON from response
        start = response.find("[")
        end = response.rfind("]") + 1

        if start != -1 and end > start:
            triples = json.loads(response[start:end])
            # Validate format
            valid_triples = []
            for t in triples:
                if isinstance(t, list) and len(t) == 3:
                    valid_triples.append(tuple(t))
            return valid_triples
    except Exception as e:
        print(f"⚠️  Triple extraction failed: {e}")

    return []

def extract_triples_spacy(text: str) -> list:
    """Fallback: rule-based triple extraction with spaCy"""
    if nlp is None:
        return []

    doc = nlp(text)
    triples = []

    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT":
                # Find subject
                subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                # Find object
                obj = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr")]

                if subj and obj:
                    triples.append((
                        subj[0].text,
                        token.lemma_,
                        obj[0].text
                    ))

    return triples

def add_triples_to_graph(triples: list):
    """Add triples to knowledge graph"""
    for s, r, o in triples:
        # Clean entity names
        s = str(s).strip()
        o = str(o).strip()
        r = str(r).strip()

        if s and o and r:
            G.add_edge(s, o, label=r)

# ===========================================================
# 5️⃣ Vector Database (FAISS)
# ===========================================================

def build_vector_db(chunks_list: list):
    """Build FAISS index from text chunks"""
    global index

    if not chunks_list:
        raise ValueError("No chunks provided")

    embeddings = np.array(
        embedder.encode(chunks_list, convert_to_numpy=True)
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index

def retrieve_similar(query: str, k: int = None) -> list:
    """Retrieve similar chunks from vector DB"""
    if k is None:
        k = Config.VECTOR_TOP_K

    if index is None or not chunks:
        return []

    q_emb = np.array(
        embedder.encode([query], convert_to_numpy=True)
    ).astype("float32")

    _, idxs = index.search(q_emb, min(k, len(chunks)))
    return [chunks[i] for i in idxs[0] if i < len(chunks)]

# ===========================================================
# 6️⃣ Knowledge Graph Queries
# ===========================================================

def query_knowledge_graph(entities: list) -> list:
    """Query graph for relationships involving entities"""
    facts = []

    for entity in entities:
        # Direct neighbors
        if entity in G.nodes():
            for neighbor in G.neighbors(entity):
                rel = G[entity][neighbor].get("label", "related_to")
                facts.append(f"{entity} [{rel}] {neighbor}")

        # Also check incoming edges
        for node in G.nodes():
            if G.has_edge(node, entity):
                rel = G[node][entity].get("label", "related_to")
                facts.append(f"{node} [{rel}] {entity}")

    return facts

def extract_entities_simple(text: str) -> list:
    """Simple entity extraction from text"""
    if nlp:
        doc = nlp(text)
        return [ent.text for ent in doc.ents]
    else:
        # Fallback: capitalize words as potential entities
        words = text.split()
        return [w for w in words if w[0].isupper() and len(w) > 2]

# ===========================================================
# 7️⃣ Graph Visualization
# ===========================================================

def visualize_graph(filename: str = None):
    """Generate interactive HTML visualization of knowledge graph"""
    if filename is None:
        filename = Config.GRAPH_HTML

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True
    )

    net.barnes_hut()

    # Add nodes and edges
    for edge in G.edges(data=True):
        source, target, data = edge
        label = data.get("label", "")
        net.add_node(source, label=source, title=source)
        net.add_node(target, label=target, title=target)
        net.add_edge(source, target, label=label, title=label)

    # Save
    net.save_graph(filename)
    print(f"📊 Graph visualization saved to: {filename}")
    print(f"   Open in browser: file://{os.path.abspath(filename)}")

# ===========================================================
# 8️⃣ Agent Decision System
# ===========================================================

def agent_decide_action(user_input: str) -> dict:
    """Determine what action/tool to use"""
    # Simple rule-based decision for reliability
    user_lower = user_input.lower()

    # Check for math keywords
    if any(word in user_lower for word in ["calculate", "compute", "math", "+", "-", "*", "/"]):
        # Extract potential expression
        expr = re.search(r'[\d\s+\-*/().]+', user_input)
        if expr:
            return {"tool": "math", "argument": expr.group().strip()}

    # Check for search keywords
    if any(word in user_lower for word in ["search", "find", "latest", "news", "current"]):
        return {"tool": "search", "argument": user_input}

    # Default: use RAG
    return {"tool": "none", "argument": user_input}

# ===========================================================
# 9️⃣ Self-Improvement Engine
# ===========================================================

def learn_from_text(new_text: str):
    """Add new information to both vector DB and knowledge graph"""
    global chunks, index

    if not new_text or len(new_text) < 10:
        return

    print(f"🧠 Learning: {new_text[:100]}...")

    # Add to chunks
    chunks.append(new_text)

    # Update vector DB
    new_emb = np.array(
        embedder.encode([new_text], convert_to_numpy=True)
    ).astype("float32")
    index.add(new_emb)

    # Extract and add triples
    triples = extract_triples_spacy(new_text)
    if triples:
        add_triples_to_graph(triples)
        print(f"   Added {len(triples)} new relationships")

# ===========================================================
# 🔟 Main RAG System
# ===========================================================

def build_rag_system(doc_path: str):
    """Build complete RAG + KG system from document"""
    global chunks

    print(f"📘 Reading document: {doc_path}")
    text = read_document(doc_path)

    print("✂️  Chunking text...")
    chunks = chunk_text(text)
    print(f"   Created {len(chunks)} chunks")

    print("🔢 Building vector database...")
    build_vector_db(chunks)

    print("🕸️  Extracting knowledge graph...")
    for i, chunk in enumerate(chunks):
        triples = extract_triples_spacy(chunk)
        add_triples_to_graph(triples)
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(chunks)} chunks")

    print(f"\n✅ RAG system ready!")
    print(f"   - {len(chunks)} text chunks")
    print(f"   - {len(G.nodes())} entities")
    print(f"   - {len(G.edges())} relationships")

    # Generate initial visualization
    visualize_graph()

def answer_query(query: str) -> str:
    """Answer query using RAG + KG + tools"""
    # Decide action
    decision = agent_decide_action(query)
    tool_name = decision["tool"]
    argument = decision["argument"]

    # Execute tool if needed
    if tool_name in TOOLS:
        result = TOOLS[tool_name](argument)
        learn_from_text(f"Query: {query}\nResult: {result}")
        return result

    # Otherwise use RAG + KG
    vector_results = retrieve_similar(query)
    entities = extract_entities_simple(query)
    graph_facts = query_knowledge_graph(entities)

    # Build context
    context_parts = []

    if vector_results:
        context_parts.append("=== Retrieved Information ===")
        context_parts.extend(vector_results)

    if graph_facts:
        context_parts.append("\n=== Knowledge Graph Facts ===")
        context_parts.extend(graph_facts)

    context = "\n".join(context_parts)

    # Generate answer (simplified for demo)
    if vector_results:
        answer = f"Based on the knowledge base: {vector_results[0][:200]}"
    else:
        answer = "I don't have enough information to answer that question."

    # Learn from interaction
    learn_from_text(f"Q: {query}\nA: {answer}")

    return answer

# ===========================================================
# 1️⃣1️⃣ Interactive Agent Loop
# ===========================================================

def run_interactive_agent(doc_path: str = None):
    """Run interactive agent loop"""
    print("\n" + "="*60)
    print("🤖 Self-Improving Agentic RAG System")
    print("="*60)

    initialize_models()

    if doc_path and os.path.exists(doc_path):
        build_rag_system(doc_path)
    else:
        print("\n⚠️  No document provided. Starting with empty knowledge base.")
        print("   The agent will learn from your interactions.\n")

    print("\nCommands:")
    print("  - Ask any question")
    print("  - 'graph' - visualize knowledge graph")
    print("  - 'stats' - show system statistics")
    print("  - 'exit' - quit")
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
                print(f"\n📊 System Statistics:")
                print(f"   - Knowledge chunks: {len(chunks)}")
                print(f"   - Graph entities: {len(G.nodes())}")
                print(f"   - Graph relationships: {len(G.edges())}")
                continue

            # Answer query
            response = answer_query(user_input)
            print(f"\n🤖 Agent: {response}")

            # Auto-visualize graph periodically
            if len(G.edges()) % 10 == 0 and len(G.edges()) > 0:
                visualize_graph()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ===========================================================
# Main Entry Point
# ===========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        document_path = sys.argv[1]
    else:
        document_path = None

    run_interactive_agent(document_path)
