#!/usr/bin/env python3
"""
Self-Improving Agentic RAG System with Gemini Integration
Enhanced with Google's Gemini for superior triple extraction and reasoning
"""

import json
import re
import os
import numpy as np
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS
from pyvis.network import Network
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# Configuration
# ===========================================================

class Config:
    """Central configuration"""
    CHUNK_SIZE = 400
    VECTOR_TOP_K = 3
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    GRAPH_HTML = "knowledge_graph.html"
    SPACY_MODEL = "en_core_web_sm"
    GEMINI_MODEL = "gemini-2.5-flash"
    GOOGLE_API_KEY = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"

# Global state
embedder = None
gemini_model = None
nlp = None
G = nx.DiGraph()
chunks = []
index = None

def initialize_models():
    """Initialize all ML models"""
    global embedder, gemini_model, nlp

    print("🔄 Loading models...")

    # Initialize Gemini
    genai.configure(api_key=Config.GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
    print("✅ Gemini initialized")

    # Initialize embeddings
    embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
    print("✅ Embedder loaded")

    # Try to load spacy
    try:
        import spacy
        nlp = spacy.load(Config.SPACY_MODEL)
        print("✅ spaCy loaded")
    except:
        print("⚠️  spaCy model not found (optional)")
        nlp = None

    print("✅ All models ready!")

# ===========================================================
# Tool System
# ===========================================================

def tool_math(expr: str) -> str:
    """Safe math evaluation tool"""
    try:
        # Only allow basic math operations
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sum": sum
        }
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
# Document Processing
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
# Gemini-Powered Triple Extraction
# ===========================================================

def extract_triples_gemini(text: str) -> list:
    """Extract subject-relation-object triples using Gemini"""
    prompt = f"""Extract knowledge graph triples from the following text.
Return ONLY a JSON array of triples in the format: [["subject", "relation", "object"], ...]

Rules:
- Extract all meaningful relationships
- Use clear, concise relation names
- Normalize entity names (proper capitalization)
- Return valid JSON only, no explanations

Text:
{text[:1000]}

JSON:"""

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract JSON from response
        start = response_text.find("[")
        end = response_text.rfind("]") + 1

        if start != -1 and end > start:
            json_str = response_text[start:end]
            triples = json.loads(json_str)

            # Validate and clean
            valid_triples = []
            for t in triples:
                if isinstance(t, list) and len(t) == 3:
                    s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                    if s and r and o:
                        valid_triples.append((s, r, o))

            return valid_triples
    except Exception as e:
        print(f"⚠️  Gemini extraction failed: {e}")

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
# Vector Database (FAISS)
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
# Knowledge Graph Queries
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

        # Incoming edges
        for node in G.nodes():
            if G.has_edge(node, entity):
                rel = G[node][entity].get("label", "related_to")
                facts.append(f"{node} [{rel}] {entity}")

    return list(set(facts))

def extract_entities_gemini(text: str) -> list:
    """Extract entities using Gemini"""
    prompt = f"""Extract all named entities from this text.
Return ONLY a JSON array of entity names: ["Entity1", "Entity2", ...]

Text: {text}

JSON:"""

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        start = response_text.find("[")
        end = response_text.rfind("]") + 1

        if start != -1 and end > start:
            entities = json.loads(response_text[start:end])
            return [str(e).strip() for e in entities if e]
    except:
        pass

    # Fallback: capitalize words
    words = text.split()
    return list(set([w for w in words if w and w[0].isupper() and len(w) > 2]))

# ===========================================================
# Graph Visualization
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

    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)

    # Add nodes and edges
    for edge in G.edges(data=True):
        source, target, data = edge
        label = data.get("label", "")
        net.add_node(source, label=source, title=source, color="#4CAF50")
        net.add_node(target, label=target, title=target, color="#2196F3")
        net.add_edge(source, target, label=label, title=label, arrows="to")

    # Save
    net.save_graph(filename)
    print(f"📊 Graph visualization saved to: {filename}")
    print(f"   Open in browser: file://{os.path.abspath(filename)}")

# ===========================================================
# Agent Decision System with Gemini
# ===========================================================

def agent_decide_action(user_input: str) -> dict:
    """Determine what action/tool to use with Gemini"""
    prompt = f"""You are a decision-making agent. Analyze this user query and decide what to do.

User query: "{user_input}"

Respond with JSON only:
{{
  "tool": "math" | "search" | "none",
  "argument": "the relevant argument for the tool or the query itself"
}}

Rules:
- Use "math" if it's a calculation or math problem
- Use "search" if it needs current/external information
- Use "none" for questions answerable from the knowledge base

JSON:"""

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        start = response_text.find("{")
        end = response_text.rfind("}") + 1

        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except:
        pass

    # Fallback: simple rule-based
    user_lower = user_input.lower()

    if any(word in user_lower for word in ["calculate", "compute", "math", "+"]):
        expr = re.search(r'[\d\s+\-*/().]+', user_input)
        if expr:
            return {"tool": "math", "argument": expr.group().strip()}

    if any(word in user_lower for word in ["search", "find", "latest", "current"]):
        return {"tool": "search", "argument": user_input}

    return {"tool": "none", "argument": user_input}

# ===========================================================
# Self-Improvement Engine
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

    # Extract and add triples with Gemini
    triples = extract_triples_gemini(new_text)
    if triples:
        add_triples_to_graph(triples)
        print(f"   Added {len(triples)} new relationships")

# ===========================================================
# Main RAG System
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

    print("🕸️  Extracting knowledge graph with Gemini...")
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}...", end='\r')
        triples = extract_triples_gemini(chunk)
        add_triples_to_graph(triples)

    print(f"\n✅ RAG system ready!")
    print(f"   - {len(chunks)} text chunks")
    print(f"   - {len(G.nodes())} entities")
    print(f"   - {len(G.edges())} relationships")

    # Generate initial visualization
    visualize_graph()

def answer_query_gemini(query: str) -> str:
    """Answer query using RAG + KG + Gemini reasoning"""
    # Decide action
    decision = agent_decide_action(query)
    tool_name = decision.get("tool", "none")
    argument = decision.get("argument", query)

    # Execute tool if needed
    if tool_name in TOOLS:
        result = TOOLS[tool_name](argument)
        learn_from_text(f"Query: {query}\nResult: {result}")
        return result

    # Otherwise use RAG + KG + Gemini
    vector_results = retrieve_similar(query)
    entities = extract_entities_gemini(query)
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

    # Use Gemini for final answer
    prompt = f"""You are a helpful AI assistant with access to a knowledge base.
Answer the user's question based ONLY on the provided context.
Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"I found some information but couldn't generate an answer: {str(e)}"

    # Learn from interaction
    learn_from_text(f"Q: {query}\nA: {answer}")

    return answer

# ===========================================================
# Interactive Agent Loop
# ===========================================================

def run_interactive_agent(doc_path: str = None):
    """Run interactive agent loop"""
    print("\n" + "="*60)
    print("🤖 Self-Improving Agentic RAG System (Gemini Edition)")
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
            response = answer_query_gemini(user_input)
            print(f"\n🤖 Agent: {response}")

            # Auto-visualize graph periodically
            if len(G.edges()) % 10 == 0 and len(G.edges()) > 0:
                visualize_graph()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ===========================================================
# Main Entry Point
# ===========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        document_path = sys.argv[1]
    else:
        document_path = "sample_knowledge.txt"

    run_interactive_agent(document_path)
