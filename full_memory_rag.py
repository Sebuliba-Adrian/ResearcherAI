#!/usr/bin/env python3
"""
FULL MEMORY RAG System
Now with REAL conversation memory and context tracking
"""

import json
import re
import os
import pickle
from datetime import datetime
import networkx as nx
from PyPDF2 import PdfReader
from pyvis.network import Network
import google.generativeai as genai

# Configuration
GOOGLE_API_KEY = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
GEMINI_MODEL = "gemini-2.5-flash"
SESSIONS_DIR = "sessions"  # Directory for multiple sessions
DEFAULT_SESSION = "default"  # Default session name

# Global state
G = nx.DiGraph()
chunks = []
conversation_history = []  # NEW: Track conversation
source_document = None  # Track source document
current_session = DEFAULT_SESSION  # Track current session name

print("🚀 Initializing Full Memory RAG System...")
print("="*60)

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)
print("✅ Gemini AI initialized")
print("✅ NetworkX Knowledge Graph ready")
print("✅ Conversation Memory enabled")
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
# Retrieval with Conversation Context
# ===========================================================

def retrieve_with_gemini(query, chunks_list):
    """Use Gemini to find relevant chunks"""

    # Create numbered chunks
    chunks_text = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk}"
        for i, chunk in enumerate(chunks_list)
    ])

    prompt = f"""Find which chunks contain information to answer this query.

CHUNKS:
{chunks_text}

QUERY: "{query}"

Return ONLY a JSON array of chunk numbers, e.g., [1, 3, 5]
If no chunks are relevant, return []

JSON:"""

    try:
        response = gemini_model.generate_content(prompt)
        resp_text = response.text.strip()

        start = resp_text.find("[")
        end = resp_text.rfind("]") + 1

        if start != -1 and end > start:
            json_str = resp_text[start:end]
            chunk_numbers = json.loads(json_str)

            relevant = []
            for num in chunk_numbers:
                if isinstance(num, int) and 1 <= num <= len(chunks_list):
                    relevant.append(chunks_list[num - 1])

            return relevant

    except Exception as e:
        print(f"⚠️  Retrieval error: {e}")

    return chunks_list[:2]  # Fallback

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
    except:
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
        for node in G.nodes():
            if entity.lower() in node.lower() or node.lower() in entity.lower():
                for neighbor in G.neighbors(node):
                    rel = G[node][neighbor].get("label", "related_to")
                    facts.append(f"{node} [{rel}] {neighbor}")

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

    words = text.split()
    return list(set([w for w in words if w and w[0].isupper() and len(w) > 2]))

# ===========================================================
# NEW: Answer with Full Conversation Memory
# ===========================================================

def answer_query_with_memory(query):
    """Answer using conversation history"""

    print(f"\n🔍 Processing with conversation memory...")

    # 1. Build conversation context
    conversation_context = ""
    if conversation_history:
        conversation_context = "PREVIOUS CONVERSATION:\n"
        for i, turn in enumerate(conversation_history[-3:], 1):  # Last 3 turns
            conversation_context += f"Turn {i}:\n"
            conversation_context += f"  User: {turn['query']}\n"
            conversation_context += f"  Agent: {turn['answer'][:200]}\n\n"

    # 2. Retrieve relevant chunks
    relevant_chunks = retrieve_with_gemini(query, chunks)
    print(f"✅ Found {len(relevant_chunks)} relevant chunks")

    # 3. Query knowledge graph
    entities = extract_entities_gemini(query)
    graph_facts = query_knowledge_graph(entities)

    # 4. Build context with conversation history
    context_parts = []

    if conversation_context:
        context_parts.append(conversation_context)

    if relevant_chunks:
        context_parts.append("=== Document Information ===")
        for i, chunk in enumerate(relevant_chunks[:3], 1):
            context_parts.append(f"{i}. {chunk}")

    if graph_facts:
        context_parts.append("\n=== Knowledge Graph Facts ===")
        context_parts.extend(graph_facts[:5])

    context = "\n".join(context_parts)

    # 5. Generate answer with conversation awareness
    prompt = f"""You are a helpful AI assistant with conversation memory.

{context}

CURRENT QUESTION: {query}

Instructions:
1. Use the PREVIOUS CONVERSATION to understand references like "that", "they", "it"
2. Answer based on the document information and knowledge graph
3. Be specific and maintain conversation continuity

Answer:"""

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error: {str(e)}"

    # 6. Save to conversation history
    conversation_history.append({
        "query": query,
        "answer": answer,
        "entities": entities,
        "retrieved_chunks": len(relevant_chunks)
    })

    # Show memory status
    print(f"💾 Conversation history: {len(conversation_history)} turns")

    return answer

# ===========================================================
# Session Management
# ===========================================================

def ensure_sessions_dir():
    """Ensure sessions directory exists"""
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR)

def get_session_path(session_name):
    """Get full path for a session file"""
    ensure_sessions_dir()
    return os.path.join(SESSIONS_DIR, f"{session_name}.pkl")

def list_sessions():
    """List all available sessions"""
    ensure_sessions_dir()
    sessions = []

    if os.path.exists(SESSIONS_DIR):
        for filename in os.listdir(SESSIONS_DIR):
            if filename.endswith(".pkl"):
                session_name = filename[:-4]  # Remove .pkl
                filepath = os.path.join(SESSIONS_DIR, filename)

                try:
                    with open(filepath, "rb") as f:
                        state = pickle.load(f)
                        sessions.append({
                            "name": session_name,
                            "source": state.get("source_document", "Unknown"),
                            "conversations": state["stats"]["num_conversations"],
                            "entities": state["stats"]["num_entities"],
                            "timestamp": state.get("timestamp", "Unknown"),
                            "current": session_name == current_session
                        })
                except:
                    pass

    return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

def create_new_session(session_name, doc_path=None):
    """Create a new session"""
    global current_session

    # Save current session if it has data
    if len(chunks) > 0 or len(conversation_history) > 0:
        print(f"\n💾 Saving current session '{current_session}'...")
        save_state(get_session_path(current_session))

    # Clear current state
    clear_current_state()

    # Set new session name
    current_session = session_name
    print(f"\n✨ Created new session: '{session_name}'")

    # Build system if document provided
    if doc_path:
        build_system(doc_path)

def switch_session(session_name):
    """Switch to a different session"""
    global current_session

    session_path = get_session_path(session_name)

    if not os.path.exists(session_path):
        print(f"\n❌ Session '{session_name}' not found")
        return False

    # Save current session if it has data
    if len(chunks) > 0 or len(conversation_history) > 0:
        print(f"\n💾 Saving current session '{current_session}'...")
        save_state(get_session_path(current_session))

    # Load new session
    current_session = session_name
    if load_state(session_path):
        print(f"\n✅ Switched to session: '{session_name}'")
        return True
    return False

def delete_session(session_name):
    """Delete a session"""
    if session_name == current_session:
        print(f"\n⚠️  Cannot delete active session. Switch to another session first.")
        return False

    session_path = get_session_path(session_name)

    if not os.path.exists(session_path):
        print(f"\n❌ Session '{session_name}' not found")
        return False

    try:
        os.remove(session_path)
        print(f"\n🗑️  Deleted session: '{session_name}'")
        return True
    except Exception as e:
        print(f"\n❌ Error deleting session: {e}")
        return False

def clear_current_state():
    """Clear all current state (for switching sessions)"""
    global G, chunks, conversation_history, source_document
    G = nx.DiGraph()
    chunks = []
    conversation_history = []
    source_document = None

# ===========================================================
# Persistence - Save/Load System State
# ===========================================================

def save_state(filename=None):
    """Save entire system state to disk"""
    if filename is None:
        filename = get_session_path(current_session)

    state = {
        "graph": nx.node_link_data(G),  # NetworkX graph as JSON-serializable dict
        "chunks": chunks,
        "conversation_history": conversation_history,
        "source_document": source_document,
        "session_name": current_session,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "num_chunks": len(chunks),
            "num_entities": len(G.nodes()),
            "num_relationships": len(G.edges()),
            "num_conversations": len(conversation_history)
        }
    }

    try:
        with open(filename, "wb") as f:
            pickle.dump(state, f)

        print(f"\n💾 Session '{current_session}' saved!")
        print(f"   File: {filename}")
        print(f"   Chunks: {state['stats']['num_chunks']}")
        print(f"   Entities: {state['stats']['num_entities']}")
        print(f"   Conversations: {state['stats']['num_conversations']}")
        print(f"   Timestamp: {state['timestamp']}")
        return True
    except Exception as e:
        print(f"\n❌ Error saving state: {e}")
        return False

def load_state(filename=None):
    """Load system state from disk"""
    global G, chunks, conversation_history, source_document, current_session

    if filename is None:
        filename = get_session_path(current_session)

    if not os.path.exists(filename):
        print(f"\n⚠️  No saved state found at {filename}")
        return False

    try:
        with open(filename, "rb") as f:
            state = pickle.load(f)

        # Restore all components
        G = nx.node_link_graph(state["graph"], directed=True)
        chunks = state["chunks"]
        conversation_history = state["conversation_history"]
        source_document = state.get("source_document")

        # Update current session name if stored
        if "session_name" in state:
            current_session = state["session_name"]

        print(f"\n📂 Session '{current_session}' loaded!")
        print(f"   File: {filename}")
        print(f"   Chunks: {state['stats']['num_chunks']}")
        print(f"   Entities: {state['stats']['num_entities']}")
        print(f"   Conversations: {state['stats']['num_conversations']}")
        print(f"   Saved: {state['timestamp']}")
        print(f"   Source: {source_document or 'Unknown'}")
        return True
    except Exception as e:
        print(f"\n❌ Error loading state: {e}")
        import traceback
        traceback.print_exc()
        return False

def auto_save():
    """Auto-save before exit"""
    if len(chunks) > 0 or len(conversation_history) > 0:
        print("\n💾 Auto-saving system state...")
        save_state()

# ===========================================================
# Visualization
# ===========================================================

def visualize_graph(filename="knowledge_graph.html"):
    """Generate interactive HTML visualization of knowledge graph"""
    if len(G.nodes()) == 0:
        print("⚠️  Knowledge graph is empty. Build the system first.")
        return

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True
    )

    # Configure physics
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.001
    )

    # Add nodes and edges
    for edge in G.edges(data=True):
        source, target, data = edge
        label = data.get("label", "")

        # Add nodes with colors
        net.add_node(source, label=source, title=source, color="#4CAF50", size=25)
        net.add_node(target, label=target, title=target, color="#2196F3", size=25)

        # Add edge
        net.add_edge(source, target, label=label, title=label, arrows="to")

    # Save
    net.save_graph(filename)
    print(f"\n📊 Knowledge Graph Visualization:")
    print(f"   File: {filename}")
    print(f"   Entities: {len(G.nodes())}")
    print(f"   Relationships: {len(G.edges())}")
    print(f"   Open: file://{os.path.abspath(filename)}")

# ===========================================================
# Build System
# ===========================================================

def build_system(doc_path):
    """Build RAG + KG system"""
    global chunks, source_document

    source_document = doc_path  # Track source
    print(f"\n📘 Reading: {doc_path}")
    text = read_document(doc_path)

    print("✂️  Chunking...")
    chunks = chunk_text(text)
    print(f"   Created {len(chunks)} chunks\n")

    print(f"🕸️  Building Knowledge Graph with Gemini...")
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

def run_interactive(doc_path=None, session_name=None):
    """Run interactive session with multi-session support"""
    global current_session

    print("\n" + "="*60)
    print("🤖 Full Memory RAG System with Multi-Session Support")
    print("="*60)
    print()

    # Set session name if provided
    if session_name:
        current_session = session_name

    # Try to load session or build new system
    if doc_path is None or doc_path == "resume":
        # Try to load default session
        if load_state():
            print(f"\n✅ Resumed session '{current_session}'!")
        else:
            print("\n⚠️  No saved session found.")
            print("\n💡 Available sessions:")
            sessions = list_sessions()
            if sessions:
                for s in sessions:
                    marker = "→" if s["current"] else " "
                    print(f"   {marker} {s['name']} ({s['conversations']} convs, {s['entities']} entities)")
                print("\nTip: Use 'sessions' command to manage sessions")
            else:
                print("   No sessions found. Please provide a document path.")
            return
    else:
        # Building new system - use session name or generate from doc
        if session_name is None:
            # Generate session name from document
            base_name = os.path.splitext(os.path.basename(doc_path))[0]
            current_session = base_name

        build_system(doc_path)

    print(f"\n📌 Current session: '{current_session}'")
    print("\n💡 Commands:")
    print("   - Ask any question (with conversation context!)")
    print("   - 'sessions' - list all sessions")
    print("   - 'new <name> [doc]' - create new session")
    print("   - 'switch <name>' - switch to different session")
    print("   - 'delete <name>' - delete a session")
    print("   - 'graph' - visualize knowledge graph")
    print("   - 'memory' - show conversation history")
    print("   - 'save' - save current session")
    print("   - 'clear' - clear conversation memory")
    print("   - 'stats' - show statistics")
    print("   - 'exit' - quit (auto-saves)")
    print()

    try:
        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    auto_save()
                    print("\n👋 Goodbye!")
                    break

                # Session management commands
                if user_input.lower() == "sessions":
                    sessions = list_sessions()
                    if sessions:
                        print(f"\n📂 Available Sessions ({len(sessions)}):")
                        for s in sessions:
                            marker = "→" if s["name"] == current_session else " "
                            print(f"\n{marker} {s['name']}")
                            print(f"   Source: {s['source']}")
                            print(f"   Conversations: {s['conversations']}")
                            print(f"   Entities: {s['entities']}")
                            print(f"   Last saved: {s['timestamp'][:19]}")
                    else:
                        print("\n📂 No sessions found")
                    continue

                if user_input.lower().startswith("new "):
                    parts = user_input[4:].split(maxsplit=1)
                    new_name = parts[0]
                    new_doc = parts[1] if len(parts) > 1 else None
                    create_new_session(new_name, new_doc)
                    continue

                if user_input.lower().startswith("switch "):
                    session_name = user_input[7:].strip()
                    switch_session(session_name)
                    continue

                if user_input.lower().startswith("delete "):
                    session_name = user_input[7:].strip()
                    confirm = input(f"⚠️  Delete session '{session_name}'? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        delete_session(session_name)
                    else:
                        print("❌ Deletion cancelled")
                    continue

                if user_input.lower() == "save":
                    save_state()
                    continue

                if user_input.lower() == "graph":
                    visualize_graph()
                    continue

                if user_input.lower() == "memory":
                    print(f"\n💾 Conversation History ({len(conversation_history)} turns):")
                    for i, turn in enumerate(conversation_history, 1):
                        print(f"\n{i}. Q: {turn['query']}")
                        print(f"   A: {turn['answer'][:150]}...")
                    continue

                if user_input.lower() == "clear":
                    conversation_history.clear()
                    print("\n🗑️  Conversation memory cleared")
                    continue

                if user_input.lower() == "stats":
                    print(f"\n📊 Statistics:")
                    print(f"   - Session: {current_session}")
                    print(f"   - Source: {source_document or 'Unknown'}")
                    print(f"   - Chunks: {len(chunks)}")
                    print(f"   - Entities: {len(G.nodes())}")
                    print(f"   - Relationships: {len(G.edges())}")
                    print(f"   - Conversation turns: {len(conversation_history)}")
                    continue

                # Answer with full memory
                answer = answer_query_with_memory(user_input)
                print(f"\n🤖 Agent:\n{answer}")

            except KeyboardInterrupt:
                auto_save()
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    finally:
        # Ensure auto-save on any exit
        auto_save()

# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Check if it's a session command
        if arg == "resume":
            run_interactive(doc_path="resume")
        else:
            # Treat as document path
            session_name = sys.argv[2] if len(sys.argv) > 2 else None
            run_interactive(doc_path=arg, session_name=session_name)
    else:
        # No arguments - try to resume or show sessions
        sessions = list_sessions()
        if sessions:
            print("\n💡 Available sessions:")
            for s in sessions[:5]:  # Show top 5
                print(f"   - {s['name']} ({s['conversations']} convs, updated {s['timestamp'][:10]})")
            print("\nTip: Run 'python3 full_memory_rag.py resume' to continue")
        else:
            print("\n⚠️  No sessions found. Starting with sample document...")
            run_interactive("sample_knowledge.txt")
