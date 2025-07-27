#!/usr/bin/env python3
"""
Complete Multi-Agent Research Assistant with Sessions & Memory
===============================================================
Combines multi-agent architecture with conversation memory and session management.

Features:
- 5 Specialized Agents (Data, Graph, Vector, Reasoning, Orchestrator)
- Conversation Memory (tracks context across turns)
- Multi-Session Support (independent research threads)
- Session Switching (seamlessly switch between sessions)
- Auto-save/Resume (never lose progress)
"""

import os
import re
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import requests
import feedparser
import networkx as nx
from pyvis.network import Network
import google.generativeai as genai

# Configuration
GOOGLE_API_KEY = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
GEMINI_MODEL = "gemini-2.5-flash"

DATA_DIR = "research_data"
SESSIONS_DIR = "research_sessions"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

print("🤖 Initializing Complete Multi-Agent Research System...")
print("   With Conversation Memory & Multi-Session Support")
print("="*70)

# ============================================================================
# Session Management
# ============================================================================

def get_session_path(session_name):
    """Get path for session file"""
    return os.path.join(SESSIONS_DIR, f"{session_name}.pkl")

def list_sessions():
    """List all available research sessions"""
    sessions = []
    if os.path.exists(SESSIONS_DIR):
        for filename in os.listdir(SESSIONS_DIR):
            if filename.endswith(".pkl"):
                session_name = filename[:-4]
                filepath = os.path.join(SESSIONS_DIR, filename)
                try:
                    with open(filepath, "rb") as f:
                        state = pickle.load(f)
                        sessions.append({
                            "name": session_name,
                            "papers_collected": state.get("metadata", {}).get("total_papers_collected", 0),
                            "conversations": len(state.get("conversation_history", [])),
                            "graph_nodes": len(state.get("graph_nodes", [])),
                            "timestamp": state.get("timestamp", "Unknown")
                        })
                except:
                    pass
    return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

# ============================================================================
# Agent Classes (Same as before, with added conversation memory)
# ============================================================================

class DataCollectorAgent:
    """Collects research papers from multiple sources"""
    
    def __init__(self):
        self.sources = {
            "arxiv": self.fetch_arxiv,
            "websearch": self.fetch_web
        }
        self.collected_count = 0
        
    def fetch_arxiv(self, category="cs.AI", days=7, max_results=10):
        """Fetch recent arXiv papers"""
        print(f"  📡 Fetching from arXiv ({category})...")
        try:
            url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
            feed = feedparser.parse(url)
            since = datetime.utcnow() - timedelta(days=days)
            
            papers = []
            for entry in feed.entries:
                try:
                    published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
                    if published >= since:
                        paper = {
                            "id": "arxiv_" + entry.id.split("/")[-1],
                            "title": entry.title,
                            "abstract": entry.summary,
                            "authors": [a.name for a in entry.authors],
                            "topics": [t.term for t in getattr(entry, "tags", [])],
                            "source": "arXiv",
                            "url": entry.link,
                            "published": published.isoformat(),
                            "text": f"Title: {entry.title}\n\nAbstract: {entry.summary}"
                        }
                        papers.append(paper)
                except Exception as e:
                    continue
                    
            print(f"    ✅ Found {len(papers)} papers from arXiv")
            self.collected_count += len(papers)
            return papers
            
        except Exception as e:
            print(f"    ❌ Error fetching arXiv: {e}")
            return []
    
    def fetch_web(self, query="latest AI research", max_results=5):
        """Fetch recent web articles"""
        print(f"  📡 Searching web for: {query}...")
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            papers = []
            for i, r in enumerate(results):
                paper = {
                    "id": f"web_{i}_{hash(r['href'])}",
                    "title": r["title"],
                    "abstract": r["body"],
                    "authors": ["Web Source"],
                    "topics": ["Web", "Current"],
                    "source": "Web",
                    "url": r["href"],
                    "published": datetime.now().isoformat(),
                    "text": f"Title: {r['title']}\n\n{r['body']}"
                }
                papers.append(paper)
            
            print(f"    ✅ Found {len(papers)} web results")
            self.collected_count += len(papers)
            return papers
            
        except Exception as e:
            print(f"    ⚠️  Web search error: {e}")
            return []
    
    def collect_all(self, sources=None):
        """Collect from all enabled sources"""
        if sources is None:
            sources = ["arxiv", "websearch"]
        
        print(f"\n🔍 DataCollectorAgent starting collection...")
        all_papers = []
        
        for source in sources:
            if source in self.sources:
                papers = self.sources[source]()
                all_papers.extend(papers)
        
        print(f"\n✅ DataCollectorAgent collected {len(all_papers)} items total")
        return all_papers

class KnowledgeGraphAgent:
    """Builds and maintains knowledge graph"""
    
    def __init__(self):
        self.G = nx.MultiDiGraph()
        
    def extract_triples_gemini(self, text):
        """Extract triples using Gemini"""
        prompt = f"""Extract knowledge triples from this research text.
Return ONLY a JSON array of [subject, relation, object] triples.

Text: {text[:1000]}

JSON:"""
        
        try:
            response = gemini_model.generate_content(prompt)
            resp_text = response.text.strip()
            
            start = resp_text.find("[")
            end = resp_text.rfind("]") + 1
            
            if start != -1 and end > start:
                triples = json.loads(resp_text[start:end])
                valid_triples = []
                for t in triples:
                    if isinstance(t, list) and len(t) == 3:
                        s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                        if s and r and o:
                            valid_triples.append((s, r, o))
                return valid_triples
                
        except Exception as e:
            pass
        
        return []
    
    def add_paper_to_graph(self, paper):
        """Add paper metadata to graph"""
        pid = paper["id"]
        
        self.G.add_node(pid, type="paper", title=paper["title"],
                       source=paper["source"], url=paper.get("url", ""))
        
        for author in paper.get("authors", []):
            self.G.add_node(author, type="author")
            self.G.add_edge(pid, author, label="authored_by")
        
        for topic in paper.get("topics", []):
            self.G.add_node(topic, type="topic")
            self.G.add_edge(pid, topic, label="about")
        
        triples = self.extract_triples_gemini(paper["text"])
        for s, r, o in triples:
            self.G.add_node(s, type="entity")
            self.G.add_node(o, type="entity")
            self.G.add_edge(s, o, label=r)
    
    def process_papers(self, papers):
        """Process multiple papers"""
        print(f"\n🕸️  KnowledgeGraphAgent processing {len(papers)} papers...")
        
        for i, paper in enumerate(papers, 1):
            print(f"  Processing {i}/{len(papers)}: {paper['title'][:60]}...")
            self.add_paper_to_graph(paper)
        
        print(f"✅ Graph updated: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")
    
    def visualize(self, filename="research_graph.html"):
        """Create interactive visualization"""
        print(f"\n📊 Generating graph visualization...")
        
        net = Network(height="750px", width="100%", bgcolor="#222222", 
                     font_color="white", directed=True)
        
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)
        
        color_map = {
            "paper": "#4CAF50", "author": "#FF9800",
            "topic": "#2196F3", "entity": "#9C27B0"
        }
        
        for node, data in self.G.nodes(data=True):
            node_type = data.get("type", "entity")
            color = color_map.get(node_type, "gray")
            title = data.get("title", node)
            net.add_node(node, label=str(node)[:50], title=title, color=color, size=25)
        
        for src, dst, data in self.G.edges(data=True):
            label = data.get("label", "")
            net.add_edge(src, dst, title=label, label=label[:20], arrows="to")
        
        net.save_graph(filename)
        print(f"✅ Graph saved to {filename}")

class VectorAgent:
    """Manages semantic search"""
    
    def __init__(self):
        self.chunks = []
    
    def chunk_text(self, text, size=400):
        """Chunk text intelligently"""
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) < size:
                current += sent + " "
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent + " "
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks
    
    def add_paper(self, paper):
        """Add paper chunks"""
        new_chunks = self.chunk_text(paper["text"])
        for chunk in new_chunks:
            self.chunks.append({
                "text": chunk,
                "paper_id": paper["id"],
                "title": paper["title"],
                "source": paper["source"]
            })
    
    def process_papers(self, papers):
        """Process multiple papers"""
        print(f"\n📚 VectorAgent processing {len(papers)} papers...")
        for paper in papers:
            self.add_paper(paper)
        print(f"✅ Total chunks: {len(self.chunks)}")
    
    def retrieve_with_gemini(self, query, k=3):
        """Use Gemini to find relevant chunks"""
        if not self.chunks:
            return []
        
        chunks_text = "\n\n".join([
            f"CHUNK {i+1} (from {chunk['title'][:50]}):\n{chunk['text'][:300]}"
            for i, chunk in enumerate(self.chunks[:min(20, len(self.chunks))])
        ])
        
        prompt = f"""Find the {k} most relevant chunks to answer this query.
        
CHUNKS:
{chunks_text}

QUERY: "{query}"

Return ONLY a JSON array of chunk numbers, e.g., [1, 3, 5]
JSON:"""
        
        try:
            response = gemini_model.generate_content(prompt)
            resp_text = response.text.strip()
            
            start = resp_text.find("[")
            end = resp_text.rfind("]") + 1
            
            if start != -1 and end > start:
                chunk_numbers = json.loads(resp_text[start:end])
                relevant = []
                for num in chunk_numbers:
                    if isinstance(num, int) and 1 <= num <= len(self.chunks):
                        relevant.append(self.chunks[num - 1])
                return relevant
                
        except Exception as e:
            pass
        
        return self.chunks[:k] if len(self.chunks) >= k else self.chunks

class ReasoningAgent:
    """Handles complex reasoning with conversation memory"""
    
    def __init__(self, graph_agent, vector_agent):
        self.graph_agent = graph_agent
        self.vector_agent = vector_agent
        self.conversation_history = []  # Track all conversations
    
    def synthesize_answer(self, query):
        """Synthesize answer with conversation memory"""
        print(f"\n🧠 ReasoningAgent processing query: {query}")
        
        # 1. Build conversation context
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "PREVIOUS CONVERSATION:\n"
            for i, turn in enumerate(self.conversation_history[-3:], 1):
                conversation_context += f"Turn {i}:\n"
                conversation_context += f"  User: {turn['query']}\n"
                conversation_context += f"  Agent: {turn['answer'][:200]}\n\n"
        
        # 2. Retrieve relevant text chunks
        text_chunks = self.vector_agent.retrieve_with_gemini(query, k=3)
        
        # 3. Build context
        context = f"""You are an AI research assistant with conversation memory.

{conversation_context}

RETRIEVED TEXT CHUNKS:
"""
        for i, chunk in enumerate(text_chunks, 1):
            context += f"\nChunk {i} (from {chunk.get('title', 'Unknown')}):\n{chunk['text'][:500]}\n"

        final_prompt = f"""{context}

CURRENT QUESTION: {query}

Instructions:
1. Use the PREVIOUS CONVERSATION to understand references like "that", "they", "it"
2. Answer based on the retrieved information
3. Be specific and maintain conversation continuity

Answer:"""
        
        try:
            response = gemini_model.generate_content(final_prompt)
            answer = response.text.strip()
            print("  ✅ Answer synthesized")
            
            # Save to conversation history
            self.conversation_history.append({
                "query": query,
                "answer": answer,
                "retrieved_chunks": len(text_chunks)
            })
            
            return answer
        except Exception as e:
            print(f"  ❌ Synthesis error: {e}")
            return f"Error generating answer: {e}"

# ============================================================================
# OrchestratorAgent with Full Session Management
# ============================================================================

class OrchestratorAgent:
    """
    Orchestrates all agents with session management
    """
    
    def __init__(self, session_name="default"):
        print(f"\n🎭 OrchestratorAgent initializing session '{session_name}'...")
        self.session_name = session_name
        self.data_collector = DataCollectorAgent()
        self.graph_agent = KnowledgeGraphAgent()
        self.vector_agent = VectorAgent()
        self.reasoning_agent = ReasoningAgent(self.graph_agent, self.vector_agent)
        self.metadata = {"total_papers_collected": 0, "last_collection": None}
        
        # Try to load existing session
        self.load_session()
        print("✅ All agents initialized\n")
    
    def save_session(self):
        """Save current session state"""
        session_path = get_session_path(self.session_name)
        
        state = {
            "session_name": self.session_name,
            "graph_nodes": list(self.graph_agent.G.nodes(data=True)),
            "graph_edges": list(self.graph_agent.G.edges(data=True)),
            "chunks": self.vector_agent.chunks,
            "conversation_history": self.reasoning_agent.conversation_history,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(session_path, "wb") as f:
                pickle.dump(state, f)
            print(f"💾 Session '{self.session_name}' saved")
            return True
        except Exception as e:
            print(f"❌ Error saving session: {e}")
            return False
    
    def load_session(self):
        """Load existing session state"""
        session_path = get_session_path(self.session_name)
        
        if not os.path.exists(session_path):
            print(f"ℹ️  No existing session '{self.session_name}', starting fresh")
            return False
        
        try:
            with open(session_path, "rb") as f:
                state = pickle.load(f)
            
            # Restore graph
            self.graph_agent.G = nx.MultiDiGraph()
            for node, data in state["graph_nodes"]:
                self.graph_agent.G.add_node(node, **data)
            for src, dst, data in state["graph_edges"]:
                self.graph_agent.G.add_edge(src, dst, **data)
            
            # Restore chunks
            self.vector_agent.chunks = state["chunks"]
            
            # Restore conversation history
            self.reasoning_agent.conversation_history = state["conversation_history"]
            
            # Restore metadata
            self.metadata = state["metadata"]
            
            print(f"📂 Session '{self.session_name}' loaded!")
            print(f"   Papers: {self.metadata.get('total_papers_collected', 0)}")
            print(f"   Graph nodes: {len(self.graph_agent.G.nodes())}")
            print(f"   Conversations: {len(self.reasoning_agent.conversation_history)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading session: {e}")
            return False
    
    def switch_session(self, new_session_name):
        """Switch to different session"""
        # Save current session
        self.save_session()
        
        # Switch to new session
        self.session_name = new_session_name
        
        # Reset agents
        self.graph_agent = KnowledgeGraphAgent()
        self.vector_agent = VectorAgent()
        self.reasoning_agent = ReasoningAgent(self.graph_agent, self.vector_agent)
        self.metadata = {"total_papers_collected": 0, "last_collection": None}
        
        # Load new session
        self.load_session()
        print(f"✅ Switched to session '{new_session_name}'")
    
    def run_collection_cycle(self, sources=None):
        """Run full data collection and processing cycle"""
        print("\n" + "="*70)
        print("🔄 STARTING COLLECTION CYCLE")
        print("="*70)
        
        start_time = time.time()
        
        # 1. Collect data
        papers = self.data_collector.collect_all(sources)
        
        if not papers:
            print("\n⚠️  No new papers collected")
            return
        
        # 2. Process with graph agent
        self.graph_agent.process_papers(papers)
        
        # 3. Process with vector agent
        self.vector_agent.process_papers(papers)
        
        # 4. Update metadata
        self.metadata["total_papers_collected"] += len(papers)
        self.metadata["last_collection"] = datetime.now().isoformat()
        
        # 5. Auto-save
        self.save_session()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Collection cycle complete in {elapsed:.1f}s")
        print("="*70)
    
    def interactive_mode(self):
        """Interactive research assistant mode with session management"""
        print("\n" + "="*70)
        print("🤖 MULTI-AGENT RESEARCH ASSISTANT")
        print(f"📌 Current Session: '{self.session_name}'")
        print("="*70)
        print("\n💡 Commands:")
        print("   - Ask any research question")
        print("   - 'sessions' - list all sessions")
        print("   - 'switch <name>' - switch to different session")
        print("   - 'collect' - run data collection")
        print("   - 'graph' - visualize knowledge graph")
        print("   - 'memory' - show conversation history")
        print("   - 'stats' - show statistics")
        print("   - 'save' - save current session")
        print("   - 'exit' - quit (auto-saves)\n")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    self.save_session()
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == "sessions":
                    sessions = list_sessions()
                    if sessions:
                        print(f"\n📂 Available Sessions ({len(sessions)}):")
                        for s in sessions:
                            marker = "→" if s["name"] == self.session_name else " "
                            print(f"\n{marker} {s['name']}")
                            print(f"   Papers: {s['papers_collected']}")
                            print(f"   Conversations: {s['conversations']}")
                            print(f"   Last update: {s['timestamp'][:19]}")
                    else:
                        print("\n📂 No sessions found")
                    continue
                
                if user_input.lower().startswith("switch "):
                    new_session = user_input[7:].strip()
                    self.switch_session(new_session)
                    continue
                
                if user_input.lower() == "collect":
                    sources = input("  Sources (arxiv,websearch or Enter for default): ").strip()
                    sources = sources.split(",") if sources else None
                    self.run_collection_cycle(sources)
                    continue
                
                if user_input.lower() == "graph":
                    self.graph_agent.visualize()
                    continue
                
                if user_input.lower() == "memory":
                    history = self.reasoning_agent.conversation_history
                    print(f"\n💾 Conversation History ({len(history)} turns):")
                    for i, turn in enumerate(history, 1):
                        print(f"\n{i}. Q: {turn['query']}")
                        print(f"   A: {turn['answer'][:150]}...")
                    continue
                
                if user_input.lower() == "save":
                    self.save_session()
                    continue
                
                if user_input.lower() == "stats":
                    print(f"\n📊 Session Statistics:")
                    print(f"   Session: {self.session_name}")
                    print(f"   Total papers: {self.metadata['total_papers_collected']}")
                    print(f"   Graph nodes: {len(self.graph_agent.G.nodes())}")
                    print(f"   Graph edges: {len(self.graph_agent.G.edges())}")
                    print(f"   Text chunks: {len(self.vector_agent.chunks)}")
                    print(f"   Conversations: {len(self.reasoning_agent.conversation_history)}")
                    continue
                
                # Process as research question
                answer = self.reasoning_agent.synthesize_answer(user_input)
                print(f"\n🤖 Agent:\n{answer}\n")
                print(f"💾 Conversation history: {len(self.reasoning_agent.conversation_history)} turns")
                
            except KeyboardInterrupt:
                self.save_session()
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for session argument
    if len(sys.argv) > 1:
        session_name = sys.argv[1]
    else:
        # Check for existing sessions
        sessions = list_sessions()
        if sessions:
            print("\n💡 Available sessions:")
            for s in sessions[:5]:
                print(f"   - {s['name']} ({s['conversations']} conversations)")
            print("\nTip: python3 multi_agent_rag_complete.py <session_name>")
        session_name = "default"
    
    print(f"\n✅ Multi-Agent Research System with Sessions initialized")
    
    orchestrator = OrchestratorAgent(session_name)
    orchestrator.interactive_mode()
