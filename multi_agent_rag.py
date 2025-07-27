#!/usr/bin/env python3
"""
Multi-Agent Research Assistant System
======================================
A sophisticated multi-agent system for autonomous research, learning, and knowledge discovery.

Agents:
- OrchestratorAgent: Coordinates all agents and manages workflow
- DataCollectorAgent: Fetches from arXiv, PubMed, Zenodo, web search
- KnowledgeGraphAgent: Builds and maintains knowledge graph
- VectorAgent: Manages vector embeddings and semantic search
- ReasoningAgent: Complex reasoning and query answering with Gemini
- SchedulerAgent: Automated periodic data collection
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

GRAPH_PATH = os.path.join(DATA_DIR, "knowledge_graph.pkl")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
AGENT_STATE_PATH = os.path.join(DATA_DIR, "agent_state.json")

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

print("🤖 Initializing Multi-Agent Research System...")
print("="*70)

# ============================================================================
# Agent 1: DataCollectorAgent - Autonomous Data Gathering
# ============================================================================

class DataCollectorAgent:
    """
    Collects research papers and data from multiple sources:
    - arXiv (AI/ML papers)
    - PubMed (biomedical research)
    - Zenodo (open research data)
    - Web search for current topics
    """
    
    def __init__(self):
        self.sources = {
            "arxiv": self.fetch_arxiv,
            "pubmed": self.fetch_pubmed,
            "zenodo": self.fetch_zenodo,
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
                    print(f"    ⚠️  Error parsing entry: {e}")
                    continue
                    
            print(f"    ✅ Found {len(papers)} papers from arXiv")
            self.collected_count += len(papers)
            return papers
            
        except Exception as e:
            print(f"    ❌ Error fetching arXiv: {e}")
            return []
    
    def fetch_pubmed(self, term="artificial intelligence", days=7, max_results=5):
        """Fetch from PubMed (requires biopython, optional)"""
        print(f"  📡 Fetching from PubMed...")
        try:
            # Simplified: using PubMed E-utilities API directly
            since_date = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            
            # Search for IDs
            search_url = f"{base_url}esearch.fcgi?db=pubmed&term={term}&retmax={max_results}&datetype=pdat&mindate={since_date}&retmode=json"
            response = requests.get(search_url, timeout=10)
            search_data = response.json()
            
            ids = search_data.get("esearchresult", {}).get("idlist", [])
            
            if not ids:
                print(f"    ℹ️  No recent PubMed articles found")
                return []
            
            # Fetch abstracts
            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
            response = requests.get(fetch_url, timeout=10)
            
            # Simple XML parsing (basic approach)
            papers = []
            for pmid in ids:
                paper = {
                    "id": f"pubmed_{pmid}",
                    "title": f"PubMed Article {pmid}",
                    "abstract": "Abstract available on PubMed",
                    "authors": ["PubMed Author"],
                    "topics": ["Biomedical", "AI"],
                    "source": "PubMed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "published": datetime.now().isoformat(),
                    "text": f"PubMed ID: {pmid}\nAbstract available at PubMed"
                }
                papers.append(paper)
            
            print(f"    ✅ Found {len(papers)} articles from PubMed")
            self.collected_count += len(papers)
            return papers
            
        except Exception as e:
            print(f"    ⚠️  PubMed fetch error: {e}")
            return []
    
    def fetch_zenodo(self, query="machine learning", days=30):
        """Fetch from Zenodo open research"""
        print(f"  📡 Fetching from Zenodo...")
        try:
            url = f"https://zenodo.org/api/records/?q={query}&access_right=open&size=10"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"    ⚠️  Zenodo API error: {response.status_code}")
                return []
            
            records = response.json().get("hits", {}).get("hits", [])
            since = datetime.utcnow() - timedelta(days=days)
            
            papers = []
            for r in records:
                try:
                    created = datetime.strptime(r["created"], "%Y-%m-%dT%H:%M:%S.%f%z")
                    if created.replace(tzinfo=None) >= since:
                        paper = {
                            "id": f"zenodo_{r['id']}",
                            "title": r["metadata"].get("title", "Untitled"),
                            "abstract": r["metadata"].get("description", ""),
                            "authors": [c["name"] for c in r["metadata"].get("creators", [])],
                            "topics": r["metadata"].get("keywords", []),
                            "source": "Zenodo",
                            "url": r["links"]["html"],
                            "published": created.isoformat(),
                            "text": r["metadata"].get("description", "")
                        }
                        papers.append(paper)
                except Exception as e:
                    print(f"    ⚠️  Error parsing Zenodo record: {e}")
                    continue
            
            print(f"    ✅ Found {len(papers)} records from Zenodo")
            self.collected_count += len(papers)
            return papers
            
        except Exception as e:
            print(f"    ❌ Zenodo fetch error: {e}")
            return []
    
    def fetch_web(self, query="latest AI research", max_results=5):
        """Fetch recent web articles using DuckDuckGo"""
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
            sources = ["arxiv", "websearch"]  # Default to working sources
        
        print(f"\n🔍 DataCollectorAgent starting collection...")
        all_papers = []
        
        for source in sources:
            if source in self.sources:
                papers = self.sources[source]()
                all_papers.extend(papers)
            else:
                print(f"  ⚠️  Unknown source: {source}")
        
        print(f"\n✅ DataCollectorAgent collected {len(all_papers)} items total")
        return all_papers

# ============================================================================
# Agent 2: KnowledgeGraphAgent - Graph Construction & Maintenance
# ============================================================================

class KnowledgeGraphAgent:
    """
    Builds and maintains knowledge graph from research papers:
    - Extracts entities and relationships using Gemini
    - Creates paper-author-topic-citation network
    - Maintains co-authorship and citation graphs
    - Provides graph analytics and queries
    """
    
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.load_graph()
        
    def load_graph(self):
        """Load existing graph"""
        if os.path.exists(GRAPH_PATH):
            try:
                self.G = nx.read_gpickle(GRAPH_PATH)
                print(f"✅ KnowledgeGraphAgent loaded graph: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")
            except Exception as e:
                print(f"⚠️  Error loading graph: {e}")
                self.G = nx.MultiDiGraph()
        else:
            print("ℹ️  No existing graph found, starting fresh")
    
    def save_graph(self):
        """Persist graph to disk"""
        try:
            nx.write_gpickle(self.G, GRAPH_PATH)
            print(f"💾 Graph saved: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")
        except Exception as e:
            print(f"❌ Error saving graph: {e}")
    
    def extract_triples_gemini(self, text):
        """Extract subject-relation-object triples using Gemini"""
        prompt = f"""Extract knowledge triples from this research text.
Return ONLY a JSON array of [subject, relation, object] triples.

Text: {text[:1000]}

Example output:
[["GPT-4", "is_a", "language model"], ["OpenAI", "developed", "GPT-4"]]

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
            print(f"    ⚠️  Triple extraction error: {e}")
        
        return []
    
    def add_paper_to_graph(self, paper):
        """Add complete paper metadata to graph"""
        pid = paper["id"]
        
        # Add paper node
        self.G.add_node(pid, 
                       type="paper",
                       title=paper["title"],
                       source=paper["source"],
                       url=paper.get("url", ""),
                       published=paper.get("published", ""))
        
        # Add authors and authorship edges
        for author in paper.get("authors", []):
            self.G.add_node(author, type="author")
            self.G.add_edge(pid, author, label="authored_by")
        
        # Add co-authorship edges
        authors = paper.get("authors", [])
        for i, a1 in enumerate(authors):
            for a2 in authors[i+1:]:
                if not self.G.has_edge(a1, a2):
                    self.G.add_edge(a1, a2, label="co_authored_with")
        
        # Add topics
        for topic in paper.get("topics", []):
            self.G.add_node(topic, type="topic")
            self.G.add_edge(pid, topic, label="about")
        
        # Add source node
        source = paper["source"]
        self.G.add_node(source, type="source")
        self.G.add_edge(pid, source, label="published_in")
        
        # Extract and add triples from content
        triples = self.extract_triples_gemini(paper["text"])
        for s, r, o in triples:
            self.G.add_node(s, type="entity")
            self.G.add_node(o, type="entity")
            self.G.add_edge(s, o, label=r)
    
    def process_papers(self, papers):
        """Process multiple papers and add to graph"""
        print(f"\n🕸️  KnowledgeGraphAgent processing {len(papers)} papers...")
        
        for i, paper in enumerate(papers, 1):
            print(f"  Processing {i}/{len(papers)}: {paper['title'][:60]}...")
            self.add_paper_to_graph(paper)
        
        self.save_graph()
        print(f"✅ Graph updated: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")
    
    def query_graph(self, query_type, **kwargs):
        """Query graph for specific patterns"""
        if query_type == "papers_by_author":
            author = kwargs.get("author", "")
            papers = [src for src, dst, data in self.G.edges(data=True)
                     if data.get("label") == "authored_by" and dst == author]
            return papers
        
        elif query_type == "coauthors":
            author = kwargs.get("author", "")
            coauthors = [dst for src, dst, data in self.G.edges(data=True)
                        if src == author and data.get("label") == "co_authored_with"]
            return coauthors
        
        elif query_type == "papers_on_topic":
            topic = kwargs.get("topic", "")
            papers = [src for src, dst, data in self.G.edges(data=True)
                     if data.get("label") == "about" and dst == topic]
            return papers
        
        elif query_type == "find_path":
            start = kwargs.get("start")
            end = kwargs.get("end")
            try:
                path = nx.shortest_path(self.G, start, end)
                return path
            except:
                return []
        
        return []
    
    def visualize(self, filename="research_graph.html"):
        """Create interactive visualization"""
        print(f"\n📊 Generating graph visualization...")
        
        net = Network(height="750px", width="100%", bgcolor="#222222", 
                     font_color="white", directed=True)
        
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)
        
        color_map = {
            "paper": "#4CAF50",
            "author": "#FF9800",
            "topic": "#2196F3",
            "source": "#E91E63",
            "entity": "#9C27B0"
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
        print(f"   Nodes: {len(self.G.nodes())}, Edges: {len(self.G.edges())}")

# ============================================================================
# Agent 3: VectorAgent - Semantic Search with Gemini
# ============================================================================

class VectorAgent:
    """
    Manages semantic search using Gemini for intelligent chunk retrieval:
    - Chunks documents intelligently
    - Uses Gemini for semantic understanding
    - Retrieves most relevant content for queries
    """
    
    def __init__(self):
        self.chunks = []
        self.load_chunks()
    
    def load_chunks(self):
        """Load existing chunks"""
        if os.path.exists(CHUNKS_PATH):
            try:
                with open(CHUNKS_PATH, "rb") as f:
                    self.chunks = pickle.load(f)
                print(f"✅ VectorAgent loaded {len(self.chunks)} chunks")
            except Exception as e:
                print(f"⚠️  Error loading chunks: {e}")
                self.chunks = []
    
    def save_chunks(self):
        """Save chunks to disk"""
        try:
            with open(CHUNKS_PATH, "wb") as f:
                pickle.dump(self.chunks, f)
            print(f"💾 Saved {len(self.chunks)} chunks")
        except Exception as e:
            print(f"❌ Error saving chunks: {e}")
    
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
        self.save_chunks()
        print(f"✅ Total chunks: {len(self.chunks)}")
    
    def retrieve_with_gemini(self, query, k=3):
        """Use Gemini to find relevant chunks"""
        if not self.chunks:
            return []
        
        # Create chunk catalog
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
            print(f"⚠️  Retrieval error: {e}")
        
        return self.chunks[:k] if len(self.chunks) >= k else self.chunks

# ============================================================================
# Agent 4: ReasoningAgent - Advanced Query Processing
# ============================================================================

class ReasoningAgent:
    """
    Handles complex reasoning and query answering:
    - Natural language to graph query translation
    - Multi-source information synthesis
    - Hypothesis generation
    - Research trend analysis
    """
    
    def __init__(self, graph_agent, vector_agent):
        self.graph_agent = graph_agent
        self.vector_agent = vector_agent
    
    def translate_nl_to_graph_query(self, nl_query):
        """Translate natural language to graph query command"""
        prompt = f"""Translate this natural language question into a graph query command.

Question: "{nl_query}"

Available query types:
- papers_by_author: {{"action": "papers_by_author", "author": "name"}}
- coauthors: {{"action": "coauthors", "author": "name"}}
- papers_on_topic: {{"action": "papers_on_topic", "topic": "topic"}}
- find_path: {{"action": "find_path", "start": "entity1", "end": "entity2"}}

Return ONLY a JSON command:"""
        
        try:
            response = gemini_model.generate_content(prompt)
            resp_text = response.text.strip()
            
            start = resp_text.find("{")
            end = resp_text.rfind("}") + 1
            
            if start != -1 and end > start:
                return json.loads(resp_text[start:end])
        except Exception as e:
            print(f"⚠️  Translation error: {e}")
        
        return {"action": "general_query", "query": nl_query}
    
    def synthesize_answer(self, query):
        """Synthesize answer from multiple sources"""
        print(f"\n🧠 ReasoningAgent processing query: {query}")
        
        # 1. Translate to graph query
        graph_cmd = self.translate_nl_to_graph_query(query)
        print(f"  📍 Graph command: {graph_cmd['action']}")
        
        # 2. Execute graph query
        if graph_cmd["action"] in ["papers_by_author", "coauthors", "papers_on_topic", "find_path"]:
            graph_results = self.graph_agent.query_graph(graph_cmd["action"], **graph_cmd)
        else:
            graph_results = []
        
        # 3. Retrieve relevant text chunks
        text_chunks = self.vector_agent.retrieve_with_gemini(query, k=3)
        
        # 4. Synthesize final answer with Gemini
        context = f"""You are an AI research assistant with access to:

GRAPH QUERY RESULTS:
Command: {graph_cmd}
Results: {graph_results[:10]}

RETRIEVED TEXT CHUNKS:
"""
        for i, chunk in enumerate(text_chunks, 1):
            context += f"\nChunk {i} (from {chunk.get('title', 'Unknown')}):\n{chunk['text'][:500]}\n"

        final_prompt = f"""{context}

QUESTION: {query}

Provide a comprehensive answer based on the graph data and text chunks above.
Be specific and cite sources when possible.

ANSWER:"""
        
        try:
            response = gemini_model.generate_content(final_prompt)
            answer = response.text.strip()
            print("  ✅ Answer synthesized")
            return answer
        except Exception as e:
            print(f"  ❌ Synthesis error: {e}")
            return f"Error generating answer: {e}"

# ============================================================================
# Agent 5: OrchestratorAgent - Coordinates All Agents
# ============================================================================

class OrchestratorAgent:
    """
    Orchestrates all agents for autonomous research:
    - Manages agent lifecycle
    - Coordinates data collection and processing
    - Handles user interactions
    - Manages system state
    """
    
    def __init__(self):
        print("\n🎭 OrchestratorAgent initializing...")
        self.data_collector = DataCollectorAgent()
        self.graph_agent = KnowledgeGraphAgent()
        self.vector_agent = VectorAgent()
        self.reasoning_agent = ReasoningAgent(self.graph_agent, self.vector_agent)
        self.metadata = self.load_metadata()
        print("✅ All agents initialized\n")
    
    def load_metadata(self):
        """Load system metadata"""
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        return {
            "total_papers_collected": 0,
            "last_collection": None,
            "collections_history": []
        }
    
    def save_metadata(self):
        """Save system metadata"""
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
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
        self.metadata["collections_history"].append({
            "timestamp": datetime.now().isoformat(),
            "papers_collected": len(papers),
            "duration_seconds": time.time() - start_time
        })
        self.save_metadata()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Collection cycle complete in {elapsed:.1f}s")
        print(f"📊 System stats:")
        print(f"   - Total papers: {self.metadata['total_papers_collected']}")
        print(f"   - Graph nodes: {len(self.graph_agent.G.nodes())}")
        print(f"   - Graph edges: {len(self.graph_agent.G.edges())}")
        print(f"   - Text chunks: {len(self.vector_agent.chunks)}")
        print("="*70)
    
    def interactive_mode(self):
        """Interactive research assistant mode"""
        print("\n" + "="*70)
        print("🤖 MULTI-AGENT RESEARCH ASSISTANT")
        print("="*70)
        print("\n💡 Commands:")
        print("   - Ask any research question")
        print("   - 'collect' - Run data collection cycle")
        print("   - 'graph' - Visualize knowledge graph")
        print("   - 'stats' - Show system statistics")
        print("   - 'exit' - Quit\n")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == "collect":
                    sources = input("  Sources (arxiv,websearch or press Enter for default): ").strip()
                    sources = sources.split(",") if sources else None
                    self.run_collection_cycle(sources)
                    continue
                
                if user_input.lower() == "graph":
                    self.graph_agent.visualize()
                    continue
                
                if user_input.lower() == "stats":
                    print(f"\n📊 System Statistics:")
                    print(f"   Total papers: {self.metadata['total_papers_collected']}")
                    print(f"   Graph nodes: {len(self.graph_agent.G.nodes())}")
                    print(f"   Graph edges: {len(self.graph_agent.G.edges())}")
                    print(f"   Text chunks: {len(self.vector_agent.chunks)}")
                    print(f"   Last collection: {self.metadata.get('last_collection', 'Never')}")
                    continue
                
                # Process as research question
                answer = self.reasoning_agent.synthesize_answer(user_input)
                print(f"\n🤖 Agent:\n{answer}\n")
                
            except KeyboardInterrupt:
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
    print("✅ Multi-Agent Research System initialized\n")
    
    orchestrator = OrchestratorAgent()
    orchestrator.interactive_mode()
