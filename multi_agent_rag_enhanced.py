#!/usr/bin/env python3
"""
Enhanced Multi-Agent Research System with Full ETL Pipeline
===========================================================
Features:
- 5 Data Sources: arXiv, Semantic Scholar, Zenodo, PubMed, Web
- Full ETL Pipeline: Extract → Transform → Load → Validate
- Data Quality Checks
- All previous features (sessions, memory, orchestration)
"""

import os
import re
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
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
ETL_CACHE_DIR = "etl_cache"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(ETL_CACHE_DIR, exist_ok=True)

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

print("🤖 Initializing Enhanced Multi-Agent Research System...")
print("   📡 Data Sources: arXiv, Semantic Scholar, Zenodo, PubMed, Web")
print("   🔄 Full ETL Pipeline: Extract → Transform → Load → Validate")
print("="*70)

# ============================================================================
# Session Management (Same as before)
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
# ETL Pipeline Components
# ============================================================================

class ETLPipeline:
    """
    Full Extract-Transform-Load pipeline for research data
    """

    def __init__(self):
        self.extraction_stats = {"success": 0, "failed": 0, "total": 0}
        self.transformation_stats = {"valid": 0, "invalid": 0, "total": 0}
        self.validation_rules = {
            "required_fields": ["id", "title", "abstract", "authors", "source"],
            "min_title_length": 10,
            "min_abstract_length": 50,
            "max_title_length": 500,
            "max_abstract_length": 10000
        }

    def extract(self, source_name: str, fetch_function, **kwargs) -> List[Dict]:
        """
        EXTRACT: Fetch raw data from source
        """
        print(f"\n[ETL-EXTRACT] Fetching from {source_name}...")
        start_time = time.time()

        try:
            raw_data = fetch_function(**kwargs)
            elapsed = time.time() - start_time

            self.extraction_stats["success"] += len(raw_data)
            self.extraction_stats["total"] += len(raw_data)

            print(f"  ✅ Extracted {len(raw_data)} items in {elapsed:.2f}s")
            return raw_data

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ Extraction failed: {e}")
            self.extraction_stats["failed"] += 1
            return []

    def transform(self, raw_data: List[Dict], source_name: str) -> List[Dict]:
        """
        TRANSFORM: Clean, normalize, and enrich data
        """
        print(f"\n[ETL-TRANSFORM] Processing {len(raw_data)} items from {source_name}...")
        transformed = []

        for item in raw_data:
            try:
                # Normalize structure
                cleaned = self._normalize_paper(item)

                # Enrich with metadata
                cleaned["etl_processed"] = datetime.now().isoformat()
                cleaned["etl_source"] = source_name
                cleaned["etl_pipeline_version"] = "1.0"

                # Clean text
                cleaned["title"] = self._clean_text(cleaned["title"])
                cleaned["abstract"] = self._clean_text(cleaned["abstract"])

                # Generate searchable text
                cleaned["text"] = f"Title: {cleaned['title']}\n\nAbstract: {cleaned['abstract']}"

                transformed.append(cleaned)
                self.transformation_stats["valid"] += 1

            except Exception as e:
                print(f"  ⚠️  Transform error: {e}")
                self.transformation_stats["invalid"] += 1
                continue

        self.transformation_stats["total"] += len(raw_data)
        print(f"  ✅ Transformed {len(transformed)}/{len(raw_data)} items")

        return transformed

    def validate(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        VALIDATE: Check data quality and filter invalid entries
        """
        print(f"\n[ETL-VALIDATE] Validating {len(data)} items...")
        valid = []
        invalid = []

        for item in data:
            issues = []

            # Check required fields
            for field in self.validation_rules["required_fields"]:
                if field not in item or not item[field]:
                    issues.append(f"Missing {field}")

            # Check title length
            if len(item.get("title", "")) < self.validation_rules["min_title_length"]:
                issues.append("Title too short")
            elif len(item.get("title", "")) > self.validation_rules["max_title_length"]:
                issues.append("Title too long")

            # Check abstract length
            if len(item.get("abstract", "")) < self.validation_rules["min_abstract_length"]:
                issues.append("Abstract too short")
            elif len(item.get("abstract", "")) > self.validation_rules["max_abstract_length"]:
                issues.append("Abstract too long")

            if issues:
                item["validation_issues"] = issues
                invalid.append(item)
            else:
                item["validated"] = True
                valid.append(item)

        print(f"  ✅ Valid: {len(valid)}")
        print(f"  ❌ Invalid: {len(invalid)}")

        return valid, invalid

    def load(self, data: List[Dict], target: str = "knowledge_base") -> bool:
        """
        LOAD: Store processed data
        """
        print(f"\n[ETL-LOAD] Loading {len(data)} items to {target}...")

        try:
            cache_file = os.path.join(ETL_CACHE_DIR, f"{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            with open(cache_file, "w") as f:
                json.dump({
                    "loaded_at": datetime.now().isoformat(),
                    "count": len(data),
                    "data": data
                }, f, indent=2)

            print(f"  ✅ Loaded to: {cache_file}")
            return True

        except Exception as e:
            print(f"  ❌ Load failed: {e}")
            return False

    def _normalize_paper(self, paper: Dict) -> Dict:
        """Normalize paper structure"""
        return {
            "id": paper.get("id", f"unknown_{hash(str(paper))}"),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "authors": paper.get("authors", []),
            "topics": paper.get("topics", []),
            "source": paper.get("source", "Unknown"),
            "url": paper.get("url", ""),
            "published": paper.get("published", datetime.now().isoformat())
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (but keep punctuation)
        text = re.sub(r'[^\w\s\.,!?;:()\-]', '', text)

        return text.strip()

    def get_stats(self) -> Dict:
        """Get ETL pipeline statistics"""
        return {
            "extraction": self.extraction_stats,
            "transformation": self.transformation_stats,
            "success_rate": (
                self.extraction_stats["success"] / max(self.extraction_stats["total"], 1)
            ) * 100
        }

# ============================================================================
# Enhanced DataCollectorAgent with 5 Sources + ETL
# ============================================================================

class DataCollectorAgent:
    """
    Collects research papers from 5 sources with full ETL pipeline
    Sources: arXiv, Semantic Scholar, Zenodo, PubMed, Web
    """

    def __init__(self):
        self.etl = ETLPipeline()
        self.sources = {
            "arxiv": self.fetch_arxiv,
            "semantic_scholar": self.fetch_semantic_scholar,
            "zenodo": self.fetch_zenodo,
            "pubmed": self.fetch_pubmed,
            "websearch": self.fetch_web
        }
        self.collected_count = 0

    # ========== ARXIV ==========
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
                            "published": published.isoformat()
                        }
                        papers.append(paper)
                except Exception as e:
                    continue

            return papers

        except Exception as e:
            print(f"    ❌ Error: {e}")
            return []

    # ========== SEMANTIC SCHOLAR ==========
    def fetch_semantic_scholar(self, query="artificial intelligence", max_results=10):
        """Fetch from Semantic Scholar API"""
        print(f"  📡 Fetching from Semantic Scholar...")
        try:
            # Semantic Scholar API v1 (no key needed for basic queries)
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "paperId,title,abstract,authors,year,publicationDate,url,citationCount"
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                papers = []

                for item in data.get("data", []):
                    if not item.get("abstract"):
                        continue

                    paper = {
                        "id": f"s2_{item['paperId']}",
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "authors": [a.get("name", "") for a in item.get("authors", [])],
                        "topics": ["AI", "ML"],  # S2 doesn't provide detailed topics in basic API
                        "source": "Semantic Scholar",
                        "url": item.get("url", f"https://www.semanticscholar.org/paper/{item['paperId']}"),
                        "published": item.get("publicationDate", datetime.now().isoformat()),
                        "citation_count": item.get("citationCount", 0)
                    }
                    papers.append(paper)

                return papers
            else:
                print(f"    ⚠️  Status code: {response.status_code}")
                return []

        except Exception as e:
            print(f"    ❌ Error: {e}")
            return []

    # ========== ZENODO ==========
    def fetch_zenodo(self, query="machine learning", max_results=10):
        """Fetch from Zenodo repository"""
        print(f"  📡 Fetching from Zenodo...")
        try:
            url = "https://zenodo.org/api/records"
            params = {
                "q": query,
                "size": max_results,
                "sort": "mostrecent",
                "type": "publication"
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                papers = []

                for item in data.get("hits", {}).get("hits", []):
                    metadata = item.get("metadata", {})

                    # Get description/abstract
                    abstract = metadata.get("description", "")
                    if len(abstract) < 50:
                        continue

                    paper = {
                        "id": f"zenodo_{item['id']}",
                        "title": metadata.get("title", ""),
                        "abstract": abstract,
                        "authors": [c.get("name", "") for c in metadata.get("creators", [])],
                        "topics": metadata.get("keywords", []),
                        "source": "Zenodo",
                        "url": item.get("links", {}).get("html", ""),
                        "published": metadata.get("publication_date", datetime.now().isoformat()),
                        "doi": metadata.get("doi", "")
                    }
                    papers.append(paper)

                return papers
            else:
                print(f"    ⚠️  Status code: {response.status_code}")
                return []

        except Exception as e:
            print(f"    ❌ Error: {e}")
            return []

    # ========== PUBMED ==========
    def fetch_pubmed(self, query="artificial intelligence", max_results=10):
        """Fetch from PubMed"""
        print(f"  📡 Fetching from PubMed...")
        try:
            # PubMed E-utilities (no API key needed for low volume)
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "pub_date"
            }

            search_response = requests.get(search_url, params=search_params, timeout=10)

            if search_response.status_code != 200:
                print(f"    ⚠️  Search failed: {search_response.status_code}")
                return []

            search_data = search_response.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            # Fetch details
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml"
            }

            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)

            if fetch_response.status_code != 200:
                print(f"    ⚠️  Fetch failed: {fetch_response.status_code}")
                return []

            # Parse XML (basic parsing)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(fetch_response.content)

            papers = []
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract data
                    title_elem = article.find(".//ArticleTitle")
                    abstract_elem = article.find(".//AbstractText")
                    pmid_elem = article.find(".//PMID")

                    if title_elem is None or abstract_elem is None:
                        continue

                    # Get authors
                    authors = []
                    for author in article.findall(".//Author"):
                        lastname = author.find("LastName")
                        forename = author.find("ForeName")
                        if lastname is not None and forename is not None:
                            authors.append(f"{forename.text} {lastname.text}")

                    paper = {
                        "id": f"pubmed_{pmid_elem.text if pmid_elem is not None else 'unknown'}",
                        "title": title_elem.text or "",
                        "abstract": abstract_elem.text or "",
                        "authors": authors,
                        "topics": ["PubMed", "Medical"],
                        "source": "PubMed",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/" if pmid_elem is not None else "",
                        "published": datetime.now().isoformat()
                    }
                    papers.append(paper)

                except Exception as e:
                    continue

            return papers

        except Exception as e:
            print(f"    ❌ Error: {e}")
            return []

    # ========== WEB SEARCH ==========
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
                    "published": datetime.now().isoformat()
                }
                papers.append(paper)

            return papers

        except Exception as e:
            print(f"    ⚠️  Web search error: {e}")
            return []

    # ========== COLLECT WITH ETL ==========
    def collect_all(self, sources=None, use_etl=True):
        """
        Collect from all enabled sources with full ETL pipeline
        """
        if sources is None:
            sources = ["arxiv", "semantic_scholar", "zenodo", "pubmed", "websearch"]

        print(f"\n🔍 DataCollectorAgent starting collection with ETL pipeline...")
        print(f"   Sources: {', '.join(sources)}")
        all_papers = []

        for source in sources:
            if source in self.sources:
                if use_etl:
                    # Full ETL pipeline
                    raw_data = self.etl.extract(source, self.sources[source])
                    transformed_data = self.etl.transform(raw_data, source)
                    valid_data, invalid_data = self.etl.validate(transformed_data)
                    self.etl.load(valid_data, target=f"{source}_papers")
                    all_papers.extend(valid_data)
                else:
                    # Direct fetch (legacy)
                    papers = self.sources[source]()
                    all_papers.extend(papers)

        # Show ETL stats
        if use_etl:
            stats = self.etl.get_stats()
            print(f"\n📊 ETL Pipeline Statistics:")
            print(f"   Extracted: {stats['extraction']['success']}")
            print(f"   Valid: {stats['transformation']['valid']}")
            print(f"   Invalid: {stats['transformation']['invalid']}")
            print(f"   Success Rate: {stats['success_rate']:.1f}%")

        print(f"\n✅ DataCollectorAgent collected {len(all_papers)} items total")
        self.collected_count = len(all_papers)
        return all_papers

# ============================================================================
# Other agents remain the same (KnowledgeGraphAgent, VectorAgent, etc.)
# ============================================================================

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
        except:
            pass

        return []

    def add_paper_to_graph(self, paper):
        """Add paper to graph"""
        pid = paper["id"]

        self.G.add_node(pid, type="paper", title=paper["title"],
                       source=paper["source"], url=paper.get("url", ""))

        for author in paper.get("authors", []):
            self.G.add_node(author, type="author")
            self.G.add_edge(pid, author, label="authored_by")

        for topic in paper.get("topics", []):
            self.G.add_node(topic, type="topic")
            self.G.add_edge(pid, topic, label="about")

        triples = self.extract_triples_gemini(paper.get("text", paper.get("abstract", "")))
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
        text = paper.get("text", f"Title: {paper['title']}\n\nAbstract: {paper.get('abstract', '')}")
        new_chunks = self.chunk_text(text)
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
        except:
            pass

        return self.chunks[:k] if len(self.chunks) >= k else self.chunks

class ReasoningAgent:
    """Handles complex reasoning with conversation memory"""

    def __init__(self, graph_agent, vector_agent):
        self.graph_agent = graph_agent
        self.vector_agent = vector_agent
        self.conversation_history = []

    def synthesize_answer(self, query):
        """Synthesize answer with conversation memory"""
        print(f"\n🧠 ReasoningAgent processing query: {query}")

        # Build conversation context
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "PREVIOUS CONVERSATION:\n"
            for i, turn in enumerate(self.conversation_history[-3:], 1):
                conversation_context += f"Turn {i}:\n"
                conversation_context += f"  User: {turn['query']}\n"
                conversation_context += f"  Agent: {turn['answer'][:200]}\n\n"

        # Retrieve relevant text chunks
        text_chunks = self.vector_agent.retrieve_with_gemini(query, k=3)

        # Build context
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
# OrchestratorAgent (Same as before)
# ============================================================================

class OrchestratorAgent:
    """Orchestrates all agents with session management"""

    def __init__(self, session_name="default"):
        print(f"\n🎭 OrchestratorAgent initializing session '{session_name}'...")
        self.session_name = session_name
        self.data_collector = DataCollectorAgent()
        self.graph_agent = KnowledgeGraphAgent()
        self.vector_agent = VectorAgent()
        self.reasoning_agent = ReasoningAgent(self.graph_agent, self.vector_agent)
        self.metadata = {"total_papers_collected": 0, "last_collection": None}

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
        self.save_session()

        self.session_name = new_session_name

        self.graph_agent = KnowledgeGraphAgent()
        self.vector_agent = VectorAgent()
        self.reasoning_agent = ReasoningAgent(self.graph_agent, self.vector_agent)
        self.metadata = {"total_papers_collected": 0, "last_collection": None}

        self.load_session()
        print(f"✅ Switched to session '{new_session_name}'")

    def run_collection_cycle(self, sources=None):
        """Run full data collection with ETL"""
        print("\n" + "="*70)
        print("🔄 STARTING COLLECTION CYCLE WITH ETL PIPELINE")
        print("="*70)

        start_time = time.time()

        # Collect with ETL
        papers = self.data_collector.collect_all(sources, use_etl=True)

        if not papers:
            print("\n⚠️  No new papers collected")
            return

        # Process with other agents
        self.graph_agent.process_papers(papers)
        self.vector_agent.process_papers(papers)

        # Update metadata
        self.metadata["total_papers_collected"] += len(papers)
        self.metadata["last_collection"] = datetime.now().isoformat()

        # Auto-save
        self.save_session()

        elapsed = time.time() - start_time
        print(f"\n✅ Collection cycle complete in {elapsed:.1f}s")
        print("="*70)

    def interactive_mode(self):
        """Interactive mode with all features"""
        print("\n" + "="*70)
        print("🤖 ENHANCED MULTI-AGENT RESEARCH ASSISTANT")
        print(f"📌 Current Session: '{self.session_name}'")
        print("📡 Data Sources: arXiv, Semantic Scholar, Zenodo, PubMed, Web")
        print("="*70)
        print("\n💡 Commands:")
        print("   - Ask any research question")
        print("   - 'collect' - run ETL collection from all sources")
        print("   - 'collect arxiv' - collect from specific source")
        print("   - 'sessions' - list all sessions")
        print("   - 'switch <name>' - switch to different session")
        print("   - 'graph' - visualize knowledge graph")
        print("   - 'memory' - show conversation history")
        print("   - 'stats' - show statistics")
        print("   - 'etl-stats' - show ETL pipeline stats")
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

                if user_input.lower().startswith("collect"):
                    parts = user_input.split()
                    if len(parts) > 1:
                        sources = [parts[1]]
                    else:
                        sources = None
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

                if user_input.lower() == "etl-stats":
                    stats = self.data_collector.etl.get_stats()
                    print(f"\n📊 ETL Pipeline Statistics:")
                    print(f"   Extracted: {stats['extraction']['success']} items")
                    print(f"   Failed: {stats['extraction']['failed']} items")
                    print(f"   Valid: {stats['transformation']['valid']} items")
                    print(f"   Invalid: {stats['transformation']['invalid']} items")
                    print(f"   Success Rate: {stats['success_rate']:.1f}%")
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

    if len(sys.argv) > 1:
        session_name = sys.argv[1]
    else:
        sessions = list_sessions()
        if sessions:
            print("\n💡 Available sessions:")
            for s in sessions[:5]:
                print(f"   - {s['name']} ({s['conversations']} conversations)")
            print("\nTip: python3 multi_agent_rag_enhanced.py <session_name>")
        session_name = "default"

    print(f"\n✅ Enhanced Multi-Agent System with 5 Data Sources + ETL initialized")

    orchestrator = OrchestratorAgent(session_name)
    orchestrator.interactive_mode()
