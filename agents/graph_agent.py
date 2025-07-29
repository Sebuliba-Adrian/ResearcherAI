"""
KnowledgeGraphAgent - Graph Database Management
==============================================

Supports both Neo4j (production) and NetworkX (development)
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai

logger = logging.getLogger(__name__)


class KnowledgeGraphAgent:
    """Knowledge graph construction and querying with dual backend support"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with config

        config = {
            "type": "neo4j" or "networkx",
            "uri": "bolt://neo4j:7687",  # for neo4j
            "user": "neo4j",
            "password": "password",
            "database": "research"
        }
        """
        self.config = config or {"type": "networkx"}
        self.db_type = self.config.get("type", "networkx")

        if self.db_type == "neo4j":
            self._init_neo4j()
        else:
            self._init_networkx()

        # Initialize Gemini for triple extraction
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            logger.info("Gemini configured for triple extraction")
        else:
            self.model = None
            logger.warning("GOOGLE_API_KEY not found - triple extraction disabled")

        logger.info(f"KnowledgeGraphAgent initialized with {self.db_type} backend")

    def _init_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            from neo4j import GraphDatabase

            uri = self.config.get("uri", "bolt://localhost:7687")
            user = self.config.get("user", "neo4j")
            password = self.config.get("password", "password")

            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()

            # Create constraints for uniqueness
            with self.driver.session(database=self.config.get("database", "neo4j")) as session:
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")

            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            logger.info("Falling back to NetworkX")
            self.db_type = "networkx"
            self._init_networkx()

    def _init_networkx(self):
        """Initialize NetworkX graph"""
        import networkx as nx
        self.G = nx.MultiDiGraph()
        logger.info("NetworkX graph initialized")

    def process_papers(self, papers: List[Dict]) -> Dict:
        """Process papers and build knowledge graph"""
        logger.info(f"Processing {len(papers)} papers for knowledge graph...")

        stats = {"nodes_added": 0, "edges_added": 0, "papers_processed": 0}

        for i, paper in enumerate(papers, 1):
            logger.info(f"  Processing {i}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")

            try:
                # Extract triples using Gemini
                triples = self._extract_triples(paper)

                # Add to graph
                if self.db_type == "neo4j":
                    result = self._add_to_neo4j(paper, triples)
                else:
                    result = self._add_to_networkx(paper, triples)

                stats["nodes_added"] += result["nodes"]
                stats["edges_added"] += result["edges"]
                stats["papers_processed"] += 1

            except Exception as e:
                logger.error(f"Failed to process paper: {e}")

        logger.info(f"✅ Graph updated: {stats['nodes_added']} nodes, {stats['edges_added']} edges")
        return stats

    def _extract_triples(self, paper: Dict) -> List[Tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples using Gemini"""
        if not self.model:
            # Fallback: extract basic metadata triples
            triples = []
            title = paper.get("title", "")
            for author in paper.get("authors", [])[:3]:
                triples.append((author, "authored", title))
            for topic in paper.get("topics", [])[:3]:
                triples.append((title, "is_about", topic))
            return triples

        # Use Gemini for intelligent triple extraction
        text = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:1000]}"

        prompt = f"""Extract knowledge graph triples from this research paper.
Return ONLY (subject, predicate, object) triples, one per line.
Focus on: entities, authors, concepts, methods, findings.
Max 20 triples.

Paper:
{text}

Format: subject | predicate | object"""

        try:
            response = self.model.generate_content(prompt)
            triples = []

            for line in response.text.strip().split("\n"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    triples.append(tuple(parts))

            return triples[:20]
        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return []

    def _add_to_neo4j(self, paper: Dict, triples: List[Tuple[str, str, str]]) -> Dict:
        """Add paper and triples to Neo4j"""
        nodes_added = 0
        edges_added = 0

        with self.driver.session(database=self.config.get("database", "neo4j")) as session:
            # Add paper node
            session.run("""
                MERGE (p:Paper {id: $id})
                SET p.title = $title,
                    p.abstract = $abstract,
                    p.source = $source,
                    p.url = $url,
                    p.published = $published
            """, id=paper.get("id"), title=paper.get("title"), abstract=paper.get("abstract", ""),
                source=paper.get("source"), url=paper.get("url", ""), published=paper.get("published", ""))
            nodes_added += 1

            # Add authors (filter nulls and empty strings)
            for author in paper.get("authors", []):
                # Skip null, empty, or non-string authors
                if not author or (isinstance(author, str) and not author.strip()):
                    continue

                # Convert to string if needed
                author_name = str(author).strip() if not isinstance(author, str) else author.strip()

                session.run("""
                    MERGE (a:Author {name: $name})
                    MERGE (p:Paper {id: $paper_id})
                    MERGE (a)-[:AUTHORED]->(p)
                """, name=author_name, paper_id=paper.get("id"))
                nodes_added += 1
                edges_added += 1

            # Add topics (filter nulls and non-primitives)
            for topic in paper.get("topics", []):
                # Skip null, empty, or complex objects
                if not topic:
                    continue

                # Skip dictionaries and other non-primitive types
                if isinstance(topic, dict) or isinstance(topic, list):
                    logger.warning(f"Skipping non-primitive topic: {topic}")
                    continue

                # Convert to string
                topic_name = str(topic).strip() if not isinstance(topic, str) else topic.strip()

                if not topic_name:
                    continue

                session.run("""
                    MERGE (t:Topic {name: $name})
                    MERGE (p:Paper {id: $paper_id})
                    MERGE (p)-[:ABOUT]->(t)
                """, name=topic_name, paper_id=paper.get("id"))
                nodes_added += 1
                edges_added += 1

            # Add extracted triples
            for subj, pred, obj in triples:
                # Clean relationship name: ASCII only, no special chars except underscore
                clean_pred = pred.upper().replace(' ', '_').replace('-', '_')
                # Remove non-ASCII characters and other special characters
                clean_pred = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_pred)
                # Remove leading/trailing underscores and collapse multiple underscores
                clean_pred = '_'.join(filter(None, clean_pred.split('_')))
                # Default to RELATED_TO if empty after cleaning
                if not clean_pred or not clean_pred[0].isalpha():
                    clean_pred = "RELATED_TO"

                session.run(f"""
                    MERGE (s:Entity {{name: $subj}})
                    MERGE (o:Entity {{name: $obj}})
                    MERGE (s)-[r:{clean_pred}]->(o)
                """, subj=subj, obj=obj)
                edges_added += 1

        return {"nodes": nodes_added, "edges": edges_added}

    def _add_to_networkx(self, paper: Dict, triples: List[Tuple[str, str, str]]) -> Dict:
        """Add paper and triples to NetworkX graph"""
        nodes_added = 0
        edges_added = 0

        # Add paper node
        paper_id = paper.get("id", "")
        if paper_id not in self.G:
            self.G.add_node(paper_id,
                          type="paper",
                          title=paper.get("title", ""),
                          source=paper.get("source", ""))
            nodes_added += 1

        # Add authors
        for author in paper.get("authors", []):
            if author not in self.G:
                self.G.add_node(author, type="author")
                nodes_added += 1
            self.G.add_edge(author, paper_id, label="authored")
            edges_added += 1

        # Add topics
        for topic in paper.get("topics", []):
            if topic not in self.G:
                self.G.add_node(topic, type="topic")
                nodes_added += 1
            self.G.add_edge(paper_id, topic, label="is_about")
            edges_added += 1

        # Add extracted triples
        for subj, pred, obj in triples:
            if subj not in self.G:
                self.G.add_node(subj, type="entity")
                nodes_added += 1
            if obj not in self.G:
                self.G.add_node(obj, type="entity")
                nodes_added += 1
            self.G.add_edge(subj, obj, label=pred)
            edges_added += 1

        return {"nodes": nodes_added, "edges": edges_added}

    def query_graph(self, entity: str, max_hops: int = 2) -> List[Dict]:
        """Query graph for entity and related information"""
        if self.db_type == "neo4j":
            return self._query_neo4j(entity, max_hops)
        else:
            return self._query_networkx(entity, max_hops)

    def _query_neo4j(self, entity: str, max_hops: int = 2) -> List[Dict]:
        """Query Neo4j graph"""
        with self.driver.session(database=self.config.get("database", "neo4j")) as session:
            result = session.run(f"""
                MATCH path = (start)-[*1..{max_hops}]-(end)
                WHERE start.name CONTAINS $entity OR start.title CONTAINS $entity
                RETURN path
                LIMIT 50
            """, entity=entity)

            paths = []
            for record in result:
                path = record["path"]
                paths.append({
                    "nodes": [node["name"] if "name" in node else node.get("title", "") for node in path.nodes],
                    "relationships": [rel.type for rel in path.relationships]
                })

            return paths

    def _query_networkx(self, entity: str, max_hops: int = 2) -> List[Dict]:
        """Query NetworkX graph"""
        import networkx as nx

        # Find matching nodes
        matching_nodes = [n for n in self.G.nodes()
                         if entity.lower() in str(n).lower()]

        if not matching_nodes:
            return []

        # Get subgraph around matches
        all_nodes = set()
        for node in matching_nodes:
            # Get neighbors within max_hops
            for target in self.G.nodes():
                try:
                    if nx.has_path(self.G, node, target):
                        path = nx.shortest_path(self.G, node, target)
                        if len(path) <= max_hops + 1:
                            all_nodes.update(path)
                except:
                    pass

        subgraph = self.G.subgraph(all_nodes)

        # Convert to paths
        paths = []
        for edge in subgraph.edges(data=True):
            paths.append({
                "nodes": [edge[0], edge[1]],
                "relationships": [edge[2].get("label", "")]
            })

        return paths

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        if self.db_type == "neo4j":
            with self.driver.session(database=self.config.get("database", "neo4j")) as session:
                result = session.run("MATCH (n) RETURN count(n) as nodes")
                node_count = result.single()["nodes"]
                result = session.run("MATCH ()-[r]->() RETURN count(r) as edges")
                edge_count = result.single()["edges"]
                return {"nodes": node_count, "edges": edge_count, "backend": "Neo4j"}
        else:
            return {
                "nodes": len(self.G.nodes()),
                "edges": len(self.G.edges()),
                "backend": "NetworkX"
            }

    def export_graph_data(self) -> Dict:
        """Export graph data in format suitable for web visualization"""
        if self.db_type == "neo4j":
            return self._export_neo4j_data()
        else:
            return self._export_networkx_data()

    def _export_neo4j_data(self) -> Dict:
        """Export Neo4j graph data"""
        with self.driver.session(database=self.config.get("database", "neo4j")) as session:
            # Get all nodes
            nodes_result = session.run("MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as props LIMIT 500")
            nodes = []
            for record in nodes_result:
                node = {
                    "id": str(record["id"]),
                    "label": record["props"].get("name") or record["props"].get("title") or f"Node {record['id']}",
                    "type": record["labels"][0] if record["labels"] else "Entity",
                    "properties": record["props"]
                }
                nodes.append(node)

            # Get all edges
            edges_result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN id(a) as source, id(b) as target, type(r) as type, properties(r) as props
                LIMIT 1000
            """)
            edges = []
            for record in edges_result:
                edge = {
                    "source": str(record["source"]),
                    "target": str(record["target"]),
                    "label": record["type"],
                    "properties": record["props"]
                }
                edges.append(edge)

            return {"nodes": nodes, "edges": edges}

    def _export_networkx_data(self) -> Dict:
        """Export NetworkX graph data"""
        nodes = []
        edges = []

        # Export nodes
        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            nodes.append({
                "id": str(node_id),
                "label": str(node_id)[:50],  # Truncate long labels
                "type": node_data.get("type", "Entity"),
                "properties": node_data
            })

        # Export edges
        for source, target, data in self.G.edges(data=True):
            edges.append({
                "source": str(source),
                "target": str(target),
                "label": data.get("label", ""),
                "properties": data
            })

        return {"nodes": nodes, "edges": edges}

    def visualize(self, filename: str = "knowledge_graph.html"):
        """Generate interactive visualization (NetworkX only)"""
        if self.db_type == "neo4j":
            logger.warning("Visualization only supported for NetworkX backend")
            return

        from pyvis.network import Network

        if len(self.G.nodes()) == 0:
            logger.warning("Graph is empty, nothing to visualize")
            return

        net = Network(height="750px", width="100%", bgcolor="#222222",
                     font_color="white", directed=True)
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)

        for edge in self.G.edges(data=True):
            source, target, data = edge
            label = data.get("label", "")
            net.add_node(source, label=str(source)[:30], title=str(source), size=25)
            net.add_node(target, label=str(target)[:30], title=str(target), size=25)
            net.add_edge(source, target, label=label, title=label, arrows="to")

        net.save_graph(filename)
        logger.info(f"Visualization saved to {filename}")

    def close(self):
        """Close database connection"""
        if self.db_type == "neo4j" and hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j connection closed")
