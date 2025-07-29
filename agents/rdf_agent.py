"""
RDFAgent - Semantic Web and RDF Export/Import
==============================================

Handles conversion between graph formats and RDF for interoperability
Supports: Turtle, RDF/XML, JSON-LD, N-Triples
"""

import os
import logging
from typing import Dict, List, Optional, Union
from io import BytesIO
import urllib.parse
import hashlib
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, FOAF, DCTERMS, XSD, OWL
from datetime import datetime

logger = logging.getLogger(__name__)


class RDFAgent:
    """RDF export/import and semantic web operations"""

    # Define namespaces
    RESEARCHAI = Namespace("http://researcherai.org/ontology#")
    BIBO = Namespace("http://purl.org/ontology/bibo/")
    SCHEMA = Namespace("http://schema.org/")

    # Supported RDF formats
    FORMATS = {
        'turtle': {'ext': '.ttl', 'media_type': 'text/turtle'},
        'xml': {'ext': '.rdf', 'media_type': 'application/rdf+xml'},
        'json-ld': {'ext': '.jsonld', 'media_type': 'application/ld+json'},
        'n3': {'ext': '.n3', 'media_type': 'text/n3'},
        'nt': {'ext': '.nt', 'media_type': 'application/n-triples'},
    }

    def __init__(self):
        """Initialize RDF agent"""
        self.graph = Graph()

        # Bind namespaces
        self.graph.bind('rai', self.RESEARCHAI)
        self.graph.bind('bibo', self.BIBO)
        self.graph.bind('foaf', FOAF)
        self.graph.bind('dcterms', DCTERMS)
        self.graph.bind('schema', self.SCHEMA)
        self.graph.bind('owl', OWL)

        logger.info("RDFAgent initialized with ontology bindings")

    def _create_uri(self, type_name: str, identifier: str) -> URIRef:
        """Create consistent URI for entities"""
        # Create hash-based URI to ensure validity
        # This handles all special characters, long strings, and edge cases
        hash_id = hashlib.md5(identifier.encode()).hexdigest()

        # Use hash for URI but keep original as rdfs:label later
        return URIRef(f"http://researcherai.org/{type_name}/{hash_id}")

    def export_to_rdf(self, graph_data: Dict) -> Graph:
        """
        Convert NetworkX/Neo4j graph to RDF

        Args:
            graph_data: Dict with 'nodes' and 'edges' keys

        Returns:
            RDFLib Graph object
        """
        logger.info(f"Exporting {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('edges', []))} edges to RDF...")

        # Clear existing graph
        self.graph = Graph()
        self._bind_namespaces()

        # Add ontology header
        self._add_ontology_metadata()

        # Process nodes
        for node in graph_data.get('nodes', []):
            self._add_node_to_rdf(node)

        # Process edges
        for edge in graph_data.get('edges', []):
            self._add_edge_to_rdf(edge)

        logger.info(f"✅ RDF graph created with {len(self.graph)} triples")
        return self.graph

    def _bind_namespaces(self):
        """Bind all namespaces to graph"""
        self.graph.bind('rai', self.RESEARCHAI)
        self.graph.bind('bibo', self.BIBO)
        self.graph.bind('foaf', FOAF)
        self.graph.bind('dcterms', DCTERMS)
        self.graph.bind('schema', self.SCHEMA)
        self.graph.bind('owl', OWL)

    def _add_ontology_metadata(self):
        """Add ontology metadata and documentation"""
        ontology_uri = URIRef("http://researcherai.org/ontology")

        self.graph.add((ontology_uri, RDF.type, OWL.Ontology))
        self.graph.add((ontology_uri, DCTERMS.title, Literal("ResearcherAI Ontology")))
        self.graph.add((ontology_uri, DCTERMS.description,
                       Literal("Ontology for representing research papers, authors, and relationships")))
        self.graph.add((ontology_uri, DCTERMS.created,
                       Literal(datetime.now().isoformat(), datatype=XSD.dateTime)))

    def _add_node_to_rdf(self, node: Dict):
        """Add a node to RDF graph with appropriate typing"""
        node_id = node.get('id', '')
        node_type = node.get('type', 'entity').lower()
        label = node.get('label', '')
        properties = node.get('properties', {})

        if node_type == 'paper':
            self._add_paper_node(node_id, label, properties)
        elif node_type == 'author':
            self._add_author_node(node_id, label, properties)
        else:
            self._add_entity_node(node_id, label, properties, node_type)

    def _add_paper_node(self, node_id: str, label: str, properties: Dict):
        """Add paper node with bibliographic metadata"""
        paper_uri = self._create_uri('paper', node_id)

        # Type declarations
        self.graph.add((paper_uri, RDF.type, self.BIBO.AcademicArticle))
        self.graph.add((paper_uri, RDF.type, self.RESEARCHAI.Paper))

        # Title
        title = properties.get('title', label)
        if title:
            self.graph.add((paper_uri, DCTERMS.title, Literal(title)))
            self.graph.add((paper_uri, RDFS.label, Literal(title)))

        # Abstract
        if 'abstract' in properties:
            self.graph.add((paper_uri, DCTERMS.abstract, Literal(properties['abstract'])))

        # Source/Publisher
        if 'source' in properties:
            self.graph.add((paper_uri, DCTERMS.source, Literal(properties['source'])))

        # Publication year
        if 'year' in properties and properties['year'] != 'N/A':
            try:
                year = int(properties['year'])
                self.graph.add((paper_uri, DCTERMS.issued, Literal(year, datatype=XSD.gYear)))
            except (ValueError, TypeError):
                pass

        # URL/DOI
        if 'url' in properties:
            self.graph.add((paper_uri, self.BIBO.uri, URIRef(properties['url'])))

        if 'doi' in properties:
            self.graph.add((paper_uri, self.BIBO.doi, Literal(properties['doi'])))

        # Add all other properties
        for key, value in properties.items():
            if key not in ['title', 'abstract', 'source', 'year', 'url', 'doi', 'type']:
                predicate = self.RESEARCHAI[f"has{key.capitalize()}"]
                self.graph.add((paper_uri, predicate, Literal(str(value))))

    def _add_author_node(self, node_id: str, label: str, properties: Dict):
        """Add author node with FOAF properties"""
        author_uri = self._create_uri('author', node_id)

        # Type declarations
        self.graph.add((author_uri, RDF.type, FOAF.Person))
        self.graph.add((author_uri, RDF.type, self.RESEARCHAI.Author))

        # Name
        name = properties.get('name', label)
        if name:
            self.graph.add((author_uri, FOAF.name, Literal(name)))
            self.graph.add((author_uri, RDFS.label, Literal(name)))

        # Other properties
        if 'email' in properties:
            self.graph.add((author_uri, FOAF.mbox, Literal(properties['email'])))

        if 'affiliation' in properties:
            self.graph.add((author_uri, self.SCHEMA.affiliation, Literal(properties['affiliation'])))

        # Add remaining properties
        for key, value in properties.items():
            if key not in ['name', 'email', 'affiliation', 'type']:
                predicate = self.RESEARCHAI[f"has{key.capitalize()}"]
                self.graph.add((author_uri, predicate, Literal(str(value))))

    def _add_entity_node(self, node_id: str, label: str, properties: Dict, entity_type: str):
        """Add generic entity node"""
        entity_uri = self._create_uri('entity', node_id)

        # Type declaration
        self.graph.add((entity_uri, RDF.type, self.RESEARCHAI.Entity))

        # Label
        if label:
            self.graph.add((entity_uri, RDFS.label, Literal(label)))

        # Entity type
        self.graph.add((entity_uri, self.RESEARCHAI.entityType, Literal(entity_type)))

        # Add all properties
        for key, value in properties.items():
            if key != 'type':
                predicate = self.RESEARCHAI[f"has{key.capitalize()}"]
                self.graph.add((entity_uri, predicate, Literal(str(value))))

    def _add_edge_to_rdf(self, edge: Dict):
        """Add an edge/relationship to RDF graph"""
        source_id = edge.get('source', '')
        target_id = edge.get('target', '')
        label = edge.get('label', '').strip()
        properties = edge.get('properties', {})

        if not source_id or not target_id:
            return

        # Determine source/target types from IDs (heuristic)
        source_uri = self._guess_uri_from_id(source_id)
        target_uri = self._guess_uri_from_id(target_id)

        # Create predicate from label
        if label:
            # Map common labels to standard predicates
            predicate = self._get_predicate_for_label(label)
        else:
            predicate = self.RESEARCHAI.relatedTo

        # Add triple
        self.graph.add((source_uri, predicate, target_uri))

        # Add edge properties if any
        if properties:
            # Create reified statement for properties
            stmt = BNode()
            self.graph.add((stmt, RDF.type, RDF.Statement))
            self.graph.add((stmt, RDF.subject, source_uri))
            self.graph.add((stmt, RDF.predicate, predicate))
            self.graph.add((stmt, RDF.object, target_uri))

            for key, value in properties.items():
                prop_predicate = self.RESEARCHAI[f"edge{key.capitalize()}"]
                self.graph.add((stmt, prop_predicate, Literal(str(value))))

    def _guess_uri_from_id(self, identifier: str) -> URIRef:
        """Guess appropriate URI type from identifier"""
        # Check if it looks like a URL (arxiv, http, etc.)
        if identifier.startswith('http://') or identifier.startswith('https://'):
            return self._create_uri('paper', identifier)

        # Check if it's an arXiv ID
        if 'arxiv' in identifier.lower():
            return self._create_uri('paper', identifier)

        # Check if it looks like a person name (contains space, starts with capital)
        if ' ' in identifier and identifier[0].isupper():
            return self._create_uri('author', identifier)

        # Default to entity
        return self._create_uri('entity', identifier)

    def _get_predicate_for_label(self, label: str) -> URIRef:
        """Map edge labels to appropriate RDF predicates"""
        label_lower = label.lower().replace(' ', '_')

        # Standard mappings
        mappings = {
            'author': DCTERMS.creator,
            'wrote': DCTERMS.creator,
            'created': DCTERMS.creator,
            'cites': self.BIBO.cites,
            'references': DCTERMS.references,
            'related': DCTERMS.relation,
            'uses': self.RESEARCHAI.uses,
            'describes': DCTERMS.description,
        }

        # Check for exact matches
        for key, predicate in mappings.items():
            if key in label_lower:
                return predicate

        # Default: create custom predicate
        return self.RESEARCHAI[label.replace(' ', '_').replace('-', '_')]

    def serialize(self, format: str = 'turtle') -> str:
        """
        Serialize RDF graph to string

        Args:
            format: One of 'turtle', 'xml', 'json-ld', 'n3', 'nt'

        Returns:
            Serialized RDF as string
        """
        if format not in self.FORMATS:
            raise ValueError(f"Unsupported format: {format}. Use one of {list(self.FORMATS.keys())}")

        logger.info(f"Serializing RDF graph to {format} format...")

        return self.graph.serialize(format=format)

    def serialize_to_file(self, filepath: str, format: str = 'turtle'):
        """
        Serialize RDF graph to file

        Args:
            filepath: Output file path
            format: RDF format
        """
        serialized = self.serialize(format=format)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(serialized)

        logger.info(f"✅ RDF exported to {filepath} ({len(self.graph)} triples)")

    def import_from_rdf(self, rdf_data: Union[str, bytes, BytesIO], format: str = 'turtle') -> Dict:
        """
        Import RDF data and convert to graph format

        Args:
            rdf_data: RDF data as string, bytes, or file-like object
            format: RDF format ('turtle', 'xml', 'json-ld', etc.)

        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        logger.info(f"Importing RDF data from {format} format...")

        # Parse RDF
        imported_graph = Graph()

        if isinstance(rdf_data, str):
            imported_graph.parse(data=rdf_data, format=format)
        elif isinstance(rdf_data, bytes):
            imported_graph.parse(data=rdf_data.decode('utf-8'), format=format)
        else:
            imported_graph.parse(source=rdf_data, format=format)

        logger.info(f"Parsed {len(imported_graph)} triples from RDF")

        # Convert to nodes and edges
        nodes = []
        edges = []
        node_set = set()

        # Extract nodes
        for s, p, o in imported_graph:
            # Add subject as node
            if isinstance(s, URIRef) and str(s) not in node_set:
                node = self._rdf_to_node(s, imported_graph)
                if node:
                    nodes.append(node)
                    node_set.add(str(s))

            # Add object as node if it's a URIRef
            if isinstance(o, URIRef) and str(o) not in node_set:
                node = self._rdf_to_node(o, imported_graph)
                if node:
                    nodes.append(node)
                    node_set.add(str(o))

            # Add edge if object is a URIRef
            if isinstance(o, URIRef) and p != RDF.type:
                edge = {
                    'source': str(s),
                    'target': str(o),
                    'label': self._get_label_for_predicate(p),
                    'properties': {}
                }
                edges.append(edge)

        logger.info(f"✅ Extracted {len(nodes)} nodes and {len(edges)} edges from RDF")

        return {'nodes': nodes, 'edges': edges}

    def _rdf_to_node(self, uri: URIRef, graph: Graph) -> Optional[Dict]:
        """Convert RDF resource to node dict"""
        # Get type
        types = list(graph.objects(uri, RDF.type))

        if not types:
            return None

        # Determine node type
        node_type = 'entity'
        for t in types:
            if t in [self.BIBO.AcademicArticle, self.RESEARCHAI.Paper]:
                node_type = 'paper'
                break
            elif t in [FOAF.Person, self.RESEARCHAI.Author]:
                node_type = 'author'
                break

        # Get label
        labels = list(graph.objects(uri, RDFS.label))
        label = str(labels[0]) if labels else str(uri).split('/')[-1]

        # Get all properties
        properties = {'type': node_type}

        for p, o in graph.predicate_objects(uri):
            if p == RDF.type or p == RDFS.label:
                continue

            prop_name = self._get_property_name(p)

            if isinstance(o, Literal):
                properties[prop_name] = str(o)
            elif isinstance(o, URIRef):
                properties[prop_name] = str(o)

        return {
            'id': str(uri),
            'label': label,
            'type': node_type,
            'properties': properties
        }

    def _get_label_for_predicate(self, predicate: URIRef) -> str:
        """Get human-readable label for predicate"""
        pred_str = str(predicate)

        # Check if it's a known namespace
        if pred_str.startswith(str(DCTERMS)):
            return pred_str.split('#')[-1].split('/')[-1]

        # Extract last part
        return pred_str.split('#')[-1].split('/')[-1]

    def _get_property_name(self, predicate: URIRef) -> str:
        """Get property name from predicate URI"""
        pred_str = str(predicate)

        # Standard mappings
        if predicate == DCTERMS.title:
            return 'title'
        elif predicate == DCTERMS.abstract:
            return 'abstract'
        elif predicate == DCTERMS.source:
            return 'source'
        elif predicate == DCTERMS.issued:
            return 'year'
        elif predicate == FOAF.name:
            return 'name'
        elif predicate == FOAF.mbox:
            return 'email'

        # Extract last part and remove 'has' prefix if present
        name = pred_str.split('#')[-1].split('/')[-1]
        if name.startswith('has'):
            name = name[3:].lower()

        return name

    def get_stats(self) -> Dict:
        """Get RDF graph statistics"""
        return {
            'triples': len(self.graph),
            'subjects': len(set(self.graph.subjects())),
            'predicates': len(set(self.graph.predicates())),
            'objects': len(set(self.graph.objects())),
            'namespaces': len(list(self.graph.namespaces()))
        }
