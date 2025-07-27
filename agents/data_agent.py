"""
DataCollectorAgent - Autonomous Data Collection
===============================================

Collects research papers from 7 sources:
1. arXiv (recent AI research)
2. Semantic Scholar (academic papers)
3. Zenodo (research datasets)
4. PubMed (biomedical literature)
5. Web Search (DuckDuckGo)
6. HuggingFace Hub (models and datasets)
7. Kaggle (datasets and competitions)
"""

import os
import time
import requests
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DataCollectorAgent:
    """Autonomous data collection from multiple research sources"""

    def __init__(self):
        self.sources = {
            "arxiv": True,
            "semantic_scholar": True,
            "zenodo": True,
            "pubmed": True,
            "websearch": True,
            "huggingface": True,
            "kaggle": False  # Requires API credentials
        }
        self.collection_stats = {
            "total_collected": 0,
            "by_source": {},
            "last_collection": None
        }
        logger.info("DataCollectorAgent initialized with 7 sources")

    def collect_all(self, query: str, max_per_source: int = 10) -> List[Dict]:
        """Collect from all enabled sources"""
        logger.info(f"Starting autonomous collection for query: '{query}'")
        all_papers = []

        for source_name, enabled in self.sources.items():
            if not enabled:
                logger.info(f"Skipping {source_name} (disabled)")
                continue

            try:
                papers = self._collect_from_source(source_name, query, max_per_source)
                all_papers.extend(papers)
                self.collection_stats["by_source"][source_name] = len(papers)
                logger.info(f"✅ {source_name}: {len(papers)} papers")
            except Exception as e:
                logger.error(f"❌ {source_name} failed: {e}")
                self.collection_stats["by_source"][source_name] = 0

        self.collection_stats["total_collected"] = len(all_papers)
        self.collection_stats["last_collection"] = datetime.now().isoformat()

        logger.info(f"Collection complete: {len(all_papers)} total papers from {len([s for s in self.sources.values() if s])} sources")
        return all_papers

    def _collect_from_source(self, source: str, query: str, max_results: int) -> List[Dict]:
        """Route to appropriate collection method"""
        methods = {
            "arxiv": self._fetch_arxiv,
            "semantic_scholar": self._fetch_semantic_scholar,
            "zenodo": self._fetch_zenodo,
            "pubmed": self._fetch_pubmed,
            "websearch": self._fetch_websearch,
            "huggingface": self._fetch_huggingface,
            "kaggle": self._fetch_kaggle
        }

        return methods[source](query, max_results)

    def _fetch_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch from arXiv API"""
        import urllib.parse
        import xml.etree.ElementTree as ET

        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        for entry in root.findall("atom:entry", namespace):
            paper = {
                "id": entry.find("atom:id", namespace).text,
                "title": entry.find("atom:title", namespace).text.strip().replace("\n", " "),
                "abstract": entry.find("atom:summary", namespace).text.strip().replace("\n", " "),
                "authors": [author.find("atom:name", namespace).text for author in entry.findall("atom:author", namespace)],
                "published": entry.find("atom:published", namespace).text,
                "url": entry.find("atom:id", namespace).text,
                "source": "arXiv",
                "topics": [cat.attrib.get("term", "") for cat in entry.findall("atom:category", namespace)]
            }
            papers.append(paper)

        return papers

    def _fetch_semantic_scholar(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch from Semantic Scholar API"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,url,publicationDate,citationCount"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("data", []):
            if not item.get("abstract"):
                continue

            paper = {
                "id": f"s2_{item.get('paperId', '')}",
                "title": item.get("title", "Unknown"),
                "abstract": item.get("abstract", ""),
                "authors": [author.get("name", "") for author in item.get("authors", [])],
                "published": item.get("publicationDate", ""),
                "url": item.get("url", ""),
                "source": "Semantic Scholar",
                "topics": [],
                "citation_count": item.get("citationCount", 0)
            }
            papers.append(paper)

        return papers

    def _fetch_zenodo(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch from Zenodo API"""
        url = "https://zenodo.org/api/records"
        params = {
            "q": query,
            "size": max_results,
            "sort": "mostrecent"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("hits", {}).get("hits", []):
            metadata = item.get("metadata", {})

            paper = {
                "id": f"zenodo_{item.get('id', '')}",
                "title": metadata.get("title", "Unknown"),
                "abstract": metadata.get("description", ""),
                "authors": [creator.get("name", "") for creator in metadata.get("creators", [])],
                "published": metadata.get("publication_date", ""),
                "url": item.get("links", {}).get("html", ""),
                "source": "Zenodo",
                "topics": metadata.get("keywords", []),
                "resource_type": metadata.get("resource_type", {}).get("type", "")
            }
            papers.append(paper)

        return papers

    def _fetch_pubmed(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch from PubMed API"""
        # Search for article IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub_date"
        }

        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        search_data = search_response.json()

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        # Fetch article details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }

        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
        fetch_response.raise_for_status()
        fetch_data = fetch_response.json()

        papers = []
        for pmid in id_list:
            article = fetch_data.get("result", {}).get(pmid, {})

            paper = {
                "id": f"pubmed_{pmid}",
                "title": article.get("title", "Unknown"),
                "abstract": article.get("source", ""),  # PubMed summary API doesn't include abstract
                "authors": [author.get("name", "") for author in article.get("authors", [])],
                "published": article.get("pubdate", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source": "PubMed",
                "topics": article.get("articleids", [])
            }
            papers.append(paper)

        return papers

    def _fetch_websearch(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch from DuckDuckGo web search"""
        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            results = ddgs.text(f"{query} research paper", max_results=max_results)

            papers = []
            for i, result in enumerate(results):
                paper = {
                    "id": f"web_{i}",
                    "title": result.get("title", "Unknown"),
                    "abstract": result.get("body", ""),
                    "authors": [],
                    "published": datetime.now().isoformat(),
                    "url": result.get("href", ""),
                    "source": "Web Search",
                    "topics": []
                }
                papers.append(paper)

            return papers
        except ImportError:
            logger.warning("duckduckgo-search not installed")
            return []

    def _fetch_huggingface(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch from HuggingFace Hub"""
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            papers = []

            # Fetch models
            models = list(api.list_models(search=query, limit=max_results // 2, sort="downloads", direction=-1))
            for model in models:
                paper = {
                    "id": f"hf_model_{model.id}",
                    "title": f"Model: {model.id}",
                    "abstract": f"HuggingFace model with {getattr(model, 'downloads', 0):,} downloads. Tags: {', '.join(getattr(model, 'tags', [])[:5])}",
                    "authors": [getattr(model, 'author', 'Unknown')],
                    "published": str(getattr(model, 'created_at', '')),
                    "url": f"https://huggingface.co/{model.id}",
                    "source": "HuggingFace",
                    "topics": getattr(model, 'tags', []),
                    "downloads": getattr(model, 'downloads', 0)
                }
                papers.append(paper)

            # Fetch datasets
            datasets = list(api.list_datasets(search=query, limit=max_results // 2, sort="downloads", direction=-1))
            for dataset in datasets:
                paper = {
                    "id": f"hf_dataset_{dataset.id}",
                    "title": f"Dataset: {dataset.id}",
                    "abstract": f"HuggingFace dataset with {getattr(dataset, 'downloads', 0):,} downloads. Tags: {', '.join(getattr(dataset, 'tags', [])[:5])}",
                    "authors": [getattr(dataset, 'author', 'Unknown')],
                    "published": str(getattr(dataset, 'created_at', '')),
                    "url": f"https://huggingface.co/datasets/{dataset.id}",
                    "source": "HuggingFace",
                    "topics": getattr(dataset, 'tags', []),
                    "downloads": getattr(dataset, 'downloads', 0)
                }
                papers.append(paper)

            return papers
        except ImportError:
            logger.warning("huggingface-hub not installed")
            return []

    def _fetch_kaggle(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch from Kaggle API"""
        try:
            from kaggle import api as kaggle_api

            # Authenticate (requires ~/.kaggle/kaggle.json or env vars)
            kaggle_api.authenticate()

            papers = []

            # Fetch datasets
            datasets = kaggle_api.dataset_list(search=query, page_size=max_results)
            for dataset in datasets[:max_results]:
                paper = {
                    "id": f"kaggle_{dataset.ref}",
                    "title": f"Dataset: {dataset.title}",
                    "abstract": dataset.subtitle or "Kaggle dataset",
                    "authors": [dataset.creator_name],
                    "published": str(dataset.lastUpdated),
                    "url": f"https://www.kaggle.com/datasets/{dataset.ref}",
                    "source": "Kaggle",
                    "topics": dataset.tags or [],
                    "votes": dataset.voteCount,
                    "downloads": dataset.downloadCount
                }
                papers.append(paper)

            return papers
        except Exception as e:
            logger.warning(f"Kaggle API failed (credentials may be missing): {e}")
            return []

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return self.collection_stats
