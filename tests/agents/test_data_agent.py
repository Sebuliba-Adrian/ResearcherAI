"""
Comprehensive tests for DataCollectorAgent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
from agents.data_agent import DataCollectorAgent


class TestDataCollectorAgentInitialization:
    """Test DataCollectorAgent initialization"""

    def test_initialization_default(self):
        """Test agent initializes with default configuration"""
        agent = DataCollectorAgent()

        assert agent.sources["arxiv"] is True
        assert agent.sources["semantic_scholar"] is True
        assert agent.sources["zenodo"] is True
        assert agent.sources["pubmed"] is True
        assert agent.sources["websearch"] is True
        assert agent.sources["huggingface"] is True
        assert agent.sources["kaggle"] is False

        assert agent.collection_stats["total_collected"] == 0
        assert agent.collection_stats["by_source"] == {}
        assert agent.collection_stats["last_collection"] is None
        assert agent.last_collected_papers == []

    def test_initialization_sources_enabled(self):
        """Test all expected sources are configured"""
        agent = DataCollectorAgent()

        expected_sources = [
            "arxiv", "semantic_scholar", "zenodo", "pubmed",
            "websearch", "huggingface", "kaggle"
        ]

        for source in expected_sources:
            assert source in agent.sources


class TestArxivCollection:
    """Test arXiv data collection"""

    @patch('requests.get')
    def test_fetch_arxiv_success(self, mock_get):
        """Test successful arXiv paper collection"""
        # Mock XML response
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.12345</id>
                <title>Test Paper Title</title>
                <summary>This is a test abstract</summary>
                <author><name>Test Author</name></author>
                <published>2023-01-15T00:00:00Z</published>
                <category term="cs.AI"/>
            </entry>
        </feed>"""

        mock_response = Mock()
        mock_response.content = xml_response.encode('utf-8')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_arxiv("machine learning", max_results=10)

        assert len(papers) == 1
        assert papers[0]["id"] == "http://arxiv.org/abs/2301.12345"
        assert papers[0]["title"] == "Test Paper Title"
        assert papers[0]["abstract"] == "This is a test abstract"
        assert papers[0]["authors"] == ["Test Author"]
        assert papers[0]["source"] == "arXiv"
        assert "cs.AI" in papers[0]["topics"]

    @patch('requests.get')
    def test_fetch_arxiv_multiple_papers(self, mock_get):
        """Test fetching multiple papers from arXiv"""
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.001</id>
                <title>Paper 1</title>
                <summary>Abstract 1</summary>
                <author><name>Author 1</name></author>
                <published>2023-01-01T00:00:00Z</published>
                <category term="cs.AI"/>
            </entry>
            <entry>
                <id>http://arxiv.org/abs/2301.002</id>
                <title>Paper 2</title>
                <summary>Abstract 2</summary>
                <author><name>Author 2</name></author>
                <author><name>Author 3</name></author>
                <published>2023-01-02T00:00:00Z</published>
                <category term="cs.LG"/>
            </entry>
        </feed>"""

        mock_response = Mock()
        mock_response.content = xml_response.encode('utf-8')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_arxiv("AI", max_results=10)

        assert len(papers) == 2
        assert papers[0]["authors"] == ["Author 1"]
        assert papers[1]["authors"] == ["Author 2", "Author 3"]

    @patch('requests.get')
    def test_fetch_arxiv_with_multiline_title(self, mock_get):
        """Test arXiv handles multiline titles and abstracts"""
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.001</id>
                <title>Test Paper
                With Multiple Lines</title>
                <summary>Abstract with
                multiple lines
                of text</summary>
                <author><name>Author 1</name></author>
                <published>2023-01-01T00:00:00Z</published>
                <category term="cs.AI"/>
            </entry>
        </feed>"""

        mock_response = Mock()
        mock_response.content = xml_response.encode('utf-8')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_arxiv("test", max_results=1)

        assert "\n" not in papers[0]["title"]
        assert "\n" not in papers[0]["abstract"]
        assert "Test Paper" in papers[0]["title"]
        assert "With Multiple Lines" in papers[0]["title"]

    @patch('requests.get')
    def test_fetch_arxiv_http_error(self, mock_get):
        """Test arXiv handles HTTP errors"""
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

        agent = DataCollectorAgent()

        with pytest.raises(requests.exceptions.HTTPError):
            agent._fetch_arxiv("test", max_results=10)


class TestSemanticScholarCollection:
    """Test Semantic Scholar data collection"""

    @patch('requests.get')
    def test_fetch_semantic_scholar_success(self, mock_get):
        """Test successful Semantic Scholar collection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "paperId": "test123",
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "authors": [{"name": "Author 1"}, {"name": "Author 2"}],
                    "publicationDate": "2023-01-15",
                    "url": "https://semanticscholar.org/paper/test123",
                    "citationCount": 42
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_semantic_scholar("machine learning", max_results=10)

        assert len(papers) == 1
        assert papers[0]["id"] == "s2_test123"
        assert papers[0]["title"] == "Test Paper"
        assert papers[0]["authors"] == ["Author 1", "Author 2"]
        assert papers[0]["source"] == "Semantic Scholar"
        assert papers[0]["citation_count"] == 42

    @patch('requests.get')
    def test_fetch_semantic_scholar_filters_no_abstract(self, mock_get):
        """Test Semantic Scholar filters papers without abstracts"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "paperId": "test1",
                    "title": "Paper 1",
                    "abstract": "Valid abstract",
                    "authors": [{"name": "Author"}],
                    "publicationDate": "2023-01-01",
                    "url": "https://example.com"
                },
                {
                    "paperId": "test2",
                    "title": "Paper 2",
                    "abstract": None,  # No abstract
                    "authors": [{"name": "Author"}],
                    "publicationDate": "2023-01-02",
                    "url": "https://example.com"
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_semantic_scholar("test", max_results=10)

        assert len(papers) == 1
        assert papers[0]["id"] == "s2_test1"

    @patch('time.sleep')
    @patch('requests.get')
    def test_fetch_semantic_scholar_rate_limit_retry(self, mock_get, mock_sleep):
        """Test Semantic Scholar retries on rate limiting"""
        # First call: 429 error, second call: success
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Rate Limited")

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "data": [
                {
                    "paperId": "test123",
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "authors": [{"name": "Author"}],
                    "publicationDate": "2023-01-01",
                    "url": "https://example.com"
                }
            ]
        }
        mock_response_success.raise_for_status = Mock()

        mock_get.side_effect = [mock_response_429, mock_response_success]

        agent = DataCollectorAgent()
        papers = agent._fetch_semantic_scholar("test", max_results=10)

        assert len(papers) == 1
        assert mock_sleep.called
        assert mock_get.call_count == 2

    @patch('time.sleep')
    @patch('requests.get')
    def test_fetch_semantic_scholar_rate_limit_exhausted(self, mock_get, mock_sleep):
        """Test Semantic Scholar gives up after max retries"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Rate Limited")
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_semantic_scholar("test", max_results=10)

        assert len(papers) == 0
        assert mock_get.call_count == 3  # Initial + 2 retries

    @patch('requests.get')
    def test_fetch_semantic_scholar_network_error(self, mock_get):
        """Test Semantic Scholar handles network errors"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        agent = DataCollectorAgent()
        papers = agent._fetch_semantic_scholar("test", max_results=10)

        assert len(papers) == 0

    @patch('time.sleep')
    @patch('requests.get')
    def test_fetch_semantic_scholar_retry_loop_exhaustion(self, mock_get, mock_sleep):
        """Test Semantic Scholar for-else clause when retries succeed on last attempt"""
        # First two calls: fail with non-429 error, third call: timeout (no response object)
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Error")

        # This will trigger the else clause of the for loop
        mock_get.side_effect = [
            requests.exceptions.Timeout("Timeout 1"),
            requests.exceptions.Timeout("Timeout 2"),
            requests.exceptions.Timeout("Timeout 3")
        ]

        agent = DataCollectorAgent()
        papers = agent._fetch_semantic_scholar("test", max_results=10)

        # Should exhaust retries and return empty list via for-else
        assert len(papers) == 0


class TestZenodoCollection:
    """Test Zenodo data collection"""

    @patch('requests.get')
    def test_fetch_zenodo_success(self, mock_get):
        """Test successful Zenodo collection"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hits": {
                "hits": [
                    {
                        "id": "12345",
                        "metadata": {
                            "title": "Test Dataset",
                            "description": "Test description",
                            "creators": [{"name": "Creator 1"}],
                            "publication_date": "2023-01-15",
                            "keywords": ["machine learning", "AI"],
                            "resource_type": {"type": "dataset"}
                        },
                        "links": {
                            "html": "https://zenodo.org/record/12345"
                        }
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_zenodo("machine learning", max_results=10)

        assert len(papers) == 1
        assert papers[0]["id"] == "zenodo_12345"
        assert papers[0]["title"] == "Test Dataset"
        assert papers[0]["source"] == "Zenodo"
        assert papers[0]["topics"] == ["machine learning", "AI"]
        assert papers[0]["resource_type"] == "dataset"

    @patch('requests.get')
    def test_fetch_zenodo_empty_results(self, mock_get):
        """Test Zenodo with empty results"""
        mock_response = Mock()
        mock_response.json.return_value = {"hits": {"hits": []}}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_zenodo("nonexistent", max_results=10)

        assert len(papers) == 0

    @patch('requests.get')
    def test_fetch_zenodo_http_error(self, mock_get):
        """Test Zenodo handles HTTP errors"""
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")

        agent = DataCollectorAgent()

        with pytest.raises(requests.exceptions.HTTPError):
            agent._fetch_zenodo("test", max_results=10)


class TestPubMedCollection:
    """Test PubMed data collection"""

    @patch('requests.get')
    def test_fetch_pubmed_success(self, mock_get):
        """Test successful PubMed collection"""
        # Mock search response
        search_response = Mock()
        search_response.json.return_value = {
            "esearchresult": {
                "idlist": ["12345", "67890"]
            }
        }
        search_response.raise_for_status = Mock()

        # Mock fetch response
        fetch_response = Mock()
        fetch_response.json.return_value = {
            "result": {
                "12345": {
                    "title": "Test Medical Paper",
                    "source": "Journal of Test",
                    "authors": [
                        {"name": "Author A"},
                        {"name": "Author B"}
                    ],
                    "pubdate": "2023 Jan 15",
                    "articleids": [
                        {"idtype": "pubmed", "value": "12345"},
                        {"idtype": "doi", "value": "10.1234/test"}
                    ]
                },
                "67890": {
                    "title": "Another Paper",
                    "source": "Test Journal",
                    "authors": [{"name": "Author C"}],
                    "pubdate": "2023 Jan 20",
                    "articleids": []
                }
            }
        }
        fetch_response.raise_for_status = Mock()

        mock_get.side_effect = [search_response, fetch_response]

        agent = DataCollectorAgent()
        papers = agent._fetch_pubmed("cancer research", max_results=10)

        assert len(papers) == 2
        assert papers[0]["id"] == "pubmed_12345"
        assert papers[0]["title"] == "Test Medical Paper"
        assert papers[0]["authors"] == ["Author A", "Author B"]
        assert papers[0]["source"] == "PubMed"
        assert "pubmed:12345" in papers[0]["topics"]
        assert "doi:10.1234/test" in papers[0]["topics"]

    @patch('requests.get')
    def test_fetch_pubmed_no_results(self, mock_get):
        """Test PubMed with no search results"""
        search_response = Mock()
        search_response.json.return_value = {
            "esearchresult": {
                "idlist": []
            }
        }
        search_response.raise_for_status = Mock()
        mock_get.return_value = search_response

        agent = DataCollectorAgent()
        papers = agent._fetch_pubmed("nonexistent", max_results=10)

        assert len(papers) == 0

    @patch('requests.get')
    def test_fetch_pubmed_cleans_empty_authors(self, mock_get):
        """Test PubMed filters out empty/null author names"""
        search_response = Mock()
        search_response.json.return_value = {
            "esearchresult": {"idlist": ["123"]}
        }
        search_response.raise_for_status = Mock()

        fetch_response = Mock()
        fetch_response.json.return_value = {
            "result": {
                "123": {
                    "title": "Test Paper",
                    "source": "Test",
                    "authors": [
                        {"name": "Valid Author"},
                        {"name": ""},  # Empty name
                        {"name": "   "},  # Whitespace only
                        None,  # Null author
                    ],
                    "pubdate": "2023",
                    "articleids": []
                }
            }
        }
        fetch_response.raise_for_status = Mock()

        mock_get.side_effect = [search_response, fetch_response]

        agent = DataCollectorAgent()
        papers = agent._fetch_pubmed("test", max_results=1)

        assert len(papers[0]["authors"]) == 1
        assert papers[0]["authors"][0] == "Valid Author"

    @patch('requests.get')
    def test_fetch_pubmed_http_error(self, mock_get):
        """Test PubMed handles HTTP errors"""
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Error")

        agent = DataCollectorAgent()

        with pytest.raises(requests.exceptions.HTTPError):
            agent._fetch_pubmed("test", max_results=10)

    @patch('requests.get')
    def test_fetch_pubmed_cleans_string_authors(self, mock_get):
        """Test PubMed handles string authors (not dict)"""
        search_response = Mock()
        search_response.json.return_value = {
            "esearchresult": {"idlist": ["123"]}
        }
        search_response.raise_for_status = Mock()

        fetch_response = Mock()
        fetch_response.json.return_value = {
            "result": {
                "123": {
                    "title": "Test Paper",
                    "source": "Test",
                    "authors": ["String Author", "", "   "],  # String authors, some empty
                    "pubdate": "2023",
                    "articleids": ["plain_string_topic"]  # Non-dict topics
                }
            }
        }
        fetch_response.raise_for_status = Mock()

        mock_get.side_effect = [search_response, fetch_response]

        agent = DataCollectorAgent()
        papers = agent._fetch_pubmed("test", max_results=1)

        # Should have one author (the valid string) and one topic
        assert len(papers[0]["authors"]) == 1
        assert papers[0]["authors"][0] == "String Author"
        assert len(papers[0]["topics"]) == 1
        assert papers[0]["topics"][0] == "plain_string_topic"


class TestWebSearchCollection:
    """Test web search data collection"""

    @patch('duckduckgo_search.DDGS')
    def test_fetch_websearch_success(self, mock_ddgs):
        """Test successful web search collection"""
        mock_results = [
            {
                "title": "Research Paper 1",
                "body": "Abstract of research paper 1",
                "href": "https://example.com/paper1"
            },
            {
                "title": "Research Paper 2",
                "body": "Abstract of research paper 2",
                "href": "https://example.com/paper2"
            }
        ]

        mock_instance = Mock()
        mock_instance.text.return_value = iter(mock_results)
        mock_ddgs.return_value = mock_instance

        agent = DataCollectorAgent()
        papers = agent._fetch_websearch("machine learning", max_results=10)

        assert len(papers) == 2
        assert papers[0]["id"] == "web_0"
        assert papers[0]["title"] == "Research Paper 1"
        assert papers[0]["abstract"] == "Abstract of research paper 1"
        assert papers[0]["source"] == "Web Search"
        assert papers[0]["url"] == "https://example.com/paper1"

    @patch('duckduckgo_search.DDGS')
    def test_fetch_websearch_filters_empty_results(self, mock_ddgs):
        """Test web search filters empty/None results"""
        mock_results = [
            {
                "title": "Valid Paper",
                "body": "Valid abstract",
                "href": "https://example.com/valid"
            },
            None,  # Empty result - will be filtered out
            {
                "title": "Another Valid Paper",
                "body": "Another abstract",
                "href": "https://example.com/valid2"
            }
        ]

        mock_instance = Mock()
        mock_instance.text.return_value = iter(mock_results)
        mock_ddgs.return_value = mock_instance

        agent = DataCollectorAgent()
        papers = agent._fetch_websearch("test", max_results=10)

        # Only 2 papers, but IDs are based on enumerate which includes None
        assert len(papers) == 2
        assert papers[0]["id"] == "web_0"
        assert papers[1]["id"] == "web_2"  # Skips web_1 because index 1 was None

    def test_fetch_websearch_import_error(self):
        """Test web search handles missing duckduckgo-search library"""
        with patch.dict('sys.modules', {'duckduckgo_search': None}):
            agent = DataCollectorAgent()
            papers = agent._fetch_websearch("test", max_results=10)

            assert len(papers) == 0


class TestHuggingFaceCollection:
    """Test HuggingFace Hub data collection"""

    @patch('huggingface_hub.HfApi')
    def test_fetch_huggingface_success(self, mock_hfapi):
        """Test successful HuggingFace collection"""
        mock_model = Mock()
        mock_model.id = "test/model"
        mock_model.downloads = 1000
        mock_model.tags = ["pytorch", "transformers"]
        mock_model.author = "test-author"
        mock_model.created_at = "2023-01-15"

        mock_dataset = Mock()
        mock_dataset.id = "test/dataset"
        mock_dataset.downloads = 500
        mock_dataset.tags = ["nlp", "text"]
        mock_dataset.author = "test-author"
        mock_dataset.created_at = "2023-01-20"

        mock_api = Mock()
        mock_api.list_models.return_value = [mock_model]
        mock_api.list_datasets.return_value = [mock_dataset]
        mock_hfapi.return_value = mock_api

        agent = DataCollectorAgent()
        papers = agent._fetch_huggingface("transformers", max_results=10)

        assert len(papers) == 2
        assert papers[0]["id"] == "hf_model_test/model"
        assert papers[0]["downloads"] == 1000
        assert papers[0]["source"] == "HuggingFace"
        assert papers[1]["id"] == "hf_dataset_test/dataset"

    @patch('huggingface_hub.HfApi')
    def test_fetch_huggingface_handles_missing_attributes(self, mock_hfapi):
        """Test HuggingFace handles models/datasets with missing attributes"""
        mock_model = Mock(spec=[])  # No attributes
        mock_model.id = "test/model"

        mock_api = Mock()
        mock_api.list_models.return_value = [mock_model]
        mock_api.list_datasets.return_value = []
        mock_hfapi.return_value = mock_api

        agent = DataCollectorAgent()
        papers = agent._fetch_huggingface("test", max_results=10)

        assert len(papers) == 1
        assert papers[0]["downloads"] == 0
        assert papers[0]["topics"] == []

    def test_fetch_huggingface_import_error(self):
        """Test HuggingFace handles missing huggingface-hub library"""
        with patch.dict('sys.modules', {'huggingface_hub': None}):
            agent = DataCollectorAgent()
            papers = agent._fetch_huggingface("test", max_results=10)

            assert len(papers) == 0


class TestKaggleCollection:
    """Test Kaggle data collection"""

    def test_fetch_kaggle_success(self):
        """Test successful Kaggle collection"""
        mock_dataset = Mock()
        mock_dataset.ref = "test/dataset"
        mock_dataset.title = "Test Dataset"
        mock_dataset.subtitle = "A test dataset"
        mock_dataset.creator_name = "Test Creator"
        mock_dataset.lastUpdated = "2023-01-15"
        mock_dataset.tags = ["machine-learning", "nlp"]
        mock_dataset.voteCount = 100
        mock_dataset.downloadCount = 500

        # Create a mock kaggle module with an api attribute
        mock_kaggle_module = Mock()
        mock_kaggle_api = Mock()
        mock_kaggle_api.authenticate = Mock()
        mock_kaggle_api.dataset_list.return_value = [mock_dataset]
        mock_kaggle_module.api = mock_kaggle_api

        # Mock the kaggle module import
        with patch.dict('sys.modules', {'kaggle': mock_kaggle_module}):
            agent = DataCollectorAgent()
            papers = agent._fetch_kaggle("machine learning", max_results=10)

            assert len(papers) == 1
            assert papers[0]["id"] == "kaggle_test/dataset"
            assert papers[0]["title"] == "Dataset: Test Dataset"
            assert papers[0]["source"] == "Kaggle"
            assert papers[0]["votes"] == 100
            assert papers[0]["downloads"] == 500

    def test_fetch_kaggle_authentication_error(self):
        """Test Kaggle handles authentication errors"""
        mock_kaggle_module = Mock()
        mock_kaggle_api = Mock()
        mock_kaggle_api.authenticate.side_effect = Exception("Authentication failed")
        mock_kaggle_module.api = mock_kaggle_api

        with patch.dict('sys.modules', {'kaggle': mock_kaggle_module}):
            agent = DataCollectorAgent()
            papers = agent._fetch_kaggle("test", max_results=10)

            assert len(papers) == 0

    def test_fetch_kaggle_import_error(self):
        """Test Kaggle handles missing kaggle library"""
        with patch.dict('sys.modules', {'kaggle': None}):
            agent = DataCollectorAgent()
            papers = agent._fetch_kaggle("test", max_results=10)

            assert len(papers) == 0


class TestCollectAll:
    """Test collect_all method"""

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    @patch.object(DataCollectorAgent, '_fetch_semantic_scholar')
    @patch.object(DataCollectorAgent, '_fetch_zenodo')
    def test_collect_all_success(self, mock_zenodo, mock_s2, mock_arxiv):
        """Test collect_all aggregates from all sources"""
        mock_arxiv.return_value = [
            {"id": "arxiv_1", "title": "ArXiv Paper", "source": "arXiv"}
        ]
        mock_s2.return_value = [
            {"id": "s2_1", "title": "S2 Paper", "source": "Semantic Scholar"}
        ]
        mock_zenodo.return_value = [
            {"id": "zenodo_1", "title": "Zenodo Dataset", "source": "Zenodo"}
        ]

        agent = DataCollectorAgent()
        # Disable some sources to speed up test
        agent.sources["pubmed"] = False
        agent.sources["websearch"] = False
        agent.sources["huggingface"] = False
        agent.sources["kaggle"] = False

        papers = agent.collect_all("machine learning", max_per_source=5)

        assert len(papers) == 3
        assert agent.collection_stats["total_collected"] == 3
        assert agent.collection_stats["by_source"]["arxiv"] == 1
        assert agent.collection_stats["by_source"]["semantic_scholar"] == 1
        assert agent.collection_stats["by_source"]["zenodo"] == 1
        assert agent.collection_stats["last_collection"] is not None
        assert len(agent.last_collected_papers) == 3

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    @patch.object(DataCollectorAgent, '_fetch_semantic_scholar')
    def test_collect_all_handles_source_failure(self, mock_s2, mock_arxiv):
        """Test collect_all continues when a source fails"""
        mock_arxiv.return_value = [
            {"id": "arxiv_1", "title": "ArXiv Paper", "source": "arXiv"}
        ]
        mock_s2.side_effect = Exception("API error")

        agent = DataCollectorAgent()
        # Disable other sources
        agent.sources["zenodo"] = False
        agent.sources["pubmed"] = False
        agent.sources["websearch"] = False
        agent.sources["huggingface"] = False
        agent.sources["kaggle"] = False

        papers = agent.collect_all("test", max_per_source=5)

        assert len(papers) == 1
        assert agent.collection_stats["by_source"]["arxiv"] == 1
        assert agent.collection_stats["by_source"]["semantic_scholar"] == 0

    def test_collect_all_skips_disabled_sources(self):
        """Test collect_all skips disabled sources"""
        agent = DataCollectorAgent()

        # Kaggle is disabled by default
        assert agent.sources["kaggle"] is False

        with patch.object(agent, '_fetch_kaggle') as mock_kaggle:
            # Disable all other sources too
            for source in agent.sources:
                agent.sources[source] = False

            papers = agent.collect_all("test", max_per_source=5)

            assert len(papers) == 0
            mock_kaggle.assert_not_called()

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    def test_collect_all_stores_papers(self, mock_arxiv):
        """Test collect_all stores papers for later retrieval"""
        test_papers = [
            {"id": "1", "title": "Paper 1"},
            {"id": "2", "title": "Paper 2"}
        ]
        mock_arxiv.return_value = test_papers

        agent = DataCollectorAgent()
        # Disable other sources
        for source in agent.sources:
            if source != "arxiv":
                agent.sources[source] = False

        papers = agent.collect_all("test", max_per_source=5)

        assert agent.last_collected_papers == papers
        assert len(agent.get_last_collection()) == 2

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    def test_collect_all_updates_timestamp(self, mock_arxiv):
        """Test collect_all updates last_collection timestamp"""
        mock_arxiv.return_value = [{"id": "1", "title": "Test"}]

        agent = DataCollectorAgent()
        for source in agent.sources:
            if source != "arxiv":
                agent.sources[source] = False

        before = datetime.now()
        agent.collect_all("test", max_per_source=5)
        after = datetime.now()

        timestamp = datetime.fromisoformat(agent.collection_stats["last_collection"])
        assert before <= timestamp <= after


class TestCollectFromSource:
    """Test _collect_from_source routing"""

    def test_collect_from_source_routes_correctly(self):
        """Test _collect_from_source routes to correct methods"""
        agent = DataCollectorAgent()

        methods = {
            "arxiv": "_fetch_arxiv",
            "semantic_scholar": "_fetch_semantic_scholar",
            "zenodo": "_fetch_zenodo",
            "pubmed": "_fetch_pubmed",
            "websearch": "_fetch_websearch",
            "huggingface": "_fetch_huggingface",
            "kaggle": "_fetch_kaggle"
        }

        for source, method_name in methods.items():
            with patch.object(agent, method_name) as mock_method:
                mock_method.return_value = []
                agent._collect_from_source(source, "test", 10)
                mock_method.assert_called_once_with("test", 10)


class TestStatistics:
    """Test statistics methods"""

    def test_get_stats_initial(self):
        """Test get_stats returns initial statistics"""
        agent = DataCollectorAgent()

        stats = agent.get_stats()

        assert stats["total_collected"] == 0
        assert stats["by_source"] == {}
        assert stats["last_collection"] is None

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    def test_get_stats_after_collection(self, mock_arxiv):
        """Test get_stats returns updated statistics after collection"""
        mock_arxiv.return_value = [
            {"id": "1", "title": "Paper 1"},
            {"id": "2", "title": "Paper 2"}
        ]

        agent = DataCollectorAgent()
        for source in agent.sources:
            if source != "arxiv":
                agent.sources[source] = False

        agent.collect_all("test", max_per_source=5)
        stats = agent.get_stats()

        assert stats["total_collected"] == 2
        assert stats["by_source"]["arxiv"] == 2
        assert stats["last_collection"] is not None

    def test_get_last_collection_initial(self):
        """Test get_last_collection returns empty list initially"""
        agent = DataCollectorAgent()

        papers = agent.get_last_collection()

        assert papers == []

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    def test_get_last_collection_after_collection(self, mock_arxiv):
        """Test get_last_collection returns collected papers"""
        test_papers = [
            {"id": "1", "title": "Paper 1"},
            {"id": "2", "title": "Paper 2"}
        ]
        mock_arxiv.return_value = test_papers

        agent = DataCollectorAgent()
        for source in agent.sources:
            if source != "arxiv":
                agent.sources[source] = False

        agent.collect_all("test", max_per_source=5)
        papers = agent.get_last_collection()

        assert len(papers) == 2
        assert papers == test_papers


class TestEdgeCases:
    """Test edge cases and error handling"""

    @patch('requests.get')
    def test_fetch_arxiv_empty_response(self, mock_get):
        """Test arXiv handles empty XML response"""
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        mock_response = Mock()
        mock_response.content = xml_response.encode('utf-8')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_arxiv("test", max_results=10)

        assert len(papers) == 0

    @patch('requests.get')
    def test_fetch_semantic_scholar_empty_data(self, mock_get):
        """Test Semantic Scholar handles empty data array"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_semantic_scholar("test", max_results=10)

        assert len(papers) == 0

    @patch('requests.get')
    def test_fetch_zenodo_missing_metadata(self, mock_get):
        """Test Zenodo handles missing metadata gracefully"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hits": {
                "hits": [
                    {
                        "id": "12345",
                        "metadata": {},  # Empty metadata
                        "links": {}
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        agent = DataCollectorAgent()
        papers = agent._fetch_zenodo("test", max_results=10)

        assert len(papers) == 1
        assert papers[0]["title"] == "Unknown"
        assert papers[0]["authors"] == []

    @patch('duckduckgo_search.DDGS')
    def test_fetch_websearch_handles_missing_fields(self, mock_ddgs):
        """Test web search handles results with missing fields"""
        mock_results = [
            {
                "title": "Paper with URL",
                "body": "Abstract",
                "href": "https://example.com"
            },
            {
                # Missing title, body, href - but not None so it won't be filtered
                "title": "",
                "body": "",
                "href": ""
            }
        ]

        mock_instance = Mock()
        mock_instance.text.return_value = iter(mock_results)
        mock_ddgs.return_value = mock_instance

        agent = DataCollectorAgent()
        papers = agent._fetch_websearch("test", max_results=10)

        assert len(papers) == 2
        assert papers[0]["title"] == "Paper with URL"
        # Empty strings become "Unknown" via dict.get() defaults
        assert papers[1]["title"] == "" or papers[1]["title"] == "Unknown"
        assert papers[1]["url"] == ""

    @patch('requests.get')
    def test_fetch_pubmed_timeout(self, mock_get):
        """Test PubMed handles timeout errors"""
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

        agent = DataCollectorAgent()

        with pytest.raises(requests.exceptions.Timeout):
            agent._fetch_pubmed("test", max_results=10)

    def test_collect_all_with_max_per_source(self):
        """Test collect_all respects max_per_source parameter"""
        agent = DataCollectorAgent()

        with patch.object(agent, '_fetch_arxiv') as mock_arxiv:
            mock_arxiv.return_value = []
            for source in agent.sources:
                if source != "arxiv":
                    agent.sources[source] = False

            agent.collect_all("test", max_per_source=20)

            mock_arxiv.assert_called_once_with("test", 20)


class TestIPv6Fix:
    """Test IPv6 network fix"""

    def test_ipv6_fix_applied(self):
        """Test that IPv6 fix is applied on module import"""
        import agents.data_agent as data_agent_module
        import requests.packages.urllib3.util.connection as urllib3_cn

        # The fix should set allowed_gai_family to return AF_INET
        import socket
        assert urllib3_cn.allowed_gai_family() == socket.AF_INET


class TestLogging:
    """Test logging behavior"""

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    def test_collect_all_logs_progress(self, mock_arxiv, caplog):
        """Test collect_all logs collection progress"""
        import logging
        caplog.set_level(logging.INFO)

        mock_arxiv.return_value = [{"id": "1", "title": "Test"}]

        agent = DataCollectorAgent()
        for source in agent.sources:
            if source != "arxiv":
                agent.sources[source] = False

        agent.collect_all("test query", max_per_source=5)

        # Check that logging occurred
        assert any("Starting autonomous collection" in record.message for record in caplog.records)
        assert any("Collection complete" in record.message for record in caplog.records)

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    def test_collect_all_logs_skipped_sources(self, mock_arxiv, caplog):
        """Test collect_all logs skipped sources"""
        import logging
        caplog.set_level(logging.INFO)

        agent = DataCollectorAgent()
        # Kaggle is disabled by default
        agent.sources = {"kaggle": False}

        agent.collect_all("test", max_per_source=5)

        assert any("Skipping kaggle (disabled)" in record.message for record in caplog.records)

    @patch.object(DataCollectorAgent, '_fetch_arxiv')
    def test_collect_all_logs_errors(self, mock_arxiv, caplog):
        """Test collect_all logs source failures"""
        import logging
        caplog.set_level(logging.ERROR)

        mock_arxiv.side_effect = Exception("Test error")

        agent = DataCollectorAgent()
        agent.sources = {"arxiv": True}

        agent.collect_all("test", max_per_source=5)

        assert any("arxiv failed" in record.message for record in caplog.records)


class TestIntegration:
    """Integration tests"""

    @patch('requests.get')
    @patch('duckduckgo_search.DDGS')
    def test_full_collection_workflow(self, mock_ddgs, mock_requests):
        """Test full collection workflow with multiple sources"""
        # Mock arXiv response
        arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.001</id>
                <title>ArXiv Paper</title>
                <summary>ArXiv abstract</summary>
                <author><name>ArXiv Author</name></author>
                <published>2023-01-01T00:00:00Z</published>
                <category term="cs.AI"/>
            </entry>
        </feed>"""

        # Mock Semantic Scholar response
        s2_response = {
            "data": [{
                "paperId": "s2_123",
                "title": "S2 Paper",
                "abstract": "S2 abstract",
                "authors": [{"name": "S2 Author"}],
                "publicationDate": "2023-01-01",
                "url": "https://example.com",
                "citationCount": 10
            }]
        }

        # Setup mock responses
        arxiv_mock = Mock()
        arxiv_mock.content = arxiv_xml.encode('utf-8')
        arxiv_mock.raise_for_status = Mock()

        s2_mock = Mock()
        s2_mock.status_code = 200
        s2_mock.json.return_value = s2_response
        s2_mock.raise_for_status = Mock()

        # Return appropriate responses based on URL
        def get_side_effect(url, *args, **kwargs):
            if "arxiv.org" in url:
                return arxiv_mock
            elif "semanticscholar.org" in url:
                return s2_mock
            return Mock()

        mock_requests.side_effect = get_side_effect

        # Mock web search
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.text.return_value = iter([{
            "title": "Web Paper",
            "body": "Web abstract",
            "href": "https://example.com/web"
        }])
        mock_ddgs.return_value = mock_ddgs_instance

        agent = DataCollectorAgent()
        # Enable only tested sources
        agent.sources = {
            "arxiv": True,
            "semantic_scholar": True,
            "websearch": True,
            "zenodo": False,
            "pubmed": False,
            "huggingface": False,
            "kaggle": False
        }

        papers = agent.collect_all("machine learning", max_per_source=5)

        assert len(papers) == 3
        assert agent.collection_stats["total_collected"] == 3
        assert agent.collection_stats["by_source"]["arxiv"] == 1
        assert agent.collection_stats["by_source"]["semantic_scholar"] == 1
        assert agent.collection_stats["by_source"]["websearch"] == 1

        # Verify paper contents
        sources = [p["source"] for p in papers]
        assert "arXiv" in sources
        assert "Semantic Scholar" in sources
        assert "Web Search" in sources
