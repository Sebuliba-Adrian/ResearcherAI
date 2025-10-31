"""
ResearcherAI ETL Pipeline - Main DAG
=====================================

Parallel data collection from 7 sources with intelligent processing.

DAG Structure:
    start
      │
      ├──────┬──────┬──────┬──────┬──────┬──────┐
      │      │      │      │      │      │      │
    arXiv  SS    Zen   PM   Web   HF   Kaggle  (Parallel)
      │      │      │      │      │      │      │
      └──────┴──────┴──────┴──────┴──────┴──────┘
                     │
              merge_papers
                     │
           check_paper_threshold
                     │
              ┌──────┴──────┐
              │             │
         (>10 papers)    (<10 papers)
              │             │
         process_data   skip_processing
              │
      ┌───────┴───────┐
      │               │
  store_vectors  extract_triples
      │               │
      └───────┬───────┘
              │
      generate_summary
              │
            end

Author: ResearcherAI Team
"""

import os
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.utils.dates import days_ago

# Add ResearcherAI to Python path
sys.path.insert(0, os.environ.get('RESEARCHER_AI_HOME', '/opt/airflow/researcher_ai'))


@lru_cache(maxsize=1)
def _get_data_collector_cls():
    from agents.data_agent import DataCollectorAgent  # pylint: disable=import-outside-toplevel

    return DataCollectorAgent


@lru_cache(maxsize=1)
def _get_orchestrator_cls():
    from agents.orchestrator_agent import OrchestratorAgent  # pylint: disable=import-outside-toplevel

    return OrchestratorAgent


def _create_data_collector():
    return _get_data_collector_cls()()


def _build_orchestrator_config_from_variables() -> Dict[str, Dict[str, Any]]:
    """Construct orchestrator configuration overrides from Airflow Variables."""
    config: Dict[str, Dict[str, Any]] = {}

    def _maybe_int(value: str) -> Any:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value

    def _maybe_bool(value: str) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    vector_type = Variable.get("vector_db_type", default_var="").strip().lower()
    if vector_type:
        vector_config: Dict[str, Any] = {"type": vector_type}

        if vector_type == "qdrant":
            host = Variable.get("qdrant_host", default_var="").strip() or None
            port = Variable.get("qdrant_port", default_var="").strip()
            grpc_port = Variable.get("qdrant_grpc_port", default_var="").strip()
            collection = Variable.get("qdrant_collection", default_var="").strip() or None
            api_key = Variable.get("qdrant_api_key", default_var="").strip() or None
            prefer_grpc = Variable.get("qdrant_prefer_grpc", default_var="").strip()
            use_https = Variable.get("qdrant_https", default_var="").strip()

            if host:
                vector_config["host"] = host
            if port:
                vector_config["port"] = _maybe_int(port)
            if grpc_port:
                vector_config["grpc_port"] = _maybe_int(grpc_port)
            if collection:
                vector_config["collection_name"] = collection
            if api_key:
                vector_config["api_key"] = api_key
            if prefer_grpc:
                vector_config["prefer_grpc"] = _maybe_bool(prefer_grpc)
            if use_https:
                vector_config["https"] = _maybe_bool(use_https)

        config["vector_db"] = vector_config

    graph_type = Variable.get("graph_db_type", default_var="").strip().lower()
    if graph_type:
        graph_config: Dict[str, Any] = {"type": graph_type}

        if graph_type == "neo4j":
            uri = Variable.get("neo4j_uri", default_var="").strip() or None
            user = Variable.get("neo4j_user", default_var="").strip() or None
            password = Variable.get("neo4j_password", default_var="").strip() or None
            database = Variable.get("neo4j_database", default_var="").strip() or None
            lifetime = Variable.get("neo4j_max_connection_lifetime", default_var="").strip()

            if uri:
                graph_config["uri"] = uri
            if user:
                graph_config["user"] = user
            if password:
                graph_config["password"] = password
            if database:
                graph_config["database"] = database
            if lifetime:
                graph_config["max_connection_lifetime"] = _maybe_int(lifetime)

        config["graph_db"] = graph_config

    return config

# ============================================================================
# Configuration
# ============================================================================

# DAG default arguments
default_args = {
    'owner': 'researcher_ai',
    'depends_on_past': False,
    'email': ['alerts@researcherai.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1),
}

# Global variables
RESEARCH_QUERY = Variable.get("research_query", default_var="artificial intelligence machine learning")
MAX_PAPERS_PER_SOURCE = int(Variable.get("max_papers_per_source", default_var=10))
MIN_PAPERS_THRESHOLD = int(Variable.get("min_papers_threshold", default_var=10))
SESSION_NAME = Variable.get("session_name", default_var="airflow_default")

# ============================================================================
# Helper Functions
# ============================================================================

def get_orchestrator(**context):
    """Get or create orchestrator instance"""
    overrides = _build_orchestrator_config_from_variables()
    orchestrator = _get_orchestrator_cls()(session_name=SESSION_NAME, config=overrides or None)
    return orchestrator

def store_papers_in_xcom(papers: List[Dict], **context):
    """Store paper count and summary in XCom (not full data)"""
    context['task_instance'].xcom_push(key='paper_count', value=len(papers))
    context['task_instance'].xcom_push(
        key='paper_summary',
        value={
            'total': len(papers),
            'sources': list(set(p.get('source', 'unknown') for p in papers)),
            'timestamp': datetime.now().isoformat()
        }
    )
    return len(papers)

# ============================================================================
# Task Functions: Data Collection
# ============================================================================

def collect_from_arxiv(**context):
    """Collect papers from arXiv"""
    collector = _create_data_collector()
    papers = collector._fetch_arxiv(RESEARCH_QUERY, MAX_PAPERS_PER_SOURCE)
    print(f"✅ arXiv: Collected {len(papers)} papers")
    return papers

def collect_from_semantic_scholar(**context):
    """Collect papers from Semantic Scholar"""
    collector = _create_data_collector()
    papers = collector._fetch_semantic_scholar(RESEARCH_QUERY, MAX_PAPERS_PER_SOURCE)
    print(f"✅ Semantic Scholar: Collected {len(papers)} papers")
    return papers

def collect_from_zenodo(**context):
    """Collect papers from Zenodo"""
    collector = _create_data_collector()
    papers = collector._fetch_zenodo(RESEARCH_QUERY, MAX_PAPERS_PER_SOURCE)
    print(f"✅ Zenodo: Collected {len(papers)} papers")
    return papers

def collect_from_pubmed(**context):
    """Collect papers from PubMed"""
    collector = _create_data_collector()
    papers = collector._fetch_pubmed(RESEARCH_QUERY, MAX_PAPERS_PER_SOURCE)
    print(f"✅ PubMed: Collected {len(papers)} papers")
    return papers

def collect_from_websearch(**context):
    """Collect papers from web search"""
    collector = _create_data_collector()
    papers = collector._fetch_websearch(RESEARCH_QUERY, MAX_PAPERS_PER_SOURCE)
    print(f"✅ WebSearch: Collected {len(papers)} papers")
    return papers

def collect_from_huggingface(**context):
    """Collect datasets from HuggingFace"""
    collector = _create_data_collector()
    papers = collector._fetch_huggingface(RESEARCH_QUERY, MAX_PAPERS_PER_SOURCE)
    print(f"✅ HuggingFace: Collected {len(papers)} papers")
    return papers

def collect_from_kaggle(**context):
    """Collect datasets from Kaggle"""
    collector = _create_data_collector()
    try:
        papers = collector._fetch_kaggle(RESEARCH_QUERY, MAX_PAPERS_PER_SOURCE)
        print(f"✅ Kaggle: Collected {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"⚠️ Kaggle: {e} (credentials may be missing)")
        return []

# ============================================================================
# Task Functions: Data Processing
# ============================================================================

def merge_collected_papers(**context):
    """Merge papers from all sources"""
    ti = context['task_instance']

    # Pull data from all collection tasks
    source_tasks = [
        'collect_papers.arxiv',
        'collect_papers.semantic_scholar',
        'collect_papers.zenodo',
        'collect_papers.pubmed',
        'collect_papers.websearch',
        'collect_papers.huggingface',
        'collect_papers.kaggle',
    ]

    all_papers = []
    for task_id in source_tasks:
        papers = ti.xcom_pull(task_ids=task_id)
        if papers:
            all_papers.extend(papers)

    print(f"📦 Merged {len(all_papers)} total papers from {len(source_tasks)} sources")

    # Store in orchestrator
    orchestrator = get_orchestrator(**context)
    orchestrator.papers = list(all_papers)
    orchestrator.save_session()

    # Push count to XCom for branching
    store_papers_in_xcom(all_papers, **context)

    return len(all_papers)

def check_paper_threshold(**context):
    """Decide whether to process or skip based on paper count"""
    ti = context['task_instance']
    paper_count = ti.xcom_pull(task_ids='merge_papers', key='paper_count')

    print(f"📊 Collected {paper_count} papers (threshold: {MIN_PAPERS_THRESHOLD})")

    if paper_count >= MIN_PAPERS_THRESHOLD:
        print("✅ Threshold met - proceeding with processing")
        return ['process_papers.store_vectors', 'process_papers.extract_triples']
    else:
        print("⚠️ Insufficient papers - skipping processing")
        return 'skip_processing'

def store_in_vector_db(**context):
    """Store papers in vector database (FAISS)"""
    orchestrator = get_orchestrator(**context)

    papers = orchestrator.papers
    print(f"🔢 Storing {len(papers)} papers in vector database...")

    if not papers:
        print("⚠️ No papers available for vector storage")
        return 0

    stats = orchestrator.vector_agent.process_papers(papers)
    stored_count = stats.get('documents_added') or len(papers)

    # Persist FAISS index
    orchestrator.save_session()

    print(f"✅ Stored {stored_count} embeddings in vector database")
    return stored_count

def extract_knowledge_triples(**context):
    """Extract knowledge graph triples from papers"""
    orchestrator = get_orchestrator(**context)

    papers = orchestrator.papers
    print(f"🕸️ Extracting knowledge triples from {len(papers)} papers...")

    if not papers:
        print("⚠️ No papers available for knowledge graph updates")
        return 0

    stats = orchestrator.graph_agent.process_papers(papers[:20])
    orchestrator.save_session()

    extracted_count = stats.get('edges_added', 0)
    print(f"✅ Extracted {extracted_count} knowledge triples")
    return extracted_count

def generate_collection_summary(**context):
    """Generate summary of collection run"""
    ti = context['task_instance']

    paper_count = ti.xcom_pull(task_ids='merge_papers', key='paper_count')
    paper_summary = ti.xcom_pull(task_ids='merge_papers', key='paper_summary')
    vector_count = ti.xcom_pull(task_ids='process_papers.store_vectors')
    triple_count = ti.xcom_pull(task_ids='process_papers.extract_triples')

    summary = {
        'run_date': datetime.now().isoformat(),
        'query': RESEARCH_QUERY,
        'papers_collected': paper_count,
        'sources': paper_summary.get('sources', []) if paper_summary else [],
        'vectors_stored': vector_count or 0,
        'triples_extracted': triple_count or 0,
        'session': SESSION_NAME
    }

    print("=" * 70)
    print("📊 COLLECTION RUN SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key:20s}: {value}")
    print("=" * 70)

    return summary

# ============================================================================
# DAG Definition
# ============================================================================

with DAG(
    dag_id='research_paper_etl',
    default_args=default_args,
    description='Parallel research paper collection and processing pipeline',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    start_date=days_ago(1),
    catchup=False,
    tags=['research', 'etl', 'production'],
    max_active_runs=2,  # Prevent concurrent runs but allow manual overlap
    dagrun_timeout=timedelta(hours=2),  # Max 2 hours per run
) as dag:

    # Start marker
    start = EmptyOperator(task_id='start')

    # Task Group: Parallel Data Collection
    with TaskGroup('collect_papers', tooltip='Collect papers from all sources in parallel') as collect_group:

        arxiv_task = PythonOperator(
            task_id='arxiv',
            python_callable=collect_from_arxiv,
            execution_timeout=timedelta(minutes=10),
        )

        semantic_task = PythonOperator(
            task_id='semantic_scholar',
            python_callable=collect_from_semantic_scholar,
            execution_timeout=timedelta(minutes=10),
        )

        zenodo_task = PythonOperator(
            task_id='zenodo',
            python_callable=collect_from_zenodo,
            execution_timeout=timedelta(minutes=10),
        )

        pubmed_task = PythonOperator(
            task_id='pubmed',
            python_callable=collect_from_pubmed,
            execution_timeout=timedelta(minutes=10),
        )

        websearch_task = PythonOperator(
            task_id='websearch',
            python_callable=collect_from_websearch,
            execution_timeout=timedelta(minutes=10),
        )

        huggingface_task = PythonOperator(
            task_id='huggingface',
            python_callable=collect_from_huggingface,
            execution_timeout=timedelta(minutes=10),
        )

        kaggle_task = PythonOperator(
            task_id='kaggle',
            python_callable=collect_from_kaggle,
            execution_timeout=timedelta(minutes=10),
        )

    # Merge results
    merge_task = PythonOperator(
        task_id='merge_papers',
        python_callable=merge_collected_papers,
    )

    # Decision point: Check if enough papers collected
    branch_task = BranchPythonOperator(
        task_id='check_threshold',
        python_callable=check_paper_threshold,
    )

    # Task Group: Process Papers (if threshold met)
    with TaskGroup('process_papers', tooltip='Process collected papers') as process_group:

        vectors_task = PythonOperator(
            task_id='store_vectors',
            python_callable=store_in_vector_db,
            execution_timeout=timedelta(minutes=30),
        )

        triples_task = PythonOperator(
            task_id='extract_triples',
            python_callable=extract_knowledge_triples,
            execution_timeout=timedelta(minutes=30),
        )

        # Parallel processing
        [vectors_task, triples_task]

    # Skip processing if threshold not met
    skip_task = EmptyOperator(
        task_id='skip_processing',
        trigger_rule='none_failed',
    )

    # Generate summary
    summary_task = PythonOperator(
        task_id='generate_summary',
        python_callable=generate_collection_summary,
        trigger_rule='none_failed',  # Run even if processing skipped
    )

    # End marker
    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed',
    )

    # Define workflow
    start >> collect_group >> merge_task >> branch_task
    branch_task >> [process_group, skip_task] >> summary_task >> end
