"""
ResearcherAI System Monitoring DAG
===================================

Monitors system health, data quality, and sends alerts.

DAG Structure:
    start
      │
      ├──────┬──────┬──────┬──────┐
      │      │      │      │      │
  check   check  check  check  check
  api    vector  graph  disk   memory
  health  db     db     space  usage
      │      │      │      │      │
      └──────┴──────┴──────┴──────┘
                     │
              aggregate_metrics
                     │
           evaluate_health_status
                     │
        ┌─────────┴─────────┐
        │                   │
   (issues found)      (all healthy)
        │                   │
   send_alerts         log_success
        │                   │
        └─────────┬─────────┘
                  │
                end

Author: ResearcherAI Team
"""

import os
import sys
import psutil
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.dates import days_ago

# Add ResearcherAI to Python path
sys.path.insert(0, os.environ.get('RESEARCHER_AI_HOME', '/opt/airflow/researcher_ai'))

from agents.orchestrator_agent import OrchestratorAgent

# ============================================================================
# Configuration
# ============================================================================

default_args = {
    'owner': 'researcher_ai',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Health check thresholds
THRESHOLDS = {
    'disk_usage_percent': 90,
    'memory_usage_percent': 95,
    'min_vector_embeddings': 10,
    'min_graph_nodes': 10,
}

# ============================================================================
# Health Check Functions
# ============================================================================

def check_api_health(**context):
    """Check if ResearcherAI API is responsive"""
    import requests

    try:
        response = requests.get('http://localhost:8000/v1/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'status': 'healthy',
                'response_time': response.elapsed.total_seconds(),
                'details': data
            }
        else:
            return {
                'status': 'unhealthy',
                'error': f'HTTP {response.status_code}',
                'response_time': response.elapsed.total_seconds()
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def check_vector_db_health(**context):
    """Check FAISS vector database health"""
    try:
        orchestrator = OrchestratorAgent(session_name='airflow_default')
        vector_agent = orchestrator.vector_agent

        stats = vector_agent.get_stats()
        embedding_count = stats.get('total_embeddings', 0)

        health_status = {
            'status': 'healthy' if embedding_count >= THRESHOLDS['min_vector_embeddings'] else 'warning',
            'embedding_count': embedding_count,
            'dimension': stats.get('dimension', 384),
        }

        if embedding_count < THRESHOLDS['min_vector_embeddings']:
            health_status['warning'] = f"Only {embedding_count} embeddings (threshold: {THRESHOLDS['min_vector_embeddings']})"

        return health_status
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def check_graph_db_health(**context):
    """Check knowledge graph database health"""
    try:
        orchestrator = OrchestratorAgent(session_name='airflow_default')
        graph_agent = orchestrator.graph_agent

        stats = graph_agent.get_stats()
        node_count = stats.get('nodes', 0)
        edge_count = stats.get('edges', 0)

        health_status = {
            'status': 'healthy' if node_count >= THRESHOLDS['min_graph_nodes'] else 'warning',
            'node_count': node_count,
            'edge_count': edge_count,
        }

        if node_count < THRESHOLDS['min_graph_nodes']:
            health_status['warning'] = f"Only {node_count} nodes (threshold: {THRESHOLDS['min_graph_nodes']})"

        return health_status
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def check_disk_space(**context):
    """Check available disk space"""
    try:
        # Check ResearcherAI volumes directory
        volumes_path = '/opt/airflow/researcher_ai/volumes'
        disk_usage = shutil.disk_usage(volumes_path)

        total_gb = disk_usage.total / (1024**3)
        used_gb = disk_usage.used / (1024**3)
        free_gb = disk_usage.free / (1024**3)
        percent_used = (disk_usage.used / disk_usage.total) * 100

        health_status = {
            'status': 'healthy' if percent_used < THRESHOLDS['disk_usage_percent'] else 'critical',
            'total_gb': round(total_gb, 2),
            'used_gb': round(used_gb, 2),
            'free_gb': round(free_gb, 2),
            'percent_used': round(percent_used, 2),
        }

        if percent_used >= THRESHOLDS['disk_usage_percent']:
            health_status['error'] = f"Disk usage at {percent_used:.1f}% (threshold: {THRESHOLDS['disk_usage_percent']}%)"

        return health_status
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def check_memory_usage(**context):
    """Check system memory usage"""
    try:
        memory = psutil.virtual_memory()

        health_status = {
            'status': 'healthy' if memory.percent < THRESHOLDS['memory_usage_percent'] else 'critical',
            'total_gb': round(memory.total / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': round(memory.percent, 2),
        }

        if memory.percent >= THRESHOLDS['memory_usage_percent']:
            health_status['error'] = f"Memory usage at {memory.percent:.1f}% (threshold: {THRESHOLDS['memory_usage_percent']}%)"

        return health_status
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

# ============================================================================
# Aggregation and Decision Functions
# ============================================================================

def aggregate_health_metrics(**context):
    """Aggregate all health check results"""
    ti = context['task_instance']

    # Pull results from all health checks
    api_health = ti.xcom_pull(task_ids='health_checks.check_api')
    vector_health = ti.xcom_pull(task_ids='health_checks.check_vector_db')
    graph_health = ti.xcom_pull(task_ids='health_checks.check_graph_db')
    disk_health = ti.xcom_pull(task_ids='health_checks.check_disk')
    memory_health = ti.xcom_pull(task_ids='health_checks.check_memory')

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'api': api_health,
        'vector_db': vector_health,
        'graph_db': graph_health,
        'disk': disk_health,
        'memory': memory_health,
    }

    # Count issues
    issues = []
    for component, health in metrics.items():
        if component == 'timestamp':
            continue
        if isinstance(health, dict):
            if health.get('status') in ['error', 'critical']:
                issues.append(f"{component}: {health.get('error', 'unknown error')}")
            elif health.get('status') == 'warning':
                issues.append(f"{component}: {health.get('warning', 'warning')}")

    metrics['issues'] = issues
    metrics['overall_status'] = 'healthy' if len(issues) == 0 else ('warning' if any('warning' in i for i in issues) else 'critical')

    # Store in XCom
    ti.xcom_push(key='health_metrics', value=metrics)
    ti.xcom_push(key='issue_count', value=len(issues))

    print("=" * 70)
    print("🏥 SYSTEM HEALTH CHECK")
    print("=" * 70)
    print(f"Overall Status: {metrics['overall_status'].upper()}")
    print(f"Issues Found: {len(issues)}")
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ All systems healthy")
    print("=" * 70)

    return metrics

def evaluate_health_status(**context):
    """Decide whether to send alerts or log success"""
    ti = context['task_instance']
    issue_count = ti.xcom_pull(task_ids='aggregate_metrics', key='issue_count')

    if issue_count > 0:
        print(f"⚠️ {issue_count} issue(s) found - sending alerts")
        return 'send_alerts'
    else:
        print("✅ All systems healthy - logging success")
        return 'log_success'

def send_health_alerts(**context):
    """Send alerts for health issues"""
    ti = context['task_instance']
    metrics = ti.xcom_pull(task_ids='aggregate_metrics', key='health_metrics')

    issues = metrics.get('issues', [])

    # In production, you would send to Slack/email/PagerDuty
    print("🚨 SENDING HEALTH ALERTS")
    print("=" * 70)
    print(f"Timestamp: {metrics['timestamp']}")
    print(f"Overall Status: {metrics['overall_status'].upper()}")
    print(f"\nIssues ({len(issues)}):")
    for issue in issues:
        print(f"  • {issue}")
    print("=" * 70)

    # TODO: Implement actual alerting
    # - Send Slack webhook
    # - Send email via SMTP
    # - Create PagerDuty incident

    return {
        'alerts_sent': len(issues),
        'timestamp': datetime.now().isoformat()
    }

def log_health_success(**context):
    """Log successful health check"""
    ti = context['task_instance']
    metrics = ti.xcom_pull(task_ids='aggregate_metrics', key='health_metrics')

    print("✅ SYSTEM HEALTH CHECK PASSED")
    print("=" * 70)
    print(f"Timestamp: {metrics['timestamp']}")
    print("All components healthy:")
    print(f"  • API: {metrics['api']['status']}")
    print(f"  • Vector DB: {metrics['vector_db']['embedding_count']} embeddings")
    print(f"  • Graph DB: {metrics['graph_db']['node_count']} nodes")
    print(f"  • Disk: {metrics['disk']['free_gb']:.2f} GB free")
    print(f"  • Memory: {metrics['memory']['available_gb']:.2f} GB available")
    print("=" * 70)

    return {'status': 'success', 'timestamp': datetime.now().isoformat()}

# ============================================================================
# DAG Definition
# ============================================================================

with DAG(
    dag_id='system_monitoring',
    default_args=default_args,
    description='Monitor ResearcherAI system health and send alerts',
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'health', 'alerts'],
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
) as dag:

    start = EmptyOperator(task_id='start')

    # Task Group: Health Checks (parallel)
    with TaskGroup('health_checks', tooltip='Run all health checks in parallel') as health_group:

        api_check = PythonOperator(
            task_id='check_api',
            python_callable=check_api_health,
            execution_timeout=timedelta(minutes=2),
        )

        vector_check = PythonOperator(
            task_id='check_vector_db',
            python_callable=check_vector_db_health,
            execution_timeout=timedelta(minutes=2),
        )

        graph_check = PythonOperator(
            task_id='check_graph_db',
            python_callable=check_graph_db_health,
            execution_timeout=timedelta(minutes=2),
        )

        disk_check = PythonOperator(
            task_id='check_disk',
            python_callable=check_disk_space,
            execution_timeout=timedelta(minutes=1),
        )

        memory_check = PythonOperator(
            task_id='check_memory',
            python_callable=check_memory_usage,
            execution_timeout=timedelta(minutes=1),
        )

    # Aggregate metrics
    aggregate = PythonOperator(
        task_id='aggregate_metrics',
        python_callable=aggregate_health_metrics,
    )

    # Decision: Alert or Success
    branch = BranchPythonOperator(
        task_id='evaluate_status',
        python_callable=evaluate_health_status,
    )

    # Alert path
    alert = PythonOperator(
        task_id='send_alerts',
        python_callable=send_health_alerts,
    )

    # Success path
    success = PythonOperator(
        task_id='log_success',
        python_callable=log_health_success,
    )

    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed',
    )

    # Define workflow
    start >> health_group >> aggregate >> branch
    branch >> [alert, success] >> end
