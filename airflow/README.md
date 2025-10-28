# Apache Airflow for ResearcherAI

Production-grade ETL pipeline orchestration for automated research paper collection and processing.

## 🎯 What This Does

Replaces the simple scheduler with Apache Airflow to provide:
- **Parallel data collection** from 7 sources (3-4x faster)
- **Visual workflow monitoring** via web UI
- **Automatic retries** with exponential backoff
- **Health monitoring** and alerting
- **DAG-based workflows** with task dependencies
- **Scalable execution** with Celery workers

## 📁 Directory Structure

```
airflow/
├── docker-compose.yml      # Infrastructure setup
├── .env                     # Environment configuration
├── setup_airflow.sh         # Setup and start script
├── requirements.txt         # Python dependencies
├── dags/                    # DAG definitions
│   ├── research_paper_etl.py      # Main ETL pipeline
│   └── system_monitoring.py       # Health checks
├── logs/                    # Task execution logs
├── plugins/                 # Custom operators (optional)
└── config/                  # Airflow config overrides
```

## 🚀 Quick Start

```bash
# 1. Navigate to airflow directory
cd airflow

# 2. Run setup script
./setup_airflow.sh

# 3. Access UI
# Airflow UI: http://localhost:8080 (airflow/airflow)
# Flower UI: http://localhost:5555
```

## 📊 DAGs Included

### 1. research_paper_etl
**Schedule**: Every 6 hours
**Purpose**: Collect research papers from multiple sources in parallel

**Workflow**:
```
collect (parallel) → merge → check_threshold → process → summary
```

**Tasks**:
- Parallel collection from 7 sources
- Merge and deduplicate papers
- Threshold check (min 10 papers)
- Store in vector DB (FAISS)
- Extract knowledge triples
- Generate run summary

### 2. system_monitoring
**Schedule**: Every 30 minutes
**Purpose**: Monitor system health and alert on issues

**Checks**:
- API health
- Vector DB status
- Graph DB status
- Disk space
- Memory usage

## 🔧 Configuration

### Environment Variables (.env)
```bash
GOOGLE_API_KEY=your-api-key
RESEARCHER_AI_SESSION=airflow_default
RESEARCHER_AI_MAX_PAPERS_PER_SOURCE=10
```

### Airflow Variables (via UI)
- `research_query`: Search query for paper collection
- `max_papers_per_source`: Papers per source
- `min_papers_threshold`: Minimum for processing
- `session_name`: ResearcherAI session name

## 📈 Performance Comparison

| Metric | Before (Sequential) | After (Airflow) | Improvement |
|--------|-------------------|-----------------|-------------|
| Collection Time | 19-38s | 5-10s | **3-4x faster** |
| Retry Logic | None | 3 retries | **Robust** |
| Monitoring | Logs only | Full UI | **Visual** |
| Parallelism | Sequential | 7 parallel | **7x concurrent** |
| Error Recovery | Manual | Automatic | **Resilient** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Airflow Ecosystem                 │
│                                             │
│  ┌──────────┐                              │
│  │ Web UI   │ (localhost:8080)             │
│  └────┬─────┘                              │
│       │                                     │
│  ┌────▼──────────┐                         │
│  │  PostgreSQL   │  Metadata DB            │
│  └────┬──────────┘                         │
│       │                                     │
│  ┌────▼──────────┐                         │
│  │  Scheduler    │  Triggers tasks         │
│  └────┬──────────┘                         │
│       │                                     │
│  ┌────▼──────────┐                         │
│  │  Redis Queue  │  Task queue             │
│  └────┬──────────┘                         │
│       │                                     │
│  ┌────▼──────────┐                         │
│  │ Workers (3x)  │  Execute tasks          │
│  └───────────────┘                         │
│                                             │
└─────────────────────────────────────────────┘
```

## 🛠️ Common Operations

### Start Services
```bash
./setup_airflow.sh
```

### Stop Services
```bash
docker compose down
```

### View Logs
```bash
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-worker
```

### Trigger DAG Manually
```bash
docker compose exec airflow-webserver airflow dags trigger research_paper_etl
```

### Access Airflow CLI
```bash
docker compose exec airflow-webserver bash
```

## 🔍 Monitoring

### Airflow UI (Port 8080)
- View DAG runs and task status
- Check logs and execution times
- Monitor historical performance
- Manage variables and connections

### Flower UI (Port 5555)
- Monitor Celery workers
- View task queue
- Check worker performance
- Track task execution

## 🐛 Troubleshooting

### DAGs Not Appearing
```bash
# Check for syntax errors
docker compose exec airflow-webserver python /opt/airflow/dags/research_paper_etl.py

# Check scheduler logs
docker compose logs airflow-scheduler
```

### Tasks Failing
1. Click on failed task in UI
2. View logs
3. Check for API timeouts or import errors
4. Adjust `execution_timeout` if needed

### Reset Everything
```bash
docker compose down -v
./setup_airflow.sh
```

## 📚 Documentation

- **Concepts**: `../AIRFLOW_COMPLETE_GUIDE.md`
- **Usage**: `../AIRFLOW_USAGE_GUIDE.md`
- **Official Docs**: https://airflow.apache.org/docs/

## 🎓 Next Steps

1. **Customize queries**: Edit Airflow Variables
2. **Add sources**: Extend `research_paper_etl.py`
3. **Configure alerts**: Set up email/Slack notifications
4. **Scale workers**: Adjust replicas in `docker-compose.yml`
5. **Monitor performance**: Use Gantt charts in UI

## 📊 System Requirements

- Docker: 20.10+
- Docker Compose: 2.0+
- RAM: 4GB minimum (8GB recommended)
- Disk: 10GB free space
- CPU: 2+ cores

## 🚀 Production Checklist

- [ ] Set strong passwords in `.env`
- [ ] Configure email alerts (SMTP)
- [ ] Set up external database (not SQLite)
- [ ] Configure persistent volumes
- [ ] Enable HTTPS for web UI
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backup strategy
- [ ] Set resource limits in Docker

## 🆘 Support

- Issues: https://github.com/Sebuliba-Adrian/ResearcherAI/issues
- Airflow Community: https://airflow.apache.org/community/

---

**Powered by Apache Airflow 2.10.2** | **Built for ResearcherAI**
