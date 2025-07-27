# ResearcherAI - Production Multi-Agent RAG System v2.0

**Containerized multi-agent research system with conversation memory, multi-session support, and autonomous data collection.**

---

## Features

### Multi-Agent Architecture
- **DataCollectorAgent**: Autonomous collection from 7 sources
- **KnowledgeGraphAgent**: Neo4j/NetworkX graph database
- **VectorAgent**: Qdrant/FAISS vector search
- **ReasoningAgent**: Complex reasoning with conversation memory
- **OrchestratorAgent**: Multi-agent coordination & sessions
- **SchedulerAgent**: Automated background collection

### Data Sources (7)
1. arXiv - Recent AI research papers
2. Semantic Scholar - Academic papers with citations
3. Zenodo - Research datasets
4. PubMed - Biomedical literature
5. Web Search - DuckDuckGo results
6. HuggingFace - Models and datasets
7. Kaggle - Datasets and competitions

### Production Features
- **Dual Database Support**: Neo4j + Qdrant (prod) or NetworkX + FAISS (dev)
- **Conversation Memory**: Maintains context across questions
- **Multi-Session Management**: Independent research threads
- **Session Persistence**: Save/load across restarts
- **Automated Collection**: Scheduled background data gathering
- **Interactive Visualization**: PyVis knowledge graph HTML

---

## Quick Start

### Using Docker (Recommended for Production)

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start all services
docker-compose up -d

# 3. View logs
docker-compose logs -f rag-app

# 4. Access the system
docker-compose exec rag-app python main.py
```

### Local Development

```bash
# 1. Clone and setup
cd ResearcherAI
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Add your GOOGLE_API_KEY

# 4. Run the system
python main.py [session_name]
```

---

## Usage

### Interactive Commands

```
help                  # Show all commands

# Research
<question>            # Ask any question
collect [query]       # Collect papers

# Session Management
sessions              # List all sessions
switch <name>         # Switch to session
new <name>            # Create new session

# Information
stats                 # System statistics
memory                # Conversation history
graph                 # Visualize knowledge graph

# Scheduler
schedule start        # Start automated collection
schedule stop         # Stop scheduler
schedule now          # Run immediately
schedule status       # Show scheduler info

# System
save                  # Save current session
quit                  # Exit (auto-saves)
```

---

## Project Structure

```
ResearcherAI/
├── agents/
│   ├── __init__.py              # Package exports
│   ├── data_agent.py            # DataCollectorAgent (7 sources)
│   ├── graph_agent.py           # KnowledgeGraphAgent (Neo4j/NetworkX)
│   ├── vector_agent.py          # VectorAgent (Qdrant/FAISS)
│   ├── reasoner_agent.py        # ReasoningAgent (memory)
│   ├── orchestrator_agent.py    # OrchestratorAgent (coordination)
│   └── scheduler_agent.py       # SchedulerAgent (automation)
│
├── config/
│   └── settings.yaml            # System configuration
│
├── volumes/
│   ├── neo4j/                   # Graph DB data
│   ├── qdrant/                  # Vector DB data
│   ├── sessions/                # Session files
│   ├── cache/                   # Downloaded papers
│   └── logs/                    # Application logs
│
├── main.py                      # CLI entry point
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-service orchestration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

---

## Verified Features

All features have been tested and verified working:

- ✅ **System Initialization** - All 6 agents initialize correctly
- ✅ **Data Collection** - All 7 sources working (18+ papers collected in tests)
- ✅ **Knowledge Graph** - Neo4j/NetworkX with triple extraction
- ✅ **Vector Search** - Qdrant/FAISS with semantic similarity
- ✅ **Conversation Memory** - Context preserved across 3+ turns
- ✅ **Multi-Session Support** - Independent sessions with switching
- ✅ **Persistence** - Save/load across restarts (tested with 9KB state)
- ✅ **Orchestration** - All agents coordinate correctly

See [COMPLETE_SYSTEM_VERIFIED.md](COMPLETE_SYSTEM_VERIFIED.md) for detailed test results.

---

## Configuration

Configuration is managed through [config/settings.yaml](config/settings.yaml) and environment variables in `.env`.

Key settings:
- Data source toggles
- Database connection strings (Neo4j/Qdrant)
- Agent parameters (conversation memory, chunking)
- Scheduler settings

---

## Development vs Production

### Development Mode
- Uses NetworkX + FAISS
- No external dependencies
- Fast setup for testing

### Production Mode
- Uses Neo4j + Qdrant
- Scalable to billions of nodes/vectors
- Distributed clustering support

---

## Troubleshooting

**No module named 'agents'**: Run from project root
**GOOGLE_API_KEY not found**: Create `.env` with your API key
**Neo4j connection failed**: Check docker-compose or switch to NetworkX
**Qdrant connection failed**: Check docker-compose or switch to FAISS

---

## Built with

- Google Gemini 2.0 Flash
- Neo4j & Qdrant
- NetworkX & FAISS
- Sentence Transformers
- Docker & Python 3.11

---

For detailed documentation, see the verification report: [COMPLETE_SYSTEM_VERIFIED.md](COMPLETE_SYSTEM_VERIFIED.md)
