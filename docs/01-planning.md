---
layout: default
title: Planning & Requirements
---

# Planning & Requirements

<div class="glass-card">
Every great project starts with a problem worth solving. Let me take you back to where this all began.
</div>

## The Problem

It was late October 2024, and I was drowning in research papers. I was trying to stay current with advances in transformer architectures, RAG systems, and multi-agent frameworks - but new papers were being published faster than I could read them.

I'd spend hours:
- Searching across arXiv, Semantic Scholar, PubMed
- Copying paper titles and abstracts into notes
- Trying to remember which paper mentioned what concept
- Re-reading papers because I forgot their key insights

**There had to be a better way.**

## The Vision

I imagined a research assistant that could:

1. **Automatically collect** papers from multiple sources based on my interests
2. **Build a knowledge graph** showing how papers, authors, and concepts relate
3. **Answer my questions** by synthesizing information across papers
4. **Remember our conversations** so I don't have to repeat context
5. **Scale** from my laptop to production without rewriting code

But I didn't want to build just another prototype. I wanted something that demonstrated **production-grade patterns** I could use in real applications.

## Core Requirements

I broke this down into functional and non-functional requirements.

<div class="glass-card">

### Functional Requirements

**Data Collection**
- ✅ Support multiple data sources (academic databases, web search)
- ✅ Automatic deduplication of papers
- ✅ Scheduled/automated collection in background
- ✅ Rate limiting and error handling

**Knowledge Organization**
- ✅ Knowledge graph with entities (papers, authors, topics)
- ✅ Relationships (authored, cites, is_about)
- ✅ Vector embeddings for semantic search
- ✅ Graph visualization

**Query & Reasoning**
- ✅ Natural language question answering
- ✅ Multi-hop reasoning across papers
- ✅ Source citation with paper references
- ✅ Conversation memory (5-turn history)

**Session Management**
- ✅ Multiple research sessions
- ✅ Save/load session state
- ✅ Session statistics and metadata

**User Interface**
- ✅ Modern, responsive web interface
- ✅ Data collection controls
- ✅ Interactive graph visualization
- ✅ Chat interface for queries

</div>

<div class="glass-card">

### Non-Functional Requirements

**Performance**
- Data collection: < 2 minutes for 10+ papers
- Query answering: < 5 seconds
- Graph queries: < 100ms
- Vector search: < 100ms

**Reliability**
- 99% uptime for production deployment
- Graceful degradation if services unavailable
- Automatic retries with exponential backoff
- Circuit breakers for external APIs

**Scalability**
- Support 1000+ papers in knowledge base
- Handle concurrent users in production
- Horizontal scaling with Kafka
- Efficient caching to reduce costs

**Maintainability**
- 90%+ test coverage
- Type safety with TypeScript/Pydantic
- Clear separation of concerns
- Comprehensive documentation

**Cost Efficiency**
- Intelligent model selection (use cheapest model for each task)
- Caching to avoid redundant API calls
- Token budgets to prevent runaway costs
- Target: < $10/month for moderate usage

</div>

## Choosing the Tech Stack

This was one of the most important decisions. I needed technologies that were:
- **Mature** enough for production
- **Well-documented** so I could move fast
- **Composable** so I could swap components
- **Cost-effective** to run

Here's how I evaluated each component:

### LLM Provider

**Candidates**: OpenAI GPT-4, Anthropic Claude, Google Gemini

**Winner**: **Google Gemini 2.0 Flash**

**Why?**
- ✅ Fast response times (< 2s average)
- ✅ Cost-effective ($0.35 per 1M tokens)
- ✅ Large context window (1M tokens)
- ✅ Good reasoning capabilities
- ✅ Free tier for development

I experimented with all three, and Gemini gave the best balance of speed, cost, and quality for research Q&A.

### Orchestration Framework

**Candidates**: LangChain, LangGraph, Custom

**Winner**: **LangGraph**

**Why?**
- ✅ Built for multi-agent workflows
- ✅ Excellent state management
- ✅ Visual workflow debugging
- ✅ Works seamlessly with LlamaIndex
- ✅ Good documentation and examples

LangChain was too linear for my multi-agent pattern. LangGraph gave me the graph-based orchestration I needed.

### RAG Framework

**Candidates**: LlamaIndex, Haystack, Custom

**Winner**: **LlamaIndex**

**Why?**
- ✅ Best-in-class retrieval strategies
- ✅ Flexible architecture
- ✅ Great integration with vector DBs
- ✅ Built-in evaluation tools
- ✅ Active community

LlamaIndex saved me weeks of work on chunking strategies, embedding management, and retrieval optimization.

### Knowledge Graph

**Candidates**: Neo4j, NetworkX, TigerGraph

**Winner**: **Both Neo4j AND NetworkX** (dual backend)

**Why?**
- ✅ Neo4j for production (persistent, scalable, visual)
- ✅ NetworkX for development (fast startup, no infrastructure)
- ✅ Same API for both (abstraction layer)
- ✅ Easy switching via environment variables

This was a game-changer. I could develop and test on my laptop with NetworkX, then deploy to production with Neo4j without changing code.

### Vector Database

**Candidates**: Pinecone, Weaviate, Qdrant, FAISS

**Winner**: **Both Qdrant AND FAISS** (dual backend)

**Why?**
- ✅ Qdrant for production (persistent, REST API, dashboard)
- ✅ FAISS for development (in-memory, no setup)
- ✅ Same abstraction layer
- ✅ Cost: $0 (self-hosted Qdrant)

Again, dual backends gave me the flexibility to move fast in development and scale in production.

### Event Streaming

**Candidates**: Kafka, RabbitMQ, Redis Streams

**Winner**: **Kafka** (optional)

**Why?**
- ✅ Industry standard for event-driven systems
- ✅ Event persistence and replay
- ✅ Horizontal scaling with consumer groups
- ✅ Rich ecosystem (Kafka UI, connectors)
- ✅ Optional: falls back to sync mode if unavailable

I made Kafka optional because it's overkill for development but essential for production scalability.

### ETL Orchestration

**Candidates**: Airflow, Prefect, Dagster

**Winner**: **Apache Airflow**

**Why?**
- ✅ Industry standard for data pipelines
- ✅ Visual DAG editor and monitoring
- ✅ Automatic retries and error handling
- ✅ Scalable with Celery workers
- ✅ Rich integrations

Airflow gave me 3-4x faster data collection through parallel execution and automatic retries.

### Frontend

**Candidates**: Next.js, Vite+React, SvelteKit

**Winner**: **Vite + React + TypeScript**

**Why?**
- ✅ Lightning fast dev server (< 1s HMR)
- ✅ React ecosystem and component libraries
- ✅ TypeScript for type safety
- ✅ Lightweight (no SSR overhead)
- ✅ Easy deployment

I didn't need SSR for this app, so Vite's simplicity and speed won.

## Architecture Philosophy

I made some key architectural decisions early on:

<div class="glass-card">

### 1. Dual-Backend Strategy

**Problem**: Setting up Neo4j, Qdrant, and Kafka for development is slow and resource-heavy.

**Solution**: Abstract backends behind interfaces, provide in-memory alternatives.

**Benefits**:
- ⚡ Instant startup in development (0s vs 30s)
- 🧪 Faster test suite (no Docker overhead)
- 💰 Lower cloud costs (single container vs 7)
- 🔄 Easy switching via env vars

</div>

<div class="glass-card">

### 2. Multi-Agent Pattern

**Problem**: A single monolithic agent becomes complex and hard to test.

**Solution**: Separate concerns into specialized agents coordinated by an orchestrator.

**Benefits**:
- 🧩 Clear separation of concerns
- 🧪 Easier unit testing
- 🔄 Can replace individual agents
- 📈 Can scale agents independently

</div>

<div class="glass-card">

### 3. Event-Driven Communication

**Problem**: Synchronous agent calls create tight coupling and bottlenecks.

**Solution**: Agents publish events to Kafka; consumers process asynchronously.

**Benefits**:
- ⚡ Parallel processing (3x faster)
- 🔌 Loose coupling
- 🔄 Event replay for debugging
- 📈 Horizontal scaling

</div>

<div class="glass-card">

### 4. Production-Grade Patterns

From day one, I implemented patterns that would matter at scale:

**Circuit Breakers**: Prevent cascade failures when APIs go down
```python
@circuit_breaker(failure_threshold=5, recovery_timeout=60)
def call_external_api():
    ...
```

**Token Budgets**: Prevent runaway LLM costs
```python
@token_budget(per_request=10000, per_user=100000)
def generate_answer():
    ...
```

**Intelligent Caching**: 40% cost reduction with dual-tier cache
```python
@cache(ttl=3600, strategy="dual-tier")
def expensive_operation():
    ...
```

**Dynamic Model Selection**: Use cheapest model that meets requirements
```python
model = select_model(task_type="summarization", max_latency=2.0)
```

</div>

## The Plan

With requirements and architecture decided, I created a development plan:

**Phase 1: Core Agents (Week 1)**
- [x] Set up project structure
- [x] Implement DataCollectorAgent with 3 sources
- [x] Implement KnowledgeGraphAgent with NetworkX
- [x] Implement VectorAgent with FAISS
- [x] Implement ReasoningAgent with Gemini
- [x] Basic OrchestratorAgent

**Phase 2: Production Features (Week 2)**
- [x] Add Neo4j backend for graphs
- [x] Add Qdrant backend for vectors
- [x] Implement Kafka event system
- [x] Add 4 more data sources
- [x] Implement SchedulerAgent
- [x] Session management and persistence
- [x] Apache Airflow integration

**Phase 3: Frontend & Testing (Week 3)**
- [x] React frontend with glassmorphism design
- [x] 7 pages (Home, Collect, Ask, Graph, Vector, Upload, Sessions)
- [x] Comprehensive test suite (90%+ coverage)
- [x] GitHub Actions CI/CD
- [x] Docker containerization
- [x] Documentation

## Lessons from Planning

Looking back, here's what I learned:

**✅ What Worked**

1. **Dual-backend strategy was brilliant** - Saved hours of dev time
2. **Starting with requirements** - Kept me focused
3. **Choosing mature tech** - Less debugging, more building
4. **Production patterns from day 1** - No painful refactoring later

**🤔 What I'd Change**

1. **Should have added Airflow earlier** - Parallel collection is much faster
2. **Could have started with fewer data sources** - 3 would have been enough to validate
3. **Frontend design took longer than expected** - Glassmorphism is tricky to get right

**💡 Key Insights**

> The best architecture is one that lets you move fast in development and scale in production without rewriting code.

> Abstractions are worth the upfront cost when they give you optionality.

> Production patterns implemented early save painful refactoring later.

## Ready for Architecture?

Now that you understand the "why" behind ResearcherAI, let's dive into the "how". In the next section, I'll walk you through the system architecture and how all these pieces fit together.

<div class="progress-nav">
  <a href="index.html">← Back to Home</a>
  <a href="02-architecture.html">Next: Architecture Design →</a>
</div>
