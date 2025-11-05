# LangGraph & LlamaIndex Integration Guide

## Overview

ResearcherAI now includes advanced agent orchestration and RAG capabilities through LangGraph and LlamaIndex integrations. These optional enhancements work seamlessly alongside the existing multi-agent architecture.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [LangGraph Orchestrator](#langgraph-orchestrator)
- [LlamaIndex RAG System](#llamaindex-rag-system)
- [Usage Examples](#usage-examples)
- [Integration with Existing System](#integration-with-existing-system)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Features

### LangGraph Integration
- **Stateful Workflows**: Graph-based agent orchestration with state persistence
- **Checkpoint System**: Save and resume agent workflows across sessions
- **Conditional Routing**: Dynamic workflow paths based on agent outputs
- **Error Recovery**: Built-in error handling and retry logic
- **Workflow Visualization**: ASCII representation of agent execution graphs

### LlamaIndex Integration
- **Advanced RAG**: Enhanced retrieval-augmented generation capabilities
- **Hybrid Search**: Combined vector and keyword search
- **Document Indexing**: Automatic chunking and metadata extraction
- **Query Optimization**: Intelligent query rewriting and response synthesis
- **Multiple Backends**: Support for Qdrant (production) and in-memory (development)

## Installation

### Prerequisites

Ensure you have Python 3.11+ and a virtual environment set up:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

All required packages are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages installed:
- `langgraph==0.2.45` - Graph-based agent orchestration
- `langchain-core==0.3.15` - Core LangChain utilities
- `llama-index==0.12.0` - Advanced RAG framework
- `llama-index-embeddings-huggingface==0.4.0` - HuggingFace embeddings
- `llama-index-vector-stores-qdrant==0.4.0` - Qdrant integration

## LangGraph Orchestrator

### Overview

The LangGraph orchestrator provides stateful, graph-based workflow management for multi-agent coordination.

### Architecture

```
┌─────────────────┐
│ Data Collection │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Graph Processing│
└────────┬────────┘
         │
         v
┌─────────────────┐
│Vector Processing│
└────────┬────────┘
         │
         v
┌─────────────────┐
│    Reasoning    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Critic Review  │────┐
└────────┬────────┘    │
         │             │
         v             │
  ┌─────────┐          │
  │   End   │    ←─────┘
  └─────────┘   (retry if needed)
```

### Basic Usage

```python
from agents.langgraph_orchestrator import create_orchestrator

# Create orchestrator
orchestrator = create_orchestrator()

# Run a workflow
result = orchestrator.run_workflow(
    query="What are the latest advances in RAG systems?",
    thread_id="session_123"
)

# Access results
print(f"Answer: {result['reasoning_result']['answer']}")
print(f"Steps: {result['messages']}")

# Retrieve saved state
saved_state = orchestrator.get_state("session_123")
```

### Advanced Configuration

```python
from agents.langgraph_orchestrator import LangGraphOrchestrator

# Custom configuration
config = {
    "max_retries": 2,
    "timeout": 300,
    "enable_checkpoints": True
}

orchestrator = LangGraphOrchestrator(config)
```

### Workflow Features

**State Persistence**
- Workflows are checkpointed at each node
- Resume from any point using thread IDs
- Multi-session support

**Conditional Logic**
- Critic review determines workflow continuation
- Automatic retry on quality issues
- Configurable retry limits

**Error Handling**
- Graceful error recovery
- Error state tracking
- Detailed error messages

## LlamaIndex RAG System

### Overview

The LlamaIndex RAG system provides enterprise-grade document indexing, retrieval, and question-answering capabilities.

### Architecture

```
┌──────────────────┐
│  Research Papers │
└────────┬─────────┘
         │
         v
┌──────────────────┐
│Document Processor│ (Chunking, Metadata)
└────────┬─────────┘
         │
         v
┌──────────────────┐
│  Embedding Model │ (HuggingFace/SentenceTransformers)
└────────┬─────────┘
         │
         v
┌──────────────────┐
│  Vector Store    │ (Qdrant/In-Memory)
└────────┬─────────┘
         │
         v
┌──────────────────┐
│  Query Engine    │ (Retriever + Synthesizer)
└──────────────────┘
```

### Basic Usage

```python
from agents.llamaindex_rag import create_rag_system

# Create RAG system (in-memory for development)
rag = create_rag_system(use_qdrant=False)

# Index documents
papers = [
    {
        "id": "paper_1",
        "title": "Attention Is All You Need",
        "abstract": "We propose the Transformer...",
        "authors": ["Vaswani et al."],
        "year": "2017",
        "source": "arXiv",
        "url": "https://arxiv.org/abs/1706.03762"
    }
]

stats = rag.index_documents(papers)
print(f"Indexed {stats['documents_indexed']} documents")

# Query the system
result = rag.query("What is the Transformer architecture?", top_k=5)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']}")

# Retrieve similar documents
similar = rag.retrieve_similar("attention mechanisms", top_k=3)
for doc in similar:
    print(f"- {doc['metadata']['title']} (score: {doc['score']:.3f})")
```

### Production Configuration (with Qdrant)

```python
import os

# Set Qdrant connection
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"

# Create RAG with Qdrant
rag = create_rag_system(use_qdrant=True)

# Index documents - stored in Qdrant
stats = rag.index_documents(papers)
```

### Advanced Features

**Document Metadata**
- Automatic metadata extraction
- Custom metadata fields
- Metadata-based filtering

**Query Modes**
- `compact`: Fast, concise responses
- `refine`: Iterative refinement
- `tree_summarize`: Hierarchical summarization

**Post-Processing**
- Similarity score filtering
- Result reranking
- Duplicate removal

## Usage Examples

### Example 1: Complete Research Workflow

```python
from agents.langgraph_orchestrator import create_orchestrator
from agents.llamaindex_rag import create_rag_system

# Setup
orchestrator = create_orchestrator()
rag_system = create_rag_system(use_qdrant=False)

# Step 1: Collect and index papers
research_query = "What are the latest trends in multi-agent systems?"
workflow_result = orchestrator.run_workflow(research_query)

# Step 2: Index collected papers
papers = workflow_result.get('papers', [])
if papers:
    rag_system.index_documents(papers)

# Step 3: Query indexed knowledge
answer = rag_system.query(research_query, top_k=10)
print(f"Final Answer: {answer['answer']}")
```

### Example 2: Multi-Session Conversation

```python
from agents.langgraph_orchestrator import create_orchestrator

orchestrator = create_orchestrator()

# Session 1
result1 = orchestrator.run_workflow(
    "Explain transformers in NLP",
    thread_id="user_alice_session1"
)

# Later: Resume session
saved_state = orchestrator.get_state("user_alice_session1")
print(f"Previous query: {saved_state['query']}")

# Continue with follow-up
result2 = orchestrator.run_workflow(
    "How do they differ from RNNs?",
    thread_id="user_alice_session1"
)
```

### Example 3: Document Retrieval and Analysis

```python
from agents.llamaindex_rag import create_rag_system

rag = create_rag_system(use_qdrant=False)

# Index corpus
corpus = load_research_papers()  # Your paper collection
rag.index_documents(corpus)

# Comparative analysis
queries = [
    "What are the key innovations in transformer architectures?",
    "How has attention mechanism evolved?",
    "What are the limitations of current models?"
]

for query in queries:
    result = rag.query(query, top_k=5)
    print(f"\nQ: {query}")
    print(f"A: {result['answer']}")
    print(f"Sources: {[s['metadata']['title'] for s in result['sources']]}")
```

## Integration with Existing System

### Compatibility

Both integrations are designed to work alongside the existing ResearcherAI architecture:

- **No Breaking Changes**: Existing agents remain fully functional
- **Optional Usage**: Can be enabled/disabled independently
- **Shared Data Models**: Compatible with existing paper schemas
- **Unified Configuration**: Uses same `.env` and `config.yaml`

### Using with Existing Agents

```python
# Combine with existing orchestrator
from agents.orchestrator_agent import ResearchOrchestrator
from agents.langgraph_orchestrator import create_orchestrator

# Traditional orchestrator
traditional = ResearchOrchestrator()
papers = traditional.collect_research_papers("machine learning", max_papers=10)

# Enhanced with LangGraph
langgraph = create_orchestrator()
result = langgraph.run_workflow("Analyze these ML papers")

# Enhanced with LlamaIndex
from agents.llamaindex_rag import create_rag_system
rag = create_rag_system()
rag.index_documents(papers)
```

## Testing

### Run Integration Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run comprehensive test suite
python test_langgraph_llamaindex.py
```

### Test Output

```
======================================================================
LangGraph & LlamaIndex Integration Test Suite
======================================================================

TEST 1: Package Imports               ✓ PASSED
TEST 2: LangGraph Orchestrator       ✓ PASSED
TEST 3: LlamaIndex RAG System        ✓ PASSED
TEST 4: Integration Compatibility    ✓ PASSED

Total: 4/4 tests passed

🎉 All tests passed!
```

### Individual Component Tests

```bash
# Test LangGraph only
python -c "from agents.langgraph_orchestrator import create_orchestrator; o = create_orchestrator(); print(o.get_workflow_graph())"

# Test LlamaIndex only
python -c "from agents.llamaindex_rag import create_rag_system; r = create_rag_system(); print(r.get_stats())"
```

## Troubleshooting

### Common Issues

**1. CUDA/GPU Issues**

If you see CUDA compatibility errors:

```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python test_langgraph_llamaindex.py
```

**2. OpenAI API Errors**

LlamaIndex uses MockLLM by default to avoid requiring OpenAI:

```python
# In llamaindex_rag.py
Settings.llm = MockLLM()  # Already configured
```

**3. Import Errors**

Ensure all packages are installed:

```bash
pip install -r requirements.txt
```

**4. Qdrant Connection Issues**

For production use with Qdrant:

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant if needed
docker-compose up -d qdrant
```

**5. Memory Issues**

For large document collections:

```python
# Reduce chunk size
from llama_index.core import Settings
Settings.chunk_size = 256  # Default: 512
Settings.chunk_overlap = 25  # Default: 50
```

### Debug Mode

Enable detailed logging:

```python
import logging

# LangGraph debug
logging.getLogger("langgraph").setLevel(logging.DEBUG)

# LlamaIndex debug
logging.getLogger("llama_index").setLevel(logging.DEBUG)
```

## Performance Optimization

### LangGraph

```python
# Disable checkpointing for speed (loses persistence)
from langgraph.graph import StateGraph
workflow = StateGraph(AgentState)
# ... build workflow without checkpointer
compiled = workflow.compile()  # No checkpointer argument
```

### LlamaIndex

```python
# Use smaller embedding model
rag = LlamaIndexRAG(
    embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2"  # Faster
)

# Reduce retrieval size
result = rag.query(question, top_k=3)  # Default: 5
```

## Next Steps

1. **Explore Advanced Features**: Try conditional workflows and custom agents
2. **Production Deployment**: Set up Qdrant for persistent vector storage
3. **Custom Workflows**: Extend `LangGraphOrchestrator` with custom nodes
4. **Hybrid Retrieval**: Combine vector search with keyword filtering
5. **Performance Tuning**: Optimize for your specific use case

## Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LlamaIndex Documentation**: https://docs.llamaindex.ai/
- **ResearcherAI Main README**: [README.md](../README.md)
- **Test Suite**: [test_langgraph_llamaindex.py](../test_langgraph_llamaindex.py)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review test suite for examples
3. Consult LangGraph/LlamaIndex documentation
4. Open an issue in the ResearcherAI repository
