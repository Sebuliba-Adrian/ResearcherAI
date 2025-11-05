# Quick Start: Advanced Features (LangGraph & LlamaIndex)

## 🚀 5-Minute Quick Start

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure packages are installed
pip install -r requirements.txt
```

### Example 1: LangGraph Stateful Workflow

```python
from agents.langgraph_orchestrator import create_orchestrator

# Create orchestrator
orchestrator = create_orchestrator()

# Run research workflow
result = orchestrator.run_workflow(
    query="What are transformer architectures?",
    thread_id="my_session"
)

# View results
print("Workflow Steps:")
for msg in result['messages']:
    print(f"  - {msg}")

print(f"\nAnswer: {result['reasoning_result']['answer']}")

# Resume later
state = orchestrator.get_state("my_session")
print(f"Saved query: {state['query']}")
```

**Output:**
```
Workflow Steps:
  - Collecting data for query: What are transformer architectures?
  - Processing papers into knowledge graph
  - Creating vector embeddings
  - Performing reasoning
  - Reviewing reasoning quality

Answer: [Generated answer about transformers]
Saved query: What are transformer architectures?
```

### Example 2: LlamaIndex RAG System

```python
from agents.llamaindex_rag import create_rag_system

# Create RAG system
rag = create_rag_system(use_qdrant=False)  # In-memory mode

# Sample papers
papers = [
    {
        "id": "1",
        "title": "Attention Is All You Need",
        "abstract": "We propose the Transformer, based solely on attention mechanisms...",
        "authors": ["Vaswani et al."],
        "year": "2017",
        "source": "arXiv",
        "url": "https://arxiv.org/abs/1706.03762"
    }
]

# Index documents
stats = rag.index_documents(papers)
print(f"✓ Indexed {stats['documents_indexed']} documents")

# Query
result = rag.query("What is the Transformer?", top_k=3)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']}")

# Find similar papers
similar = rag.retrieve_similar("attention mechanisms", top_k=2)
for doc in similar:
    print(f"- {doc['metadata']['title']} (score: {doc['score']:.2f})")
```

**Output:**
```
✓ Indexed 1 documents
Answer: [Generated answer about Transformer architecture]
Sources: 1
- Attention Is All You Need (score: 0.85)
```

### Example 3: Combined Workflow

```python
from agents.langgraph_orchestrator import create_orchestrator
from agents.llamaindex_rag import create_rag_system

# Setup
orchestrator = create_orchestrator()
rag = create_rag_system(use_qdrant=False)

# Step 1: Collect papers via LangGraph workflow
workflow_result = orchestrator.run_workflow(
    "Find papers on neural architecture search",
    thread_id="nas_research"
)

# Step 2: Index papers with LlamaIndex
papers = workflow_result.get('papers', [])
if papers:
    print(f"Indexing {len(papers)} papers...")
    rag.index_documents(papers)

# Step 3: Query indexed knowledge
questions = [
    "What are the main approaches to NAS?",
    "Which methods are most efficient?",
    "What are the future directions?"
]

for q in questions:
    result = rag.query(q, top_k=5)
    print(f"\nQ: {q}")
    print(f"A: {result['answer'][:200]}...")
```

## 🎯 Common Use Cases

### Use Case 1: Research Paper Analysis

```python
from agents.llamaindex_rag import create_rag_system

rag = create_rag_system(use_qdrant=False)

# Index your paper collection
papers = [...]  # Your papers
rag.index_documents(papers)

# Ask comparative questions
result = rag.query(
    "Compare BERT and GPT approaches to language modeling",
    top_k=10
)

print(result['answer'])
```

### Use Case 2: Multi-Step Research Process

```python
from agents.langgraph_orchestrator import create_orchestrator

orchestrator = create_orchestrator()

# Step 1: Initial research
result1 = orchestrator.run_workflow(
    "Overview of reinforcement learning",
    thread_id="rl_study"
)

# Step 2: Deep dive
result2 = orchestrator.run_workflow(
    "Focus on policy gradient methods",
    thread_id="rl_study"
)

# Step 3: Compare approaches
result3 = orchestrator.run_workflow(
    "Compare with actor-critic methods",
    thread_id="rl_study"
)

# All steps are saved and can be retrieved
state = orchestrator.get_state("rl_study")
```

### Use Case 3: Document Search and Retrieval

```python
from agents.llamaindex_rag import create_rag_system

rag = create_rag_system(use_qdrant=False)

# Index corpus
papers = load_papers_from_csv("research_papers.csv")
rag.index_documents(papers)

# Find relevant papers without generating answers
similar = rag.retrieve_similar(
    "graph neural networks for molecule generation",
    top_k=10
)

# Display results
for i, doc in enumerate(similar, 1):
    print(f"{i}. {doc['metadata']['title']}")
    print(f"   Authors: {', '.join(doc['metadata']['authors'][:3])}")
    print(f"   Similarity: {doc['score']:.3f}\n")
```

## 🔧 Configuration

### Development Mode (Fast, In-Memory)

```python
# LangGraph - No checkpointing
from langgraph.graph import StateGraph
workflow = StateGraph(AgentState)
# ... build workflow
compiled = workflow.compile()  # No persistence

# LlamaIndex - In-memory vector store
rag = create_rag_system(use_qdrant=False)
```

### Production Mode (Persistent, Scalable)

```bash
# Start Qdrant
docker-compose up -d qdrant

# Set environment
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
```

```python
# LangGraph - With checkpointing (default)
orchestrator = create_orchestrator()  # Auto-saves state

# LlamaIndex - Qdrant backend
rag = create_rag_system(use_qdrant=True)
```

## 🧪 Testing

### Quick Test

```bash
source venv/bin/activate
python test_langgraph_llamaindex.py
```

### Manual Testing

```python
# Test LangGraph
from agents.langgraph_orchestrator import create_orchestrator
orch = create_orchestrator()
print(orch.get_workflow_graph())

# Test LlamaIndex
from agents.llamaindex_rag import create_rag_system
rag = create_rag_system(use_qdrant=False)
print(rag.get_stats())
```

## 📊 Performance Tips

### Speed Optimization

```python
# Use smaller embedding model
rag = LlamaIndexRAG(
    embedding_model="sentence-transformers/all-MiniLM-L3-v2"
)

# Reduce chunk size
from llama_index.core import Settings
Settings.chunk_size = 256
Settings.chunk_overlap = 20

# Limit retrieval
result = rag.query(question, top_k=3)
```

### Memory Optimization

```python
# Process in batches
batch_size = 100
for i in range(0, len(papers), batch_size):
    batch = papers[i:i+batch_size]
    rag.index_documents(batch)
```

### Force CPU (Avoid CUDA Issues)

```bash
export CUDA_VISIBLE_DEVICES=""
python your_script.py
```

## 🐛 Troubleshooting

### Issue: Import Errors

```bash
# Solution
pip install -r requirements.txt
```

### Issue: CUDA Errors

```bash
# Solution
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Qdrant Not Connected

```bash
# Solution
docker-compose up -d qdrant
# Wait 5 seconds
docker ps | grep qdrant
```

### Issue: Slow Indexing

```python
# Solution: Use smaller model
rag = LlamaIndexRAG(
    embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2"
)
```

## 📚 Next Steps

1. **Read Full Documentation**: [LANGGRAPH_LLAMAINDEX_INTEGRATION.md](LANGGRAPH_LLAMAINDEX_INTEGRATION.md)
2. **Explore Examples**: Check [test_langgraph_llamaindex.py](../test_langgraph_llamaindex.py)
3. **Integrate with UI**: Connect to frontend/streamlit
4. **Production Deploy**: Set up with Qdrant and Neo4j

## 🔗 Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [ResearcherAI README](../README.md)
