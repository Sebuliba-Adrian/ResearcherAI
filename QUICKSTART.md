# Quick Start Guide

## 🚀 Two Ways to Run

### Option 1: Simple Demo (No Setup Required)

The simplest way to understand the concepts - runs immediately with no dependencies:

```bash
python3 demo_simple.py
```

**Features:**
- ✅ Reads documents and builds knowledge graph
- ✅ Pattern-based triple extraction
- ✅ Simple word-overlap vector search
- ✅ Graph querying and visualization
- ✅ Interactive Q&A

**No installation needed!** Just Python 3.6+

---

### Option 2: Full System (Production-Ready)

Complete self-improving system with ML models:

```bash
# 1. Run setup script
./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Run with a document
python self_improving_rag.py sample_knowledge.txt
```

**Additional Features:**
- 🧠 Sentence-Transformers embeddings
- 🔧 Tool integration (math, web search)
- 📊 Interactive HTML graph visualization
- 🌐 Web search integration
- 💾 FAISS vector database
- 🤖 Self-learning from interactions

---

## 📖 Example Session

```
👤 You: Who designed the Eiffel Tower?

🔍 Processing query: Who designed the Eiffel Tower?

============================================================
📄 Retrieved Information:
1. The Eiffel Tower was designed by Gustave Eiffel and is located in Paris...

🕸️  Knowledge Graph Facts:
  • Eiffel Tower [designed_by] Gustave Eiffel
  • Eiffel Tower [located_in] Paris
============================================================

🤖 Answer: Based on the knowledge base:
The Eiffel Tower was designed by Gustave Eiffel and is located in Paris.
```

---

## 📊 Available Commands

### In Interactive Mode:

| Command | Description |
|---------|-------------|
| `<question>` | Ask anything about the loaded document |
| `graph` | Show knowledge graph statistics |
| `stats` | Show system statistics (full version only) |
| `exit` | Quit the program |

---

## 🧪 Test Queries

Try these questions with [sample_knowledge.txt](sample_knowledge.txt):

1. **Entity Questions:**
   - "Who designed the Eiffel Tower?"
   - "Where is Berlin located?"
   - "Who created Python?"

2. **Relationship Questions:**
   - "What is the capital of France?"
   - "Who developed CRISPR?"
   - "What is Claude?"

3. **Graph Traversal:**
   - "What is located in Paris?"
   - "Who works at Berkeley?"
   - "What did Einstein develop?"

4. **Tools (Full version only):**
   - "Calculate 15 * 23"
   - "Search for latest AI news"

---

## 🏗️ How It Works

### Architecture Overview

```
Document → Chunks → Vector DB
                 ↓
                Triples → Knowledge Graph
                          ↓
                    Query Processing
                          ↓
              Vector Search + Graph Query
                          ↓
                  Combined Context
                          ↓
                   LLM Reasoning
                          ↓
                 Self-Improvement
```

### Key Components

1. **Document Processing**
   - Reads PDF/TXT files
   - Chunks into ~300-400 character segments
   - Each chunk maintains context

2. **Triple Extraction**
   - **Simple Demo**: Pattern-based (regex)
   - **Full System**: spaCy NLP or LLM-based
   - Format: `(Subject, Relation, Object)`

3. **Vector Database**
   - **Simple Demo**: Word overlap scoring
   - **Full System**: FAISS with SentenceTransformers
   - Finds semantically similar content

4. **Knowledge Graph**
   - NetworkX directed graph
   - Stores entity relationships
   - Enables graph traversal

5. **Self-Learning** (Full system only)
   - Learns from each interaction
   - Updates both vector DB and graph
   - Expands knowledge continuously

---

## 📁 Project Structure

```
ResearcherAI/
│
├── demo_simple.py           # ⚡ Simple demo (no dependencies)
├── self_improving_rag.py    # 🚀 Full system
│
├── sample_knowledge.txt     # 📚 Sample document
├── requirements.txt         # 📦 Python dependencies
├── setup.sh                 # 🔧 Automated setup
│
├── README.md                # 📖 Complete documentation
├── QUICKSTART.md            # ⚡ This file
│
└── knowledge_graph.html     # 📊 Generated visualization
```

---

## 🔄 Next Steps

1. **Try the simple demo** to understand concepts
2. **Run the full system** for production features
3. **Load your own documents** (PDF or TXT)
4. **Extend with custom tools**
5. **Visualize the knowledge graph** in browser

---

## 🎯 Use Cases

- **Research Assistants**: Query academic papers
- **Documentation**: Interactive company knowledge base
- **Education**: Learn from textbooks
- **Personal Knowledge**: Organize notes and research
- **Customer Support**: Answer FAQs from manuals

---

## 🆘 Troubleshooting

### Simple Demo Issues

**Error: File not found**
```bash
# Make sure you're in the right directory
cd /home/adrian/Desktop/Projects/ResearcherAI
python3 demo_simple.py
```

### Full System Issues

**Dependencies not installing**
```bash
# Make sure you're in virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

**spaCy model missing**
```bash
python -m spacy download en_core_web_sm
```

**Memory issues**
- Use smaller documents
- Reduce chunk size
- Use lighter models

---

## 💡 Tips

1. **Start small**: Test with small documents first
2. **Ask specific questions**: Better results than vague queries
3. **Check the graph**: Use `graph` command to see what was learned
4. **Custom documents**: Replace sample_knowledge.txt with your own
5. **Iterate**: The system learns - ask follow-up questions!

---

## 🌟 What Makes This Special?

Unlike traditional RAG systems:

- ✅ **Hybrid retrieval**: Combines vector + graph search
- ✅ **Structured reasoning**: Uses entity relationships
- ✅ **Self-improving**: Learns from interactions
- ✅ **Explainable**: See what knowledge was used
- ✅ **Extensible**: Easy to add tools and features
- ✅ **No vendor lock-in**: Works locally, no APIs required

---

## 📚 Learn More

- [Full README](README.md) - Complete documentation
- [Main Script](self_improving_rag.py) - Source code with comments
- [Simple Demo](demo_simple.py) - Educational implementation

---

Ready to get started? Run:

```bash
python3 demo_simple.py
```

Happy querying! 🚀
