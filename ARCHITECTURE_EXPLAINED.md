# 🏗️ Full System Architecture - What Each Component Does

## The Complete Pipeline in `self_improving_rag_gemini.py`

### 🎯 Overview

The full system has **4 major components** working together:

```
User Query
    ↓
┌───────────────────────────────────────────────┐
│ 1️⃣ VECTOR DATABASE (FAISS)                   │
│    → Semantic similarity search               │
│    → Finds relevant chunks by meaning         │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│ 2️⃣ KNOWLEDGE GRAPH (NetworkX)                │
│    → Stores relationships as triples          │
│    → Enables multi-hop reasoning              │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│ 3️⃣ GEMINI AI                                 │
│    → Extracts triples from text               │
│    → Generates final answer                   │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│ 4️⃣ SELF-LEARNING                             │
│    → Updates vector DB with new info          │
│    → Expands knowledge graph                  │
└───────────────────────────────────────────────┘
```

---

## 📦 Component 1: Vector Database (FAISS)

### What It Does:
Converts text into **384-dimensional vectors** (embeddings) and finds similar chunks by **semantic meaning**.

### Code Location:
```python
# Line 11: Import
import faiss

# Lines 189-201: Build Vector DB
def build_vector_db(chunks_list: list):
    embeddings = np.array(
        embedder.encode(chunks_list, convert_to_numpy=True)
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Lines 203-213: Search
def retrieve_similar(query: str, k: int = 3):
    q_emb = embedder.encode([query])
    _, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0]]
```

### Example:
```python
Query: "Who designed the Eiffel Tower?"
       ↓ [Embedding converts to vector]
       ↓ [0.23, -0.15, 0.44, ... 384 dims]
       ↓ [FAISS finds similar vectors]
Result: ["The Eiffel Tower was designed by Gustave Eiffel..."]
```

### Why It's Needed:
- ❌ **Without it:** Would use word matching (like simple demo)
- ✅ **With it:** Understands "designed" = "created" = "built"

---

## 🕸️ Component 2: Knowledge Graph (NetworkX)

### What It Does:
Stores **relationships as triples**: `(Subject, Relation, Object)`

### Code Location:
```python
# Line 12: Import
import networkx as nx

# Line 39: Global graph
G = nx.DiGraph()

# Lines 144-173: Gemini Triple Extraction
def extract_triples_gemini(text: str) -> list:
    prompt = f"""Extract knowledge triples from: {text}
    Return JSON: [["subject", "relation", "object"], ...]"""

    response = gemini_model.generate_content(prompt)
    # Parse JSON and return triples

# Lines 187-195: Add to Graph
def add_triples_to_graph(triples: list):
    for s, r, o in triples:
        G.add_edge(s, o, label=r)

# Lines 217-236: Query Graph
def query_knowledge_graph(entities: list) -> list:
    facts = []
    for entity in entities:
        if entity in G.nodes():
            for neighbor in G.neighbors(entity):
                rel = G[entity][neighbor].get("label", "related_to")
                facts.append(f"{entity} [{rel}] {neighbor}")
    return facts
```

### Example:
```python
Input Text: "The Eiffel Tower was designed by Gustave Eiffel"
           ↓ [Gemini extracts]
Triple: ("Eiffel Tower", "designed_by", "Gustave Eiffel")
           ↓ [Add to graph]
Graph: Eiffel Tower ──designed_by──> Gustave Eiffel

Query: "Who designed Eiffel Tower?"
      ↓ [Extract entity: "Eiffel Tower"]
      ↓ [Query graph]
Result: "Eiffel Tower [designed_by] Gustave Eiffel"
```

### Why It's Needed:
- ❌ **Without it:** Can't traverse relationships
- ✅ **With it:** Multi-hop reasoning: Tower → Paris → France → Europe

---

## 🤖 Component 3: Gemini AI

### What It Does:
1. **Extracts triples** intelligently (not just patterns)
2. **Generates final answers** with reasoning

### Code Location:
```python
# Line 17: Import
import google.generativeai as genai

# Lines 49-51: Initialize
genai.configure(api_key=Config.GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)

# Lines 144-173: Triple Extraction
def extract_triples_gemini(text: str):
    prompt = """Extract triples..."""
    response = gemini_model.generate_content(prompt)
    return parsed_triples

# Lines 333-371: Answer Generation
def answer_query_gemini(query: str):
    # Get context from vector DB + graph
    context = retrieve_similar(query) + query_graph(entities)

    # Ask Gemini to answer
    prompt = f"""Context: {context}
    Question: {query}
    Answer based ONLY on context."""

    response = gemini_model.generate_content(prompt)
    return response.text
```

### Example:
```python
Input: "Einstein developed the theory of relativity"

Gemini Extraction:
[["Albert Einstein", "developed", "theory of relativity"]]

vs Simple Pattern:
Might miss "developed" or extract wrong subject
```

### Why It's Needed:
- ❌ **Without it:** Pattern-based extraction misses complex sentences
- ✅ **With it:** AI understands context and extracts accurately

---

## 🧠 Component 4: Self-Learning Loop

### What It Does:
After each interaction, the system **learns** by:
1. Adding new info to vector DB
2. Extracting triples from new info
3. Expanding knowledge graph

### Code Location:
```python
# Lines 283-300: Self-Learning
def learn_from_text(new_text: str):
    # 1. Add to chunks
    chunks.append(new_text)

    # 2. Update vector DB
    new_emb = embedder.encode([new_text])
    index.add(new_emb)

    # 3. Extract and add triples
    triples = extract_triples_gemini(new_text)
    add_triples_to_graph(triples)
```

### Example:
```python
Interaction 1:
User: "Calculate 15 * 23"
Agent: "345" [via math tool]
System: learn_from_text("Q: 15*23, A: 345")
       ↓
       [Adds to vector DB]
       [Extracts triple: ("15*23", "equals", "345")]
       [Adds to graph]

Interaction 2:
User: "What's 15 times 23?"
System: [Finds in vector DB or graph]
Agent: "345" [from memory!]
```

### Why It's Needed:
- ❌ **Without it:** Static knowledge base
- ✅ **With it:** Gets smarter with every interaction

---

## 🎯 How They Work Together

### Full Query Flow:

```python
User: "Who designed the Eiffel Tower?"
      ↓
┌─────────────────────────────────────────┐
│ Step 1: VECTOR SEARCH (FAISS)          │
│ - Convert query to embedding            │
│ - Find 3 most similar chunks            │
│ Result: ["Eiffel Tower designed by..."] │
└────────────┬────────────────────────────┘
             ↓
┌────────────▼────────────────────────────┐
│ Step 2: ENTITY EXTRACTION (Gemini)     │
│ - Extract: ["Eiffel Tower"]            │
└────────────┬────────────────────────────┘
             ↓
┌────────────▼────────────────────────────┐
│ Step 3: GRAPH QUERY (NetworkX)         │
│ - Query: "Eiffel Tower" relationships   │
│ Result: "Eiffel Tower [designed_by]     │
│          Gustave Eiffel"                │
└────────────┬────────────────────────────┘
             ↓
┌────────────▼────────────────────────────┐
│ Step 4: MERGE CONTEXT                  │
│ Vector: "...designed by Gustave..."     │
│ Graph: "Eiffel Tower [designed_by]      │
│        Gustave Eiffel"                  │
└────────────┬────────────────────────────┘
             ↓
┌────────────▼────────────────────────────┐
│ Step 5: GEMINI REASONING               │
│ Prompt: "Based on context, answer..."  │
│ Answer: "Gustave Eiffel designed the   │
│         Eiffel Tower"                   │
└────────────┬────────────────────────────┘
             ↓
┌────────────▼────────────────────────────┐
│ Step 6: SELF-LEARNING                  │
│ - Add interaction to vector DB          │
│ - Extract any new triples               │
│ - Update knowledge graph                │
└─────────────────────────────────────────┘
```

---

## 📊 Why All 4 Are Needed

### Comparison:

| Question Type | Vector Only | + Graph | + Gemini | + Self-Learning |
|--------------|-------------|---------|----------|-----------------|
| Simple: "Who designed X?" | ✅ Works | ✅ Better | ✅ Best | ✅ Remembers |
| Complex: "Why did Y do Z?" | ❌ Fails | ✅ Better | ✅ Best | ✅ Improves |
| Multi-hop: "Where is X located?" | ❌ Fails | ✅ Works | ✅ Better | ✅ Expands |
| Follow-up questions | ❌ No memory | ❌ No memory | ❌ No memory | ✅ Learns! |

---

## 🔍 Where Is Each Component Used?

### In `self_improving_rag_gemini.py`:

```python
# VECTOR DATABASE (FAISS)
Line 11:   import faiss
Line 189:  def build_vector_db()      # Creates index
Line 203:  def retrieve_similar()     # Searches

# KNOWLEDGE GRAPH (NetworkX)
Line 12:   import networkx as nx
Line 39:   G = nx.DiGraph()           # Global graph
Line 187:  def add_triples_to_graph() # Stores triples
Line 217:  def query_knowledge_graph()# Queries

# GEMINI AI
Line 17:   import google.generativeai
Line 50:   gemini_model = ...         # Initialize
Line 144:  def extract_triples_gemini()# Extract triples
Line 333:  def answer_query_gemini()  # Generate answer

# SELF-LEARNING
Line 283:  def learn_from_text()      # Updates KB
Line 365:  learn_from_text(...)       # After each query
```

---

## 🎯 Bottom Line

The **full system** (`self_improving_rag_gemini.py`) has:

1. ✅ **FAISS Vector DB** - Semantic search (line 11, 189-213)
2. ✅ **NetworkX Graph** - Relationship storage (line 12, 187-236)
3. ✅ **Gemini AI** - Intelligent extraction + reasoning (line 17, 144-371)
4. ✅ **Self-Learning** - Continuous improvement (line 283-300)

The **demo** (`demo_gemini_live.py`) has:
- ✅ Gemini AI only
- ❌ No vector DB
- ❌ No knowledge graph
- ❌ No self-learning

**Use the full system to get ALL features!** 🚀

---

*Once packages install, run:*
```bash
source venv/bin/activate
python3 self_improving_rag_gemini.py sample_knowledge.txt
```
