# Retail Knowledge Agent — LangGraph Edition

A **Retrieval-Augmented Generation (RAG)** chatbot for retail domain knowledge, powered by **LangGraph** for orchestration, **Qdrant** for vector search, and **Groq** (Llama 3.3 70B) for generation.

## Architecture

The pipeline is built as three **LangGraph `StateGraph`** instances:

```
┌─────────────────────────────────────────────────────────────┐
│  Index Graph                                                │
│  START → [chunking] → [embedding] → END                    │
│  State: IndexState (chunks, embedder, qdrant_client, ...)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Query Graph (RAG)                                          │
│  START → [expand] → [retrieve] → [rerank] → [generate] → END│
│  State: QueryState (question, expanded_queries, answer, ...)│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Test Graph                                                 │
│  START → [load_questions] → [evaluate] → END                │
│  State: TestState (questions, results, avg_latency, ...)    │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
Retail_agent/
├── data/                        # Source .docx documents
├── output/                      # Generated artifacts
│   ├── phase_0/                 # Extracted JSON knowledge base
│   ├── phase_1/                 # Chunked JSONL
│   └── qdrant_storage/          # Vector DB storage
├── pipeline/
│   ├── __init__.py
│   ├── config.py                # Configuration & environment
│   ├── ssl_setup.py             # SSL workarounds for corp proxies
│   ├── chunking.py              # Document chunking logic
│   ├── indexing.py              # Embedding & Qdrant indexing
│   ├── retriever.py             # Multi-query retrieval + reranking
│   ├── generator.py             # Groq LLM answer generation
│   ├── test_set.py              # Test questions & evaluation
│   ├── state.py          ← NEW # LangGraph typed state definitions
│   ├── nodes.py           ← NEW # LangGraph node functions
│   ├── graph.py           ← NEW # LangGraph StateGraph definitions
│   └── runner.py        UPDATED # Delegates to LangGraph graphs
├── tests/
│   ├── test_chunking.py
│   ├── test_data_prep.py
│   ├── test_generator.py
│   ├── test_test_set.py
│   └── test_langgraph.py  ← NEW # Tests for LangGraph components
├── phase_0_data_preparation.py  # Docx → JSON extraction
├── streamlit_app.py     UPDATED # Uses LangGraph graphs
├── requirements.txt     UPDATED # Added langgraph, langchain-core
├── run_chatbot.bat      UPDATED
└── .env                         # GROQ_API_KEY=gsk_...
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file:
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at https://console.groq.com/keys

### 3. Prepare Data (Phase 0)

```bash
python phase_0_data_preparation.py
```

### 4. Build Index & Chat (via Streamlit)

```bash
streamlit run streamlit_app.py
```

Or use the batch file:
```bash
run_chatbot.bat
```

### 5. CLI Usage

```python
from pipeline.runner import build_pipeline, run_query

# Build the index (Phase 1 + 2)
retriever, generator = build_pipeline()

# Ask a question through the LangGraph query graph
result = run_query("What is a SKU?")
print(result["answer"])
print(result["sources"])
```

### 6. Run Tests

```bash
pytest tests/ -v
```

## Key Dependencies

| Package              | Purpose                            |
|---------------------|------------------------------------|
| `langgraph`         | Graph-based pipeline orchestration |
| `langchain-core`    | Required by LangGraph              |
| `sentence-transformers` | Embedding & reranking models   |
| `qdrant-client`     | Vector database                    |
| `groq`              | LLM API (Llama 3.3 70B)           |
| `streamlit`         | Web UI                             |
| `python-docx`       | Document parsing                   |

## How LangGraph Works Here

1. **State**: Each graph has a typed `TypedDict` state schema (`IndexState`, `QueryState`, `TestState`) defined in `pipeline/state.py`.

2. **Nodes**: Each pipeline step is a pure function that takes state and returns a partial state update. Defined in `pipeline/nodes.py`.

3. **Graphs**: `StateGraph` instances are built in `pipeline/graph.py` with explicit edges connecting nodes in sequence.

4. **Memory**: `QueryState.chat_history` uses an `operator.add` reducer, so conversation history accumulates across invocations when passed as initial state.

5. **Execution**: Graphs are compiled once and invoked with `.invoke(initial_state)`. The Streamlit app and CLI both use the same compiled graphs.

## License

MIT
