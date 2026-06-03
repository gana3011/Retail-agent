# Retail Knowledge Agent

RAG pipeline for retail domain knowledge: ingests `.docx` training documents, chunks them, indexes into Qdrant (vector DB), and answers questions via Groq LLM with source attribution.

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at https://console.groq.com/keys

## Pipeline

| Phase | Step | Description | Output |
|-------|------|-------------|--------|
| 0 | Data Prep | Extract structured elements from `.docx` | `output/phase_0/*.jsonl`, `retail_knowledge_base.json` |
| 1 | Chunking | Split elements into text chunks (word-level, configurable overlap) | `output/phase_1/chunks.jsonl` |
| 2 | Indexing | Embed chunks via `all-MiniLM-L6-v2`, store in Qdrant | `output/qdrant_storage/` |
| 3 | Retrieval | Doc-type filter + vector search + cross-encoder reranking | — |
| 4 | Generation | Groq LLM answers with source citations | — |

## Usage

### CLI pipeline
```bash
python -c "from pipeline.runner import build_pipeline; build_pipeline()"
```

Or step by step:
```bash
python phase_0_data_preparation.py
python -c "from pipeline.chunking import chunk_all; chunk_all()"
python -c "from pipeline.indexing import run_indexing; run_indexing()"
```

### Web UI
```bash
streamlit run streamlit_app.py
```

### Run test suite
```bash
python -c "from pipeline.runner import build_pipeline, run_test; r, g = build_pipeline(); run_test(r, g)"
```

## Project Structure

```
Retail_agent/
├── data/                          # Source .docx files
├── output/
│   ├── phase_0/                   # Extracted elements (JSONL + merged JSON)
│   ├── phase_1/                   # Chunks (JSONL)
│   └── qdrant_storage/            # Vector index
├── pipeline/
│   ├── config.py                  # Paths, models, constants
│   ├── chunking.py                # Element → chunk conversion
│   ├── indexing.py                # Embedding + Qdrant index
│   ├── retriever.py               # Vector search + doc-type filter + reranker
│   ├── generator.py               # Groq LLM answer generation (streaming)
│   ├── runner.py                  # CLI orchestration
│   └── test_set.py                # 35 test questions + keyword evaluation
├── phase_0_data_preparation.py    # .docx → structured elements
├── streamlit_app.py               # Web UI
├── requirements.txt
└── .env.example
```

## Key Design Decisions

- **Doc-type filter**: An LLM call classifies the question's domain, then filters Qdrant results by `doc_type` payload field
- **Cross-encoder reranker**: Optional second-pass reranking via `cross-encoder/ms-marco-MiniLM-L-6-v2` to improve top-k relevance
- **Streaming**: Groq responses stream token-by-token into the Streamlit UI
- **Dispatch-based metadata builder**: Element-type specific metadata is built via a registry dict instead of if-else chains
