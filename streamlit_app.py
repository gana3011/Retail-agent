import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import streamlit as st

from pipeline.ssl_setup import configure_ssl

configure_ssl()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from pipeline.chunking import chunk_all
from pipeline.indexing import (
    load_chunks, get_embedder, get_qdrant_client,
    recreate_collection, embed_and_index,
)
from pipeline.retriever import Retriever
from pipeline.generator import AnswerGenerator
from pipeline.graph import build_index_graph, build_full_index_graph, build_query_graph
from pipeline.nodes import (
    get_shared_retriever, get_shared_generator,
    reset_shared_components,
)
from pipeline.config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, QDRANT_COLLECTION, QDRANT_PATH, DATA_DIR,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Retail Knowledge Bot",
    page_icon="🛒",
    layout="wide",
)

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "generator" not in st.session_state:
    st.session_state.generator = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = OLLAMA_MODEL


def _force_release_qdrant_lock():
    """Kill orphan Python/Streamlit processes holding the Qdrant lock,
    then delete all Qdrant lock files so a fresh client can be created."""
    current_pid = os.getpid()

    # Kill all other python/streamlit processes (Windows)
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 2:
                try:
                    pid = int(parts[1])
                    if pid != current_pid:
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                       capture_output=True, timeout=3)
                except (ValueError, Exception):
                    pass
    except Exception as e:
        logger.warning("Could not kill processes: %s", e)

    # Delete all .lock files inside the Qdrant storage folder
    if QDRANT_PATH.exists():
        for lock_file in QDRANT_PATH.rglob("*.lock"):
            try:
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass
        # Also try the top-level .lock
        top_lock = QDRANT_PATH / ".lock"
        try:
            top_lock.unlink(missing_ok=True)
        except Exception:
            pass

    time.sleep(1)  # give OS time to release handles


@st.cache_resource
def get_qdrant():
    """Return a QdrantClient, auto-recovering from stale locks."""
    # Pre-emptively remove any stale top-level lock file
    for lock_file in QDRANT_PATH.rglob("*.lock"):
        try:
            lock_file.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        return get_qdrant_client()
    except Exception as e:
        if "already accessed" in str(e) or "AlreadyLocked" in str(e):
            logger.warning("Qdrant lock detected — releasing and retrying...")
            _force_release_qdrant_lock()
            return get_qdrant_client()   # retry once after cleanup
        raise


@st.cache_resource
def get_embeddings_model():
    return get_embedder()


def try_load_index():
    if not (QDRANT_PATH / "collection" / QDRANT_COLLECTION).exists():
        st.info("No existing index found. Click 'Build Index' to create one.")
        return False
    try:
        client = get_qdrant()
        count = client.count(QDRANT_COLLECTION)
        if count.count == 0:
            return False
        model = get_embeddings_model()

        reset_shared_components()
        retriever = get_shared_retriever(client=client, embedder=model)
        st.session_state.retriever = retriever
        st.session_state.indexed = True

        generator = get_shared_generator()
        st.session_state.generator = generator
        return True
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return False


def _finalize_index_result(result: dict):
    """Shared helper: update session state after an index graph completes."""
    if result.get("error"):
        st.error(f"Index build failed: {result['error']}")
        return False

    st.session_state.indexed = True
    reset_shared_components()
    client = result.get("qdrant_client") or get_qdrant()
    model = result.get("embedder") or get_embeddings_model()
    retriever = get_shared_retriever(client=client, embedder=model)
    generator = get_shared_generator()
    st.session_state.retriever = retriever
    st.session_state.generator = generator
    return True


def _list_data_documents() -> list[str]:
    """Return sorted list of .docx filenames currently in data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(f.name for f in DATA_DIR.glob("*.docx"))


if not st.session_state.indexed:
    try_load_index()

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 Retail KB")
    st.markdown("---")

    # ── Ollama model selector ────────────────────────────────
    st.subheader("Ollama Model")
    selected_model = st.selectbox(
        "LLM model",
        options=["llama3.2:3b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:2b"],
        index=["llama3.2:3b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:2b"].index(
            st.session_state.ollama_model
        ) if st.session_state.ollama_model in ["llama3.2:3b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:2b"] else 0,
        help="Choose which Ollama model to use. Make sure it's pulled: ollama pull <model>",
    )
    if selected_model != st.session_state.ollama_model:
        st.session_state.ollama_model = selected_model
        import os as _os
        _os.environ["OLLAMA_MODEL"] = selected_model
        reset_shared_components()
        st.success(f"Switched to {selected_model}")

    st.caption(f"Base URL: `{OLLAMA_BASE_URL}`")
    st.markdown("---")

    # ── Document upload section ──────────────────────────────
    st.subheader("Documents")

    existing_docs = _list_data_documents()
    if existing_docs:
        with st.expander(f"Loaded documents ({len(existing_docs)})", expanded=False):
            for doc_name in existing_docs:
                st.markdown(f"- `{doc_name}`")
    else:
        st.caption("No documents in data/ folder yet.")

    uploaded_files = st.file_uploader(
        "Upload .docx files",
        type=["docx"],
        accept_multiple_files=True,
        help="Upload retail knowledge documents (.docx). They will be saved to the data/ folder.",
    )

    if uploaded_files:
        # Show what will be uploaded
        new_names = [f.name for f in uploaded_files]
        already_exist = [n for n in new_names if n in existing_docs]
        if already_exist:
            st.warning(
                f"Will overwrite: {', '.join(already_exist)}"
            )

        if st.button(
            f"Upload & Rebuild ({len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''})",
            use_container_width=True,
            type="primary",
        ):
            # Step 1: Save files to data/
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            saved = []
            for uf in uploaded_files:
                dest = DATA_DIR / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                saved.append(uf.name)
            st.success(f"Saved {len(saved)} file(s) to data/")

            # Step 2: Run the full pipeline (Phase 0 → 1 → 2)
            with st.spinner("Running full pipeline (Phase 0 → Phase 1 → Phase 2)..."):
                graph = build_full_index_graph()
                result = graph.invoke({})

            if _finalize_index_result(result):
                count = result.get("vector_count", 0)
                st.success(f"Pipeline complete — {count} chunks indexed!")
            st.rerun()

    st.markdown("---")

    # ── Rebuild index (without upload) ───────────────────────
    if st.button("Rebuild Index (Phase 1+2)", use_container_width=True):
        with st.spinner("Running LangGraph index pipeline..."):
            graph = build_index_graph()
            result = graph.invoke({})

        if _finalize_index_result(result):
            count = result.get("vector_count", 0)
            st.success(f"Indexed {count} chunks ready!")

    if st.button("Full Rebuild (Phase 0+1+2)", use_container_width=True):
        with st.spinner("Running full pipeline (Phase 0 → Phase 1 → Phase 2)..."):
            graph = build_full_index_graph()
            result = graph.invoke({})

        if _finalize_index_result(result):
            count = result.get("vector_count", 0)
            st.success(f"Full rebuild complete — {count} chunks indexed!")

    st.markdown("---")

    # ── Document deletion ────────────────────────────────────
    if existing_docs:
        with st.expander("Remove documents"):
            docs_to_remove = st.multiselect(
                "Select documents to remove",
                options=existing_docs,
                help="Removing documents requires a full rebuild to update the index.",
            )
            if docs_to_remove and st.button(
                f"Remove {len(docs_to_remove)} & Rebuild",
                use_container_width=True,
            ):
                for doc_name in docs_to_remove:
                    doc_path = DATA_DIR / doc_name
                    if doc_path.exists():
                        doc_path.unlink()
                        st.info(f"Removed: {doc_name}")

                with st.spinner("Rebuilding index without removed documents..."):
                    graph = build_full_index_graph()
                    result = graph.invoke({})

                if _finalize_index_result(result):
                    count = result.get("vector_count", 0)
                    st.success(f"Rebuild complete — {count} chunks indexed!")
                st.rerun()

    st.markdown("---")

    # ── Status metrics ───────────────────────────────────────
    col1, col2 = st.columns(2)
    col1.metric("Index", "Ready" if st.session_state.indexed else "Not built")
    col2.metric("Docs", str(len(existing_docs)))
    st.metric("Ollama", st.session_state.get("ollama_model", OLLAMA_MODEL))

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    if st.button("🔓 Fix Lock Error", use_container_width=True, help="Use this if you see 'Qdrant storage already accessed' error"):
        with st.spinner("Releasing Qdrant lock..."):
            # Clear the cached client so it gets recreated
            get_qdrant.clear()
            _force_release_qdrant_lock()
            st.session_state.indexed = False
            reset_shared_components()
        st.success("Lock released! Reloading...")
        st.rerun()

# ──────────────────────────────────────────────────────────────
# Main chat area
# ──────────────────────────────────────────────────────────────
st.title("Retail Knowledge Assistant")
st.caption("Ask questions about retail operations, scenarios, terms, and processes")

if not st.session_state.indexed:
    st.info("Click 'Build Index' in the sidebar or upload documents to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    parts = []
                    if s.get("title"):
                        parts.append(f"Scenario: {s['title']}")
                    if s.get("term"):
                        parts.append(f"Term: {s['term']}")
                    if s.get("domain"):
                        parts.append(f"Domain: {s['domain']}")
                    label = f"**{s['source_doc']}**"
                    if parts:
                        label += " - " + " | ".join(parts)
                    st.markdown(f"- {label} (score: {s['relevance_score']})")


def render_sources(sources: list[dict]):
    if sources:
        with st.expander("Sources"):
            for s in sources:
                parts = []
                if s.get("title"):
                    parts.append(f"Scenario: {s['title']}")
                if s.get("term"):
                    parts.append(f"Term: {s['term']}")
                if s.get("domain"):
                    parts.append(f"Domain: {s['domain']}")
                label = f"**{s['source_doc']}**"
                if parts:
                    label += " - " + " | ".join(parts)
                st.markdown(f"- {label} (score: {s['relevance_score']})")


def answer_question(prompt: str):
    if not st.session_state.indexed:
        with st.chat_message("assistant"):
            st.error("Please build the index first using the sidebar button.")
        st.session_state.messages.append({"role": "assistant", "content": "Index not built."})
        return

    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            retriever = get_shared_retriever()
            generator = get_shared_generator()

            with st.spinner("Searching knowledge base..."):
                chunks = retriever.retrieve(
                    prompt,
                )
                sources = generator._extract_sources(chunks)
                search_time = time.time() - t0

            if not chunks:
                st.warning("No relevant information found in the knowledge base.")
                return

            st.caption(f"Found {len(chunks)} relevant chunks in {search_time:.1f}s · Streaming answer...")

            answer_tokens = []

            def token_stream():
                for token in generator.generate_stream(
                    prompt, chunks,
                    chat_history=st.session_state.chat_history,
                ):
                    answer_tokens.append(token)
                    yield token

            answer = st.write_stream(token_stream())
            elapsed = time.time() - t0

            st.caption(f"{elapsed:.1f}s total · {len(chunks)} chunks · {st.session_state.get('ollama_model', 'llama3.2:3b')}")
            render_sources(sources)

            # Update conversation memory
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})



if st.session_state.get("pending_question"):
    prompt = st.session_state.pop("pending_question")
    with st.chat_message("user"):
        st.markdown(prompt)
    answer_question(prompt)

if prompt := st.chat_input("Ask about retail..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    answer_question(prompt)
