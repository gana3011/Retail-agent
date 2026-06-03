import logging
import os
import time

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
from pipeline.config import GROQ_API_KEY, QDRANT_COLLECTION, QDRANT_PATH

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
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_qdrant():
    _lock_file = QDRANT_PATH / ".lock"
    if _lock_file.exists():
        try:
            _lock_file.unlink()
        except Exception:
            pass
    return get_qdrant_client()


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
        retriever = Retriever(client=client, embedder=model)
        st.session_state.retriever = retriever
        st.session_state.indexed = True
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            st.session_state.generator = AnswerGenerator(api_key=api_key)
            st.session_state.api_key_set = True
        return True
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return False


if not st.session_state.indexed:
    try_load_index()

with st.sidebar:
    st.title("🛒 Retail KB")
    st.markdown("---")

    if st.button("Build Index (Phase 1+2)", use_container_width=True):
        with st.spinner("Chunking documents..."):
            chunk_all()
        with st.spinner("Loading embedding model..."):
            chunks = load_chunks()
            model = get_embeddings_model()
        with st.spinner("Indexing into Qdrant..."):
            client = get_qdrant()
            recreate_collection(client)
            embed_and_index(chunks, model, client)
        st.session_state.indexed = True
        retriever = Retriever(client=client, embedder=model)
        st.session_state.retriever = retriever
        generator = AnswerGenerator(api_key=os.environ.get("GROQ_API_KEY", ""))
        st.session_state.generator = generator
        count = client.count(QDRANT_COLLECTION)
        st.success(f"Indexed {count.count} chunks ready!")

    st.metric("Index", "Ready" if st.session_state.indexed else "Not built")
    st.metric("API Key", "Set" if os.environ.get("GROQ_API_KEY") else "Missing (.env)")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Retail Knowledge Assistant")
st.caption("Ask questions about retail operations, scenarios, terms, and processes")

if not st.session_state.api_key_set:
    st.info("Set your Groq API key in the .env file to enable AI answers.")
if not st.session_state.indexed:
    st.info("Click 'Build Index' in the sidebar to index the retail knowledge base.")

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
    if not st.session_state.api_key_set:
        with st.chat_message("assistant"):
            st.error("Please set your Groq API key in the .env file first.")
        st.session_state.messages.append({"role": "assistant", "content": "API key missing."})
        return

    if not st.session_state.indexed:
        with st.chat_message("assistant"):
            st.error("Please build the index first using the sidebar button.")
        st.session_state.messages.append({"role": "assistant", "content": "Index not built."})
        return

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Retrieving..."):
            t0 = time.time()
            try:
                chunks = st.session_state.retriever.retrieve(prompt)
                elapsed_retrieval = time.time() - t0

                full_response = ""
                stream = st.session_state.generator.generate_stream(prompt, chunks)
                for token in stream:
                    full_response += token
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)
                elapsed = time.time() - t0
                st.caption(f"Retrieved {len(chunks)} chunks in {elapsed_retrieval:.1f}s, generated in {elapsed - elapsed_retrieval:.1f}s")

                sources = st.session_state.generator._extract_sources(chunks)
                render_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })
            except Exception as e:
                placeholder.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                })


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
