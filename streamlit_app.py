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
    OLLAMA_MODEL, OLLAMA_BASE_URL, QDRANT_COLLECTION, DATA_DIR,
)
from pipeline.auth import bootstrap_admin, signup, signin, is_admin, list_users, change_role, delete_user

logger = logging.getLogger(__name__)

# ── Bootstrap default admin on first run ─────────────────────────────────────
bootstrap_admin()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Knowledge Bot",
    page_icon="🛒",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Auth card ── */
.auth-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
}

.auth-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    box-shadow: 0 25px 50px rgba(0,0,0,0.5);
    max-width: 440px;
    width: 100%;
    margin: auto;
}

.auth-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    text-align: center;
    margin-bottom: 0.25rem;
}

.auth-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 1.75rem;
}

/* ── Role badge ── */
.badge-admin {
    background: linear-gradient(90deg,#f59e0b,#ef4444);
    color: white;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.badge-user {
    background: linear-gradient(90deg,#3b82f6,#6366f1);
    color: white;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ── Sidebar tweaks ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Stacked tab look ── */
div[data-testid="stTabs"] button[aria-selected="true"] {
    background: #3b82f6 !important;
    color: white !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session-state initialisation ──────────────────────────────────────────────
for key, default in [
    ("authenticated", False),
    ("username", None),
    ("role", None),
    ("auth_page", "signin"),   # "signin" | "signup"
    ("retriever", None),
    ("generator", None),
    ("indexed", False),
    ("messages", []),
    ("chat_history", []),
    ("ollama_model", OLLAMA_MODEL),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGES
# ══════════════════════════════════════════════════════════════════════════════

def render_auth_page():
    """Render the sign-in / sign-up UI (full page, no sidebar)."""
    # Centered logo / branding
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 0.5rem 0;'>
        <div style='font-size:3.5rem;'>🛒</div>
        <h1 style='font-size:2.2rem;font-weight:700;color:#e2e8f0;margin:0;'>Retail Knowledge Bot</h1>
        <p style='color:#94a3b8;font-size:1rem;margin-top:0.3rem;'>Your AI-powered retail intelligence assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Tab switcher
    tab_login, tab_signup = st.tabs(["🔐 Sign In", "📝 Sign Up"])

    # ── Sign In ──────────────────────────────────────────────────────────────
    with tab_login:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("signin_form", clear_on_submit=False):
            st.markdown("#### Welcome back!")
            username = st.text_input("Username", placeholder="Enter your username", key="si_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="si_pass")
            submitted = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

        if submitted:
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                ok, msg, role = signin(username, password)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.success(msg)
                    time.sleep(0.6)
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("""
        <div style='text-align:center;margin-top:1.2rem;color:#64748b;font-size:0.85rem;'>
            🔒 Default admin: <code>admin</code> / <code>Admin@1234</code>
        </div>
        """, unsafe_allow_html=True)

    # ── Sign Up ──────────────────────────────────────────────────────────────
    with tab_signup:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("signup_form", clear_on_submit=True):
            st.markdown("#### Create an account")
            new_username = st.text_input("Username", placeholder="Choose a username", key="su_user")
            new_password = st.text_input("Password", type="password", placeholder="Min. 6 characters", key="su_pass")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_confirm")
            submitted_su = st.form_submit_button("Create Account →", use_container_width=True, type="primary")

        if submitted_su:
            if not new_username or not new_password or not confirm_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, msg = signup(new_username, new_password, role="user")
                if ok:
                    st.success(msg + " Please sign in.")
                else:
                    st.error(msg)


# ══════════════════════════════════════════════════════════════════════════════
# QDRANT HELPERS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _force_release_qdrant_lock():
    """Kill orphan Python/Streamlit processes holding the Qdrant lock,
    then delete all Qdrant lock files so a fresh client can be created."""
    current_pid = os.getpid()

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

    time.sleep(1)


@st.cache_resource
def get_qdrant():
    """Return a QdrantClient."""
    try:
        return get_qdrant_client()
    except Exception as e:
        if "already accessed" in str(e) or "AlreadyLocked" in str(e):
            logger.warning("Qdrant lock detected — releasing and retrying...")
            _force_release_qdrant_lock()
            return get_qdrant_client()
        raise


@st.cache_resource
def get_embeddings_model():
    return get_embedder()


def try_load_index():
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (authenticated)
# ══════════════════════════════════════════════════════════════════════════════

def render_main_app():
    """Render the full app for authenticated users."""
    user_is_admin = is_admin(st.session_state.role)

    if not st.session_state.indexed:
        try_load_index()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<div style='margin-bottom:0.5rem;'>"
            f"<span style='font-size:1.4rem;'>🛒</span> "
            f"<strong style='font-size:1.1rem;'>Retail KB</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # User info + role badge
        badge_class = "badge-admin" if user_is_admin else "badge-user"
        badge_label = "Admin" if user_is_admin else "User"
        st.markdown(
            f"<div style='margin-bottom:1rem;'>"
            f"👤 <strong>{st.session_state.username}</strong>&nbsp;"
            f"<span class='{badge_class}'>{badge_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("🚪 Sign Out", use_container_width=True):
            for key in ["authenticated", "username", "role", "messages",
                        "chat_history", "indexed", "retriever", "generator"]:
                st.session_state[key] = None if key in ("username", "role") else False if key in ("authenticated", "indexed") else []
            st.rerun()

        st.markdown("---")

        # ── Ollama model selector ────────────────────────────────────────────
        st.subheader("Ollama Model")
        model_options = ["llama3.2:3b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:2b"]
        selected_model = st.selectbox(
            "LLM model",
            options=model_options,
            index=model_options.index(st.session_state.ollama_model)
                  if st.session_state.ollama_model in model_options else 0,
            help="Choose which Ollama model to use. Make sure it's pulled: ollama pull <model>",
        )
        if selected_model != st.session_state.ollama_model:
            st.session_state.ollama_model = selected_model
            os.environ["OLLAMA_MODEL"] = selected_model
            reset_shared_components()
            st.success(f"Switched to {selected_model}")

        st.caption(f"Base URL: `{OLLAMA_BASE_URL}`")
        st.markdown("---")

        existing_docs = _list_data_documents()

        # ── ADMIN-ONLY: Document management ───────────────────────────────────
        if user_is_admin:
            st.subheader("📂 Documents  *(Admin)*")

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
                help="Upload retail knowledge documents (.docx). Saved to data/.",
            )

            if uploaded_files:
                new_names = [f.name for f in uploaded_files]
                already_exist = [n for n in new_names if n in existing_docs]
                if already_exist:
                    st.warning(f"Will overwrite: {', '.join(already_exist)}")

                if st.button(
                    f"Upload & Rebuild ({len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''})",
                    use_container_width=True,
                    type="primary",
                ):
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    saved = []
                    for uf in uploaded_files:
                        dest = DATA_DIR / uf.name
                        with open(dest, "wb") as f:
                            f.write(uf.getbuffer())
                        saved.append(uf.name)
                    st.success(f"Saved {len(saved)} file(s) to data/")

                    with st.spinner("Running full pipeline (Phase 0 → Phase 1 → Phase 2)..."):
                        graph = build_full_index_graph()
                        result = graph.invoke({})

                    if _finalize_index_result(result):
                        count = result.get("vector_count", 0)
                        st.success(f"Pipeline complete — {count} chunks indexed!")
                    st.rerun()

            st.markdown("---")

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

            if existing_docs:
                with st.expander("🗑️ Remove Documents"):
                    docs_to_remove = st.multiselect(
                        "Select documents to remove",
                        options=existing_docs,
                        help="Removing documents requires a full rebuild.",
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

            # ── User management ───────────────────────────────────────────────
            with st.expander("👥 User Management"):
                users = list_users()
                for u in users:
                    col_u, col_r, col_del = st.columns([3, 2, 1])
                    col_u.markdown(f"**{u['username']}**")
                    new_role = col_r.selectbox(
                        "Role",
                        options=["user", "admin"],
                        index=0 if u["role"] == "user" else 1,
                        key=f"role_{u['username']}",
                        label_visibility="collapsed",
                    )
                    if new_role != u["role"]:
                        ok, msg = change_role(u["username"], new_role)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    if col_del.button("✕", key=f"del_{u['username']}",
                                      help=f"Delete {u['username']}"):
                        if u["username"] == st.session_state.username:
                            st.error("You cannot delete your own account.")
                        else:
                            ok, msg = delete_user(u["username"])
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

            st.markdown("---")

        else:
            # Regular users: show read-only doc count
            st.subheader("📂 Documents")
            if existing_docs:
                with st.expander(f"Loaded documents ({len(existing_docs)})", expanded=False):
                    for doc_name in existing_docs:
                        st.markdown(f"- `{doc_name}`")
            else:
                st.caption("No documents available yet.")
            st.markdown("---")

        # ── Status metrics ────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        col1.metric("Index", "Ready" if st.session_state.indexed else "Not built")
        col2.metric("Docs", str(len(existing_docs)))
        st.metric("Ollama", st.session_state.get("ollama_model", OLLAMA_MODEL))

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        if user_is_admin:
            st.markdown("---")
            if st.button("🔓 Fix Lock Error", use_container_width=True,
                         help="Use if you see 'Qdrant storage already accessed' error"):
                with st.spinner("Releasing Qdrant lock..."):
                    get_qdrant.clear()
                    _force_release_qdrant_lock()
                    st.session_state.indexed = False
                    reset_shared_components()
                st.success("Lock released! Reloading...")
                st.rerun()

    # ── Main chat area ────────────────────────────────────────────────────────
    st.title("Retail Knowledge Assistant")
    st.caption("Ask questions about retail operations, scenarios, terms, and processes")

    if not st.session_state.indexed:
        if user_is_admin:
            st.info("Upload documents and click **Upload & Rebuild** in the sidebar, or click **Rebuild Index**.")
        else:
            st.info("The knowledge base is not ready yet. Please ask an admin to build the index.")

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
                    chunks = retriever.retrieve(prompt)
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


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.authenticated:
    render_auth_page()
else:
    render_main_app()
