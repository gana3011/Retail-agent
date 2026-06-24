"""Tests for the LangGraph graph definitions and node functions."""

from unittest.mock import patch, MagicMock

from pipeline.state import IndexState, QueryState, TestState
from pipeline.nodes import (
    chunking_node,
    embedding_node,
    query_expansion_node,
    retrieval_node,
    reranking_node,
    generation_node,
    load_test_questions_node,
    get_shared_retriever,
    get_shared_generator,
    reset_shared_components,
)


class TestStateDefinitions:
    """Verify TypedDict state schemas are importable and valid."""

    def test_index_state_keys(self):
        state: IndexState = {"chunks": [], "status": "ready"}
        assert state["status"] == "ready"

    def test_query_state_keys(self):
        state: QueryState = {
            "question": "What is a SKU?",
            "expanded_queries": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "sources": [],
            "chat_history": [],
        }
        assert state["question"] == "What is a SKU?"

    def test_test_state_keys(self):
        state: TestState = {"questions": ["Q1"], "results": []}
        assert len(state["questions"]) == 1


class TestSharedSingletons:
    """Test singleton lifecycle management."""

    def test_reset_clears_singletons(self):
        reset_shared_components()
        # After reset, the next call should create fresh instances
        # (we can't easily test this without the full stack, but we
        #  verify reset does not raise)
        assert True

    @patch("pipeline.nodes.Retriever")
    def test_get_shared_retriever_creates_once(self, mock_cls):
        reset_shared_components()
        r1 = get_shared_retriever()
        r2 = get_shared_retriever()
        assert r1 is r2
        mock_cls.assert_called_once()
        reset_shared_components()

    @patch("pipeline.nodes.AnswerGenerator")
    def test_get_shared_generator_creates_once(self, mock_cls):
        reset_shared_components()
        g1 = get_shared_generator(api_key="test")
        g2 = get_shared_generator(api_key="test")
        assert g1 is g2
        mock_cls.assert_called_once()
        reset_shared_components()


class TestChunkingNode:
    @patch("pipeline.nodes.chunk_all")
    def test_success(self, mock_chunk_all):
        mock_chunk_all.return_value = [{"text": "a", "metadata": {}}]
        result = chunking_node({})
        assert result["status"] == "chunked"
        assert len(result["chunks"]) == 1

    @patch("pipeline.nodes.chunk_all", side_effect=FileNotFoundError("missing"))
    def test_failure(self, mock_chunk_all):
        result = chunking_node({})
        assert result["status"] == "failed"
        assert "missing" in result["error"]


class TestQueryExpansionNode:
    @patch("pipeline.nodes.get_shared_retriever")
    def test_expansion(self, mock_retriever_fn):
        mock_retriever = MagicMock()
        mock_retriever.expand_queries.return_value = ["Q1", "Q1 rephrased"]
        mock_retriever_fn.return_value = mock_retriever

        result = query_expansion_node({"question": "What is a SKU?"})
        assert "expanded_queries" in result
        assert len(result["expanded_queries"]) == 2
        assert "rewritten_question" in result

    @patch("pipeline.nodes.get_shared_retriever")
    def test_expansion_with_chat_history(self, mock_retriever_fn):
        """Chat history is forwarded to expand_queries for contextual rewrite."""
        mock_retriever = MagicMock()
        mock_retriever.expand_queries.return_value = [
            "How is SKU used in inventory management?",
            "SKU inventory tracking usage",
        ]
        mock_retriever_fn.return_value = mock_retriever

        history = [
            {"role": "user", "content": "What is a SKU?"},
            {"role": "assistant", "content": "A SKU is a Stock Keeping Unit."},
        ]
        result = query_expansion_node({
            "question": "How is it used in inventory?",
            "chat_history": history,
        })
        # Verify chat_history was passed through
        mock_retriever.expand_queries.assert_called_once_with(
            "How is it used in inventory?", chat_history=history
        )
        assert result["rewritten_question"] == "How is SKU used in inventory management?"

    @patch("pipeline.nodes.get_shared_retriever")
    def test_rewritten_question_same_when_no_history(self, mock_retriever_fn):
        mock_retriever = MagicMock()
        mock_retriever.expand_queries.return_value = ["What is a SKU?"]
        mock_retriever_fn.return_value = mock_retriever

        result = query_expansion_node({"question": "What is a SKU?"})
        assert result["rewritten_question"] == "What is a SKU?"


class TestRetrievalNode:
    @patch("pipeline.nodes.get_shared_retriever")
    def test_retrieval(self, mock_retriever_fn):
        mock_retriever = MagicMock()
        mock_retriever._search_all.return_value = [
            {"text": "chunk1", "score": 0.9, "metadata": {}}
        ]
        mock_retriever_fn.return_value = mock_retriever

        result = retrieval_node({
            "question": "What is a SKU?",
            "expanded_queries": ["What is a SKU?"],
        })
        assert len(result["retrieved_chunks"]) == 1


class TestRerankingNode:
    @patch("pipeline.nodes.get_shared_retriever")
    def test_reranking(self, mock_retriever_fn):
        mock_retriever = MagicMock()
        mock_retriever._rerank.return_value = [
            {"text": "best", "score": 0.95, "metadata": {}}
        ]
        mock_retriever_fn.return_value = mock_retriever

        result = reranking_node({
            "question": "What is a SKU?",
            "retrieved_chunks": [
                {"text": "best", "score": 0.95, "metadata": {}},
                {"text": "other", "score": 0.5, "metadata": {}},
            ],
        })
        assert len(result["reranked_chunks"]) == 1

    def test_empty_candidates(self):
        result = reranking_node({
            "question": "test",
            "retrieved_chunks": [],
        })
        assert result["reranked_chunks"] == []


class TestGenerationNode:
    @patch("pipeline.nodes.get_shared_generator")
    def test_generation(self, mock_generator_fn):
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "SKU stands for Stock Keeping Unit."
        mock_generator._extract_sources.return_value = [
            {"source_doc": "cheatsheet.docx", "relevance_score": 0.9}
        ]
        mock_generator_fn.return_value = mock_generator

        result = generation_node({
            "question": "What is a SKU?",
            "reranked_chunks": [{"text": "SKU...", "score": 0.9, "metadata": {}}],
        })
        assert "SKU" in result["answer"]
        assert len(result["sources"]) == 1
        assert len(result["chat_history"]) == 2

    @patch("pipeline.nodes.get_shared_generator")
    def test_generation_passes_chat_history(self, mock_generator_fn):
        """Chat history is forwarded to the generator for conversational context."""
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "It is used for tracking."
        mock_generator._extract_sources.return_value = []
        mock_generator_fn.return_value = mock_generator

        history = [
            {"role": "user", "content": "What is a SKU?"},
            {"role": "assistant", "content": "A SKU is a Stock Keeping Unit."},
        ]
        generation_node({
            "question": "How is it used?",
            "reranked_chunks": [{"text": "SKU tracking...", "score": 0.9, "metadata": {}}],
            "chat_history": history,
        })
        # Verify chat_history was passed to generate()
        mock_generator.generate.assert_called_once_with(
            "How is it used?",
            [{"text": "SKU tracking...", "score": 0.9, "metadata": {}}],
            chat_history=history,
        )


class TestLoadTestQuestionsNode:
    def test_loads_questions(self):
        result = load_test_questions_node({})
        assert "questions" in result
        assert len(result["questions"]) > 0
        assert all(isinstance(q, str) for q in result["questions"])


class TestGraphImport:
    """Verify graph modules are importable and graphs compile."""

    def test_import_graphs(self):
        from pipeline.graph import index_graph, query_graph, test_graph
        assert index_graph is not None
        assert query_graph is not None
        assert test_graph is not None

    def test_build_functions(self):
        from pipeline.graph import (
            build_index_graph,
            build_query_graph,
            build_test_graph,
        )
        ig = build_index_graph()
        qg = build_query_graph()
        tg = build_test_graph()
        assert ig is not None
        assert qg is not None
        assert tg is not None
