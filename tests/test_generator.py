from pipeline.generator import AnswerGenerator


class TestExtractSources:
    def test_deduplicates_identical_sources(self):
        chunks = [
            {"text": "a", "metadata": {"source_doc": "doc1.docx", "title": "Title1", "term": ""}, "score": 0.9},
            {"text": "b", "metadata": {"source_doc": "doc1.docx", "title": "Title1", "term": ""}, "score": 0.8},
        ]
        sources = AnswerGenerator._extract_sources(chunks)
        assert len(sources) == 1

    def test_different_sources_both_included(self):
        chunks = [
            {"text": "a", "metadata": {"source_doc": "doc1.docx", "title": "", "term": "SKU"}, "score": 0.9},
            {"text": "b", "metadata": {"source_doc": "doc2.docx", "title": "", "term": ""}, "score": 0.8},
        ]
        sources = AnswerGenerator._extract_sources(chunks)
        assert len(sources) == 2

    def test_handles_missing_score(self):
        chunks = [
            {"text": "a", "metadata": {"source_doc": "doc1.docx", "title": "", "term": ""}, "score": None},
        ]
        sources = AnswerGenerator._extract_sources(chunks)
        assert sources[0]["relevance_score"] == 0

    def test_empty_chunks(self):
        assert AnswerGenerator._extract_sources([]) == []
