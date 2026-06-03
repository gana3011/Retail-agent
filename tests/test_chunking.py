from pipeline.chunking import chunk_text, make_text, build_metadata


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "hello world"
        assert chunk_text(text, max_words=500) == [text]

    def test_long_text_split(self):
        text = " ".join(str(i) for i in range(100))
        chunks = chunk_text(text, max_words=30, overlap=0)
        assert len(chunks) == 4  # 100 / 30 = 4 chunks

    def test_overlap_produces_more_chunks(self):
        text = " ".join(str(i) for i in range(100))
        no_overlap = chunk_text(text, max_words=30, overlap=0)
        with_overlap = chunk_text(text, max_words=30, overlap=10)
        assert len(with_overlap) >= len(no_overlap)

    def test_overlap_does_not_exceed_max_words(self):
        text = " ".join(str(i) for i in range(100))
        chunks = chunk_text(text, max_words=30, overlap=30)
        assert all(len(c.split()) <= 30 for c in chunks)

    def test_empty_text(self):
        assert chunk_text("") == [""]


class TestMakeText:
    def test_term_definition(self):
        elem = {
            "element_type": "term_definition",
            "term": "SKU",
            "full_form": "Stock Keeping Unit",
            "explanation": "Unique ID for each product",
            "example": "Red Medium T-shirt",
            "why_needed": "Granular tracking",
            "used_by": ["Inventory", "Merchandising"],
        }
        result = make_text(elem)
        assert "Term: SKU" in result
        assert "Full Form: Stock Keeping Unit" in result
        assert "Used By: Inventory, Merchandising" in result

    def test_scenario(self):
        elem = {
            "element_type": "scenario",
            "scenario_number": 1,
            "title": "Test Scenario",
            "domain": "Store Operations",
            "scenario_text": "A customer had an issue",
            "what_went_wrong": "System error",
            "impact": "Lost sale",
        }
        result = make_text(elem)
        assert "Scenario 1: Test Scenario" in result
        assert "What Went Wrong: System error" in result
        assert "Impact: Lost sale" in result

    def test_qa_pair(self):
        elem = {
            "element_type": "qa_pair",
            "question": "What is a SKU?",
            "answer": "Stock Keeping Unit",
        }
        result = make_text(elem)
        assert "Q: What is a SKU?" in result
        assert "A: Stock Keeping Unit" in result

    def test_paragraph_with_context(self):
        elem = {
            "element_type": "paragraph",
            "text": "Some content here",
            "process_name": "Procurement",
            "slide": "Slide 1",
        }
        result = make_text(elem)
        assert "[Procurement]" in result
        assert "[Slide 1]" in result
        assert "Some content here" in result

    def test_unknown_type_falls_back_to_text(self):
        elem = {"element_type": "unknown", "text": "fallback"}
        assert make_text(elem) == "fallback"


class TestBuildMetadata:
    def test_term_definition(self):
        elem = {
            "source_doc": "cheatsheet.docx",
            "doc_type": "cheatsheet",
            "element_type": "term_definition",
            "term": "SKU",
            "section_title": "Product Terms",
        }
        meta = build_metadata(elem)
        assert meta["domain"] == "Glossary"
        assert meta["term"] == "SKU"
        assert meta["source_doc"] == "cheatsheet.docx"

    def test_scenario(self):
        elem = {
            "source_doc": "scenarios.docx",
            "doc_type": "scenarios",
            "element_type": "scenario",
            "scenario_number": 1,
            "title": "Test",
            "domain": "Store Ops",
        }
        meta = build_metadata(elem)
        assert meta["domain"] == "Store Ops"
        assert meta["title"] == "Test"

    def test_qa_pair(self):
        elem = {
            "source_doc": "faq.docx",
            "doc_type": "faq",
            "element_type": "qa_pair",
            "question_number": 1,
            "section": "A. Basics",
        }
        meta = build_metadata(elem)
        assert meta["domain"] == "FAQ"
        assert meta["section"] == "A. Basics"

    def test_unknown_type_no_crash(self):
        elem = {
            "source_doc": "x.docx",
            "doc_type": "unknown",
            "element_type": "something_new",
        }
        meta = build_metadata(elem)
        assert meta["source_doc"] == "x.docx"
        assert meta["element_type"] == "something_new"
