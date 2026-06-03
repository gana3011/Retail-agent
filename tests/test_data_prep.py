from phase_0_data_preparation import clean_text, get_doc_metadata, extract_terminology_from_text


class TestCleanText:
    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_normalizes_newlines(self):
        assert clean_text("line1\r\nline2") == "line1\nline2"

    def test_collapses_multiple_newlines(self):
        assert clean_text("a\n\n\n\nb") == "a\n\nb"

    def test_normalizes_unicode_quotes(self):
        assert clean_text("\u2018hello\u2019") == "'hello'"
        assert clean_text("\u201chello\u201d") == '"hello"'

    def test_normalizes_dashes(self):
        assert clean_text("a\u2013b") == "a-b"
        assert clean_text("a\u2014b") == "a--b"

    def test_removes_control_chars(self):
        assert clean_text("\uf0b7bullet") == "-bullet"

    def test_handles_none(self):
        assert clean_text(None) == ""
        assert clean_text("") == ""


class TestGetDocMetadata:
    def test_detects_cheatsheet(self):
        meta = get_doc_metadata(r"C:\path\Cheatsheet Retail.docx")
        assert meta["doc_type"] == "cheatsheet"

    def test_detects_scenarios(self):
        meta = get_doc_metadata(r"C:\path\Retail Domain Scenarios.docx")
        assert meta["doc_type"] == "scenarios"

    def test_detects_faq(self):
        meta = get_doc_metadata(r"C:\path\Retail Domain FAQs.docx")
        assert meta["doc_type"] == "faq"

    def test_returns_version(self):
        meta = get_doc_metadata(r"C:\path\any.docx")
        assert meta["version"] == "1.0"
        assert meta["source_doc"] == "any.docx"


class TestExtractTerminology:
    def test_basic_term(self):
        result = extract_terminology_from_text("Term: SKU - Stock Keeping Unit")
        assert result is not None
        assert result["term"] == "SKU"
        assert "Stock Keeping Unit" in result["definition"]

    def test_with_explained_keyword(self):
        result = extract_terminology_from_text("Terminology Explained: ATP - Available to Promise")
        assert result is not None
        assert result["term"] == "ATP"

    def test_non_term_line(self):
        result = extract_terminology_from_text("Just a normal paragraph")
        assert result is None
