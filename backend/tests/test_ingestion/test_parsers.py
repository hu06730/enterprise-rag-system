import pytest
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "fixtures"


def test_txt_parser_returns_parsed_document():
    from app.ingestion.parsers import TxtParser
    parser = TxtParser()
    filepath = SAMPLE_DIR / "sample.txt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("Hello RAG.\nThis is a test document.", encoding="utf-8")

    result = parser.parse(str(filepath))
    assert result.text == "Hello RAG.\nThis is a test document."
    assert result.title == "sample.txt"


def test_md_parser():
    from app.ingestion.parsers import MdParser
    filepath = SAMPLE_DIR / "sample.md"
    filepath.write_text("# Title\n\nContent here.", encoding="utf-8")
    result = MdParser().parse(str(filepath))
    assert "# Title" in result.text
    assert result.metadata["format"] == "markdown"


def test_get_parser_returns_correct_type():
    from app.ingestion.parsers import get_parser, TxtParser, PdfParser, WordParser, MdParser
    assert isinstance(get_parser("a.txt"), TxtParser)
    assert isinstance(get_parser("a.pdf"), PdfParser)
    assert isinstance(get_parser("a.docx"), WordParser)
    assert isinstance(get_parser("a.md"), MdParser)


def test_get_parser_raises_for_unknown():
    import pytest
    from app.ingestion.parsers import get_parser
    with pytest.raises(ValueError, match="No parser"):
        get_parser("a.xyz")
