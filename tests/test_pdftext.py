from __future__ import annotations

from conftest import LONG_LINES

from pdftext import MIN_PDF_CHARS, chunk_text, extract_text_from_pdf


def test_extract_from_path_and_file_object(make_pdf):
    pdf = make_pdf(LONG_LINES)
    text = extract_text_from_pdf(pdf)
    assert "Line 0: we train the model" in text
    assert "Line 69:" in text
    assert len(text) >= MIN_PDF_CHARS

    with open(pdf, "rb") as f:
        assert extract_text_from_pdf(f) == text
        assert f.tell() == 0  # rewound so the caller can read it again


def test_chunk_ids_are_strings_and_cover_the_text():
    text = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert [c["chunk_id"] for c in chunks] == [str(i) for i in range(len(chunks))]
    assert chunks[0]["text"].startswith("word0 ")
    assert chunks[-1]["text"].endswith("word2999")
    # consecutive chunks overlap
    assert chunks[1]["text"].split()[0] in chunks[0]["text"]


def test_chunking_terminates_on_a_long_unbroken_run():
    # A space near the start of the window followed by a run longer than the window
    # used to make the overlap step move backwards forever.
    text = "abc " + "x" * 9000
    chunks = chunk_text(text)
    assert 1 <= len(chunks) <= 5
    assert "".join(c["text"] for c in chunks).count("x") >= 9000
