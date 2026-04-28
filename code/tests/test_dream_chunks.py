from __future__ import annotations

import pytest

from dream_chunks import split_poem


def test_split_poem_stanza_mode_basic():
    lines = ["a", "b", "", "c", "", "d", "e"]
    chunks = split_poem(lines, fallback_chunk_size=2)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert [c.split_mode for c in chunks] == ["stanza", "stanza", "stanza"]
    assert [c.text for c in chunks] == ["a / b", "c", "d / e"]


def test_split_poem_strips_whitespace_and_ignores_extra_blanks():
    lines = ["  a  ", "", "", "   ", " b ", "", "  "]
    chunks = split_poem(lines)
    assert len(chunks) == 2
    assert chunks[0].text == "a"
    assert chunks[1].text == "b"


def test_split_poem_empty_input_returns_empty():
    assert split_poem([]) == []


def test_split_poem_all_blanks_returns_empty():
    assert split_poem(["", " ", "   "]) == []


def test_split_poem_no_blank_markers_falls_back_to_fixed():
    lines = ["a", "b", "c", "d", "e"]
    chunks = split_poem(lines, fallback_chunk_size=2)
    assert [c.split_mode for c in chunks] == ["fixed", "fixed", "fixed"]
    assert [c.text for c in chunks] == ["a / b", "c / d", "e"]


def test_split_poem_bad_chunk_size_raises():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_poem(["a", "b"], fallback_chunk_size=0)

