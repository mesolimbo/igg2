"""Tests for src/tokenize.py."""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.tokenize import STOP_WORDS, tokenize  # noqa: E402


def test_basic_multi_word():
    assert tokenize("hello world foo") == ["hello", "world", "foo"]


def test_uppercase_is_lowercased():
    assert tokenize("Hello World") == ["hello", "world"]


def test_punctuation_is_stripped():
    assert tokenize("hello, world!") == ["hello", "world"]


def test_numbers_are_kept():
    assert tokenize("room 42") == ["room", "42"]


def test_empty_string_returns_empty_list():
    assert tokenize("") == []


def test_whitespace_only_returns_empty_list():
    assert tokenize("   \t\n  ") == []


def test_stop_words_are_included():
    # tokenize() does not filter stop words; that's the caller's job.
    result = tokenize("the quick and the dead")
    assert result == ["the", "quick", "and", "the", "dead"]
    assert "the" in result
    assert "and" in result


def test_only_stop_words_tokenizes_to_non_empty():
    result = tokenize("the and is")
    assert result == ["the", "and", "is"]
    assert len(result) > 0


def test_stop_words_is_a_set_with_expected_members():
    assert isinstance(STOP_WORDS, set)
    assert "the" in STOP_WORDS
    assert "and" in STOP_WORDS
    assert "is" in STOP_WORDS


def test_apostrophes_are_stripped():
    assert tokenize("it's") == ["it", "s"]
