"""Tests for the igg2 MCP server helpers and tool registration."""
import asyncio
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src import server  # noqa: E402


def test_safe_name_strips_directories():
    assert server._safe_name("foo.csv") == "foo.csv"
    assert server._safe_name("/data/foo.csv") == "foo.csv"
    assert server._safe_name("../../etc/passwd") == "passwd"


def test_safe_name_rejects_empty_or_dotted():
    with pytest.raises(server.Igg2Error):
        server._safe_name("")
    with pytest.raises(server.Igg2Error):
        server._safe_name("..")


def test_last_lines_keeps_tail():
    text = "a\n\nb\nc\n"
    assert server._last_lines(text, 2) == "b\nc"
    assert server._last_lines(text, 10) == "a\nb\nc"


def test_train_args_minimal():
    args = server._train_args("in.csv", "m.json", None, 42, True, None)
    assert args == ["train", "in.csv", "m.json", "--seed", "42"]


def test_train_args_all_options():
    args = server._train_args("in.csv", "m.json", 8, 7, False, 5)
    assert args == [
        "train", "in.csv", "m.json", "--seed", "7",
        "--k", "8", "--no-boundaries", "--min-cluster-size", "5",
    ]


def test_generate_args_minimal():
    args = server._generate_args("m.json", 10, None, 1.0, "text")
    assert args == [
        "generate", "m.json", "--count", "10",
        "--diversity", "1.0", "--format", "text",
    ]


def test_generate_args_with_seed():
    args = server._generate_args("m.json", 5, 3, 2.5, "json")
    assert args[:8] == [
        "generate", "m.json", "--count", "5",
        "--diversity", "2.5", "--format", "json",
    ]
    assert args[-2:] == ["--seed", "3"]


def test_tools_registered():
    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} == {
        "igg2_train", "igg2_generate", "igg2_list_models",
    }


def test_summarise_model(tmp_path):
    model = {
        "schema_version": 3,
        "metadata": {"row_count": 100, "k": 5, "column_names": ["Invention"]},
        "topics": [],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    summary = server._summarise_model(path)
    assert summary is not None
    assert "m.json" in summary
    assert "100 rows" in summary
    assert "5 topics" in summary
    assert "Invention" in summary


def test_summarise_model_rejects_non_models(tmp_path):
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    assert server._summarise_model(plain) is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not valid", encoding="utf-8")
    assert server._summarise_model(broken) is None


def test_list_models(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    assert "No trained models" in server._list_models()

    (tmp_path / "a.json").write_text(json.dumps({
        "metadata": {"row_count": 10, "k": 2, "column_names": ["X"]},
        "topics": [],
    }), encoding="utf-8")
    listing = server._list_models()
    assert "a.json" in listing
    assert "10 rows" in listing
