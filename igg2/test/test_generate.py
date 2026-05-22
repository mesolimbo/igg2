"""Tests for src/generate.py and the CLI format options."""
import json
import os
import random
import sys

from click.testing import CliRunner

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.train import train  # noqa: E402
from src.generate import generate_rows  # noqa: E402
from src.cli import cli  # noqa: E402


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")


def _trained_model():
    return train(FIXTURE, k=3, seed=42)


def test_generate_row_count():
    model = _trained_model()
    rows = generate_rows(model, count=10, seed=0)
    assert len(rows) == 10


def test_generate_column_count():
    model = _trained_model()
    rows = generate_rows(model, count=5, seed=0)
    n_cols = model["metadata"]["column_count"]
    for r in rows:
        assert len(r) == n_cols


def test_generate_non_empty_cells():
    model = _trained_model()
    rows = generate_rows(model, count=20, seed=0)
    for r in rows:
        for cell in r:
            assert isinstance(cell, str)
            assert len(cell) > 0


def test_generate_cells_do_not_end_in_stop_words():
    from src.tokenize import STOP_WORDS
    model = _trained_model()
    rows = generate_rows(model, count=50, seed=0)
    for r in rows:
        for cell in r:
            last = cell.split()[-1].lower()
            assert last not in STOP_WORDS, f"cell ends in glue word: {cell!r}"


def test_generate_deterministic_with_seed():
    model = _trained_model()
    a = generate_rows(model, count=10, seed=123)
    b = generate_rows(model, count=10, seed=123)
    assert a == b


def test_generate_different_seeds_differ():
    model = _trained_model()
    a = generate_rows(model, count=10, seed=1)
    b = generate_rows(model, count=10, seed=2)
    # Extremely unlikely to be identical with two different seeds.
    assert a != b


def test_generate_accepts_diversity():
    model = _trained_model()
    rows = generate_rows(model, count=10, seed=0, diversity=2.5)
    assert len(rows) == 10
    for r in rows:
        for cell in r:
            assert len(cell) > 0


def test_diversity_flattens_topic_selection():
    from src.generate import _pick_topic
    # id 1 is the rare topic; flattening must select it far more often.
    topics = [{"prior": 0.95, "id": 0}, {"prior": 0.05, "id": 1}]
    rng = random.Random(0)
    skewed = sum(_pick_topic(rng, topics, diversity=1.0)["id"] for _ in range(1000))
    rng = random.Random(0)
    flat = sum(_pick_topic(rng, topics, diversity=12.0)["id"] for _ in range(1000))
    assert flat > skewed


def _write_model(tmp_path):
    model = _trained_model()
    p = tmp_path / "model.json"
    p.write_text(json.dumps(model))
    return str(p)


def test_as_sentence_joins_and_capitalises():
    from src.cli import _as_sentence
    assert _as_sentence(["quiet librarian", "reveals prophecy"]) == (
        "Quiet librarian reveals prophecy"
    )


def test_cli_format_text_is_default(tmp_path):
    mp = _write_model(tmp_path)
    runner = CliRunner()
    res = runner.invoke(cli, ["generate", mp, "--count", "3", "--seed", "1"])
    assert res.exit_code == 0, res.output
    lines = [ln for ln in res.output.strip().splitlines() if ln]
    assert len(lines) == 3  # sentences only, no header row
    for ln in lines:
        assert "," not in ln  # spaces between columns, not commas
        assert ln[:1] == ln[:1].upper()  # first word capitalised
        assert ln[1:] == ln[1:].lower()  # ...and only the first word


def test_cli_format_csv(tmp_path):
    mp = _write_model(tmp_path)
    runner = CliRunner()
    res = runner.invoke(
        cli, ["generate", mp, "--count", "3", "--seed", "1", "--format", "csv"]
    )
    assert res.exit_code == 0, res.output
    lines = [ln for ln in res.output.strip().splitlines() if ln]
    assert len(lines) == 4  # header + 3 rows
    assert "," in lines[0]


def test_cli_format_tsv(tmp_path):
    mp = _write_model(tmp_path)
    runner = CliRunner()
    res = runner.invoke(
        cli, ["generate", mp, "--count", "3", "--seed", "1", "--format", "tsv"]
    )
    assert res.exit_code == 0, res.output
    lines = [ln for ln in res.output.strip().splitlines() if ln]
    assert len(lines) == 4
    assert "\t" in lines[0]


def test_cli_format_json(tmp_path):
    mp = _write_model(tmp_path)
    runner = CliRunner()
    res = runner.invoke(
        cli, ["generate", mp, "--count", "3", "--seed", "1", "--format", "json"]
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert isinstance(data, list)
    assert len(data) == 3
    for entry in data:
        assert isinstance(entry, dict)


def test_cli_diversity_flag(tmp_path):
    mp = _write_model(tmp_path)
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["generate", mp, "--count", "3", "--seed", "1", "--diversity", "3.0"],
    )
    assert res.exit_code == 0, res.output
    lines = [ln for ln in res.output.strip().splitlines() if ln]
    assert len(lines) == 3  # 3 sentences (text is the default format)


def test_cli_inspect(tmp_path):
    mp = _write_model(tmp_path)
    runner = CliRunner()
    res = runner.invoke(cli, ["inspect", mp])
    assert res.exit_code == 0, res.output
    assert "row_count" in res.output
    assert "topics" in res.output
