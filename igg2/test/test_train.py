"""Tests for src/train.py."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.train import train  # noqa: E402


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")


def test_train_schema_version():
    model = train(FIXTURE, k=4, seed=42)
    assert model["schema_version"] == 3


def test_train_metadata_column_count():
    model = train(FIXTURE, k=4, seed=42)
    # sample.csv has Character, Action, Discovery -> 3 columns
    assert model["metadata"]["column_count"] == 3
    assert len(model["metadata"]["column_names"]) == 3


def test_train_number_of_topics_matches_k():
    model = train(FIXTURE, k=4, seed=42, min_cluster_size=1)
    assert len(model["topics"]) == 4
    assert model["metadata"]["k"] == 4


def test_train_priors_sum_to_one():
    model = train(FIXTURE, k=4, seed=42)
    total = sum(t["prior"] for t in model["topics"])
    assert abs(total - 1.0) < 1e-6


def test_train_topic_columns_correct_length():
    model = train(FIXTURE, k=3, seed=42)
    n_cols = model["metadata"]["column_count"]
    for topic in model["topics"]:
        assert len(topic["columns"]) == n_cols
        for chain in topic["columns"]:
            assert set(chain.keys()) >= {
                "transitions", "transitions2",
                "start_words", "end_words", "lengths",
            }


def test_train_global_columns_length():
    model = train(FIXTURE, k=3, seed=42)
    assert len(model["global_columns"]) == model["metadata"]["column_count"]


def test_train_builds_order2_transitions():
    model = train(FIXTURE, k=3, seed=42)
    # At least one global column should have order-2 (word-pair) entries.
    assert any(col.get("transitions2") for col in model["global_columns"])
    # Every order-2 key is exactly two whitespace-joined tokens.
    for col in model["global_columns"]:
        for key in col["transitions2"]:
            assert len(key.split(" ")) == 2


def test_train_boundary_transitions_length_and_null_first():
    model = train(FIXTURE, k=3, seed=42)
    bt = model["boundary_transitions"]
    assert len(bt) == model["metadata"]["column_count"]
    assert bt[0] is None
    for entry in bt[1:]:
        assert isinstance(entry, dict)


def test_train_no_boundaries_flag():
    model = train(FIXTURE, k=3, seed=42, use_boundaries=False)
    for entry in model["boundary_transitions"]:
        assert entry is None


def test_train_lengths_keys_are_strings():
    model = train(FIXTURE, k=3, seed=42)
    for topic in model["topics"]:
        for chain in topic["columns"]:
            for key in chain["lengths"].keys():
                assert isinstance(key, str)


def test_load_sidecar_missing(tmp_path):
    from src.train import load_sidecar
    result = load_sidecar(str(tmp_path / "nonexistent.csv"))
    assert result == {}


def test_load_sidecar_present(tmp_path):
    from src.train import load_sidecar
    sidecar = tmp_path / "data.igg.yaml"
    sidecar.write_text("k: 5\nmin_cluster_size: 10\n")
    result = load_sidecar(str(tmp_path / "data.csv"))
    assert result == {"k": 5, "min_cluster_size": 10}
