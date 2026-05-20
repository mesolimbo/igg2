"""Tests for src/cluster.py."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.cluster import cluster_rows, default_k  # noqa: E402


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")


def test_default_k_small():
    assert default_k(4) == 2


def test_default_k_medium():
    # sqrt(16)/2 = 2 -> rounded 2; max(2, 2) == 2
    assert default_k(16) == 2


def test_default_k_large():
    # sqrt(100)/2 = 5
    assert default_k(100) == 5


def test_default_k_minimum_floor():
    assert default_k(1) == 2


def test_cluster_rows_returns_ints():
    df = pd.read_csv(FIXTURE, dtype=str).dropna(how="any")
    rows = df.astype(str).values.tolist()
    k = 4
    labels, km = cluster_rows(rows, k, seed=42)
    assert isinstance(labels, list)
    assert len(labels) == len(rows)
    for lbl in labels:
        assert isinstance(lbl, int)


def test_cluster_rows_labels_in_range():
    df = pd.read_csv(FIXTURE, dtype=str).dropna(how="any")
    rows = df.astype(str).values.tolist()
    k = 4
    labels, _ = cluster_rows(rows, k, seed=42)
    for lbl in labels:
        assert 0 <= lbl < k


def test_cluster_rows_deterministic():
    df = pd.read_csv(FIXTURE, dtype=str).dropna(how="any")
    rows = df.astype(str).values.tolist()
    a, _ = cluster_rows(rows, 3, seed=42)
    b, _ = cluster_rows(rows, 3, seed=42)
    assert a == b
