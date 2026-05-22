"""Train a topic-conditioned Markov-chain model from a CSV."""
import datetime
import os
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

from src.tokenize import tokenize
from src.cluster import cluster_rows, default_k


def load_sidecar(csv_path: str) -> dict:
    """Load <stem>.igg.yaml next to csv_path if it exists; return {} otherwise."""
    csv_path_str = os.fspath(csv_path)
    if "://" in csv_path_str:
        # Mirror the URL-rejection in train(); a URL has no meaningful sibling
        # sidecar on the local filesystem.
        return {}
    p = pathlib.Path(csv_path_str)
    sidecar = p.with_suffix(".igg.yaml")
    if sidecar.is_file():
        with open(sidecar, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _normalise(counter):
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _empty_chain():
    return {
        "transitions": {},
        "transitions2": {},
        "start_words": {},
        "end_words": {},
        "lengths": {},
    }


def _build_chain(cells):
    """Build a Markov chain dict from a list of cell strings.

    Two transition tables are estimated: an order-2 table keyed on the
    previous word pair ("w1 w2" -> next-word counts) and an order-1 table
    keyed on the single previous word. Generation prefers the order-2
    table for local fluency and backs off to order-1 when a pair is unseen.
    """
    transitions = defaultdict(Counter)
    transitions2 = defaultdict(Counter)
    start_counts = Counter()
    end_counts = Counter()
    length_counts = Counter()

    for cell in cells:
        toks = tokenize(cell)
        if not toks:
            continue
        length_counts[str(len(toks))] += 1
        start_counts[toks[0]] += 1
        end_counts[toks[-1]] += 1
        for a, b in zip(toks, toks[1:]):
            transitions[a][b] += 1
        for a, b, c in zip(toks, toks[1:], toks[2:]):
            transitions2[a + " " + b][c] += 1

    return {
        "transitions": {
            w: _normalise(nexts) for w, nexts in transitions.items()
        },
        "transitions2": {
            pair: _normalise(nexts) for pair, nexts in transitions2.items()
        },
        "start_words": _normalise(start_counts),
        "end_words": _normalise(end_counts),
        "lengths": _normalise(length_counts),
    }


def _topic_labels(km, top_n=3):
    """Top-N TF-IDF terms per cluster centroid."""
    vec = getattr(km, "vectorizer_", None)
    if vec is None:
        return {i: "" for i in range(len(km.cluster_centers_))}
    feat = vec.get_feature_names_out()
    centers = km.cluster_centers_
    labels = {}
    for i, c in enumerate(centers):
        idx = np.argsort(c)[::-1][:top_n]
        terms = [feat[j] for j in idx if c[j] > 0]
        labels[i] = " ".join(terms)
    return labels


def _merge_small_clusters(labels, min_size):
    """Reassign small clusters to the largest remaining cluster."""
    counts = Counter(labels)
    small = {c for c, n in counts.items() if n < min_size}
    if not small or len(small) == len(counts):
        return labels
    keep = [c for c in counts if c not in small]
    if not keep:
        return labels
    target = max(keep, key=lambda c: counts[c])
    return [target if lbl in small else lbl for lbl in labels]


def _build_boundary_transitions(rows, labels_per_col):
    """boundary_transitions[c] maps last-word-of-col(c-1) -> next-word counts."""
    n_cols = len(rows[0]) if rows else 0
    out = [None] * n_cols
    for c in range(1, n_cols):
        bucket = defaultdict(Counter)
        for row in rows:
            prev_toks = tokenize(row[c - 1])
            cur_toks = tokenize(row[c])
            if not prev_toks or not cur_toks:
                continue
            bucket[prev_toks[-1]][cur_toks[0]] += 1
        out[c] = {
            w: _normalise(nexts) for w, nexts in bucket.items()
        }
    return out


def train(csv_path, k=None, use_boundaries=True, seed=42, min_cluster_size=3):
    """Train a model and return the model dict."""
    # Reject URL-like inputs: pandas.read_csv transparently fetches over
    # http(s)/s3/gs/ftp/etc. via fsspec, which would let a caller turn this
    # CLI into an SSRF primitive or pull untrusted data at training time.
    # Require a real local file path.
    csv_path_str = os.fspath(csv_path)
    if "://" in csv_path_str:
        raise ValueError(
            "csv_path must be a local file path, not a URL-like string"
        )
    resolved = pathlib.Path(csv_path_str)
    if not resolved.is_file():
        raise FileNotFoundError(f"CSV not found or not a regular file: {csv_path_str}")
    df = pd.read_csv(str(resolved), header="infer", dtype=str, keep_default_na=False)
    df = df.dropna(how="any").reset_index(drop=True)
    df = df.astype(str)

    column_names = [str(c) for c in df.columns]
    rows = df.values.tolist()
    n_rows = len(rows)
    n_cols = len(column_names)

    if n_rows == 0:
        raise ValueError("CSV has no usable rows after cleaning")

    if k is None:
        k = default_k(n_rows)

    labels, km = cluster_rows(rows, k, seed=seed)
    labels = _merge_small_clusters(labels, min_cluster_size)

    topic_label_map = _topic_labels(km)

    unique_topics = sorted(set(labels))
    topics = []
    for tid in unique_topics:
        topic_rows = [rows[i] for i, lbl in enumerate(labels) if lbl == tid]
        col_chains = []
        for c in range(n_cols):
            cells = [r[c] for r in topic_rows]
            col_chains.append(_build_chain(cells))
        prior = len(topic_rows) / n_rows
        topics.append({
            "id": int(tid),
            "prior": prior,
            "label": topic_label_map.get(tid, ""),
            "row_count": len(topic_rows),
            "columns": col_chains,
        })

    global_columns = [
        _build_chain([r[c] for r in rows]) for c in range(n_cols)
    ]

    if use_boundaries and n_cols > 1:
        boundary_transitions = _build_boundary_transitions(rows, labels)
    else:
        boundary_transitions = [None] * n_cols

    model = {
        "schema_version": 3,
        "metadata": {
            "source_file": os.path.basename(csv_path),
            "row_count": n_rows,
            "column_count": n_cols,
            "column_names": column_names,
            "k": len(unique_topics),
            "trained_at": datetime.datetime.now(datetime.timezone.utc)
                          .replace(tzinfo=None).isoformat() + "Z",
        },
        "topics": topics,
        "global_columns": global_columns,
        "boundary_transitions": boundary_transitions,
    }
    return model
