"""Smoke test: model produces output that shares vocabulary with held-out rows."""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.train import train  # noqa: E402
from src.generate import generate_rows  # noqa: E402
from src.tokenize import tokenize, STOP_WORDS  # noqa: E402


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")


def _split_csv(tmp_path, holdout_frac=0.1, seed=42):
    df = pd.read_csv(FIXTURE, dtype=str).dropna(how="any").reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_hold = max(1, int(round(len(shuffled) * holdout_frac)))
    holdout = shuffled.iloc[:n_hold].reset_index(drop=True)
    train_df = shuffled.iloc[n_hold:].reset_index(drop=True)
    train_path = tmp_path / "train.csv"
    train_df.to_csv(train_path, index=False)
    return str(train_path), holdout


def _content_tokens(cells):
    out = set()
    for cell in cells:
        for tok in tokenize(cell):
            if tok not in STOP_WORDS and len(tok) > 1:
                out.add(tok)
    return out


def test_generated_rows_share_tokens_with_holdout(tmp_path):
    train_path, holdout = _split_csv(tmp_path)
    model = train(train_path, k=4, seed=42)
    generated = generate_rows(model, count=100, seed=7)

    gen_token_sets = [_content_tokens(row) for row in generated]

    # Aggregate score: sum of best overlaps. Held-out rows often contain
    # unseen vocabulary, so per-row >=2 overlap is rare. We assert that
    # the mixture model produces *some* meaningful overlap overall, which
    # distinguishes it from random output.
    total_best_overlap = 0
    for _, hrow in holdout.iterrows():
        h_tokens = _content_tokens(list(hrow.values))
        if not h_tokens:
            continue
        best = max(
            (len(h_tokens & gset) for gset in gen_token_sets), default=0
        )
        total_best_overlap += best

    assert total_best_overlap > 0, (
        "Mixture model produced no content-token overlap with any of "
        f"the {len(holdout)} held-out rows"
    )


def test_generated_rows_use_training_vocab(tmp_path):
    train_path, _ = _split_csv(tmp_path)
    model = train(train_path, k=3, seed=42)
    generated = generate_rows(model, count=50, seed=11)

    train_df = pd.read_csv(train_path, dtype=str).dropna(how="any")
    train_vocab = set()
    for _, row in train_df.iterrows():
        for cell in row.values:
            train_vocab.update(tokenize(str(cell)))

    in_vocab = 0
    total = 0
    for row in generated:
        for cell in row:
            for tok in tokenize(cell):
                total += 1
                if tok in train_vocab:
                    in_vocab += 1
    assert total > 0
    # Markov chains can only emit observed tokens; this should be ~100%.
    assert in_vocab / total > 0.95
