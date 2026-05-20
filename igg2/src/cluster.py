"""TF-IDF + MiniBatchKMeans clustering for row topic assignment."""
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


def default_k(n_rows: int) -> int:
    """Heuristic for number of clusters given row count."""
    return max(2, round(math.sqrt(n_rows) / 2))


def cluster_rows(rows, k, seed=42):
    """
    Cluster rows by TF-IDF on their joined text.

    rows: list of rows, each row is a list of cell strings.
    Returns (labels, vectorizer) where labels[i] is the cluster id for row i.
    """
    docs = [" ".join(cell for cell in row) for row in rows]

    # Be tolerant of tiny corpora: max_df=0.5 can drop everything.
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", max_df=0.5, min_df=1
        )
        X = vectorizer.fit_transform(docs)
        if X.shape[1] == 0:
            raise ValueError("empty vocabulary after pruning")
    except ValueError:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        X = vectorizer.fit_transform(docs)

    n_clusters = min(k, X.shape[0])
    if n_clusters < 2:
        n_clusters = 1

    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, n_init=3
    )
    km.fit(X)
    labels = [int(lbl) for lbl in km.labels_]
    # Stash centroids/vectorizer for downstream label extraction.
    km.vectorizer_ = vectorizer
    return labels, km
