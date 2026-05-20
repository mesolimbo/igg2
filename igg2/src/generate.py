"""Generate rows by walking topic-conditioned Markov chains."""
import random


def _weighted_choice(rng, d):
    """Sample a key from a {key: prob} dict."""
    if not d:
        return None
    keys = list(d.keys())
    weights = [d[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def _pick_topic(rng, topics):
    priors = {i: t["prior"] for i, t in enumerate(topics) if t["prior"] > 0}
    if not priors:
        priors = {i: 1.0 for i in range(len(topics))}
    idx = _weighted_choice(rng, priors)
    return topics[idx]


def _pick_length(rng, lengths_dict):
    if not lengths_dict:
        return 3
    key = _weighted_choice(rng, lengths_dict)
    try:
        return max(1, int(key))
    except (ValueError, TypeError):
        return 3


def _next_word(rng, current, topic_chain, global_chain):
    nexts = topic_chain.get("transitions", {}).get(current)
    if not nexts:
        nexts = global_chain.get("transitions", {}).get(current)
    if not nexts:
        return None
    return _weighted_choice(rng, nexts)


def _generate_cell(rng, topic_chain, global_chain, boundary_map, prev_last):
    """Generate one cell as a list of tokens."""
    start_word = None
    if boundary_map and prev_last and prev_last in boundary_map:
        start_word = _weighted_choice(rng, boundary_map[prev_last])
    if start_word is None:
        start_word = _weighted_choice(rng, topic_chain.get("start_words", {}))
    if start_word is None:
        start_word = _weighted_choice(rng, global_chain.get("start_words", {}))
    if start_word is None:
        return []

    target_len = _pick_length(rng, topic_chain.get("lengths", {}))
    if target_len < 1:
        target_len = 1
    safety_cap = max(target_len * 2, target_len + 1)

    end_words = topic_chain.get("end_words", {}) or global_chain.get("end_words", {})

    tokens = [start_word]
    while True:
        if len(tokens) >= safety_cap:
            break
        if len(tokens) >= target_len and tokens[-1] in end_words:
            break
        nxt = _next_word(rng, tokens[-1], topic_chain, global_chain)
        if nxt is None:
            break
        tokens.append(nxt)
        if len(tokens) >= safety_cap:
            break
    return tokens


def generate_rows(model, count=10, seed=None):
    """Generate `count` rows from a trained model dict."""
    rng = random.Random(seed)
    topics = model.get("topics", [])
    if not topics:
        raise ValueError("Model has no topics")

    global_cols = model.get("global_columns", [])
    boundaries = model.get("boundary_transitions", [])
    n_cols = model["metadata"]["column_count"]

    output = []
    for _ in range(count):
        topic = _pick_topic(rng, topics)
        topic_cols = topic.get("columns", [])
        row = []
        prev_last = None
        for c in range(n_cols):
            t_chain = topic_cols[c] if c < len(topic_cols) else {}
            g_chain = global_cols[c] if c < len(global_cols) else {}
            b_map = boundaries[c] if c < len(boundaries) else None
            if c == 0:
                b_map = None
            tokens = _generate_cell(rng, t_chain, g_chain, b_map, prev_last)
            if not tokens:
                # Final fallback: pick any start word from the global chain.
                fallback = _weighted_choice(rng, g_chain.get("start_words", {}))
                tokens = [fallback] if fallback else ["unknown"]
            cell = " ".join(tokens)
            row.append(cell)
            prev_last = tokens[-1] if tokens else None
        output.append(row)
    return output
