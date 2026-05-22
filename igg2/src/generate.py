"""Generate rows by walking topic-conditioned Markov chains."""
import random

from src.tokenize import STOP_WORDS


def _weighted_choice(rng, d):
    """Sample a key from a {key: prob} dict."""
    if not d:
        return None
    keys = list(d.keys())
    weights = [d[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def _pick_topic(rng, topics, diversity=1.0):
    """Pick a topic, optionally flattening the priors.

    `diversity` raises each prior to the power 1/diversity: 1.0 leaves the
    as-trained distribution untouched, higher values even it out so smaller
    topics get picked more often, and very large values approach uniform.
    """
    priors = {i: t["prior"] for i, t in enumerate(topics) if t["prior"] > 0}
    if not priors:
        priors = {i: 1.0 for i in range(len(topics))}
    if diversity > 0 and diversity != 1.0:
        priors = {i: p ** (1.0 / diversity) for i, p in priors.items()}
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


# How much weight the order-2 (word-pair) table gets when it is mixed with
# the order-1 table. The order-1 share is always blended in so that a
# deterministic order-2 context still yields word-level variety instead of
# reproducing a training phrase verbatim.
ORDER2_WEIGHT = 0.75


def _lookup(chains, table, key):
    """Return the first non-empty `table[key]` across the given chains."""
    for chain in chains:
        nexts = chain.get(table, {}).get(key)
        if nexts:
            return nexts
    return {}


def _mix(dists_weights):
    """Linearly mix {word: prob} dicts: [(dist, weight), ...] -> {word: score}."""
    merged = {}
    for dist, weight in dists_weights:
        if not dist or weight <= 0:
            continue
        for word, prob in dist.items():
            merged[word] = merged.get(word, 0.0) + weight * prob
    return merged


def _next_word(rng, tokens, topic_chain, global_chain):
    """Pick the next word from an order-2 / order-1 interpolation.

    The order-2 table (keyed on the previous word pair) drives local
    fluency; the order-1 table (keyed on the single previous word) is
    always mixed in at weight 1 - ORDER2_WEIGHT, so a deterministic order-2
    context produces word-level variety rather than a verbatim training
    phrase. Topic chains are preferred over the global chain within each
    order.
    """
    chains = (topic_chain, global_chain)
    order1 = _lookup(chains, "transitions", tokens[-1])
    order2 = {}
    if len(tokens) >= 2:
        order2 = _lookup(chains, "transitions2", tokens[-2] + " " + tokens[-1])

    if order2 and order1:
        merged = _mix([
            (order2, ORDER2_WEIGHT),
            (order1, 1.0 - ORDER2_WEIGHT),
        ])
    else:
        merged = order2 or order1
    return _weighted_choice(rng, merged)


def _trim_glue(tokens):
    """Drop trailing function words so a cell never ends on a glue word."""
    while tokens and tokens[-1] in STOP_WORDS:
        tokens.pop()
    return tokens


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
        nxt = _next_word(rng, tokens, topic_chain, global_chain)
        if nxt is None:
            break
        tokens.append(nxt)
        if len(tokens) >= safety_cap:
            break
    return _trim_glue(tokens)


def generate_rows(model, count=10, seed=None, diversity=1.0):
    """Generate `count` rows from a trained model dict.

    `diversity` flattens the topic priors at selection time; see
    `_pick_topic`. The default of 1.0 leaves the as-trained mix untouched.
    """
    rng = random.Random(seed)
    topics = model.get("topics", [])
    if not topics:
        raise ValueError("Model has no topics")

    global_cols = model.get("global_columns", [])
    boundaries = model.get("boundary_transitions", [])
    n_cols = model["metadata"]["column_count"]

    output = []
    for _ in range(count):
        topic = _pick_topic(rng, topics, diversity)
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
                # Final fallback: pick a global start word, skipping glue
                # words so the cell still ends on something meaningful.
                fallback = _weighted_choice(rng, g_chain.get("start_words", {}))
                if not fallback or fallback in STOP_WORDS:
                    fallback = "unknown"
                tokens = [fallback]
            cell = " ".join(tokens)
            row.append(cell)
            prev_last = tokens[-1] if tokens else None
        output.append(row)
    return output
