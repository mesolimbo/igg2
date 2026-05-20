# igg2 — A Micro Language Model for Coherent Multi-Column Generation

## Problem

Given a CSV with N columns (e.g. `Topic, Format` or
`Place, Character, Surprise`), we want to synthesise *new* rows by
sampling each column from the distribution of values seen in that
column — but in a way that keeps cells in the same generated row
**thematically aligned with each other**.

A prior experiment ("igg") trained one independent Markov chain per
column and concatenated samples. This is vertically valid but
horizontally incoherent — column 2 has no idea what column 1 just
produced, so a row like `Birdhouse Building, Cards` is just as likely
as `Birthday Party, Decorations`. Most output reads as nonsense.

The goal of igg2 is the **simplest possible upgrade** that makes
generated rows feel internally coherent, while staying nowhere near
the complexity of a neural language model.

## Core insight

Independent per-column chains throw away the only signal that links
columns: **co-occurrence within a row**. Any fix has to put a shared
latent variable across the whole row so all columns sample from a
compatible distribution.

The smallest model that does this is a **mixture of per-column Markov
chains, indexed by a discrete row-topic**. Formally it is a hidden-
variable language model:

```
P(row) = Σ_t  P(topic=t) · Π_c  P(cell_c | topic=t)
```

where `P(cell_c | topic=t)` is a small Markov chain over the words in
column `c`, trained only on rows assigned to topic `t`.

This is genuinely a "micro LM" — it has a latent state, a generative
story, and degrades gracefully to a plain per-column Markov chain when
K=1 — but training needs no gradients, no embeddings, no neural net.

## Approach

### Training pipeline

1. **Tokenise** every cell. Lowercase, strip non-alphanumerics, split
   on whitespace. (NLTK or a 20-line regex tokeniser both work; we
   prefer the regex tokeniser to keep the Docker image small.)
2. **Row vectors**: build a TF-IDF vector per row over the union of
   all tokens across all columns. Drop stop-words and any token that
   appears in > 50 % of rows — those carry no topic signal.
3. **Cluster** rows into K topics with mini-batch k-means
   (`sklearn.cluster.MiniBatchKMeans`).
   - Default `K = max(2, round(sqrt(N_rows) / 2))` — for a 200-row CSV
     this gives K ≈ 7, which is in the sweet spot for human-recognisable
     themes.
   - K is overridable per-CSV via a sidecar `<name>.igg.yaml` for
     power users.
4. **Per-(topic, column) Markov chain**: for each cluster, build a
   standard first-order Markov chain over each column's cells from
   just that cluster's rows. Each chain stores:
   - `transitions`: `{word: {next_word: prob}}`
   - `start_words`: `{word: prob}` (first word of cell)
   - `end_words`: `{word: prob}` (last word of cell)
   - `lengths`: `{length: prob}` (token count distribution)
5. **Global fallback chain**: also build a per-column chain over
   *all* rows (ignoring topic). Used at generation time when a chosen
   topic's chain dead-ends on a word with no known transition.
   Prevents fragile output from small clusters.
6. **Boundary conditioning (optional, cheap)**: while sweeping rows,
   also count `last_word(col_{c-1}) → first_word(col_c)` transitions.
   Store as `boundary_transitions[c]`. Used at generation to bias the
   start word of column c on what just came out of column c-1.

### Generation

```
sample topic t ~ P(topic)                       # multinomial weighted by cluster size
for c in columns:
    if c == 0 or boundary_transitions[c] empty for last word:
        start_word ~ topic_models[t][c].start_words
    else:
        start_word ~ boundary_transitions[c][last_word_of_prev_cell]
                     (back off to topic start_words if unseen)
    walk topic_models[t][c]  (back off to global_columns[c] on dead-ends)
    emit cell
```

The Markov walk per column is standard: pick start word, sample next
word from `transitions`, stop when a sampled length is hit AND the
current word is a known `end_word`. The whole *new* mechanism is:
**pick a topic once per row, then condition every column on it.**

## File formats

### Input: CSV (any shape)

```
Topic,Format
Birthday Party,Decorations
Birdhouse Building,Plans
...
```

First row may be a header (auto-detected) or data. No restrictions on
column count. Cells may be multi-word.

### Optional sidecar: `<name>.igg.yaml`

```yaml
k: 8                       # override default cluster count
boundary_conditioning: true
min_cluster_size: 5        # smaller clusters fold into "global" topic
```

### Output: trained model JSON

```jsonc
{
  "schema_version": 2,
  "metadata": {
    "source_file": "prompts.csv",
    "row_count": 223,
    "column_count": 2,
    "column_names": ["Topic", "Format"],
    "k": 7,
    "trained_at": "2026-05-16T12:00:00Z"
  },
  "topics": [
    {
      "id": 0,
      "prior": 0.18,                   // fraction of rows in this cluster
      "label": "party planning",       // top-N TF-IDF terms, for debugging
      "columns": [
        { "transitions": {...}, "start_words": {...},
          "end_words": {...}, "lengths": {...} }
      ]
    }
  ],
  "global_columns": [ /* per-column fallback chain over all rows */ ],
  "boundary_transitions": [ null, {"Party":{"Decorations":0.4, ...}} ]
}
```

Schema is versioned so future revisions can extend without breaking
existing consumers.

## Why this is the right "simplest"

| Option | Complexity | Cross-column coherence | Verdict |
|---|---|---|---|
| Independent per-column Markov (prior experiment) | trivial | none | baseline |
| **Topic-mixture Markov (this plan)** | ~150 lines + scikit-learn | strong | ✅ proposed |
| Concatenate columns into one Markov with `<SEP>` token | trivial | medium, but loses column structure and length control | rejected |
| LDA topic model | medium | strong | overkill; k-means is enough |
| Tiny neural LM (RNN / transformer) | high — gradients, batching, training infra | strongest | violates "ultra simple"; deferred |

Mixture-of-Markov is the sweet spot: it adds **one** sklearn dependency
and roughly **one** new function (`cluster_rows`), but it changes the
generative model from "independent columns" to "rows sampled from a
shared latent" — which is exactly what's needed for coherence.

## Project layout

```
igg2/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── Pipfile
├── Pipfile.lock
├── README.md
├── plan.md                       # this file
├── src/
│   ├── __init__.py
│   ├── tokenize.py               # cell → tokens (regex tokeniser, no NLTK)
│   ├── cluster.py                # TF-IDF + MiniBatchKMeans → topic per row
│   ├── train.py                  # csv → topic-aware json model
│   ├── generate.py               # json model → synthesised rows
│   └── cli.py                    # `igg2 train` and `igg2 generate` entrypoints
└── test/
    ├── fixtures/
    │   └── sample.csv
    ├── test_cluster.py
    ├── test_train.py
    ├── test_generate.py
    └── test_coherence.py         # mixture beats independent on held-out metric
```

## CLI surface

```
# Train a model
igg2 train INPUT.csv OUTPUT.json [--k N] [--no-boundaries] [--seed S]

# Generate rows
igg2 generate MODEL.json [--count N] [--seed S] [--format csv|json|tsv]

# Inspect a model (topic labels, cluster sizes, vocab stats)
igg2 inspect MODEL.json
```

`igg2 generate` writes synthesised rows to stdout. Default format
matches the input CSV's column layout.

## Suggested coherence metric (regression guard)

Hold out 10 % of rows from training. For each held-out row, ask: of
the top-N generations from the model, how often does at least one
generation share ≥ 2 tokens with the held-out row across multiple
columns?

- Independent-per-column Markov scores near chance.
- Topic-mixture should score materially higher.

This metric lives in `test/test_coherence.py` and is the regression
test that justifies the extra complexity. It also gives us a knob:
if a future model variant doesn't beat mixture-Markov on this metric,
it's not worth adopting.

## Docker + pipenv setup

Project uses **pipenv** for dependency management *inside the image*,
a **Makefile** as the single user-facing task runner, and ships as a
**Docker image** for every operation — building, testing, training,
generating, even dependency locking. Every common task has a `make`
target so contributors don't need to remember docker invocations.

### Why Docker is the only execution path

Pipenv (and `pip` underneath it) executes arbitrary code from
downloaded packages at install time — `setup.py`, build backends,
post-install hooks. A single malicious or compromised transitive
dependency could read host secrets (`~/.ssh`, `~/.aws`,
browser profiles, shell history) the instant `pipenv install` runs on
a developer laptop.

To shrink that blast radius, **igg2 never runs Python or pipenv on
the host**. All dependency resolution, installation, linting, testing,
training, and generation happens inside the container, which has no
access to anything outside the mounted `./data` directory. Contributors
only ever invoke `make` and `docker`; the Python toolchain is sealed
inside the image.

### `Pipfile`

```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
scikit-learn = "*"
pandas = "*"
pyyaml = "*"
click = "*"

[dev-packages]
pytest = "*"
pytest-cov = "*"

[requires]
python_version = "3.14"
```

The Pipfile intentionally has **no `[scripts]` section**. Scripts
would be invoked via `pipenv run …` on the host, which is exactly
what this project's supply-chain policy forbids. All command shortcuts
live in the Makefile, which routes through Docker.

### `Dockerfile`

```dockerfile
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIPENV_VENV_IN_PROJECT=1

RUN pip install --no-cache-dir pipenv

WORKDIR /app

# Install deps in their own layer for cache friendliness
COPY Pipfile Pipfile.lock ./
RUN pipenv sync --system

# Copy source last so code changes don't bust the deps layer
COPY src/ ./src/

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
```

### `docker-compose.yml`

Convenience for local use — mounts the host's working directory so
CSVs and output JSONs round-trip without rebuilds.

```yaml
services:
  igg2:
    build: .
    image: igg2:local
    volumes:
      - ./data:/data
    working_dir: /data
    # Override on the command line, e.g.:
    #   docker compose run --rm igg2 train input.csv model.json
    #   docker compose run --rm igg2 generate model.json --count 20
```

### `Makefile`

The Makefile is the canonical entrypoint for every task. **Every
target shells out to Docker** — there is no host-Python codepath, by
design (see "Why Docker is the only execution path" above). `make
help` lists everything.

```makefile
# Configurable: override on the command line, e.g. `make generate ARGS="--count 50"`
IMAGE   ?= igg2:local
DATA    ?= $(PWD)/data
INPUT   ?= input.csv
MODEL   ?= model.json
ARGS    ?=

# Runtime container: app image with the project's CLI as entrypoint.
DOCKER_RUN  = docker run --rm -v "$(DATA):/data" -w /data $(IMAGE)

# Tooling container: same image, but override entrypoint so we can invoke
# pipenv / pytest / python directly inside it. Mounts the *repo* (not just
# ./data) so test/, src/, and Pipfile* are visible during dev tasks.
DOCKER_TOOL = docker run --rm -v "$(PWD):/app" -w /app \
              --entrypoint "" $(IMAGE)

.PHONY: help build lock test lint clean train generate inspect shell

help:  ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## Build the Docker image (only host-side step)
	docker build -t $(IMAGE) .

lock: build  ## Regenerate Pipfile.lock inside the container
	$(DOCKER_TOOL) pipenv lock

test: build  ## Run the test suite inside the container
	$(DOCKER_TOOL) pytest test/ -v

lint: build  ## Static checks inside the container
	$(DOCKER_TOOL) python -m compileall src/

clean:  ## Remove build artefacts and caches (host-side files only)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info

# Application commands — all run inside Docker, mounting ./data as /data
train: build  ## Train: make train INPUT=foo.csv MODEL=foo.json
	$(DOCKER_RUN) train $(INPUT) $(MODEL) $(ARGS)

generate: build  ## Generate: make generate MODEL=foo.json ARGS="--count 20"
	$(DOCKER_RUN) generate $(MODEL) $(ARGS)

inspect: build  ## Inspect a model: make inspect MODEL=foo.json
	$(DOCKER_RUN) inspect $(MODEL)

shell: build  ## Drop into a shell inside the image (debugging)
	docker run --rm -it -v "$(PWD):/app" -w /app \
		--entrypoint /bin/bash $(IMAGE)
```

There are deliberately **no** `pipenv install`, `pipenv run`, or
host-`python` targets. Adding one would defeat the supply-chain
isolation that the rest of the setup exists to provide.

### Typical workflows

```
# First-time setup — the only host-side step
make build

# Iterate (every command runs inside the container)
make train    INPUT=sample.csv MODEL=sample.json
make generate MODEL=sample.json ARGS="--count 20"
make test

# Refresh the pinned dependency set (also containerised)
make lock
```

The Docker image is the unit of distribution **and** the unit of
execution. Any environment that can run `make` and `docker` can train,
generate, test, and lock dependencies without ever invoking Python or
pipenv on the host. CI runs exactly the same `make` targets — there
is no "local fast path" that could diverge from the shipped image or
expose the host to package-install code execution.

## Phasing

1. **Phase 1 — algorithm + CLI** (1–2 days): `train.py`, `generate.py`,
   `cluster.py`, `cli.py`, coherence test passing on a sample CSV.
2. **Phase 2 — Docker image**: Dockerfile + compose, smoke-test
   end-to-end inside the container.
3. **Phase 3 — tuning knobs**: per-CSV sidecar yaml, topic labels in
   `igg2 inspect`, boundary-conditioning toggle.
4. **Phase 4 — optional integrations** (deferred): pluggable export to
   downstream tools (e.g. MCP server, web UI). Not part of v1.

## Out of scope (deliberately)

- Word embeddings, neural LMs, transformers — not needed at this
  vocab size and unlikely to beat the mixture model until CSVs grow
  past ~10k rows.
- Cross-row dependencies (every row stays i.i.d. given its topic).
- Multi-language / non-English tokenisation.
- Cloud deployment, web UIs, API servers — out of v1 scope; the
  Docker image is the only delivery target.

## Construction

You have at your disposal several expert subagents familiar with all aspects of the Software Development Life Cycle:

| Subagent                 | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| **product-manager**      | Plans product features, defines user stories, manages stakeholder requirements, and creates project roadmaps. Use to translate business needs into actionable development plans. |
| **software-architect**   | Designs system architecture, makes technology decisions, plans for scalability, and reviews architectural approaches. Use when choosing between patterns, technologies, or designing robust system structures. |
| **software-developer**   | Implements features, writes code, fixes bugs, refactors existing code, and reviews implementations. Use for all hands-on coding work requiring clean, tested, maintainable code. Aims for 80% test coverage. |
| **ux-designer**          | Designs user interfaces, creates wireframes, plans user flows, and ensures accessibility compliance. Use for user-centered design decisions and interface planning. |
| **qa-engineer**          | Designs test strategies, creates test plans, writes test cases, and identifies edge cases. Use to ensure comprehensive quality assurance coverage. |
| **security-engineer**    | Reviews code for security vulnerabilities, designs secure systems, plans security controls, and ensures compliance. Use to identify and mitigate security risks. |
| **performance-engineer** | Optimizes application performance, conducts load testing, analyzes bottlenecks, and plans for scalability. Use when systems need to be made faster or validated for capacity. |
| **devops-engineer**      | Sets up CI/CD pipelines, configures infrastructure, manages deployments, and implements observability. Use for automation and operational excellence. |
| **release-manager**      | Plans releases, coordinates deployments, manages release pipelines, and handles rollback procedures. Use to ensure smooth, reliable software releases. |
| **technical-writer**     | Creates documentation, writes API references, develops tutorials, and improves existing docs. Use for clear, user-focused technical communication. |

Begin by assigning a subagent to perform any needed research, then have the agent break down your goal into granular, actionable tasks. Make sure you keep the plan as simple as possible while still satisfying your goal. IMPORTANT: You and your agents should avoid over-engineering, keep it simple. Avoid monolithic tasks, opt instead for a set of smaller SMART subtasks.

Use your subagents as you see fit to help you refine your tasks, and keep the scope of each task as small as possible. You should keep track of your tasks in a markdown file in the .ralph directory. You can use that directory to store your work files and any other information that might be useful to your agents and/or that you may need to resume your work after a restart. Be succinct in your notes.

IMPORTANT: Be sure to list the work files you add in .ralph/prompt.md.

After you've identified all your tasks, pick the single most important and useful one, then assign it to the most appropriate subagent. IMPORTANT: Work on just one task at a time. VERY IMPORTANT: Every time a subagent completes a task, update your task list.

You may also revise prompt.md in .ralph when you reach significant milestones, or when you need a major shift in strategy. You may install additional tools that your subagents require, so long as you remain security conscious. You have full sudo access to your system.  You also have a working `docker` CLI talking to the host daemon (docker-out-of-docker) — if the launcher mounted the socket, `docker ps`, `docker build`, `docker run` etc. all work. When you bind-mount paths into containers you launch, use `$RALPH_HOST_WORKSPACE` (set by the launcher) instead of `/workspace`, because the host daemon resolves volume paths against the host filesystem, not against the ralph container.

Once all tasks are complete, select one or more agents to review the work, then create a final clean-up task list and once again assign the clean-up steps, one at a time, to the appropriate agent(s). IMPORTANT: Subagents should handle all the work, you are mainly tasked with coordinating subagents.

After clean-up, perform one last in-depth review and update your status document(s) to wrap up.

Do not assume anything is unimplemented. Review your workspace carefully after reading this prompt, and if you find all tasks have been completed already, identify the single most important incremental improvement task you can perform, then assign it to one of your subagents. Improvement categories include (but are not limited to): revenue growth, security, accessibility, and maintainability.
