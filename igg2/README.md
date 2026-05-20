# igg2

`igg2` is a micro language model for generating coherent multi-column CSV data.
It learns the joint vocabulary, phrasing, and inter-column coupling of a small
tabular dataset by clustering rows into topics and training a Markov chain per
topic per column. The result is a tiny, inspectable JSON model that produces
new rows which read like they came from the same source, without hauling in a
neural stack or sending your data to an external API.

## Quick start

```
make build
make train INPUT=mydata.csv MODEL=mymodel.json
make generate MODEL=mymodel.json ARGS="--count 20"
make test
```

`INPUT` and `MODEL` are resolved relative to `./data/` by default (the directory
mounted into the container). Drop your CSV into `./data/` and the commands above
just work.

## How it works

- **Tokenize** each cell in the CSV into lowercased word tokens.
- **Cluster** rows into `k` topics using TF-IDF features over the concatenated
  row text. `k` is chosen automatically based on row count if not specified,
  and tiny clusters are merged into the largest neighbour.
- **Train per-topic Markov chains** for every column: start-word, end-word,
  length, and word-to-word transition distributions are estimated from the
  rows assigned to that topic.
- **Boundary conditioning** records, for each column, how the last word of the
  previous column biases the first word of the next column. This is what keeps
  generated rows internally coherent across columns.
- **Generate** by sampling a topic from the prior, then walking the topic's
  Markov chain for each column in turn, falling back to the global chain when
  a topic-local transition is unseen.

The model is plain JSON: every probability, transition, and topic label is
inspectable and diff-able.

## CLI reference

The CLI is exposed as three subcommands. Inside Docker the entrypoint is
`igg2`; locally you can run `python -m src.cli`.

### `train`

```
igg2 train INPUT_CSV OUTPUT_JSON [OPTIONS]
```

Trains a model from `INPUT_CSV` and writes it to `OUTPUT_JSON`.

| Option                    | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `--k INTEGER`             | auto    | Number of topic clusters. Auto-picked from row count.  |
| `--no-boundaries`         | off     | Disable inter-column boundary transitions.             |
| `--seed INTEGER`          | `42`    | Random seed for clustering.                            |
| `--min-cluster-size INT`  | `3`     | Clusters smaller than this are merged into the largest.|

If a `<input>.igg.yaml` sidecar exists next to the CSV, its values are used as
defaults; explicit CLI flags always win.

### `generate`

```
igg2 generate MODEL_JSON [OPTIONS]
```

Emits synthesized rows from a trained model to stdout.

| Option              | Default | Description                                  |
| ------------------- | ------- | -------------------------------------------- |
| `--count INTEGER`   | `10`    | Number of rows to generate.                  |
| `--seed INTEGER`    | none    | Seed for deterministic generation.           |
| `--format {csv,tsv,json}` | `csv` | Output format.                          |

### `inspect`

```
igg2 inspect MODEL_JSON
```

Prints a human-readable summary: source file, row count, topic count, column
names, and a per-topic breakdown of size, prior, and the top TF-IDF terms used
as the topic label.

## Sidecar YAML

You can ship per-dataset training defaults alongside the CSV by placing a
`<stem>.igg.yaml` file next to it. For example, `sample.csv` is paired with
`sample.igg.yaml`. Supported keys:

```yaml
k: 5                       # number of topic clusters
boundary_conditioning: true  # whether to learn inter-column transitions
min_cluster_size: 3        # merge clusters smaller than this
```

All keys are optional. CLI flags always override sidecar values.

## Output formats

Given the included `test/fixtures/sample.csv` (columns: `Character`, `Action`,
`Discovery`), `generate` produces output like:

**CSV** (default):

```
Character,Action,Discovery
quiet librarian,organising dusty archives,reveals ancient prophecy
stoned postal carrier,curing pandemic,discovers werewolf lair
forgetful wizard,brewing morning coffee,creates new spell accidentally
```

**TSV** (`--format tsv`):

```
Character	Action	Discovery
quiet librarian	organising dusty archives	reveals ancient prophecy
stoned postal carrier	curing pandemic	discovers werewolf lair
```

**JSON** (`--format json`):

```json
[
  {
    "Character": "quiet librarian",
    "Action": "organising dusty archives",
    "Discovery": "reveals ancient prophecy"
  },
  {
    "Character": "stoned postal carrier",
    "Action": "curing pandemic",
    "Discovery": "discovers werewolf lair"
  }
]
```

## Docker

Every workflow (`build`, `train`, `generate`, `inspect`, `test`, `lint`, `lock`)
runs inside a Docker image for supply-chain safety: no Python packages, model
artifacts, or training scripts touch your host interpreter. Your data is
mounted from `./data/` into `/data/` in the container.

For interactive debugging:

```
make shell
```

drops you into a bash prompt inside the same image with the repository mounted
at `/app`.

## Repository layout

```
src/
  cli.py        Click entrypoint (train / generate / inspect)
  train.py      Clustering and Markov-chain estimation
  generate.py   Topic-conditioned row sampling
  tokenize.py   Cell tokenization
  cluster.py    TF-IDF + k-means row clustering
test/
  fixtures/     Sample CSVs used by the test suite
Makefile        Wrapper around the Docker workflow
```

## License

See `LICENSE` for terms.
