# igg2-mcp

`igg2-mcp` exposes the [`igg2`](../README.md) text generator as a locally run
[Model Context Protocol](https://modelcontextprotocol.io) (MCP) server, so an
MCP client (Claude Code, Claude Desktop, …) can train models and generate text
through two tools.

Everything runs in Docker. The server image is built **on top of the igg2
image**, so it invokes the igg2 CLI directly inside its own container — there
is no Docker-in-Docker and nothing is installed on your host.

## Tools

| Tool | Description |
| ---- | ----------- |
| `igg2_train` | Train a model from a CSV in the mounted data directory. |
| `igg2_generate` | Generate text from a trained model, with options. |
| `igg2_list_models` | List the trained models available in the data directory. |

Both tools return **only the igg2 CLI's final output** — the trained-model
summary or the generated text. Container build logs and diagnostics are never
surfaced.

### `igg2_train`

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `input_csv` | — | CSV filename inside `/data`. |
| `model_name` | `model.json` | Output model filename, written to `/data`. |
| `k` | auto | Number of topic clusters. |
| `seed` | `42` | Clustering seed. |
| `boundaries` | `true` | Learn inter-column boundary transitions. |
| `min_cluster_size` | `3` | Merge clusters smaller than this. |

### `igg2_generate`

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `model_name` | `model.json` | Model filename inside `/data`. |
| `count` | `10` | Number of rows to generate. |
| `seed` | none | Seed for reproducible output. |
| `diversity` | `1.0` | Topic-prior flattening; `>1` spreads themes. |
| `output_format` | `text` | `text`, `csv`, `tsv`, or `json`. |

### `igg2_list_models`

Takes no parameters. Returns a summary line for each trained model JSON found
in `/data` — row count, topic count and columns — so you can pick a
`model_name` for `igg2_generate`.

## Build

```
make build
```

This builds the base `igg2:local` image (from `../igg2`) and then the
`igg2-mcp:local` server image on top of it.

Other targets: `make lock` (regenerate `Pipfile.lock`), `make test` (run the
test suite in the container), `make help`.

## Configure your MCP client

The server speaks MCP over stdio. Point your client at a `docker run` command,
mounting a host directory as `/data`:

```json
{
  "mcpServers": {
    "igg2": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "C:/path/to/your/data:/data",
        "igg2-mcp:local"
      ]
    }
  }
}
```

For Claude Code:

```
claude mcp add igg2 -- docker run -i --rm -v C:/path/to/your/data:/data igg2-mcp:local
```

## Data directory

The host directory you mount at `/data` is the server's workspace:

- Put input CSV files there before calling `igg2_train`.
- Trained models are written there and persist for `igg2_generate`.

Filenames passed to the tools are treated as bare names inside `/data` —
directory components are stripped, so a tool call cannot read or write
anywhere else.
