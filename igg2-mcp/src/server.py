"""MCP server exposing the igg2 text generator as train/generate tools.

This server runs inside a Docker image built on top of ``igg2:local``, so it
invokes the igg2 CLI directly within its own container -- there is no nested
Docker and nothing is installed on the host. Tools return only the igg2 CLI's
final output; build noise and diagnostics are never surfaced.
"""
import asyncio
import json
import pathlib
import sys
from enum import Enum
from typing import Annotated

from pydantic import Field
from mcp.server.fastmcp import FastMCP

# Container directory bind-mounted from the host. Input CSVs are read from
# here and trained models are written here, so models persist between the
# train and generate tools.
DATA_DIR = pathlib.Path("/data")

# The igg2 CLI, invoked as `python -m src.cli` within this container.
CLI_MODULE = "src.cli"

# Clustering a large CSV can take a while; generation is quick.
TRAIN_TIMEOUT = 600.0
GENERATE_TIMEOUT = 120.0

mcp = FastMCP(
    "igg2_mcp",
    instructions=(
        "Train tiny igg2 text-generation models from CSV files and generate "
        "new lines of text from them. Place input CSVs in the mounted /data "
        "directory; trained models are saved there and reused by igg2_generate."
    ),
)


class OutputFormat(str, Enum):
    """Output format for generated rows."""

    TEXT = "text"
    CSV = "csv"
    TSV = "tsv"
    JSON = "json"


class Igg2Error(RuntimeError):
    """Raised when an igg2 invocation fails; carries a concise message."""


def _safe_name(name: str) -> str:
    """Reduce a caller-supplied filename to a bare basename within /data.

    Directory components are stripped so a tool call cannot read or write
    outside the mounted data directory.
    """
    base = pathlib.Path(name).name
    if not base or base in {".", ".."}:
        raise Igg2Error(f"invalid filename: {name!r}")
    return base


def _last_lines(text: str, count: int) -> str:
    """Return the last ``count`` non-empty lines of ``text``, joined."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-count:])


def _train_args(csv, model, k, seed, boundaries, min_cluster_size):
    """Build the ``igg2 train`` argument list."""
    args = ["train", csv, model, "--seed", str(seed)]
    if k is not None:
        args += ["--k", str(k)]
    if not boundaries:
        args += ["--no-boundaries"]
    if min_cluster_size is not None:
        args += ["--min-cluster-size", str(min_cluster_size)]
    return args


def _generate_args(model, count, seed, diversity, output_format):
    """Build the ``igg2 generate`` argument list."""
    args = [
        "generate", model,
        "--count", str(count),
        "--diversity", str(diversity),
        "--format", output_format,
    ]
    if seed is not None:
        args += ["--seed", str(seed)]
    return args


async def _run_cli(args: list[str], timeout: float) -> str:
    """Run the igg2 CLI inside this container and return its stdout.

    Raises Igg2Error with a concise message on failure. The raw stderr /
    diagnostic stream is not surfaced beyond a short tail for context.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", CLI_MODULE, *args,
        cwd=str(DATA_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise Igg2Error(f"igg2 timed out after {timeout:.0f}s")

    if proc.returncode != 0:
        detail = _last_lines(stderr.decode(errors="replace"), 5)
        raise Igg2Error(detail or "igg2 command failed")
    return stdout.decode(errors="replace")


@mcp.tool(
    name="igg2_train",
    annotations={
        "title": "Train an igg2 model",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def igg2_train(
    input_csv: Annotated[str, Field(
        description="Filename of the source CSV inside the mounted /data "
                    "directory, e.g. 'inventions.csv'.",
        min_length=1,
    )],
    model_name: Annotated[str, Field(
        description="Filename for the trained model, written into /data and "
                    "reused later by igg2_generate.",
        min_length=1,
    )] = "model.json",
    k: Annotated[int | None, Field(
        description="Number of topic clusters. Auto-picked from the row "
                    "count if omitted.",
        ge=1,
    )] = None,
    seed: Annotated[int, Field(
        description="Random seed for clustering.",
    )] = 42,
    boundaries: Annotated[bool, Field(
        description="Whether to learn inter-column boundary transitions.",
    )] = True,
    min_cluster_size: Annotated[int | None, Field(
        description="Clusters smaller than this are merged into the largest.",
        ge=1,
    )] = None,
) -> str:
    """Train an igg2 text-generation model from a CSV file.

    Reads ``input_csv`` from the mounted /data directory, clusters its rows
    into topics, fits per-topic Markov chains, and writes the model JSON to
    /data/<model_name>. Use igg2_generate afterwards to produce text.

    Returns:
        A one-line summary of the trained model, e.g.
        "Trained model: 1570 rows, 20 topics -> model.json".
    """
    csv = _safe_name(input_csv)
    if not (DATA_DIR / csv).is_file():
        raise Igg2Error(
            f"CSV not found in /data: {csv}. Place the file in the mounted "
            f"data directory and try again."
        )
    model = _safe_name(model_name)
    args = _train_args(csv, model, k, seed, boundaries, min_cluster_size)
    return (await _run_cli(args, TRAIN_TIMEOUT)).strip()


@mcp.tool(
    name="igg2_generate",
    annotations={
        "title": "Generate text from an igg2 model",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def igg2_generate(
    model_name: Annotated[str, Field(
        description="Filename of a model previously produced by igg2_train, "
                    "inside the mounted /data directory.",
        min_length=1,
    )] = "model.json",
    count: Annotated[int, Field(
        description="Number of rows to generate.",
        ge=1, le=1000,
    )] = 10,
    seed: Annotated[int | None, Field(
        description="Seed for reproducible output. Omit for fresh randomness "
                    "on every call.",
    )] = None,
    diversity: Annotated[float, Field(
        description="Topic-prior flattening. 1.0 uses the trained topic mix; "
                    "higher values spread output across more topics.",
        ge=0.0,
    )] = 1.0,
    output_format: Annotated[OutputFormat, Field(
        description="Output format: 'text' (one sentence per row), 'csv', "
                    "'tsv', or 'json'.",
    )] = OutputFormat.TEXT,
) -> str:
    """Generate new rows of text from a trained igg2 model.

    Loads /data/<model_name> and emits ``count`` synthesized rows. With the
    default 'text' format each row is one sentence; 'csv', 'tsv' and 'json'
    return the columns separately for tabular use.

    Returns:
        The generated output in the requested format.
    """
    model = _safe_name(model_name)
    if not (DATA_DIR / model).is_file():
        raise Igg2Error(
            f"Model not found in /data: {model}. Train it first with "
            f"igg2_train."
        )
    fmt = output_format.value if isinstance(output_format, OutputFormat) \
        else str(output_format)
    args = _generate_args(model, count, seed, diversity, fmt)
    return (await _run_cli(args, GENERATE_TIMEOUT)).rstrip("\n")


def _summarise_model(path: pathlib.Path) -> str | None:
    """Return a one-line summary of an igg2 model file.

    Returns None if ``path`` is unreadable or is not a recognisable igg2
    model (a JSON object carrying ``metadata`` and ``topics``).
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "metadata" not in data \
            or "topics" not in data:
        return None
    md = data.get("metadata", {})
    cols = ", ".join(md.get("column_names", []) or []) or "?"
    summary = (
        f"{path.name} — {md.get('row_count', '?')} rows, "
        f"{md.get('k', '?')} topics; columns: {cols}"
    )
    trained = md.get("trained_at")
    if trained:
        summary += f"; trained {trained}"
    return summary


def _list_models() -> str:
    """Build the model listing for the /data directory."""
    if not DATA_DIR.is_dir():
        return "No trained models found: /data is not mounted."
    summaries = [
        s for s in (_summarise_model(p) for p in sorted(DATA_DIR.glob("*.json")))
        if s
    ]
    if not summaries:
        return "No trained models found in /data. Use igg2_train to create one."
    return "\n".join(summaries)


@mcp.tool(
    name="igg2_list_models",
    annotations={
        "title": "List trained igg2 models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def igg2_list_models() -> str:
    """List the trained igg2 models available in the /data directory.

    Scans the mounted data directory for model JSON files produced by
    igg2_train and summarises each one (row count, topic count, columns).
    Pass a listed filename as the ``model_name`` argument to igg2_generate.

    Returns:
        A newline-separated list of models with summaries, or a message
        when no models are present.
    """
    return await asyncio.to_thread(_list_models)


def main() -> None:
    """Entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
