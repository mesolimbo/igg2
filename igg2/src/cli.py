"""Click-based CLI for igg2."""
import csv as csv_mod
import json
import sys

import click

from src.train import train as train_model, load_sidecar
from src.generate import generate_rows


@click.group()
def cli():
    """igg2 - micro language model for coherent CSV generation."""


@cli.command("train")
@click.argument("input_csv")
@click.argument("output_json")
@click.option("--k", default=None, type=int, help="Number of topic clusters.")
@click.option("--no-boundaries", is_flag=True, help="Disable boundary transitions.")
@click.option("--seed", default=42, type=int)
@click.option("--min-cluster-size", default=None, type=int)
def train_cmd(input_csv, output_json, k, no_boundaries, seed, min_cluster_size):
    """Train a model from INPUT_CSV and write to OUTPUT_JSON."""
    sidecar = load_sidecar(input_csv)

    # CLI flags win; sidecar provides defaults
    effective_k = k if k is not None else sidecar.get("k")
    effective_boundaries = (not no_boundaries) if no_boundaries else sidecar.get("boundary_conditioning", True)
    effective_min_cluster = min_cluster_size if min_cluster_size is not None else sidecar.get("min_cluster_size", 3)

    model = train_model(
        input_csv,
        k=effective_k,
        use_boundaries=effective_boundaries,
        seed=seed,
        min_cluster_size=effective_min_cluster,
    )
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    click.echo(
        f"Trained model: {model['metadata']['row_count']} rows, "
        f"{model['metadata']['k']} topics -> {output_json}"
    )


@cli.command("generate")
@click.argument("model_json")
@click.option("--count", default=10, type=int)
@click.option("--seed", default=None, type=int)
@click.option(
    "--format", "fmt",
    default="csv", type=click.Choice(["csv", "tsv", "json"]),
)
def generate_cmd(model_json, count, seed, fmt):
    """Generate rows from a trained MODEL_JSON file."""
    with open(model_json, "r", encoding="utf-8") as fh:
        model = json.load(fh)
    rows = generate_rows(model, count=count, seed=seed)
    columns = model["metadata"]["column_names"]

    if fmt == "json":
        out = [dict(zip(columns, r)) for r in rows]
        click.echo(json.dumps(out, indent=2))
    else:
        delim = "\t" if fmt == "tsv" else ","
        writer = csv_mod.writer(sys.stdout, delimiter=delim)
        writer.writerow(columns)
        for r in rows:
            writer.writerow(r)


@cli.command("inspect")
@click.argument("model_json")
def inspect_cmd(model_json):
    """Print a summary of MODEL_JSON."""
    with open(model_json, "r", encoding="utf-8") as fh:
        model = json.load(fh)
    md = model["metadata"]
    click.echo(f"source_file: {md['source_file']}")
    click.echo(f"row_count:   {md['row_count']}")
    click.echo(f"K:           {md['k']}")
    click.echo(f"columns:     {', '.join(md['column_names'])}")
    click.echo("")
    click.echo("topics:")
    for t in model["topics"]:
        rc = t.get("row_count", "?")
        click.echo(
            f"  [{t['id']:>2}] {t['prior'] * 100:5.1f}%  "
            f"({rc} rows)  {t.get('label', '')}"
        )


if __name__ == "__main__":
    cli()
