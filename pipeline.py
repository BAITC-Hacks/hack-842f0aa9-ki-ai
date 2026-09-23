#!/usr/bin/env python3
"""Run the complete reproducible pipeline from parquet inputs to CSV outputs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from checks.check_outputs import validate_outputs, validate_delivery_outputs
from src.data_io import DataValidationError, load_data, save_outputs, validate_input_data
from src.data_io import build_delivery_outputs


def _load_analyzer():
    try:
        from src.analysis import analyze
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "src/analysis.py is not integrated yet. It must define "
            "analyze(edges, nodes, transactions) and return "
            "nodes_roles, clusters, top_nodes, node_metrics."
        ) from exc
    return analyze


def _require_dataframes(result: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(result, tuple) or len(result) != 4:
        raise DataValidationError(
            "analyze(...) must return exactly four DataFrames: "
            "nodes_roles, clusters, top_nodes, node_metrics"
        )
    names = ("nodes_roles", "clusters", "top_nodes", "node_metrics")
    for name, value in zip(names, result, strict=True):
        if not isinstance(value, pd.DataFrame):
            raise DataValidationError(f"analyze(...) returned non-DataFrame value for {name}")
    return result


def run_pipeline(data_dir: str | Path, out_dir: str | Path) -> dict[str, float | int | str]:
    started = time.perf_counter()
    edges, nodes, transactions = load_data(data_dir)
    input_summary = validate_input_data(edges, nodes, transactions)

    analyze = _load_analyzer()
    outputs = _require_dataframes(analyze(edges, nodes, transactions))
    nodes_roles, clusters, top_nodes, node_metrics = outputs

    output_summary = validate_outputs(
        nodes_roles,
        clusters,
        top_nodes,
        node_metrics,
        source_nodes=nodes,
        source_edges=edges,
    )
    extras = build_delivery_outputs(nodes_roles, node_metrics)
    validate_delivery_outputs(nodes_roles, extras['node_insights.csv'], extras['ranking_stability.csv'])
    save_outputs(
        out_dir,
        nodes_roles=nodes_roles,
        clusters=clusters,
        top_nodes=top_nodes,
        node_metrics=node_metrics,
        extra_outputs=extras,
    )

    elapsed = time.perf_counter() - started
    if elapsed > 300:
        raise DataValidationError(f"pipeline exceeded the 5-minute limit: {elapsed:.2f}s")

    return {
        **{f"input_{key}": value for key, value in input_summary.items()},
        **{f"output_{key}": value for key, value in output_summary.items()},
        "elapsed_seconds": elapsed,
        "out_dir": str(Path(out_dir).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="directory with the three parquet files")
    parser.add_argument("--out", default="out", help="directory for generated CSV files")
    args = parser.parse_args()

    try:
        summary = run_pipeline(args.data, args.out)
    except (DataValidationError, FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"PIPELINE FAILED: {exc}", file=sys.stderr)
        return 1

    print("PIPELINE COMPLETED")
    for key, value in summary.items():
        if key == "elapsed_seconds":
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
