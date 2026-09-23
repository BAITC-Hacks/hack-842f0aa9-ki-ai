"""Input/output helpers and source-data validation for the money graph case."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_INPUT_COLUMNS: dict[str, set[str]] = {
    "edges": {"src", "dst", "sum_kzt", "n_tx", "depth"},
    "nodes": {"gid", "depth", "is_seed"},
    "transactions": {"src", "dst", "date", "sum_kzt"},
}


class DataValidationError(ValueError):
    """Raised when source data or generated results violate the agreed contract."""


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three case parquet files and return edges, nodes and transactions."""

    data_path = Path(data_dir)
    files = {
        "edges": data_path / "edges.parquet",
        "nodes": data_path / "nodes.parquet",
        "transactions": data_path / "transactions.parquet",
    }
    missing_files = [str(path) for path in files.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Required data files are missing:\n- " + "\n- ".join(missing_files)
        )

    edges = pd.read_parquet(files["edges"])
    nodes = pd.read_parquet(files["nodes"])
    transactions = pd.read_parquet(files["transactions"])
    transactions = transactions.copy()
    transactions["date"] = pd.to_datetime(transactions["date"], errors="raise")
    return edges, nodes, transactions


def _require_columns(frame: pd.DataFrame, name: str) -> None:
    required = REQUIRED_INPUT_COLUMNS[name]
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"{name} is missing columns: {', '.join(missing)}")


def _require_no_nulls(frame: pd.DataFrame, name: str, columns: set[str]) -> None:
    null_counts = frame[list(columns)].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        details = ", ".join(f"{column}={count}" for column, count in bad.items())
        raise DataValidationError(f"{name} contains null values: {details}")


def validate_input_data(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    enforce_case_counts: bool = True,
) -> dict[str, int | float | str]:
    """Validate schema, graph integrity and edge/transaction consistency.

    Returns a compact summary suitable for console logging.
    """

    frames: Mapping[str, pd.DataFrame] = {
        "edges": edges,
        "nodes": nodes,
        "transactions": transactions,
    }
    for name, frame in frames.items():
        _require_columns(frame, name)
        _require_no_nulls(frame, name, REQUIRED_INPUT_COLUMNS[name])

    if nodes["gid"].duplicated().any():
        duplicate_count = int(nodes["gid"].duplicated(keep=False).sum())
        raise DataValidationError(f"nodes.gid contains {duplicate_count} duplicate rows")

    if edges.duplicated(["src", "dst"]).any():
        duplicate_count = int(edges.duplicated(["src", "dst"], keep=False).sum())
        raise DataValidationError(
            f"edges contains {duplicate_count} rows with duplicate src/dst pairs"
        )

    for frame_name, frame, columns in (
        ("nodes", nodes, ("gid",)),
        ("edges", edges, ("src", "dst")),
        ("transactions", transactions, ("src", "dst")),
    ):
        for column in columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
                raise DataValidationError(f"{frame_name}.{column} must contain integer ids")

    node_ids = set(nodes["gid"].astype("int64"))
    referenced_ids = (
        set(edges["src"].astype("int64"))
        | set(edges["dst"].astype("int64"))
        | set(transactions["src"].astype("int64"))
        | set(transactions["dst"].astype("int64"))
    )
    unknown_ids = referenced_ids - node_ids
    if unknown_ids:
        sample = ", ".join(map(str, sorted(unknown_ids)[:10]))
        raise DataValidationError(f"edges/transactions reference unknown gid values: {sample}")

    if (edges["sum_kzt"] <= 0).any() or (transactions["sum_kzt"] <= 0).any():
        raise DataValidationError("sum_kzt must be positive in edges and transactions")
    if (edges["n_tx"] <= 0).any():
        raise DataValidationError("edges.n_tx must be positive")

    aggregated = (
        transactions.groupby(["src", "dst"], as_index=False)
        .agg(tx_sum_kzt=("sum_kzt", "sum"), tx_count=("sum_kzt", "size"))
    )
    comparison = edges[["src", "dst", "sum_kzt", "n_tx"]].merge(
        aggregated,
        on=["src", "dst"],
        how="outer",
        indicator=True,
    )
    if not (comparison["_merge"] == "both").all():
        counts = comparison["_merge"].value_counts().to_dict()
        raise DataValidationError(f"edges and transactions contain different pairs: {counts}")
    if not np.allclose(
        comparison["sum_kzt"].astype(float),
        comparison["tx_sum_kzt"].astype(float),
        rtol=1e-9,
        atol=0.01,
    ):
        raise DataValidationError("edge sums do not match aggregated transaction sums")
    if not np.array_equal(
        comparison["n_tx"].astype("int64"), comparison["tx_count"].astype("int64")
    ):
        raise DataValidationError("edges.n_tx does not match transaction counts")

    if enforce_case_counts:
        expected = {"nodes": 2248, "edges": 3119, "transactions": 4840, "seeds": 81}
        actual = {
            "nodes": len(nodes),
            "edges": len(edges),
            "transactions": len(transactions),
            "seeds": int(nodes["is_seed"].astype(bool).sum()),
        }
        mismatches = [
            f"{key}: expected {expected[key]}, got {actual[key]}"
            for key in expected
            if actual[key] != expected[key]
        ]
        if mismatches:
            raise DataValidationError("Unexpected case dataset size: " + "; ".join(mismatches))

    edge_nodes = set(edges["src"].astype("int64")) | set(edges["dst"].astype("int64"))
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "transactions": len(transactions),
        "seeds": int(nodes["is_seed"].astype(bool).sum()),
        "orphans": len(node_ids - edge_nodes),
        "turnover_kzt": float(edges["sum_kzt"].sum()),
        "period": f"{transactions['date'].min().date()} — {transactions['date'].max().date()}",
    }


def save_outputs(
    out_dir: str | Path,
    *,
    nodes_roles: pd.DataFrame,
    clusters: pd.DataFrame,
    top_nodes: pd.DataFrame,
    node_metrics: pd.DataFrame,
) -> None:
    """Save the four agreed result tables using stable UTF-8 CSV output."""

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "nodes_roles.csv": nodes_roles,
        "clusters.csv": clusters,
        "top_nodes.csv": top_nodes,
        "node_metrics.csv": node_metrics,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_path / filename, index=False, encoding="utf-8")
