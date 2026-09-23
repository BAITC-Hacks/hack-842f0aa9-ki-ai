#!/usr/bin/env python3
"""Validate the generated CSV files against the hackathon contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import DataValidationError  # noqa: E402


ALLOWED_ROLES = {
    "consolidator",
    "transit",
    "distributor",
    "terminal",
    "coordinator",
    "peripheral",
}

REQUIRED_COLUMNS = {
    "nodes_roles": {
        "gid",
        "role",
        "role_score",
        "cluster_id",
        "priority_score",
        "evidence",
    },
    "clusters": {
        "cluster_id",
        "n_nodes",
        "n_seed",
        "sum_kzt_internal",
        "top_gids",
        "hypothesis",
    },
    "top_nodes": {"rank", "gid", "role", "priority_score", "why"},
    "node_metrics": {"gid"},
}


def _fail(message: str) -> None:
    raise DataValidationError(message)


def _require_columns(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(REQUIRED_COLUMNS[name] - set(frame.columns))
    if missing:
        _fail(f"{name}.csv is missing columns: {', '.join(missing)}")


def _integer_series(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
        _fail(f"{label} must contain integers without nulls")
    return numeric.astype("int64")


def _score_series(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        _fail(f"{label} must contain finite numeric values")
    if not numeric.between(0, 1, inclusive="both").all():
        _fail(f"{label} must be in the 0..1 range")
    return numeric.astype(float)


def _nonempty_text(series: pd.Series, label: str, max_length: int | None = None) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    if text.eq("").any():
        _fail(f"{label} contains empty values")
    if max_length is not None and text.str.len().gt(max_length).any():
        longest = int(text.str.len().max())
        _fail(f"{label} exceeds {max_length} characters (maximum found: {longest})")
    return text


def _read_outputs(out_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "nodes_roles": out_dir / "nodes_roles.csv",
        "clusters": out_dir / "clusters.csv",
        "top_nodes": out_dir / "top_nodes.csv",
        "node_metrics": out_dir / "node_metrics.csv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        _fail("Missing output files:\n- " + "\n- ".join(missing))
    return {name: pd.read_csv(path) for name, path in files.items()}


def validate_outputs(
    nodes_roles: pd.DataFrame,
    clusters: pd.DataFrame,
    top_nodes: pd.DataFrame,
    node_metrics: pd.DataFrame,
    *,
    source_nodes: pd.DataFrame | None = None,
    source_edges: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Validate generated tables and cross-file relationships."""

    frames = {
        "nodes_roles": nodes_roles,
        "clusters": clusters,
        "top_nodes": top_nodes,
        "node_metrics": node_metrics,
    }
    for name, frame in frames.items():
        _require_columns(frame, name)

    role_gids = _integer_series(nodes_roles["gid"], "nodes_roles.gid")
    if role_gids.duplicated().any():
        _fail("nodes_roles.gid must be unique")
    if len(nodes_roles) != 2248:
        _fail(f"nodes_roles.csv must contain 2248 rows, got {len(nodes_roles)}")

    roles = _nonempty_text(nodes_roles["role"], "nodes_roles.role")
    invalid_roles = sorted(set(roles) - ALLOWED_ROLES)
    if invalid_roles:
        _fail("nodes_roles.role contains invalid values: " + ", ".join(invalid_roles))
    _score_series(nodes_roles["role_score"], "nodes_roles.role_score")
    _score_series(nodes_roles["priority_score"], "nodes_roles.priority_score")
    cluster_ids = _integer_series(nodes_roles["cluster_id"], "nodes_roles.cluster_id")
    _nonempty_text(nodes_roles["evidence"], "nodes_roles.evidence", max_length=200)

    metric_gids = _integer_series(node_metrics["gid"], "node_metrics.gid")
    if metric_gids.duplicated().any():
        _fail("node_metrics.gid must be unique")
    if set(metric_gids) != set(role_gids):
        _fail("node_metrics.gid must match nodes_roles.gid exactly")

    cluster_table_ids = _integer_series(clusters["cluster_id"], "clusters.cluster_id")
    if cluster_table_ids.duplicated().any():
        _fail("clusters.cluster_id must be unique")
    if set(cluster_ids) != set(cluster_table_ids):
        _fail("cluster ids in nodes_roles and clusters must match exactly")
    for column in ("n_nodes", "n_seed"):
        values = _integer_series(clusters[column], f"clusters.{column}")
        if (values < 0).any():
            _fail(f"clusters.{column} cannot be negative")
    internal_sum = pd.to_numeric(clusters["sum_kzt_internal"], errors="coerce")
    if internal_sum.isna().any() or not np.isfinite(internal_sum).all() or (internal_sum < 0).any():
        _fail("clusters.sum_kzt_internal must contain finite non-negative values")
    _nonempty_text(clusters["top_gids"], "clusters.top_gids")
    _nonempty_text(clusters["hypothesis"], "clusters.hypothesis")

    observed_cluster_sizes = cluster_ids.value_counts().sort_index()
    declared_cluster_sizes = pd.Series(
        _integer_series(clusters["n_nodes"], "clusters.n_nodes").to_numpy(),
        index=cluster_table_ids,
    ).sort_index()
    if not observed_cluster_sizes.equals(declared_cluster_sizes):
        _fail("clusters.n_nodes does not match nodes_roles cluster membership")

    if len(top_nodes) < 20:
        _fail(f"top_nodes.csv must contain at least 20 rows, got {len(top_nodes)}")
    top_gids = _integer_series(top_nodes["gid"], "top_nodes.gid")
    if top_gids.duplicated().any():
        _fail("top_nodes.gid must be unique")
    unknown_top = set(top_gids) - set(role_gids)
    if unknown_top:
        _fail("top_nodes contains gid values absent from nodes_roles")
    ranks = _integer_series(top_nodes["rank"], "top_nodes.rank")
    if ranks.tolist() != list(range(1, len(top_nodes) + 1)):
        _fail("top_nodes.rank must be sequential from 1 in file order")
    top_roles = _nonempty_text(top_nodes["role"], "top_nodes.role")
    if not set(top_roles).issubset(ALLOWED_ROLES):
        _fail("top_nodes.role contains invalid values")
    top_scores = _score_series(top_nodes["priority_score"], "top_nodes.priority_score")
    if not top_scores.is_monotonic_decreasing:
        _fail("top_nodes must be sorted by priority_score descending")
    _nonempty_text(top_nodes["why"], "top_nodes.why")

    roles_by_gid = nodes_roles.set_index(role_gids)
    for row in top_nodes.itertuples(index=False):
        reference = roles_by_gid.loc[int(row.gid)]
        if str(row.role).strip() != str(reference["role"]).strip():
            _fail(f"top_nodes role differs from nodes_roles for gid={row.gid}")
        if not np.isclose(float(row.priority_score), float(reference["priority_score"])):
            _fail(f"top_nodes priority_score differs for gid={row.gid}")

    if source_nodes is not None:
        source_gids = set(_integer_series(source_nodes["gid"], "source nodes.gid"))
        if set(role_gids) != source_gids:
            _fail("nodes_roles.gid must match all source nodes exactly")
        seed_map = source_nodes.assign(gid=source_nodes["gid"].astype("int64")).set_index("gid")[
            "is_seed"
        ].astype(bool)
        expected_seed_counts = (
            pd.DataFrame({"gid": role_gids, "cluster_id": cluster_ids})
            .assign(is_seed=lambda frame: frame["gid"].map(seed_map))
            .groupby("cluster_id")["is_seed"]
            .sum()
            .astype("int64")
            .sort_index()
        )
        declared_seed_counts = pd.Series(
            _integer_series(clusters["n_seed"], "clusters.n_seed").to_numpy(),
            index=cluster_table_ids,
        ).sort_index()
        if not expected_seed_counts.equals(declared_seed_counts):
            _fail("clusters.n_seed does not match source seed membership")

    if source_edges is not None:
        gid_to_cluster = dict(zip(role_gids, cluster_ids, strict=True))
        edge_clusters_src = source_edges["src"].astype("int64").map(gid_to_cluster)
        edge_clusters_dst = source_edges["dst"].astype("int64").map(gid_to_cluster)
        internal_edges = source_edges[edge_clusters_src == edge_clusters_dst].copy()
        internal_edges["cluster_id"] = edge_clusters_src[edge_clusters_src == edge_clusters_dst]
        expected_internal = (
            internal_edges.groupby("cluster_id")["sum_kzt"].sum().reindex(cluster_table_ids, fill_value=0)
        )
        declared_internal = pd.Series(
            internal_sum.to_numpy(), index=cluster_table_ids, dtype=float
        )
        if not np.allclose(expected_internal.to_numpy(), declared_internal.to_numpy(), atol=0.01):
            _fail("clusters.sum_kzt_internal does not match source internal edges")

    return {
        "nodes": len(nodes_roles),
        "clusters": len(clusters),
        "top_nodes": len(top_nodes),
        "metrics": len(node_metrics),
    }


def check_output_directory(out_dir: str | Path, data_dir: str | Path | None = None) -> dict[str, int]:
    frames = _read_outputs(Path(out_dir))
    source_nodes = None
    source_edges = None
    if data_dir is not None:
        data_path = Path(data_dir)
        source_nodes = pd.read_parquet(data_path / "nodes.parquet")
        source_edges = pd.read_parquet(data_path / "edges.parquet")
    return validate_outputs(
        **frames,
        source_nodes=source_nodes,
        source_edges=source_edges,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out", help="directory containing generated CSV files")
    parser.add_argument(
        "--data",
        default=None,
        help="optional source data directory for stronger cross-checks",
    )
    args = parser.parse_args()

    try:
        summary = check_output_directory(args.out, args.data)
    except (DataValidationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 1

    print("OUTPUT CHECKS PASSED")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
