"""Explainable transaction-network analytics; no file I/O or UI dependencies."""
from __future__ import annotations

import math
import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

ROLES = ("consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral")
ROLE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLUMNS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]
RANDOM_SEED = 42
MAX_DEPTH = 4
PRIORITY_WEIGHTS = {
    "flow": .25, "transactions": .15, "degree": .15,
    "pagerank": .15, "bridge": .15, "role": .15,
}


def _prepare(edges, nodes, transactions):
    """Validate the contract, reconcile both sources, and copy/sort inputs."""
    required = {
        "edges": (edges, ["src", "dst", "sum_kzt", "n_tx"]),
        "nodes": (nodes, ["gid", "depth", "is_seed"]),
        "transactions": (transactions, ["src", "dst", "sum_kzt", "date"]),
    }
    for name, (frame, cols) in required.items():
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame[cols].isna().any().any():
            raise ValueError(f"{name}: required values must not be null")
        for col in set(cols) & {"gid", "src", "dst", "depth", "n_tx"}:
            if not is_integer_dtype(frame[col].dtype) or is_bool_dtype(frame[col].dtype):
                raise ValueError(f"{name}.{col}: integer dtype required; do not convert gid to float")
    if nodes.empty:
        raise ValueError("nodes must contain at least one client")
    if nodes.gid.duplicated().any():
        raise ValueError("nodes.gid must be unique")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges must contain one aggregate per src/dst pair")
    if not is_bool_dtype(nodes.is_seed.dtype):
        raise ValueError("nodes.is_seed must have boolean dtype")
    if not nodes.depth.between(0, MAX_DEPTH).all():
        raise ValueError("nodes.depth must be between 0 and 4")
    if not (nodes.is_seed == nodes.depth.eq(0)).all():
        raise ValueError("is_seed must match depth == 0")
    ids = set(nodes.gid)
    for name, frame in (("edges", edges), ("transactions", transactions)):
        if not (set(frame.src) | set(frame.dst)) <= ids:
            raise ValueError(f"{name}: endpoints absent from nodes")
        if not is_numeric_dtype(frame.sum_kzt.dtype):
            raise ValueError(f"{name}.sum_kzt must be numeric")
        if not np.isfinite(frame.sum_kzt).all() or not frame.sum_kzt.gt(0).all():
            raise ValueError(f"{name}.sum_kzt must be finite and positive")
    if not edges.n_tx.gt(0).all():
        raise ValueError("edges.n_tx must be positive")
    e = edges.sort_values(["src", "dst"]).reset_index(drop=True).copy()
    n = nodes.sort_values("gid").reset_index(drop=True).copy()
    t = transactions.copy()
    t["date"] = pd.to_datetime(t.date, errors="raise").dt.normalize()
    if t.date.isna().any():
        raise ValueError("transactions.date must contain valid dates")
    t = t.sort_values(["src", "dst", "date", "sum_kzt"]).reset_index(drop=True)
    agg = t.groupby(["src", "dst"], as_index=False).agg(
        tx_sum=("sum_kzt", "sum"), tx_count=("sum_kzt", "size"))
    joined = e.merge(agg, on=["src", "dst"], how="outer", indicator=True, validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise ValueError("edges and transactions have different src/dst pairs")
    if not np.isclose(joined.sum_kzt, joined.tx_sum, rtol=0, atol=.01).all():
        raise ValueError("edges and transactions disagree on amounts (tolerance 0.01 KZT)")
    if not joined.n_tx.eq(joined.tx_count).all():
        raise ValueError("edges and transactions disagree on transaction counts")
    return e, n, t


def build_graph(edges, nodes):
    """Keep direction, weights and all clients, including isolates."""
    graph = nx.DiGraph()
    graph.add_nodes_from(int(gid) for gid in sorted(nodes.gid))
    for row in edges.sort_values(["src", "dst"]).itertuples(index=False):
        graph.add_edge(int(row.src), int(row.dst),
                       sum_kzt=float(row.sum_kzt), n_tx=int(row.n_tx))
    return graph


def _communities(graph):
    """Symmetrize only for Louvain; sum opposite directions, never overwrite."""
    undirected = nx.Graph()
    undirected.add_nodes_from(graph)
    for src, dst, attrs in graph.edges(data=True):
        if src != dst:  # Self-transfers do not establish a community relationship.
            old = undirected.get_edge_data(src, dst, {}).get("weight", 0.)
            undirected.add_edge(src, dst, weight=old + attrs["sum_kzt"])
    isolated = sorted(nx.isolates(undirected))
    active = undirected.subgraph(sorted(set(undirected) - set(isolated))).copy()
    groups = nx.community.louvain_communities(
        active, weight="weight", resolution=1, seed=RANDOM_SEED
    ) if active.number_of_edges() else []
    groups += [{gid} for gid in isolated]
    # Canonical labels do not depend on the order returned by the algorithm.
    groups = sorted(groups, key=lambda group: min(group))
    return {gid: cid for cid, group in enumerate(groups) for gid in sorted(group)}


def _positive_quantile(values, q, floor=0.):
    positive = values[values > 0]
    return max(float(positive.quantile(q)) if len(positive) else 0., floor)


def _scale(values):
    """Robust log scaling: 0 maps to 0, positive p95 maps to 1, cap above."""
    cap = _positive_quantile(values, .95)
    if cap == 0:
        return values.astype(float) * 0.
    return (np.log1p(values) / np.log1p(cap)).clip(0, 1)


def _metrics(graph, nodes, tx, membership):
    frame = nodes[["gid", "depth", "is_seed"]].copy()
    for name, degree in (
        ("in_deg", graph.in_degree()), ("out_deg", graph.out_degree()),
        ("in_kzt", graph.in_degree(weight="sum_kzt")),
        ("out_kzt", graph.out_degree(weight="sum_kzt")),
        ("in_tx", graph.in_degree(weight="n_tx")), ("out_tx", graph.out_degree(weight="n_tx")),
    ):
        frame[name] = frame.gid.map(dict(degree)).fillna(0)
    for name in ("in_deg", "out_deg", "in_tx", "out_tx"):
        frame[name] = frame[name].astype("int64")
    frame["pagerank"] = frame.gid.map(
        nx.pagerank(graph, weight="sum_kzt", alpha=.85, max_iter=1000, tol=1e-10))
    # Direction retained. Amount is not a path distance; use unweighted hop distances.
    bridge = nx.betweenness_centrality(
        graph, k=min(128, len(graph)), seed=RANDOM_SEED, normalized=True, weight=None)
    frame["betweenness"] = frame.gid.map(bridge)
    frame["cluster_id"] = frame.gid.map(membership).astype("int64")
    seeds = set(nodes.loc[nodes.is_seed, "gid"])
    frame["seed_payers"] = [len(set(graph.predecessors(int(gid))) & seeds) for gid in frame.gid]
    frame["external_clusters"] = [
        len({membership[other] for other in
             set(graph.predecessors(int(gid))) | set(graph.successors(int(gid)))
             if membership[other] != membership[int(gid)]})
        for gid in frame.gid
    ]
    frame["is_isolated"] = (frame.in_deg + frame.out_deg).eq(0)
    frame["truncated_by_depth"] = frame.depth.eq(MAX_DEPTH) & frame.out_deg.eq(0)
    frame["pass_through_defined"] = frame.in_kzt.gt(0)
    # 0 is a placeholder when undefined; always check the flag in the interface.
    frame["pass_through"] = np.divide(
        frame.out_kzt, frame.in_kzt,
        out=np.zeros(len(frame)), where=frame.pass_through_defined.to_numpy())
    frame["balance_similarity"] = np.divide(
        np.minimum(frame.in_kzt, frame.out_kzt), np.maximum(frame.in_kzt, frame.out_kzt),
        out=np.zeros(len(frame)), where=(frame.in_kzt + frame.out_kzt).gt(0).to_numpy())
    endpoints = pd.concat([
        tx[["src", "date", "sum_kzt"]].rename(columns={"src": "gid"}),
        tx[["dst", "date", "sum_kzt"]].rename(columns={"dst": "gid"}),
    ], ignore_index=True)
    frame["active_days"] = frame.gid.map(endpoints.groupby("gid").date.nunique()).fillna(0).astype(int)
    daily = endpoints.groupby(["gid", "date"]).sum_kzt.sum()
    peak = daily.groupby(level=0).max() / daily.groupby(level=0).sum()
    frame["peak_day_share"] = frame.gid.map(peak).fillna(0.)
    incoming_days = set(zip(tx.dst, tx.date))
    day = pd.Timedelta(days=1)
    close = [(src, date) in incoming_days or (src, date - day) in incoming_days
             for src, date in zip(tx.src, tx.date)]
    close_amount = tx.loc[close].groupby("src").sum_kzt.sum()
    close_sum = frame.gid.map(close_amount).fillna(0.)
    frame["near_inflow_out_share"] = np.divide(
        close_sum, frame.out_kzt, out=np.zeros(len(frame)), where=frame.out_kzt.gt(0).to_numpy())
    return frame


def _assign_roles(frame):
    # Fixed quantile choices; thresholds are derived once for the whole input.
    thresholds = {
        "in_degree": int(math.ceil(_positive_quantile(frame.in_deg, .90, 3))),
        "out_degree": int(math.ceil(_positive_quantile(frame.out_deg, .90, 3))),
        "in_kzt": _positive_quantile(frame.in_kzt, .75),
        "betweenness": _positive_quantile(frame.betweenness, .90),
    }
    norm = {name: _scale(frame[name]) for name in
            ("in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx", "betweenness", "pagerank")}
    eligible = {
        "coordinator": (frame.in_deg.ge(2) & frame.out_deg.ge(2)
                        & frame.external_clusters.ge(2) & frame.betweenness.gt(0)
                        & frame.betweenness.ge(thresholds["betweenness"])),
        "consolidator": (frame.in_deg.ge(thresholds["in_degree"])
                         & frame.in_kzt.ge(thresholds["in_kzt"])
                         & frame.in_deg.ge(2 * frame.out_deg)),
        # For seeds, incomplete inbound degree must not determine this role.
        "distributor": (frame.out_deg.ge(thresholds["out_degree"])
                        & (frame.is_seed | frame.out_deg.ge(2 * frame.in_deg))),
        "transit": (~frame.is_seed & ~frame.truncated_by_depth
                    & frame.pass_through_defined & frame.pass_through.between(.7, 1.3)
                    & frame.near_inflow_out_share.ge(.5)),
        "terminal": (~frame.is_seed & ~frame.truncated_by_depth
                      & frame.in_kzt.gt(0) & frame.pass_through.le(.1)),
    }
    scores = {
        "coordinator": .5 * norm["betweenness"] + .25 * norm["pagerank"]
                       + .25 * (frame.external_clusters / 3).clip(0, 1),
        "consolidator": .5 * norm["in_deg"] + .3 * norm["in_kzt"] + .2 * norm["in_tx"],
        "distributor": .5 * norm["out_deg"] + .3 * norm["out_kzt"] + .2 * norm["out_tx"],
        "transit": .6 * frame.balance_similarity + .4 * frame.near_inflow_out_share,
        "terminal": .6 * (1 - frame.pass_through).clip(0, 1) + .4 * norm["in_tx"],
    }
    frame["role"] = "peripheral"
    frame["role_score"] = .2
    # First matching role wins; order is part of the published methodology.
    for role in ("coordinator", "consolidator", "distributor", "transit", "terminal"):
        chosen = eligible[role] & frame.role.eq("peripheral")
        frame.loc[chosen, "role"] = role
        frame.loc[chosen, "role_score"] = scores[role][chosen]
    frame.loc[frame.is_isolated, "role_score"] = .8
    frame.loc[frame.truncated_by_depth, "role_score"] = frame.loc[
        frame.truncated_by_depth, "role_score"].clip(upper=.35)
    frame.loc[frame.truncated_by_depth & frame.role.eq("peripheral"), "role_score"] = .1
    frame["role_score"] = frame.role_score.clip(0, 1)
    return thresholds


def _evidence(row):
    incoming = f"in={row.in_kzt:.6g} KZT/{row.in_deg} payers/{row.in_tx} tx"
    outgoing = f"out={row.out_kzt:.6g} KZT/{row.out_deg} recipients/{row.out_tx} tx"
    if row.role == "coordinator":
        text = (f"Bridge candidate: BC={row.betweenness:.4g}; external_clusters={row.external_clusters}; "
                f"in_deg={row.in_deg}; out_deg={row.out_deg}")
    elif row.role == "consolidator":
        text = f"Collecting: {incoming}; out_deg={row.out_deg}"
    elif row.role == "distributor":
        text = f"Distributing: {outgoing}; in_deg={row.in_deg}"
    elif row.role == "transit":
        text = (f"Transit candidate: in={row.in_kzt:.6g}; out={row.out_kzt:.6g} KZT; "
                f"out/in={row.pass_through:.3f}; near_inflow_out={row.near_inflow_out_share:.0%}")
    elif row.role == "terminal":
        text = (f"Observed retention: {incoming}; out={row.out_kzt:.6g} KZT; "
                f"out/in={row.pass_through:.3f}")
    elif row.is_isolated:
        text = "No observed edges: in_deg=0; out_deg=0; in=0; out=0 KZT"
    else:
        text = f"No specific role rule: {incoming}; {outgoing}"
    if row.is_seed:
        text += "; seed: incomplete inflow"
    if row.truncated_by_depth:
        text += "; depth=4: outflow unknown"
    if len(text) > 200:
        raise ValueError("Evidence exceeds 200 characters")
    return text


def _priority_inputs(frame):
    return {
        "flow": _scale(frame.in_kzt + frame.out_kzt),
        "transactions": _scale(frame.in_tx + frame.out_tx),
        "degree": _scale(frame.in_deg + frame.out_deg),
        "pagerank": _scale(frame.pagerank.where(~frame.is_isolated, 0)),
        "bridge": _scale(frame.betweenness),
        "role": frame.role_score.where(frame.role.ne("peripheral"), 0),
    }


def _priority(frame):
    inputs = _priority_inputs(frame)
    for name, weight in PRIORITY_WEIGHTS.items():
        frame[f"priority_{name}"] = weight * inputs[name]
    frame["priority_score"] = frame[
        [f"priority_{name}" for name in PRIORITY_WEIGHTS]].sum(axis=1).clip(0, 1)
    frame.loc[frame.is_isolated, "priority_score"] = 0.
    def explain(row):
        parts = sorted(
            ((name, getattr(row, f"priority_{name}")) for name in PRIORITY_WEIGHTS),
            key=lambda item: (-item[1], item[0]))
        terms = "; ".join(f"{name}={value:.3f}" for name, value in parts)
        return f"priority={row.priority_score:.4f}; contributions: {terms}. {row.evidence}"
    frame["why"] = [explain(row) for row in frame.itertuples(index=False)]


def _cluster_summary(frame, edges):
    lookup = frame.set_index("gid").cluster_id
    grouped_edges = edges.assign(src_cluster=edges.src.map(lookup), dst_cluster=edges.dst.map(lookup))
    internal = grouped_edges.loc[grouped_edges.src_cluster.eq(grouped_edges.dst_cluster)]
    amounts = internal.groupby("src_cluster").sum_kzt.sum()
    records = []
    for cid, group in frame.groupby("cluster_id", sort=True):
        ranked = group.sort_values(["priority_score", "gid"], ascending=[False, True])
        counts = group.role.value_counts()
        if group.is_isolated.all():
            hypothesis = "Isolated client: 0 observed transfers"
        else:
            dominant = sorted(counts.index, key=lambda role: (-int(counts[role]), role))[0]
            hypothesis = (f"Observed community: {dominant}={int(counts[dominant])}/{len(group)}; "
                          f"boundary={int(group.truncated_by_depth.sum())}; "
                          f"coordinator candidates={int(group.role.eq('coordinator').sum())}")
        records.append({
            "cluster_id": int(cid), "n_nodes": len(group), "n_seed": int(group.is_seed.sum()),
            "sum_kzt_internal": float(amounts.get(cid, 0)),
            "top_gids": ";".join(str(int(gid)) for gid in ranked.gid.head(5)),
            "hypothesis": hypothesis,
        })
    return pd.DataFrame(records, columns=CLUSTER_COLUMNS)


def analyze(edges, nodes, transactions):
    """Return (nodes_roles, clusters, top_nodes, node_metrics) as DataFrames.

    Input frames are not modified. Every supplied node survives. Fewer than 20
    input nodes produce all available nodes in top_nodes (no invented clients).
    See docs/methodology.md for role precedence, thresholds and limitations.
    """
    edges, nodes, transactions = _prepare(edges, nodes, transactions)
    graph = build_graph(edges, nodes)
    membership = _communities(graph)
    frame = _metrics(graph, nodes, transactions, membership)
    thresholds = _assign_roles(frame)
    frame["evidence"] = [_evidence(row) for row in frame.itertuples(index=False)]
    _priority(frame)
    nodes_roles = frame[ROLE_COLUMNS].copy()
    clusters = _cluster_summary(frame, edges)
    top = frame.sort_values(["priority_score", "gid"], ascending=[False, True]).head(20).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top_nodes = top[TOP_COLUMNS].reset_index(drop=True)
    node_metrics = frame.drop(columns=["role", "role_score", "evidence", "why", "priority_score"]).copy()
    node_metrics.attrs["thresholds"] = thresholds
    node_metrics.attrs["random_seed"] = RANDOM_SEED
    return nodes_roles, clusters, top_nodes, node_metrics


def _join_results(nodes_roles, node_metrics):
    """Validate and join a complete analysis snapshot by integer gid."""
    numeric = (
        "depth", "in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx",
        "pagerank", "betweenness", "external_clusters", "near_inflow_out_share",
    )
    flags = ("is_seed", "is_isolated", "truncated_by_depth")
    for name, frame, columns in (
        ("nodes_roles", nodes_roles, ROLE_COLUMNS),
        ("node_metrics", node_metrics, ["gid", *numeric, *flags]),
    ):
        if not frame.columns.is_unique:
            raise ValueError(f"{name}: duplicate column names")
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame.empty or frame[columns].isna().any().any():
            raise ValueError(f"{name}: empty frame or missing values")
        if not is_integer_dtype(frame.gid.dtype) or is_bool_dtype(frame.gid.dtype):
            raise ValueError(f"{name}.gid must have integer dtype")
        if frame.gid.duplicated().any():
            raise ValueError(f"{name}.gid must be unique")
    if set(nodes_roles.gid) != set(node_metrics.gid):
        raise ValueError("nodes_roles and node_metrics must contain the same gids")
    if not nodes_roles.role.isin(ROLES).all():
        raise ValueError("Unknown role")
    for col in ("role_score", "priority_score"):
        values = nodes_roles[col]
        if (not is_numeric_dtype(values.dtype) or not np.isfinite(values).all()
                or not values.between(0, 1).all()):
            raise ValueError(f"{col} must be finite and in [0, 1]")
    for col in numeric:
        values = node_metrics[col]
        if (not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype)
                or not np.isfinite(values).all() or values.lt(0).any()):
            raise ValueError(f"node_metrics.{col} must be finite and nonnegative")
    for col in ("depth", "in_deg", "out_deg", "in_tx", "out_tx", "external_clusters"):
        if not is_integer_dtype(node_metrics[col].dtype):
            raise ValueError(f"node_metrics.{col} must have integer dtype")
    for col in flags:
        if not is_bool_dtype(node_metrics[col].dtype):
            raise ValueError(f"node_metrics.{col} must have boolean dtype")
    m = node_metrics
    if not m.depth.between(0, MAX_DEPTH).all() or not m.is_seed.eq(m.depth.eq(0)).all():
        raise ValueError("Inconsistent depth/is_seed flags")
    if not m.is_isolated.eq((m.in_deg + m.out_deg).eq(0)).all():
        raise ValueError("Inconsistent is_isolated flag")
    if not m.truncated_by_depth.eq(m.depth.eq(MAX_DEPTH) & m.out_deg.eq(0)).all():
        raise ValueError("Inconsistent truncated_by_depth flag")
    if not m.near_inflow_out_share.between(0, 1 + 1e-12).all():
        raise ValueError("near_inflow_out_share must be in [0, 1]")
    return nodes_roles[ROLE_COLUMNS].merge(
        m[["gid", *numeric, *flags]], on="gid", validate="one_to_one"
    ).sort_values("gid").reset_index(drop=True)


def priority_features(nodes_roles, node_metrics):
    """Return gid and the six normalized, unweighted priority components.

    Uses exactly the same normalization as analyze(). No input is modified.
    """
    joined = _join_results(nodes_roles, node_metrics)
    features = pd.DataFrame(_priority_inputs(joined))
    features.loc[joined.is_isolated, :] = 0.
    features.insert(0, "gid", joined.gid)
    return features
