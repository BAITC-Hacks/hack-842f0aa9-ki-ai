"""Deterministic weight sensitivity and hypothetical node-removal scenarios."""
from __future__ import annotations

from itertools import product
from numbers import Integral, Real

import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from src.analysis import PRIORITY_WEIGHTS, build_graph, priority_features


def analyze_weight_sensitivity(nodes_roles, node_metrics, *, relative_change=.10, top_k=20):
    """Return per-node summary, scenario weights/overlap, and all node ranks.

    Vary each of six raw weights by +/- relative_change simultaneously, then
    normalize to sum to one: 64 deterministic corners plus a separate baseline.
    Counts/rank ranges exclude baseline. This is a sensitivity grid, not a
    probabilistic estimate. Features, roles and normalization stay fixed.
    """
    if (isinstance(relative_change, bool) or not isinstance(relative_change, Real)
            or not np.isfinite(relative_change) or not 0 <= relative_change < 1):
        raise ValueError("relative_change must be finite and in [0, 1)")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    features = priority_features(nodes_roles, node_metrics)
    names = list(PRIORITY_WEIGHTS)
    base_weights = np.array([PRIORITY_WEIGHTS[name] for name in names])
    values = features[names]
    base_scores = values.mul(base_weights, axis=1).sum(axis=1).clip(0, 1)
    reference = nodes_roles.set_index("gid").loc[features.gid, "priority_score"].to_numpy()
    if not np.allclose(base_scores, reference, rtol=0, atol=1e-12):
        raise ValueError("Snapshot priority scores disagree with the current formula; rerun analyze()")
    effective_k = min(int(top_k), len(features))
    baseline = pd.DataFrame({"gid": features.gid, "score": base_scores}).sort_values(
        ["score", "gid"], ascending=[False, True])
    base_top = set(baseline.gid.head(effective_k))
    baseline["rank"] = range(1, len(baseline) + 1)
    baseline = baseline.set_index("gid")

    configurations = [("baseline", base_weights, "baseline")]
    signs_grid = product((-1, 1), repeat=len(names)) if relative_change else [(0,) * len(names)]
    for index, signs in enumerate(signs_grid, 1):
        weights = base_weights * (1 + relative_change * np.array(signs))
        # Equal multipliers cancel exactly; retain exact ties from the baseline.
        weights = base_weights.copy() if len(set(signs)) == 1 else weights / weights.sum()
        configurations.append((f"variation_{index:02d}", weights, ",".join(map(str, signs))))

    rankings, scenarios = [], []
    for name, weights, signs in configurations:
        ranked = pd.DataFrame({
            "gid": features.gid,
            "priority_score": values.mul(weights, axis=1).sum(axis=1).clip(0, 1),
        }).sort_values(["priority_score", "gid"], ascending=[False, True]).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["in_top_k"] = ranked["rank"].le(effective_k)
        ranked.insert(0, "scenario", name)
        rankings.append(ranked)
        selected = set(ranked.loc[ranked.in_top_k, "gid"])
        overlap = len(selected & base_top)
        scenarios.append({
            "scenario": name, "raw_change_pattern": signs,
            "relative_change": float(relative_change), "top_k": effective_k,
            **{f"weight_{key}": float(value) for key, value in zip(names, weights)},
            "overlap_with_baseline": overlap,
            "jaccard_with_baseline": overlap / len(selected | base_top),
            "entered_gids": ";".join(map(str, sorted(selected - base_top))),
            "exited_gids": ";".join(map(str, sorted(base_top - selected))),
        })
    rankings = pd.concat(rankings, ignore_index=True)
    varied = rankings.loc[rankings.scenario.ne("baseline")]
    summary = varied.groupby("gid", sort=True).agg(
        top_k_count=("in_top_k", "sum"),
        best_rank=("rank", "min"), worst_rank=("rank", "max"),
        min_score=("priority_score", "min"), max_score=("priority_score", "max"),
    ).reset_index()
    summary["scenario_count"] = len(configurations) - 1
    summary["top_k_rate"] = summary.top_k_count / summary.scenario_count
    summary["top_k"] = effective_k
    summary["baseline_rank"] = summary.gid.map(baseline["rank"]).astype("int64")
    summary["baseline_score"] = summary.gid.map(baseline.score)
    summary["baseline_in_top_k"] = summary.baseline_rank.le(effective_k)
    return {"summary": summary, "scenarios": pd.DataFrame(scenarios), "rankings": rankings}


def _validate_graph_inputs(edges, nodes):
    for name, frame, columns in (
        ("nodes", nodes, ("gid", "is_seed")),
        ("edges", edges, ("src", "dst", "sum_kzt", "n_tx")),
    ):
        if not frame.columns.is_unique or not set(columns) <= set(frame.columns):
            raise ValueError(f"{name}: missing or duplicate columns")
        if frame[list(columns)].isna().any().any():
            raise ValueError(f"{name}: missing values")
        for col in set(columns) & {"gid", "src", "dst", "n_tx"}:
            if not is_integer_dtype(frame[col].dtype) or is_bool_dtype(frame[col].dtype):
                raise ValueError(f"{name}.{col} must have integer dtype")
    if nodes.gid.duplicated().any() or edges.duplicated(["src", "dst"]).any():
        raise ValueError("Duplicate gid or edge pair")
    if not is_bool_dtype(nodes.is_seed.dtype):
        raise ValueError("is_seed must have boolean dtype")
    if not (set(edges.src) | set(edges.dst)) <= set(nodes.gid):
        raise ValueError("Unknown edge endpoint")
    if (not is_numeric_dtype(edges.sum_kzt.dtype) or is_bool_dtype(edges.sum_kzt.dtype)
            or not np.isfinite(edges.sum_kzt).all() or not edges.sum_kzt.gt(0).all()):
        raise ValueError("sum_kzt must be finite and positive")
    if not edges.n_tx.gt(0).all():
        raise ValueError("n_tx must be positive")


def _seed_reachable(graph, seeds):
    reached = set(graph) & seeds
    pending = list(sorted(reached))
    while pending:
        current = pending.pop()
        for neighbor in graph.successors(current):
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


def _network_stats(graph, reachable):
    weak = list(nx.weakly_connected_components(graph))
    strong = list(nx.strongly_connected_components(graph))
    return {
        "nodes": len(graph), "edges": graph.number_of_edges(),
        "transactions": sum(row["n_tx"] for _, _, row in graph.edges(data=True)),
        "turnover_kzt": sum(row["sum_kzt"] for _, _, row in graph.edges(data=True)),
        "weak_components": len(weak), "largest_weak_component": max(map(len, weak), default=0),
        "strong_components": len(strong), "largest_strong_component": max(map(len, strong), default=0),
        "isolated_nodes": sum(1 for _ in nx.isolates(graph)),
        "seed_reachable_nodes": len(reachable),
    }


def simulate_node_removal(edges, nodes, gid):
    """Remove a node on a graph copy and quantify observed structural changes.

    Does not remove files, alter inputs, reroute money, or predict behavior.
    Disconnected pairs count surviving unordered weakly-connected node pairs,
    excluding the removed node so isolated-node removal creates no false split.
    """
    _validate_graph_inputs(edges, nodes)
    if isinstance(gid, bool) or not isinstance(gid, Integral):
        raise ValueError("gid must be an integer, not a float or string")
    gid = int(gid)
    if gid not in set(nodes.gid):
        raise ValueError("Selected gid is absent from nodes")
    graph = build_graph(edges, nodes)
    seeds = set(nodes.loc[nodes.is_seed, "gid"])
    before_reachable = _seed_reachable(graph, seeds)
    original_component = nx.node_connected_component(graph.to_undirected(as_view=True), gid)
    after = graph.copy()
    after.remove_node(gid)
    after_reachable = _seed_reachable(after, seeds - {gid})
    before_stats = _network_stats(graph, before_reachable)
    after_stats = _network_stats(after, after_reachable)
    changes = pd.DataFrame([
        {"metric": key, "before": value, "after": after_stats[key],
         "change": after_stats[key] - value}
        for key, value in before_stats.items()
    ])
    removed = edges.loc[edges.src.eq(gid) | edges.dst.eq(gid)].sort_values(
        ["src", "dst"]).reset_index(drop=True).copy()
    fragments = list(nx.weakly_connected_components(after.subgraph(original_component - {gid})))
    survivors = len(original_component) - 1
    disconnected = survivors * (survivors - 1) // 2 - sum(
        len(part) * (len(part) - 1) // 2 for part in fragments)
    newly_isolated = set(nx.isolates(after)) - set(nx.isolates(graph))
    lost_reachability = (before_reachable - {gid}) - after_reachable
    neighbors = (set(graph.predecessors(gid)) | set(graph.successors(gid))) - {gid}
    affected = sorted(neighbors | lost_reachability)
    records = []
    for other in affected:
        lost_in = removed.loc[removed.dst.eq(other)]
        lost_out = removed.loc[removed.src.eq(other)]
        records.append({
            "gid": other,
            "in_deg_before": graph.in_degree(other), "in_deg_after": after.in_degree(other),
            "out_deg_before": graph.out_degree(other), "out_deg_after": after.out_degree(other),
            "removed_in_kzt": float(lost_in.sum_kzt.sum()),
            "removed_out_kzt": float(lost_out.sum_kzt.sum()),
            "removed_in_tx": int(lost_in.n_tx.sum()), "removed_out_tx": int(lost_out.n_tx.sum()),
            "direct_neighbor": other in neighbors, "became_isolated": other in newly_isolated,
            "seed_reachable_before": other in before_reachable,
            "seed_reachable_after": other in after_reachable,
            "lost_seed_reachability": other in lost_reachability,
        })
    columns = [
        "gid", "in_deg_before", "in_deg_after", "out_deg_before", "out_deg_after",
        "removed_in_kzt", "removed_out_kzt", "removed_in_tx", "removed_out_tx",
        "direct_neighbor", "became_isolated", "seed_reachable_before",
        "seed_reachable_after", "lost_seed_reachability",
    ]
    affected_nodes = pd.DataFrame(records, columns=columns)
    affected_nodes["gid"] = affected_nodes.gid.astype(nodes.gid.dtype)
    for col in columns[1:]:
        dtype = bool if col in columns[9:] else (float if col.endswith("kzt") else "int64")
        affected_nodes[col] = affected_nodes[col].astype(dtype)
    impact = {
        "removed_edges": len(removed), "removed_transactions": int(removed.n_tx.sum()),
        "removed_turnover_kzt": float(removed.sum_kzt.sum()),
        "removed_turnover_share": float(removed.sum_kzt.sum()) / before_stats["turnover_kzt"]
                                  if before_stats["turnover_kzt"] else 0.,
        "direct_neighbors": len(neighbors), "newly_isolated_nodes": len(newly_isolated),
        "lost_seed_reachable_nodes": len(lost_reachability),
        "surviving_component_fragments": len(fragments),
        "additional_fragments": max(0, len(fragments) - 1),
        "disconnected_survivor_pairs": disconnected,
    }
    note = (
        f"Бұл виртуалды сценарий: {len(removed)} байланыс пен "
        f"{float(removed.sum_kzt.sum()):.2f} ₸ көрінетін айналым графтан алынды. "
        f"Қалған {len(newly_isolated)} клиент оқшауланды; "
        f"{len(lost_reachability)} клиентке seed-тен бағытталған жол жоғалды. "
        "Ақша басқа жолмен жүреді немесе нақты шығын болады деген болжам жасалмайды. "
        "Кластерлер мен рөлдер қайта есептелмейді; әсер тек берілген таңдамаға қатысты."
    )
    return {
        "gid": gid, "network_changes": changes, "impact": impact,
        "affected_nodes": affected_nodes, "removed_edges": removed, "note_kz": note,
    }
