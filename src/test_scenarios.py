"""Behavioral checks for weight sensitivity and node-removal scenarios."""
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.analysis import analyze, priority_features, PRIORITY_WEIGHTS
from src.scenarios import analyze_weight_sensitivity, simulate_node_removal
from src.test_analysis import fixture, BASE


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.roles, _, _, self.metrics = analyze(*fixture())

    def test_baseline_matches_and_each_rank_is_accounted_for(self):
        result = analyze_weight_sensitivity(self.roles, self.metrics, top_k=2)
        summary, scenarios, ranks = (result[k] for k in ("summary", "scenarios", "rankings"))
        self.assertEqual(len(summary), len(self.roles))
        self.assertEqual(len(scenarios), 65)
        self.assertTrue(summary.scenario_count.eq(64).all())
        self.assertEqual(int(summary.top_k_count.sum()), 64 * 2)
        self.assertTrue(summary.best_rank.le(summary.worst_rank).all())
        weights = scenarios.filter(regex="^weight_")
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertTrue(weights.ge(0).all().all())
        baseline = ranks[ranks.scenario.eq("baseline")].sort_values("rank")
        expected = self.roles.sort_values(["priority_score", "gid"], ascending=[False, True])
        self.assertEqual(baseline.gid.tolist(), expected.gid.tolist())
        np.testing.assert_allclose(baseline.priority_score, expected.priority_score, rtol=0, atol=1e-15)
        for _, group in ranks.groupby("scenario"):
            self.assertEqual(group["rank"].tolist(), list(range(1, len(self.roles) + 1)))
            self.assertTrue(group.priority_score.is_monotonic_decreasing)
            self.assertEqual(int(group.in_top_k.sum()), 2)
        for row in summary.itertuples():
            values = ranks[(ranks.gid == row.gid) & ranks.scenario.ne("baseline")]
            self.assertEqual(row.top_k_count, int(values.in_top_k.sum()))
            self.assertEqual(row.best_rank, values["rank"].min())
            self.assertEqual(row.worst_rank, values["rank"].max())
        features = priority_features(self.roles, self.metrics)
        reconstructed = features[list(PRIORITY_WEIGHTS)].mul(PRIORITY_WEIGHTS).sum(axis=1)
        np.testing.assert_allclose(reconstructed, self.roles.sort_values("gid").priority_score)

    def test_zero_change_and_small_graph(self):
        result = analyze_weight_sensitivity(self.roles, self.metrics, relative_change=0)
        s = result["summary"]
        self.assertTrue(s.scenario_count.eq(1).all())
        self.assertTrue(s.top_k.eq(len(self.roles)).all())
        self.assertTrue(s.top_k_count.eq(1).all())
        self.assertTrue(s.best_rank.eq(s.baseline_rank).all())
        self.assertTrue(s.worst_rank.eq(s.baseline_rank).all())

    def test_ties_preserve_large_ids_and_zero_count_clients(self):
        e, n, t = fixture()
        roles, _, _, metrics = analyze(e.iloc[:0], n[n.is_seed], t.iloc[:0])
        result = analyze_weight_sensitivity(roles, metrics, top_k=1)
        s = result["summary"].set_index("gid")
        self.assertEqual(s.loc[BASE, "top_k_count"], 64)
        self.assertEqual(s.loc[BASE + 4, "top_k_count"], 0)
        self.assertEqual(s.loc[BASE + 4, "best_rank"], 2)
        self.assertEqual(s.loc[BASE + 4, "worst_rank"], 2)

    def test_deterministic_and_no_mutation(self):
        old_roles, old_metrics = self.roles.copy(deep=True), self.metrics.copy(deep=True)
        first = analyze_weight_sensitivity(self.roles, self.metrics)
        second = analyze_weight_sensitivity(self.roles.sample(frac=1, random_state=7),
                                            self.metrics.sample(frac=1, random_state=8))
        for key in first:
            assert_frame_equal(first[key], second[key])
        assert_frame_equal(self.roles, old_roles)
        assert_frame_equal(self.metrics, old_metrics)

    def test_invalid_parameters_and_stale_scores(self):
        for delta in (-.1, 1, float("nan"), float("inf"), True, "0.1"):
            with self.assertRaises(ValueError):
                analyze_weight_sensitivity(self.roles, self.metrics, relative_change=delta)
        for k in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                analyze_weight_sensitivity(self.roles, self.metrics, top_k=k)
        roles = self.roles.copy()
        roles.loc[roles.gid == BASE + 4, "priority_score"] = .1
        with self.assertRaises(ValueError):
            analyze_weight_sensitivity(roles, self.metrics)


class RemovalTests(unittest.TestCase):
    def test_bridge_removal_has_exact_structural_and_money_effects(self):
        e, n, _ = fixture()
        old_e, old_n = e.copy(deep=True), n.copy(deep=True)
        result = simulate_node_removal(e, n, BASE + 1)
        impact = result["impact"]
        self.assertEqual(impact["removed_edges"], 2)
        self.assertEqual(impact["removed_transactions"], 2)
        self.assertEqual(impact["removed_turnover_kzt"], 19000.)
        self.assertEqual(impact["newly_isolated_nodes"], 1)
        self.assertEqual(impact["lost_seed_reachable_nodes"], 1)
        self.assertEqual(impact["additional_fragments"], 1)
        self.assertEqual(impact["disconnected_survivor_pairs"], 2)
        affected = result["affected_nodes"].set_index("gid")
        self.assertTrue(affected.loc[BASE + 2, "became_isolated"])
        self.assertEqual(affected.loc[BASE + 2, "removed_in_kzt"], 9000.)
        self.assertEqual(affected.loc[BASE, "removed_out_kzt"], 10000.)
        self.assertNotIn(BASE + 1, affected.index)
        assert_frame_equal(e, old_e)
        assert_frame_equal(n, old_n)

    def test_leaf_and_isolate_do_not_create_false_fragmentation(self):
        e, n, _ = fixture()
        for gid in (BASE + 3, BASE + 4):
            result = simulate_node_removal(e, n, gid)
            self.assertEqual(result["impact"]["additional_fragments"], 0)
            self.assertEqual(result["impact"]["disconnected_survivor_pairs"], 0)
            self.assertEqual(result["impact"]["lost_seed_reachable_nodes"], 0)
        isolated = simulate_node_removal(e, n, BASE + 4)
        self.assertEqual(isolated["impact"]["removed_edges"], 0)
        self.assertTrue(isolated["affected_nodes"].empty)
        self.assertEqual(isolated["affected_nodes"].gid.dtype, n.gid.dtype)
        change = isolated["network_changes"].set_index("metric")
        self.assertEqual(change.loc["weak_components", "change"], -1)

    def test_direction_matters_even_without_weak_fragmentation(self):
        nodes = pd.DataFrame({"gid": [1, 2, 3, 4], "is_seed": [True, False, False, False]})
        edges = pd.DataFrame({"src": [1, 2, 4, 4], "dst": [2, 3, 3, 1],
                              "sum_kzt": [5000.] * 4, "n_tx": [1] * 4})
        result = simulate_node_removal(edges, nodes, 2)
        self.assertEqual(result["impact"]["additional_fragments"], 0)
        self.assertEqual(result["impact"]["lost_seed_reachable_nodes"], 1)
        lost = result["affected_nodes"].query("lost_seed_reachability").gid.tolist()
        self.assertEqual(lost, [3])
        nodes.loc[nodes.gid == 4, "is_seed"] = True
        alternative = simulate_node_removal(edges, nodes, 2)
        self.assertEqual(alternative["impact"]["lost_seed_reachable_nodes"], 0)

    def test_seed_removal_and_last_node(self):
        e, n, _ = fixture()
        result = simulate_node_removal(e, n, BASE)
        self.assertEqual(result["impact"]["lost_seed_reachable_nodes"], 3)
        last = simulate_node_removal(e.iloc[:0], n[n.gid == BASE + 4], BASE + 4)
        self.assertTrue(last["network_changes"]["after"].eq(0).all())
        self.assertEqual(last["impact"]["disconnected_survivor_pairs"], 0)

    def test_reciprocal_and_self_edges_count_once(self):
        nodes = pd.DataFrame({"gid": [1, 2], "is_seed": [True, False]})
        edges = pd.DataFrame({"src": [1, 2, 1], "dst": [2, 1, 1],
                              "sum_kzt": [5000., 7000., 3000.], "n_tx": [2, 1, 1]})
        result = simulate_node_removal(edges, nodes, 1)
        self.assertEqual(result["impact"]["removed_edges"], 3)
        self.assertEqual(result["impact"]["removed_turnover_kzt"], 15000.)
        self.assertEqual(result["impact"]["removed_transactions"], 4)
        self.assertEqual(result["impact"]["direct_neighbors"], 1)
        affected = result["affected_nodes"].iloc[0]
        self.assertEqual(affected.removed_in_kzt, 5000.)
        self.assertEqual(affected.removed_out_kzt, 7000.)

    def test_reject_bad_gid_and_inputs(self):
        e, n, _ = fixture()
        for gid in (999, float(BASE), str(BASE), True):
            with self.assertRaises(ValueError):
                simulate_node_removal(e, n, gid)
        with self.assertRaises(ValueError):
            simulate_node_removal(pd.concat([e, e.iloc[:1]]), n, BASE)
        bad = e.copy()
        bad.loc[0, "sum_kzt"] = float("inf")
        with self.assertRaises(ValueError):
            simulate_node_removal(bad, n, BASE)
        with self.assertRaises(ValueError):
            simulate_node_removal(e, n[n.gid != BASE + 1], BASE)

    def test_row_order_is_irrelevant(self):
        e, n, _ = fixture()
        a = simulate_node_removal(e, n, BASE + 1)
        b = simulate_node_removal(e.sample(frac=1, random_state=3),
                                  n.sample(frac=1, random_state=4), BASE + 1)
        self.assertEqual(a["impact"], b["impact"])
        for key in ("network_changes", "affected_nodes", "removed_edges"):
            assert_frame_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
