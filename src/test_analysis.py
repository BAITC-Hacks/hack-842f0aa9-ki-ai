"""Regression checks for analytics; no private data is required."""
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from src.analysis import analyze, build_graph, _communities, ROLE_COLUMNS, CLUSTER_COLUMNS, TOP_COLUMNS

BASE = 2**60 + 17


def fixture():
    ids = [BASE + offset for offset in range(5)]
    nodes = pd.DataFrame({
        "gid": ids, "depth": [0, 1, 2, 4, 0], "is_seed": [True, False, False, False, True],
    })
    tx = pd.DataFrame({
        "src": [ids[0], ids[1], ids[0]], "dst": [ids[1], ids[2], ids[3]],
        "sum_kzt": [10000., 9000., 5000.], "date": ["2026-07-01"] * 3,
    })
    edges = tx.groupby(["src", "dst"], as_index=False).agg(
        sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    return edges, nodes, tx


class AnalyticsTests(unittest.TestCase):
    def test_full_contract_and_large_ids(self):
        e, n, t = fixture()
        before = [x.copy(deep=True) for x in (e, n, t)]
        roles, clusters, top, metrics = analyze(e, n, t)
        self.assertEqual(list(roles.columns), ROLE_COLUMNS)
        self.assertEqual(list(clusters.columns), CLUSTER_COLUMNS)
        self.assertEqual(list(top.columns), TOP_COLUMNS)
        self.assertEqual(set(roles.gid), set(n.gid))
        self.assertEqual(roles.gid.dtype, n.gid.dtype)
        self.assertEqual(len(top), len(n))  # Small fixtures cannot provide 20 clients.
        self.assertEqual(top["rank"].tolist(), list(range(1, len(n) + 1)))
        self.assertTrue(roles[["role_score", "priority_score"]].ge(0).all().all())
        self.assertTrue(roles[["role_score", "priority_score"]].le(1).all().all())
        self.assertTrue(roles.evidence.str.len().between(1, 200).all())
        self.assertTrue(roles.evidence.str.contains(r"\d").all())
        self.assertFalse(metrics.isna().any().any())
        for original, saved in zip((e, n, t), before):
            assert_frame_equal(original, saved)
        contributions = metrics.filter(regex=r"^priority_").sum(axis=1)
        self.assertTrue((contributions - roles.priority_score).abs().lt(1e-12).all())

    def test_transit_terminal_boundary_and_isolate(self):
        r, c, _, m = analyze(*fixture())
        r, m = r.set_index("gid"), m.set_index("gid")
        self.assertEqual(r.loc[BASE + 1, "role"], "transit")
        self.assertEqual(r.loc[BASE + 2, "role"], "terminal")
        self.assertNotEqual(r.loc[BASE + 3, "role"], "terminal")
        self.assertIn("outflow unknown", r.loc[BASE + 3, "evidence"])
        self.assertTrue(m.loc[BASE + 3, "truncated_by_depth"])
        self.assertEqual(r.loc[BASE + 4, "role"], "peripheral")
        self.assertEqual(r.loc[BASE + 4, "priority_score"], 0)
        isolated_cluster = c[c.cluster_id == r.loc[BASE + 4, "cluster_id"]].iloc[0]
        self.assertEqual(isolated_cluster.n_nodes, 1)
        self.assertEqual(isolated_cluster.sum_kzt_internal, 0)
        self.assertEqual(m.loc[BASE + 1, "in_tx"], 1)
        self.assertEqual(m.loc[BASE + 1, "near_inflow_out_share"], 1)

    def test_time_pattern_is_required_for_transit(self):
        e, n, t = fixture()
        t.loc[t.src == BASE + 1, "date"] = "2026-07-10"
        r, _, _, m = analyze(e, n, t)
        self.assertNotEqual(r.set_index("gid").loc[BASE + 1, "role"], "transit")
        self.assertEqual(m.set_index("gid").loc[BASE + 1, "near_inflow_out_share"], 0)

    def test_seed_not_classified_using_balance(self):
        e, n, t = fixture()
        n.loc[n.gid == BASE + 1, ["is_seed", "depth"]] = [True, 0]
        r, _, _, _ = analyze(e, n, t)
        self.assertNotEqual(r.set_index("gid").loc[BASE + 1, "role"], "transit")

    def test_input_permutation_and_repeat_are_stable(self):
        e, n, t = fixture()
        first = analyze(e, n, t)
        second = analyze(e.sample(frac=1, random_state=9), n.sample(frac=1, random_state=3),
                         t.sample(frac=1, random_state=2))
        for a, b in zip(first, second):
            assert_frame_equal(a, b)

    def test_all_isolated_and_single_client(self):
        e, n, t = fixture()
        for subset in (n[n.is_seed], n[n.is_seed].head(1)):
            r, c, top, m = analyze(e.iloc[:0], subset, t.iloc[:0])
            self.assertEqual(len(r), len(subset))
            self.assertEqual(len(c), len(subset))
            self.assertTrue(r.role.eq("peripheral").all())
            self.assertTrue(r.priority_score.eq(0).all())
            self.assertTrue(m.in_kzt.eq(0).all())

    def test_reciprocal_amounts_and_self_loop(self):
        nodes = pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 1, 0],
                              "is_seed": [True, False, True]})
        tx = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 1, 3],
                           "sum_kzt": [5000., 7000., 6000.], "date": ["2026-07-01"] * 3})
        edges = tx.groupby(["src", "dst"], as_index=False).agg(
            sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
        r, c, _, m = analyze(edges, nodes, tx)
        self.assertEqual(c.sum_kzt_internal.sum(), 18000.)
        self.assertEqual(m.set_index("gid").loc[1, "out_kzt"], 5000.)
        self.assertEqual(m.set_index("gid").loc[1, "in_kzt"], 7000.)
        membership = _communities(build_graph(edges, nodes))
        self.assertEqual(membership[1], membership[2])
        self.assertNotEqual(membership[1], membership[3])

    def test_reject_mismatched_amount_and_count(self):
        for col, delta in (("sum_kzt", 1.), ("n_tx", 1)):
            e, n, t = fixture()
            e.loc[0, col] += delta
            with self.assertRaises(ValueError):
                analyze(e, n, t)

    def test_reject_missing_pair_and_endpoint(self):
        e, n, t = fixture()
        with self.assertRaises(ValueError):
            analyze(e.iloc[1:], n, t)
        with self.assertRaises(ValueError):
            analyze(e, n[n.gid != BASE + 1], t)

    def test_reject_duplicate_and_float_ids(self):
        e, n, t = fixture()
        with self.assertRaises(ValueError):
            analyze(e, pd.concat([n, n.iloc[:1]]), t)
        with self.assertRaises(ValueError):
            analyze(pd.concat([e, e.iloc[:1]]), n, t)
        n["gid"] = n.gid.astype(float)
        with self.assertRaises(ValueError):
            analyze(e, n, t)

    def test_reject_null_infinite_and_bad_date(self):
        for value in (float("nan"), float("inf"), -1.):
            e, n, t = fixture()
            e.loc[0, "sum_kzt"] = value
            with self.assertRaises(ValueError):
                analyze(e, n, t)
        e, n, t = fixture()
        t.loc[0, "date"] = "bad-date"
        with self.assertRaises(ValueError):
            analyze(e, n, t)


if __name__ == "__main__":
    unittest.main()
