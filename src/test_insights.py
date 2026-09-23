"""Kazakh insight contracts and independent data-limit flags."""
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from src.analysis import analyze
from src.insights import build_node_insights, INSIGHT_COLUMNS
from src.test_analysis import fixture, BASE


class InsightTests(unittest.TestCase):
    def setUp(self):
        self.roles, _, _, self.metrics = analyze(*fixture())

    def test_explanations_cover_all_clients_and_preserve_inputs(self):
        old_roles, old_metrics = self.roles.copy(deep=True), self.metrics.copy(deep=True)
        result = build_node_insights(self.roles, self.metrics)
        self.assertEqual(list(result.columns), INSIGHT_COLUMNS)
        self.assertEqual(result.gid.tolist(), sorted(self.roles.gid))
        self.assertTrue(result.gid.is_unique)
        self.assertEqual(result.gid.dtype, self.roles.gid.dtype)
        for col in ("review_reason_kz", "missing_data_kz", "next_step_kz"):
            self.assertTrue(result[col].str.len().gt(0).all())
            self.assertTrue(result[col].str.contains("[әғқңөұүі]").all())
        self.assertTrue(result.review_reason_kz.str.contains(r"\d").all())
        assert_frame_equal(self.roles, old_roles)
        assert_frame_equal(self.metrics, old_metrics)

    def test_exact_amounts_and_time_evidence(self):
        row = build_node_insights(self.roles, self.metrics).set_index("gid").loc[BASE + 1]
        self.assertIn("10 000.00 ₸", row.review_reason_kz)
        self.assertIn("9 000.00 ₸", row.review_reason_kz)
        self.assertIn("0.900", row.review_reason_kz)
        self.assertIn("100.0%", row.review_reason_kz)
        self.assertIn("сағат/минут", row.missing_data_kz)
        self.assertIn("нақты уақытын", row.next_step_kz)

    def test_seed_boundary_and_isolate_are_independent(self):
        result = build_node_insights(self.roles, self.metrics).set_index("gid")
        seed = result.loc[BASE]
        self.assertTrue(seed.seed_inflow_incomplete)
        self.assertIn("толық кіріс емес", seed.missing_data_kz)
        boundary = result.loc[BASE + 3]
        self.assertTrue(boundary.depth_boundary)
        self.assertIn("кейінгі алушылар", boundary.missing_data_kz)
        self.assertIn("5-қадам", boundary.next_step_kz)
        isolated = result.loc[BASE + 4]
        self.assertTrue(isolated.is_isolated)
        self.assertTrue(isolated.seed_inflow_incomplete)
        self.assertFalse(isolated.depth_boundary)
        self.assertIn("басым тексеруге ағындық дәлел жоқ", isolated.review_reason_kz)
        self.assertIn("Seed", isolated.data_flags_kz)
        self.assertIn("Байланыссыз", isolated.data_flags_kz)

    def test_all_role_templates_use_numeric_metrics(self):
        for role, expected in (
            ("consolidator", "жинақталу"), ("distributor", "тарату"),
            ("coordinator", "делдалдық"), ("terminal", "ұстап қалу"),
            ("peripheral", "дәлел жеткіліксіз"),
        ):
            roles = self.roles.copy()
            roles.loc[roles.gid == BASE + 1, "role"] = role
            text = build_node_insights(roles, self.metrics).set_index("gid").loc[BASE + 1, "review_reason_kz"]
            self.assertIn(expected, text)
            self.assertIn("10 000.00", text)
            self.assertIn("9 000.00", text)

    def test_join_is_by_gid_and_rejects_bad_snapshot(self):
        expected = build_node_insights(self.roles, self.metrics)
        shuffled = build_node_insights(self.roles.sample(frac=1, random_state=1),
                                       self.metrics.sample(frac=1, random_state=2))
        assert_frame_equal(expected, shuffled)
        with self.assertRaises(ValueError):
            build_node_insights(self.roles, self.metrics.iloc[1:])
        with self.assertRaises(ValueError):
            build_node_insights(pd.concat([self.roles, self.roles.iloc[:1]]), self.metrics)
        changed = self.metrics.copy()
        changed["gid"] = changed.gid.astype(float)
        with self.assertRaises(ValueError):
            build_node_insights(self.roles, changed)
        changed = self.metrics.copy()
        changed["is_seed"] = changed.is_seed.astype(str)
        with self.assertRaises(ValueError):
            build_node_insights(self.roles, changed)
        changed = self.metrics.copy()
        changed.loc[0, "truncated_by_depth"] = True
        with self.assertRaises(ValueError):
            build_node_insights(self.roles, changed)

    def test_roundoff_in_temporal_share_is_accepted_but_bad_data_is_not(self):
        metrics = self.metrics.copy()
        metrics.loc[metrics.gid == BASE + 1, "near_inflow_out_share"] = 1 + 2e-16
        build_node_insights(self.roles, metrics)
        metrics.loc[metrics.gid == BASE + 1, "near_inflow_out_share"] = 1.01
        with self.assertRaises(ValueError):
            build_node_insights(self.roles, metrics)


if __name__ == "__main__":
    unittest.main()
