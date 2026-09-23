"""UI integration checks; fixtures are temporary and never presented as real results."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class InterfaceTests(unittest.TestCase):
    def test_upgraded_cards_stability_and_removal(self):
        from src.analysis import analyze
        from src.test_analysis import fixture

        edges, nodes, transactions = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "MONEY_GRAPH_DATA": tmp, "MONEY_GRAPH_OUT": tmp,
        }):
            for name, frame in (("nodes", nodes), ("edges", edges)):
                frame.to_parquet(Path(tmp) / f"{name}.parquet")
            for name, frame in zip(("nodes_roles", "clusters", "top_nodes", "node_metrics"),
                                   analyze(edges, nodes, transactions)):
                frame.to_csv(Path(tmp) / f"{name}.csv", index=False)
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(at.exception)
            self.assertFalse(at.error)
            self.assertEqual(len(at.get("download_button")), 6)
            target = int(nodes.gid.iloc[1])
            at.text_input[0].set_value(str(target)).run()
            self.assertFalse(at.exception)
            self.assertTrue(any("Неге тексеру керек?" in item.value for item in at.markdown))
            self.assertTrue(any("64" in item.value for item in at.info))
            at.button(key="run_removal").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any(m.label == "Seed-тен жолы жоғалған" for m in at.metric))
            self.assertEqual(len(at.get("download_button")), 7)
            # A newly selected client must not display the previous simulation.
            at.text_input[0].set_value(str(int(nodes.gid.iloc[-1]))).run()
            self.assertFalse(any(m.label == "Seed-тен жолы жоғалған" for m in at.metric))
            self.assertFalse(at.exception)

    def test_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = {key: os.environ.get(key) for key in ("MONEY_GRAPH_DATA", "MONEY_GRAPH_OUT")}
            os.environ["MONEY_GRAPH_DATA"] = tmp
            os.environ["MONEY_GRAPH_OUT"] = tmp
            try:
                at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
                self.assertFalse(at.exception)
                self.assertTrue(at.info)
                pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 4, 0], "is_seed": [True, False, True]}).to_parquet(Path(tmp) / "nodes.parquet")
                pd.DataFrame({"src": [1], "dst": [2], "sum_kzt": [5000.0], "n_tx": [1]}).to_parquet(Path(tmp) / "edges.parquet")
                at.run()
                self.assertFalse(at.exception)
                at.text_input[0].set_value("2").run()
                self.assertFalse(at.exception)
                self.assertTrue(any("4-" in w.value for w in at.warning))
                at.text_input[0].set_value("3").run()
                self.assertFalse(at.exception)
                at.text_input[0].set_value("999").run()
                self.assertTrue(at.warning)
                at.text_input[0].set_value("abc").run()
                self.assertTrue(at.warning)
                pd.DataFrame({"gid": [1, 2, 3], "role": ["distributor", "peripheral", "peripheral"],
                              "role_score": [0.5, 0.2, 0.1], "cluster_id": [0, 0, 1],
                              "priority_score": [0.7, 0.3, 0.1], "evidence": ["fixture"] * 3}).to_csv(Path(tmp) / "nodes_roles.csv", index=False)
                pd.DataFrame({"rank": [1], "gid": [1], "role": ["distributor"], "priority_score": [0.7], "why": ["fixture"]}).to_csv(Path(tmp) / "top_nodes.csv", index=False)
                pd.DataFrame({"cluster_id": [0], "n_nodes": [2], "n_seed": [1], "sum_kzt_internal": [5000], "top_gids": ["1"], "hypothesis": ["fixture"]}).to_csv(Path(tmp) / "clusters.csv", index=False)
                at.text_input[0].set_value("").run()
                self.assertFalse(at.exception)
                at.selectbox[0].select(1).run()
                self.assertFalse(at.exception)
                at.selectbox[1].select("consolidator").run()
                self.assertFalse(at.exception)
                self.assertEqual(len(at.get("download_button")), 3)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
