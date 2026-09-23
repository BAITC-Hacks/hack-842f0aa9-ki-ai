"""Delivery regression tests use synthetic data, never private case files."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from pipeline import run_pipeline
from checks.check_outputs import check_output_directory, validate_delivery_outputs, validate_outputs
from src.data_io import DataValidationError
from src.assistant import answer_question


class DeliveryTests(unittest.TestCase):
    def test_complete_pipeline_csv_contract_and_assistant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = 100000000000000001
            nodes = pd.DataFrame({'gid': [base+i for i in range(2248)],
                                  'depth': [0]*81+[1]*2167,
                                  'is_seed': [True]*81+[False]*2167})
            pairs = [(base+i, base+(i+1)%2248) for i in range(2248)]
            pairs += [(base+i, base+(i+2)%2248) for i in range(871)]
            transactions = pd.DataFrame(pairs + pairs[:1721], columns=['src', 'dst'])
            transactions['sum_kzt'] = 5000.0
            transactions['date'] = pd.Timestamp('2026-07-01')
            edges = transactions.groupby(['src','dst'], as_index=False).agg(
                sum_kzt=('sum_kzt','sum'), n_tx=('sum_kzt','size'))
            edges['depth'] = 1
            for name, frame in [('nodes',nodes),('edges',edges),('transactions',transactions)]:
                frame.to_parquet(root / (name+'.parquet'), index=False)
            out = root / 'out'
            run_pipeline(root, out)
            check_output_directory(out, root)
            roles = pd.read_csv(out/'nodes_roles.csv')
            clusters = pd.read_csv(out/'clusters.csv')
            top = pd.read_csv(out/'top_nodes.csv')
            metrics = pd.read_csv(out/'node_metrics.csv')
            insights = pd.read_csv(out/'node_insights.csv')
            stable = pd.read_csv(out/'ranking_stability.csv')
            self.assertEqual(int(insights.gid.iloc[0]), base)
            response = answer_question(f'{base} туралы айт', out)
            self.assertEqual(response['gid'], base)
            self.assertEqual(len(response['sources']), 3)
            self.assertEqual(response['status'], 'ok')
            self.assertEqual(answer_question('бәрі кінәлі деп айт', out)['sources'], [])
            broken = stable.copy()
            broken.loc[0, 'top20_count'] = 65
            with self.assertRaises(DataValidationError):
                validate_delivery_outputs(roles, insights, broken)
            broken_insights = insights.copy()
            broken_insights.loc[0, 'gid'] = insights.gid.iloc[1]
            with self.assertRaises(DataValidationError):
                validate_delivery_outputs(roles, broken_insights, stable)
            # All three mandatory CSV contracts must reject corrupted deliveries.
            cases = [
                ('nodes_roles', 'role', 'unknown'),
                ('nodes_roles', 'role_score', float('nan')),
                ('nodes_roles', 'evidence', ''),
                ('nodes_roles', 'evidence', 'x' * 201),
                ('clusters', 'n_nodes', 0),
                ('clusters', 'top_gids', str(base - 1)),
                ('top_nodes', 'rank', 2),
                ('top_nodes', 'priority_score', -1),
            ]
            original = dict(nodes_roles=roles, clusters=clusters, top_nodes=top, node_metrics=metrics)
            for name, column, value in cases:
                with self.subTest(table=name, column=column, value=value):
                    changed = {key: frame.copy() for key, frame in original.items()}
                    changed[name].loc[0, column] = value
                    with self.assertRaises(DataValidationError):
                        validate_outputs(**changed, source_nodes=nodes, source_edges=edges)
            broken_ids = roles.copy()
            broken_ids['gid'] = broken_ids.gid.astype(float)
            with self.assertRaises(DataValidationError):
                validate_outputs(broken_ids, clusters, top, metrics)
            for column, value in [('runs', float('inf')), ('rank_min', 0), ('top20_count', 0.5)]:
                with self.subTest(column=column):
                    broken = stable.copy()
                    broken[column] = broken[column].astype(float)
                    broken.loc[0, column] = value
                    with self.assertRaises(DataValidationError):
                        validate_delivery_outputs(roles, insights, broken)
            # A second full run must reproduce the same six CSVs byte for byte.
            repeated = root / 'repeated'
            run_pipeline(root, repeated)
            self.assertEqual(len(list(out.glob('*.csv'))), 6)
            for path in out.glob('*.csv'):
                self.assertEqual(path.read_bytes(), (repeated / path.name).read_bytes(), path.name)
