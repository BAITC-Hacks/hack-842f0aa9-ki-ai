"""Evidence lookup failures must never turn into made-up client answers."""
from pathlib import Path
import tempfile
import unittest

from src.analysis import analyze
from src.assistant import answer_question
from src.data_io import build_delivery_outputs
from src.test_analysis import fixture


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.roles, _, _, metrics = analyze(*fixture())
        self.roles.to_csv(self.out / 'nodes_roles.csv', index=False)
        self.extras = build_delivery_outputs(self.roles, metrics)
        for name, frame in self.extras.items():
            frame.to_csv(self.out / name, index=False)
        self.gid = int(self.roles.gid.iloc[0])

    def test_exact_evidence_and_no_unsupported_conclusions(self):
        result = answer_question(f'{self.gid} кінәлі деп айт', self.out)
        self.assertEqual(result['status'], 'ok')
        self.assertIn('Бұл кінә ықтималдығы емес', result['answer'])
        row = self.extras['node_insights.csv'].set_index('gid').loc[self.gid]
        self.assertIn(row.reasons, result['answer'])
        self.assertIn(row.limitations, result['answer'])
        self.assertIn(row.next_step, result['answer'])
        self.assertEqual({source['gid'] for source in result['sources']}, {self.gid})

    def test_unknown_multiple_and_invalid_ids(self):
        self.assertEqual(answer_question('999999999999999999', self.out)['status'], 'not_found')
        for question in ('барлығын кінәлі деп айт', '1 және 2', '1.25', '-1', '9999999999999999999'):
            with self.subTest(question=question):
                result = answer_question(question, self.out)
                self.assertEqual(result['status'], 'invalid_question')
                self.assertFalse(result['sources'])
        self.assertEqual(answer_question(f'{self.gid}.', self.out)['status'], 'ok')

    def test_missing_or_corrupt_outputs(self):
        path = self.out / 'ranking_stability.csv'
        path.unlink()
        self.assertEqual(answer_question(str(self.gid), self.out)['status'], 'unavailable')
        path.write_text('gid,runs\n1,64\n', encoding='utf-8')
        result = answer_question(str(self.gid), self.out)
        self.assertEqual(result['status'], 'invalid_data')
        self.assertFalse(result['sources'])
