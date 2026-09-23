"""No-network tests for real provider request formats and evidence grounding."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.analysis import analyze
from src.assistant import answer_with_ai, build_context
from src.data_io import build_delivery_outputs
from src.llm import AIConfig, AIError, generate_answer
from src.test_analysis import fixture


def response(answer='Көрсеткіш бойынша тексеру ұсынылады. [N1]', citations=None, provider='OpenAI'):
    payload = json.dumps({'answer': answer, 'citations': citations or ['N1']}, ensure_ascii=False)
    if provider == 'OpenAI':
        body = {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': payload}]}]}
    else:
        body = {'choices': [{'finish_reason': 'stop', 'message': {'content': payload}}]}
    return Mock(status_code=200, json=Mock(return_value=body))


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [{'id': 'N1', 'gid': '100000000000000001', 'priority_score': 0.6}]
        self.config = AIConfig('OpenAI', 'test-only-not-a-real-key')

    @patch('src.llm.requests.post')
    def test_openai_request_and_valid_citations(self, post):
        post.return_value = response()
        result = generate_answer('Неге тексеру керек?', self.evidence, self.config)
        self.assertEqual(result['citations'], ['N1'])
        self.assertEqual(post.call_args.args[0], 'https://api.openai.com/v1/responses')
        params = post.call_args.kwargs
        self.assertFalse(params['allow_redirects'])
        self.assertEqual(params['timeout'], (5, 45))
        self.assertFalse(params['json']['store'])
        self.assertTrue(params['json']['text']['format']['strict'])
        self.assertNotIn(self.config.api_key, json.dumps(params['json']))
        self.assertNotIn(self.config.api_key, repr(self.config))
        post.return_value.close.assert_called_once()

    @patch('src.llm.requests.post')
    def test_nvidia_request_and_response(self, post):
        post.return_value = response(provider='NVIDIA')
        result = generate_answer('Түсіндір', self.evidence, AIConfig('NVIDIA', 'test-only-key'))
        self.assertEqual(result['citations'], ['N1'])
        self.assertEqual(post.call_args.args[0], 'https://integrate.api.nvidia.com/v1/chat/completions')
        self.assertFalse(post.call_args.kwargs['json']['stream'])
        self.assertEqual(post.call_args.kwargs['json']['model'], 'meta/llama-3.3-70b-instruct')

    @patch('src.llm.requests.post')
    def test_missing_key_and_bad_provider_never_send(self, post):
        for config in (AIConfig('OpenAI', ''), AIConfig('unknown', 'test')):
            with self.subTest(provider=config.provider), self.assertRaises(AIError):
                generate_answer('Сұрақ', self.evidence, config)
        post.assert_not_called()

    @patch('src.llm.requests.post')
    def test_timeout_and_network_errors_are_sanitized(self, post):
        for error in (requests.Timeout('secret-request'), requests.ConnectionError('secret-request')):
            post.side_effect = error
            with self.assertRaises(AIError) as caught:
                generate_answer('Сұрақ', self.evidence, self.config)
            self.assertNotIn('secret-request', str(caught.exception))

    @patch('src.llm.requests.post')
    def test_http_failures_do_not_expose_provider_body(self, post):
        for status in (302, 400, 401, 403, 429, 500):
            with self.subTest(status=status):
                post.return_value = Mock(status_code=status, text='secret-key')
                with self.assertRaises(AIError) as caught:
                    generate_answer('Сұрақ', self.evidence, self.config)
                self.assertNotIn('secret-key', str(caught.exception))
                post.return_value.json.assert_not_called()

    @patch('src.llm.requests.post')
    def test_unknown_citation_missing_citation_and_invented_gid_are_rejected(self, post):
        cases = [('Жауап [X9]', ['X9']), ('Жауап', ['N1']),
                 ('100000000000000009 тексеру [N1]', ['N1'])]
        for answer, citations in cases:
            with self.subTest(answer=answer):
                post.return_value = response(answer, citations)
                with self.assertRaises(AIError):
                    generate_answer('Сұрақ', self.evidence, self.config)

    @patch('src.llm.requests.post')
    def test_truncated_and_malformed_outputs_are_rejected(self, post):
        for body in ({'status': 'incomplete', 'output': []}, {'status': 'completed', 'output': []}):
            post.return_value = Mock(status_code=200, json=Mock(return_value=body))
            with self.assertRaises(AIError):
                generate_answer('Сұрақ', self.evidence, self.config)


class GroundedAssistantTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.edges, self.nodes, transactions = fixture()
        first = int(self.nodes.gid.iloc[0])
        self.nodes = pd.concat([self.nodes, pd.DataFrame({
            'gid': [first + 5], 'depth': [1], 'is_seed': [False],
        })], ignore_index=True)
        transactions = pd.concat([transactions, pd.DataFrame({
            'src': [first], 'dst': [first + 2],
            'sum_kzt': [7000.], 'date': ['2026-07-01'],
        })], ignore_index=True)
        self.edges = transactions.groupby(['src', 'dst'], as_index=False).agg(
            sum_kzt=('sum_kzt', 'sum'), n_tx=('sum_kzt', 'size'))
        self.roles, _, _, metrics = analyze(self.edges, self.nodes, transactions)
        self.roles.to_csv(self.root / 'nodes_roles.csv', index=False)
        for name, table in build_delivery_outputs(self.roles, metrics).items():
            table.to_csv(self.root / name, index=False)
        self.edges.to_parquet(self.root / 'edges.parquet', index=False)
        self.ids = self.roles.gid.astype(int).tolist()

    def test_top_scope_and_selected_scope_are_bounded(self):
        evidence, sources = build_context('Кімді бірінші тексереміз?', self.root)
        self.assertEqual(len(evidence), 5)
        self.assertTrue(all(row['scope'] == 'top_5_only' for row in evidence))
        self.assertTrue(all(isinstance(row['gid'], str) for row in evidence))
        evidence, sources = build_context('Неге?', self.root, self.ids[-1])
        self.assertEqual([row['gid'] for row in evidence], [str(self.ids[-1])])

    def test_explicit_ids_override_selection_and_compute_common_recipients(self):
        incoming = self.edges.groupby('dst').src.nunique()
        target = int(incoming[incoming.ge(2)].index[0])
        payers = self.edges.loc[self.edges.dst.eq(target), 'src'].head(2).astype(int).tolist()
        evidence, sources = build_context(' және '.join(map(str, payers)), self.root, self.ids[-1], self.root)
        graph = evidence[-1]
        self.assertEqual(graph['id'], 'G1')
        row = next(row for row in graph['recipients'] if row['gid'] == str(target))
        expected = self.edges.loc[self.edges.src.isin(payers) & self.edges.dst.eq(target)]
        self.assertEqual(row['sum_kzt'], round(float(expected.sum_kzt.sum()), 2))
        self.assertEqual(row['payers'], 2)

    @patch('src.llm.requests.post')
    def test_unknown_or_excessive_ids_never_call_api(self, post):
        for question in ('999999999999999999 туралы', ' '.join(map(str, self.ids[:6]))):
            result = answer_with_ai(question, self.root, config=AIConfig('OpenAI', 'test'))
            self.assertEqual(result['status'], 'invalid_data')
        post.assert_not_called()

    @patch('src.llm.requests.post')
    def test_llm_mode_reports_actual_model_evidence_and_sources(self, post):
        post.return_value = response()
        result = answer_with_ai('Неге?', self.root, config=AIConfig('OpenAI', 'test'), selected_gid=self.ids[0])
        self.assertEqual(result['mode'], 'llm')
        self.assertEqual(result['sources'][0]['gid'], self.ids[0])
        self.assertEqual(result['evidence'][0]['gid'], str(self.ids[0]))
        self.assertIn('gpt-', result['model'])

    @patch('src.llm.requests.post')
    def test_api_failure_is_marked_as_offline_fallback(self, post):
        post.side_effect = requests.Timeout()
        result = answer_with_ai(str(self.ids[0]), self.root, config=AIConfig('OpenAI', 'test'))
        self.assertEqual(result['mode'], 'offline')
        self.assertEqual(result['status'], 'fallback')
        self.assertIn('жергілікті', result['notice'])
        self.assertTrue(result['sources'])
