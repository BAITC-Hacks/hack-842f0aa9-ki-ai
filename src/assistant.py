"""Local evidence lookup with an explicit optional grounded LLM mode."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import os
import numpy as np
import pandas as pd

from checks.check_outputs import ALLOWED_ROLES, validate_delivery_outputs
from src.data_io import DataValidationError
from src.llm import AIConfig, AIError, PROVIDERS, generate_answer

SOURCES = ('nodes_roles.csv', 'node_insights.csv', 'ranking_stability.csv')


def _read_frames(out_dir: str | Path) -> list[pd.DataFrame]:
    frames = []
    for name in SOURCES:
        frame = pd.read_csv(Path(out_dir) / name, dtype={'gid': 'string'})
        if 'gid' not in frame or not frame.gid.str.fullmatch(r'\d{1,19}').fillna(False).all():
            raise DataValidationError(f'{name}: invalid gid')
        frame['gid'] = frame.gid.astype('int64')
        frames.append(frame)
    roles, insights, stability = frames
    if roles.gid.duplicated().any() or not set(roles.role).issubset(ALLOWED_ROLES):
        raise DataValidationError('Invalid or duplicate roles')
    for name in ('priority_score', 'role_score'):
        roles[name] = pd.to_numeric(roles[name], errors='raise')
        if not roles[name].between(0, 1).all():
            raise DataValidationError(f'Invalid {name}')
    validate_delivery_outputs(roles, insights, stability)
    return frames


def _response(answer: str, status: str, gid: int | None = None) -> dict:
    result = {'mode': 'offline', 'status': status, 'answer': answer, 'sources': []}
    if gid is not None:
        result['gid'] = gid
    return result


def answer_question(question: str, out_dir: str | Path = 'out') -> dict:
    """Return a factual card for one explicit gid, with safe missing-data states."""
    ids = re.findall(r'(?<![\w.+-])\d{1,19}(?!\w|\.\d)', question)
    if len(set(ids)) != 1:
        return _response('Сұрақта бір клиенттің толық gid мәнін көрсетіңіз.', 'invalid_question')
    gid = int(ids[0])
    if gid > 2**63 - 1:
        return _response('gid int64 шегінен аспайтын бүтін сан болуы керек.', 'invalid_question')
    try:
        frames = _read_frames(out_dir)
    except FileNotFoundError:
        return _response('Көмекшіге CSV нәтижелері жетіспейді. Алдымен python pipeline.py --data data --out out командасын орындаңыз.', 'unavailable', gid)
    except (OSError, ValueError, KeyError, AttributeError, OverflowError):
        return _response('CSV нәтижелері оқылмады немесе өзара сәйкес емес. Пайплайнды қайта іске қосып, checks/check_outputs.py арқылы тексеріңіз.', 'invalid_data', gid)
    matches = [frame.loc[frame.gid.eq(gid)] for frame in frames]
    if any(len(frame) != 1 for frame in matches):
        return _response('Бұл gid үшін нәтиже табылмады. Толық идентификаторды тексеріңіз.', 'not_found', gid)
    role, insight, stable = [frame.iloc[0] for frame in matches]
    answer = (
        f"gid={gid}; рөл={role.role}; тексеру басымдығы={role.priority_score:.4f}; "
        f"рөл ережелеріне сәйкестік={role.role_score:.4f}.\n\n"
        f"Неге тексеру керек: {insight.reasons}\n\n"
        f"Не белгісіз: {insight.limitations}\n\n"
        f"Келесі қадам: {insight.next_step}\n\n"
        f"Топ-20: {int(stable.top20_count)}/{int(stable.runs)} сынақ; "
        f"орын {int(stable.rank_min)}–{int(stable.rank_max)}. "
        'Бұл кінә ықтималдығы емес. Жауап сақталған деректерден құрастырылды.'
    )
    result = _response(answer, 'ok', gid)
    result['sources'] = [{'file': name, 'gid': gid} for name in SOURCES]
    return result


def build_context(question: str, out_dir: str | Path, selected_gid: int | None = None,
                  data_dir: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    """Bound context locally: up to five cards and computed common recipients."""
    roles, insights, stability = _read_frames(out_dir)
    known = set(roles.gid)
    tokens = re.findall(r'(?<![\w.+-])\d{1,19}(?!\w|\.\d)', question)
    # Counts such as 'top 5' are not ids unless they occur in the data.
    ids = list(dict.fromkeys(int(value) for value in tokens if len(value) >= 10 or int(value) in known))
    if len(ids) > 5:
        raise DataValidationError('Бір сұрақта ең көбі 5 клиентті салыстырыңыз.')
    if any(gid not in known for gid in ids):
        raise DataValidationError('Сұрақтағы gid деректерден табылмады.')
    scope = 'explicit_clients'
    if not ids and selected_gid is not None:
        if selected_gid not in known:
            raise DataValidationError('Таңдалған gid деректерден табылмады.')
        ids = [selected_gid]
        scope = 'selected_client'
    if not ids:
        ids = roles.sort_values(['priority_score', 'gid'], ascending=[False, True]).gid.head(5).tolist()
        scope = 'top_5_only'
    evidence, sources = [], []
    for index, gid in enumerate(ids, 1):
        row = roles.loc[roles.gid.eq(gid)].iloc[0]
        insight = insights.loc[insights.gid.eq(gid)].iloc[0]
        stable = stability.loc[stability.gid.eq(gid)].iloc[0]
        item = {'id': f'N{index}', 'gid': str(gid), 'scope': scope,
                'role': row.role, 'role_score': float(row.role_score),
                'priority_score': float(row.priority_score), 'cluster_id': int(row.cluster_id),
                'reasons': insight.reasons, 'limitations': insight.limitations, 'next_step': insight.next_step,
                'runs': int(stable.runs), 'top20_count': int(stable.top20_count),
                'rank_min': int(stable.rank_min), 'rank_max': int(stable.rank_max)}
        evidence.append(item)
        sources.append({'id': item['id'], 'gid': int(gid), 'file': ', '.join(SOURCES)})
    if len(ids) > 1 and data_dir is not None and (Path(data_dir) / 'edges.parquet').is_file():
        edges = pd.read_parquet(Path(data_dir) / 'edges.parquet')
        # Analytics already enforces full data integrity in the pipeline. Here check
        # only the columns and exact ids needed for this read-only graph query.
        if not {'src', 'dst', 'sum_kzt', 'n_tx'}.issubset(edges.columns):
            raise DataValidationError('Байланыс деректерінің бағандары толық емес.')
        for col in ('src', 'dst'):
            if not pd.api.types.is_integer_dtype(edges[col]) or not set(edges[col]).issubset(known):
                raise DataValidationError('Байланыс gid мәндері CSV клиенттерімен сәйкес емес.')
        if edges.duplicated(['src', 'dst']).any() or edges[['sum_kzt', 'n_tx']].isna().any().any():
            raise DataValidationError('Байланыс деректері жарамсыз.')
        for col in ('sum_kzt', 'n_tx'):
            if not pd.api.types.is_numeric_dtype(edges[col]) or not np.isfinite(edges[col]).all() or (edges[col] <= 0).any():
                raise DataValidationError('Байланыс сомасы мен саны жарамсыз.')
        if not pd.api.types.is_integer_dtype(edges.n_tx):
            raise DataValidationError('Аударым саны бүтін сан болуы керек.')
        outgoing = edges.loc[edges.src.isin(ids)]
        recipients = outgoing.groupby('dst').agg(payers=('src', 'nunique'), sum_kzt=('sum_kzt', 'sum'), n_tx=('n_tx', 'sum'))
        common = recipients.loc[recipients.payers.eq(len(ids))].sort_values(['sum_kzt'], ascending=False).head(10)
        rows = [{'gid': str(int(gid)), 'payers': int(row.payers), 'sum_kzt': round(float(row.sum_kzt), 2),
                 'n_tx': int(row.n_tx)} for gid, row in common.iterrows()]
        evidence.append({'id': 'G1', 'operation': 'common_direct_recipients', 'from_gids': [str(g) for g in ids],
                         'all_selected_payers_required': True, 'limit': 10, 'recipients': rows,
                         'limitation': 'Тек тікелей бақыланған аударымдар. Ортақ алушы болуы рөлін не кінәсін дәлелдемейді.'})
        sources.append({'id': 'G1', 'gid': ', '.join(map(str, ids)), 'file': 'edges.parquet (ортақ тікелей алушылар)'})
    return evidence, sources


def answer_with_ai(question: str, out_dir: str | Path = 'out', *, config: AIConfig,
                   selected_gid: int | None = None, data_dir: str | Path | None = None) -> dict:
    """One model call after local retrieval. Errors visibly fall back to local evidence."""
    try:
        evidence, sources = build_context(question, out_dir, selected_gid, data_dir)
    except (OSError, ValueError, KeyError, AttributeError, OverflowError):
        return _response('ЖИ контексті дайындалмады. gid мәндерін, ең көбі 5 клиент шегін және пайплайн нәтижелерін тексеріңіз.', 'invalid_data')
    try:
        generated = generate_answer(question, evidence, config)
    except AIError as exc:
        fallback = answer_question(evidence[0]['gid'], out_dir)
        fallback.update(status='fallback', notice=str(exc) + ' Төменде бірінші клиенттің жергілікті карточкасы көрсетілді.')
        return fallback
    cited = [source for source in sources if source['id'] in generated['citations']]
    return {'mode': 'llm', 'status': 'ok', 'provider': config.provider, 'model': config.model_name,
            'answer': generated['answer'], 'sources': cited, 'evidence': evidence,
            'notice': 'ЖИ түсіндірмесі — тексеруге арналған жоба. Сандарды дереккөзбен салыстырыңыз.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--out', default='out')
    parser.add_argument('--ai', choices=list(PROVIDERS), help='Explicitly send bounded evidence to this provider')
    parser.add_argument('--model', default='')
    parser.add_argument('--data', default='data')
    args = parser.parse_args()
    if args.ai:
        config = AIConfig(args.ai, os.environ.get(PROVIDERS[args.ai][2], ''), args.model)
        result = answer_with_ai(args.question, args.out, config=config, data_dir=args.data)
    else:
        result = answer_question(args.question, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
