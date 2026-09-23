"""Offline evidence lookup. No network calls or generative language model."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import pandas as pd

from checks.check_outputs import ALLOWED_ROLES, validate_delivery_outputs
from src.data_io import DataValidationError

SOURCES = ('nodes_roles.csv', 'node_insights.csv', 'ranking_stability.csv')


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
        frames = []
        for name in SOURCES:
            # Never round an 18-digit gid through float.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--out', default='out')
    args = parser.parse_args()
    result = answer_question(args.question, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
