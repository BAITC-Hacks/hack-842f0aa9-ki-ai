"""Offline evidence assistant. No API, network calls, or language model."""
from pathlib import Path
import argparse
import json
import re
import pandas as pd


def answer_question(question: str, out_dir='out') -> dict:
    """Answer about one explicit gid using saved evidence; never infer guilt."""
    ids = re.findall(r'(?<!\d)\d{10,19}(?!\d)', question)
    if len(set(ids)) != 1:
        return {'mode': 'offline', 'answer': 'Сұрақта бір клиенттің толық gid мәнін көрсетіңіз.', 'sources': []}
    gid = int(ids[0])
    root = Path(out_dir)
    roles = pd.read_csv(root / 'nodes_roles.csv')
    insights = pd.read_csv(root / 'node_insights.csv')
    stability = pd.read_csv(root / 'ranking_stability.csv')
    matches = [frame.loc[frame.gid.eq(gid)] for frame in (roles, insights, stability)]
    if any(len(frame) != 1 for frame in matches):
        return {'mode': 'offline', 'gid': gid, 'answer': 'Бұл gid үшін толық, бірегей нәтиже табылмады.', 'sources': []}
    role, insight, stable = [frame.iloc[0] for frame in matches]
    return {'mode': 'offline', 'gid': gid,
        'answer': f"gid={gid}; рөл={role.role}; басымдық={role.priority_score:.4f}. "
                  f"{insight.reasons}\nШектеулер: {insight.limitations}\n"
                  f"Келесі қадам: {insight.next_step}\n"
                  f"Топ-20: {int(stable.top20_count)}/{int(stable.runs)} сынақ; "
                  f"орын {int(stable.rank_min)}–{int(stable.rank_max)}. "
                  'Бұл кінә ықтималдығы емес. Жауап сақталған деректерден құрастырылды.',
        'sources': [{'file': name, 'gid': gid} for name in
                    ('nodes_roles.csv', 'node_insights.csv', 'ranking_stability.csv')]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--out', default='out')
    args = parser.parse_args()
    print(json.dumps(answer_question(args.question, args.out), ensure_ascii=False, indent=2))
