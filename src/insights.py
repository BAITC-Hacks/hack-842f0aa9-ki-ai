"""Kazakh analyst explanations built from a validated analysis snapshot."""
from __future__ import annotations

import pandas as pd
from src.analysis import _join_results

INSIGHT_COLUMNS = [
    "gid", "role", "review_reason_kz", "missing_data_kz", "next_step_kz",
    "seed_inflow_incomplete", "depth_boundary", "is_isolated", "data_flags_kz",
]


def _money(value):
    return f"{value:,.2f}".replace(",", " ")


def build_node_insights(nodes_roles, node_metrics):
    """Return one Kazakh explanation per gid without changing mandatory outputs.

    The three limitation flags are independent: an isolated seed has two flags.
    Text describes review hypotheses, never guilt or a full account balance.
    """
    frame = _join_results(nodes_roles, node_metrics)
    records = []
    for r in frame.itertuples(index=False):
        observed = (
            f"Көрінетін кіріс {_money(r.in_kzt)} ₸: {r.in_deg} жіберуші, {r.in_tx} аударым; "
            f"шығыс {_money(r.out_kzt)} ₸: {r.out_deg} алушы, {r.out_tx} аударым. "
        )
        if r.is_isolated:
            reason = "Байланыс саны 0; осы графта басым тексеруге ағындық дәлел жоқ. "
            next_step = "Алдымен 0 байланыстың себебін: таңдаманы, кезеңді және 5 000 ₸ сүзгісін тексеріңіз."
        elif r.role == "consolidator":
            reason = (
                f"{r.in_deg} жіберушіден {r.out_deg} шығыс алушыға шоғырланған құрылым бар; "
                "қаражаттың жинақталу мақсатын тексеруге негіз. "
            )
            next_step = (
                f"{r.in_deg} жіберушінің аударым мақсаттарын салыстырып, "
                f"{r.out_deg} алушымен байланысын тексеріңіз."
            )
        elif r.role == "distributor":
            reason = f"{r.out_deg} алушыға тарату байқалады; аударымдардың ортақ мақсатын тексеру қажет. "
            next_step = f"{r.out_deg} алушыға жасалған {r.out_tx} аударымның мақсаты мен күндерін салыстырыңыз."
        elif r.role == "transit":
            ratio = r.out_kzt / r.in_kzt if r.in_kzt else 0.
            reason = (
                f"Шығыс/кіріс={ratio:.3f}; шығыстың {100 * r.near_inflow_out_share:.1f}%-ы "
                "кіріс болған күні не келесі күні жасалған. Бұл транзитке кандидат белгісі. "
            )
            next_step = (
                f"{r.in_tx} кіріс пен {r.out_tx} шығыс аударымның нақты уақытын және "
                "сомалық сәйкестігін тексеріңіз; бір күндегі оқиғалар реті әзірге белгісіз."
            )
        elif r.role == "terminal":
            share = 100 * r.out_kzt / r.in_kzt if r.in_kzt else 0.
            reason = (
                f"Көрінетін шығыс кірістің {share:.1f}%-ы; осы кезеңдегі ұстап қалу белгісін "
                "толық үзіндімен тексеру қажет. "
            )
            next_step = (
                f"{_money(r.in_kzt)} ₸ кірістен кейінгі күндердің шығысын толық үзіндіден "
                "тексеріңіз; мұны шот қалдығы деп қабылдамаңыз."
            )
        elif r.role == "coordinator":
            reason = (
                f"Өз тобынан тыс {r.external_clusters} кластермен байланыс бар; "
                f"делдалдық көрсеткіші {r.betweenness:.6f}. Желідегі байланыстырушы рөлді тексеру қажет. "
            )
            next_step = (
                f"{r.external_clusters} сыртқы кластерге баратын байланысты қараңыз; "
                "түйінді алып тастау сценарийімен құрылымдық әсерін салыстырыңыз."
            )
        else:
            reason = "Арнайы рөлге дәлел жеткіліксіз; басымдықты тек көрінетін ағындармен салыстырыңыз. "
            next_step = (
                f"{r.in_tx + r.out_tx} кіріс/шығыс жазбасын және тікелей көршілерді қараңыз; "
                "рөлді жаңа дерек түскенде қайта бағалаңыз."
            )
        reason = observed + reason + f"Тексеру басымдығы {r.priority_score:.4f}."
        missing = ["Таңдама сыртындағы аударымдар мен толық шот үзіндісі берілмеген."]
        flags = []
        extra_steps = []
        if r.is_seed:
            flags.append("Seed: кіріс толық емес")
            missing.append(
                f"Seed клиенттің {_money(r.in_kzt)} ₸ көрінетін кірісі толық кіріс емес; "
                "таңдамаға кірмеген көздер жетіспейді."
            )
            extra_steps.append("Seed-тің таңдамадан тыс кірісін толық үзіндімен салыстырыңыз.")
        if r.truncated_by_depth:
            flags.append("4-қадам: шығыс белгісіз")
            missing.append(
                "4-қадамда жинау тоқтаған: көрінетін шығыс 0, кейінгі алушылар мен аударымдар белгісіз."
            )
            extra_steps.append("5-қадамдағы шығыс байланыстарын сұратып, содан кейін рөлді қайта бағалаңыз.")
        if r.is_isolated:
            flags.append("Байланыссыз клиент")
            missing.append("Байланыс саны 0; бұл клиенттің шынайы белсенділігі жоқ екенін дәлелдемейді.")
        if r.role == "transit":
            missing.append("Тек күн берілген; сағат/минут және қаражаттың нақты сәйкестігі жоқ.")
        records.append({
            "gid": r.gid, "role": r.role, "review_reason_kz": reason,
            "missing_data_kz": " ".join(missing),
            "next_step_kz": " ".join([next_step, *extra_steps]),
            "seed_inflow_incomplete": bool(r.is_seed),
            "depth_boundary": bool(r.truncated_by_depth),
            "is_isolated": bool(r.is_isolated),
            "data_flags_kz": "; ".join(flags) or "Арнайы шектеу белгісі жоқ; жалпы таңдама шектеулері сақталады",
        })
    return pd.DataFrame(records, columns=INSIGHT_COLUMNS)
