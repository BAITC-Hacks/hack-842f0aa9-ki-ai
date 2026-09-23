"""Presentation adapters for the analytics upgrade; no duplicate scoring logic."""
from html import escape

import pandas as pd
import streamlit as st

from src.insights import build_node_insights
from src.scenarios import analyze_weight_sensitivity, simulate_node_removal


@st.cache_data(show_spinner=False, max_entries=4)
def prepare_panels(roles, metrics):
    insights = build_node_insights(roles, metrics)
    sensitivity = analyze_weight_sensitivity(roles, metrics)
    return insights, sensitivity["summary"], sensitivity["scenarios"]


@st.cache_data(show_spinner=False, max_entries=12)
def removal_result(edges, nodes, gid):
    return simulate_node_removal(edges, nodes, gid)


def client_insights(gid, record, insights, stability):
    st.markdown("#### Тексеру карточкасы")
    score_cols = st.columns(2)
    score_cols[0].metric("Тексеру басымдығы", f"{record['priority_score']:.3f}")
    score_cols[1].metric("Рөл ережелеріне сәйкестік", f"{record['role_score']:.3f}")
    st.caption("Бұл бағалар кінә ықтималдығы емес. Басымдық тексеру кезегін, рөл бағасы ережелерге сәйкестікті көрсетеді.")
    matched = insights.loc[insights.gid == gid] if not insights.empty else insights
    if matched.empty:
        st.info("Толық қазақша карточка үшін осы нұсқаның пайплайнын қайта іске қосыңыз.")
        return
    insight = matched.iloc[0]
    flags = []
    for key, label in [("seed_inflow_incomplete", "Seed: кіріс толық емес"),
                       ("depth_boundary", "4-қадам: шығыс белгісіз"),
                       ("is_isolated", "Байланыссыз клиент")]:
        if bool(insight[key]):
            flags.append(label)
    st.markdown("**Дерек шектеулері:** " + (" · ".join(flags) if flags else "Жалпы таңдама шектеулері сақталады"))
    columns = st.columns(3)
    for col, title, field in zip(columns, ["Неге тексеру керек?", "Не белгісіз?", "Келесі қадам"],
                                  ["review_reason_kz", "missing_data_kz", "next_step_kz"]):
        col.markdown(f'<div class="insight-card"><h4>{title}</h4><p>{escape(str(insight[field]))}</p></div>', unsafe_allow_html=True)
    selected = stability.loc[stability.gid == gid] if not stability.empty else stability
    if not selected.empty:
        row = selected.iloc[0]
        st.info(f"Топ-{int(row.top_k)} тұрақтылығы: {int(row.top_k_count)}/{int(row.scenario_count)} нұсқада тізімге кірді. "
                f"Орын аралығы: {int(row.best_rank)}–{int(row.worst_rank)}.")
        st.caption("Алты салмақтың ±10% өзгерісі нормалауға дейін тексеріледі. Бұл жиілік статистикалық сенімділік немесе қылмыс ықтималдығы емес.")


def stability_panel(top, summary, scenarios):
    st.markdown("#### Рейтинг салмақтарға қаншалықты тәуелді?")
    st.caption("64 комбинацияда тек басымдық салмақтары өзгереді. Граф, рөлдер және метрикалар тұрақты қалады. Тең бағалар gid арқылы реттеледі.")
    if summary.empty:
        st.info("Тұрақтылық есебіне толық node_metrics.csv қажет. Пайплайнды қайта іске қосыңыз.")
        return
    stable = int(summary.top_k_count.eq(summary.scenario_count).sum())
    varied = scenarios.loc[scenarios.scenario.ne("baseline")]
    cols = st.columns(3)
    cols[0].metric("Барлық нұсқада топта", stable)
    cols[1].metric("Кемінде бір рет кірген", int(summary.top_k_count.gt(0).sum()))
    cols[2].metric("Бастапқы топпен ең аз ортақ", int(varied.overlap_with_baseline.min()))
    joined = top[["rank", "gid"]].merge(summary, on="gid", validate="one_to_one")
    table = joined[["rank", "gid", "top_k_count", "scenario_count", "best_rank", "worst_rank"]].copy()
    # Browser tables must not round these 18-digit identifiers.
    table["gid"] = table.gid.astype(str)
    table = table.rename(columns={"rank": "Бастапқы орын", "gid": "Клиент (gid)",
                                  "top_k_count": "Топқа кіргені", "scenario_count": "Нұсқа саны",
                                  "best_rank": "Ең жоғары орын", "worst_rank": "Ең төмен орын"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Бірдей бастапқы бағалар жасанды түрде ажыратылмайды. Тұрақтылық бағалау әдісінің сезімталдығын көрсетеді.")


def removal_panel(gid, edges, nodes):
    with st.expander("Түйінді алып тастағанда не өзгереді?"):
        st.write("Таңдалған клиентті графтың көшірмесінен алып, желі құрылымын салыстырыңыз.")
        st.caption("Бастапқы деректер өзгермейді. Бұл нақты бұғаттау, ақша жоғалуы немесе қайта бағытталуы туралы болжам емес.")
        if nodes.empty or "is_seed" not in nodes:
            st.info("Сценарий үшін бастапқы nodes.parquet қажет.")
            return
        if st.button("Сценарийді есептеу", key="run_removal"):
            st.session_state["removal_gid"] = gid
        if st.session_state.get("removal_gid") != gid:
            return
        try:
            with st.spinner("Құрылымдық әсер есептелуде…"):
                result = removal_result(edges, nodes, int(gid))
        except (ValueError, KeyError) as exc:
            st.error(f"Сценарий есептелмеді: {exc}")
            return
        impact = result["impact"]
        cols = st.columns(3)
        cols[0].metric("Seed-тен жолы жоғалған", impact["lost_seed_reachable_nodes"])
        cols[1].metric("Жаңадан оқшауланған", impact["newly_isolated_nodes"])
        cols[2].metric("Қосымша фрагмент", impact["additional_fragments"])
        labels = {"nodes": "Клиенттер", "edges": "Бағытталған байланыстар",
                  "weak_components": "Әлсіз байланысқан компоненттер",
                  "largest_weak_component": "Ең үлкен компонент көлемі",
                  "seed_reachable_nodes": "Seed-тен жетуге болатын клиенттер"}
        changes = result["network_changes"]
        changes = changes.loc[changes.metric.isin(labels)].copy()
        changes["metric"] = changes.metric.map(labels)
        for col in ("before", "after", "change"):
            changes[col] = changes[col].astype("int64")
        st.dataframe(changes.rename(columns={"metric": "Көрсеткіш", "before": "Бұрын", "after": "Кейін", "change": "Өзгеріс"}),
                     hide_index=True, width="stretch")
        st.write(result["note_kz"])
        st.caption("Жолы жоғалғандар санына алынып тасталған клиент кірмейді. Балама seed-тен жетуге болатын клиенттер жоғалды деп саналмайды.")
        affected = result["affected_nodes"].copy()
        affected["gid"] = affected.gid.astype(str)
        st.markdown("**Әсер еткен клиенттер**")
        st.dataframe(affected, hide_index=True, width="stretch")
        st.download_button("Сценарий нәтижесін жүктеу", result["network_changes"].to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"removal_{gid}.csv", mime="text/csv", key=f"download_removal_{gid}")
