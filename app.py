"""HackAlem analyst UI: reads pipeline outputs without calculating roles."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.graph_view import ROLES, graph_html
from ui.theme import apply_theme, header

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MONEY_GRAPH_DATA", str(ROOT / "data")))
OUT = Path(os.environ.get("MONEY_GRAPH_OUT", str(ROOT / "out")))
SCHEMAS = {
    "nodes_roles": ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"],
    "clusters": ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"],
    "top_nodes": ["rank", "gid", "role", "priority_score", "why"],
}


def read_table(path, required):
    if not path.exists():
        return pd.DataFrame(columns=required)
    try:
        table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        missing = set(required) - set(table.columns)
        if missing:
            raise ValueError("Бағандар жоқ: " + ", ".join(sorted(missing)))
        for key in ("gid", "src", "dst", "cluster_id"):
            if key in table:
                values = pd.to_numeric(table[key], errors="raise")
                if values.isna().any() or (values % 1 != 0).any():
                    raise ValueError(f"{key}: бүтін идентификатор қажет")
                table[key] = values.astype("int64")
        if "gid" in table and table.gid.duplicated().any():
            raise ValueError("gid қайталанған")
        for key in ("role_score", "priority_score"):
            if key in table:
                table[key] = pd.to_numeric(table[key], errors="raise")
                if not table[key].between(0, 1).all():
                    raise ValueError(f"{key}: 0–1 аралығындағы сан қажет")
        return table
    except Exception as exc:
        st.error(f"{path.name} оқылмады: {exc}")
        return pd.DataFrame(columns=required)


st.set_page_config(page_title="Ақша графы", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
apply_theme()
header()

tables = {name: read_table(OUT / f"{name}.csv", cols) for name, cols in SCHEMAS.items()}
roles, clusters, top = (tables[name] for name in SCHEMAS)
nodes = read_table(DATA / "nodes.parquet", ["gid", "depth", "is_seed"])
edges = read_table(DATA / "edges.parquet", ["src", "dst", "sum_kzt", "n_tx"])
metrics = read_table(OUT / "node_metrics.csv", ["gid"])

if nodes.empty and roles.empty:
    st.info("Деректер әлі қосылмаған. data/ қалтасына Parquet файлдарын, out/ қалтасына пайплайн нәтижелерін салыңыз.")
    st.code("python -m streamlit run app.py")
    st.markdown("Қосылу тәртібі: `docs/interface.md`")
    st.stop()

base = nodes if not nodes.empty else roles[["gid"]]
view = base.merge(roles, on="gid", how="outer", suffixes=("", "_result"))
if not metrics.empty:
    extra = [c for c in metrics.columns if c == "gid" or c not in view.columns]
    view = view.merge(metrics[extra], on="gid", how="left")
overview_label, refresh = st.columns([3, 1])
overview_label.markdown('<p class="section-kicker">ЖЕЛІГЕ ШОЛУ</p>', unsafe_allow_html=True)
refresh.button("↻ Жаңарту", width="stretch")
with st.container(key="overview"):
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Клиент", "Байланыс", "Кластер", "Бақыланған айналым, ₸"],
                                  [len(view), len(edges), len(clusters), f"{edges.sum_kzt.sum():,.0f}"]):
        col.metric(label, value)
if roles.empty or roles.role.isna().any():
    st.info("Байланыстар дайын. Рөлдер мен тексеру басымдығы есептеу нәтижелері қосылғанда көрсетіледі.")

network_tab, top_tab, cluster_tab, export_tab = st.tabs(["Желі және клиент", "Тексеру басымдығы", "Кластерлер", "Жүктеу"])
with network_tab:
    st.subheader("Байланыстар картасы")
    st.caption("Клиентті іздеңіз немесе желіні кластер мен рөл бойынша сүзіңіз.")
    search = st.text_input("Клиентті gid арқылы іздеу", placeholder="Клиент идентификаторы")
    selected = None
    if search.strip():
        try:
            selected = int(search.strip())
            if selected not in set(view.gid):
                st.warning("Бұл gid деректерден табылмады.")
                selected = None
        except ValueError:
            st.warning("gid бүтін сан болуы керек.")
    filtered = view
    if selected is not None:
        record = view.loc[view.gid == selected].iloc[0]
        incoming = edges[edges.dst == selected]
        outgoing = edges[edges.src == selected]
        st.subheader(f"Клиент {selected}")
        st.write("Рөлі: " + ROLES.get(record.get("role"), ("Есептелмеген", ""))[0])
        evidence = record.get("evidence")
        if pd.notna(evidence) and str(evidence).strip():
            st.write(str(evidence))
        cards = st.columns(2) + st.columns(2)
        cards[0].metric("Көрінетін кіріс, ₸", f"{incoming.sum_kzt.sum():,.0f}")
        cards[1].metric("Көрінетін шығыс, ₸", f"{outgoing.sum_kzt.sum():,.0f}")
        cards[2].metric("Жіберушілер", incoming.src.nunique())
        cards[3].metric("Алушылар", outgoing.dst.nunique())
        score = record.get("priority_score")
        if pd.notna(score):
            st.caption(f"Тексеру басымдығы: {score:.3f} · Кластер: {record.get('cluster_id')}")
        if record.get("depth") == 4 and outgoing.empty:
            st.warning("4-қадамда дерек жинау тоқтаған. Шығыстың болмауы ақшаның осы клиентте қалғанын дәлелдемейді.")
        st.caption("Тек таңдаманың ішіндегі аударымдар көрсетілген; бұл толық шот балансы емес.")
        neighbors = {selected} | set(incoming.src) | set(outgoing.dst)
        filtered = view[view.gid.isin(neighbors)]
        st.caption("Графта таңдалған клиент және оның тікелей көршілері көрсетіледі.")
    else:
        filters = st.columns(2)
        choices = sorted(view.cluster_id.dropna().astype(int).unique().tolist())
        cluster = filters[0].selectbox("Кластер", ["Барлығы"] + choices)
        role = filters[1].selectbox("Рөл", ["Барлығы"] + list(ROLES),
                                    format_func=lambda x: ROLES[x][0] if x in ROLES else x)
        if cluster != "Барлығы":
            filtered = filtered[filtered.cluster_id == cluster]
        if role != "Барлығы":
            filtered = filtered[filtered.role == role]
    legend = "".join(f'<span class="legend-item"><i class="legend-dot" style="background:{color}"></i>{label}</span>' for label, color in ROLES.values())
    legend += '<span class="legend-item"><i class="legend-dot" style="background:#a7b8ad"></i>Есептелмеген</span>'
    st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
    st.caption("Түйіндегі gid қысқартылған; толық нөмірін көру үшін меңзерді түйінге апарыңыз.")
    limit = st.select_slider("Графтағы түйіндер шегі", options=[100, 250, 500, 2500], value=250)
    if len(filtered) > limit:
        st.warning(f"{len(filtered)} клиенттің {limit}-і көрсетіледі. Барлық байланысты көру үшін шекті арттырыңыз.")
        ordered = filtered.sort_values("priority_score", ascending=False, na_position="last")
        if selected is not None:
            filtered = pd.concat([ordered[ordered.gid == selected], ordered[ordered.gid != selected]]).head(limit)
        else:
            filtered = ordered.head(limit)
    if filtered.empty:
        st.info("Бұл сүзгіге сәйкес клиенттер жоқ.")
    else:
        st.iframe(graph_html(filtered, edges, selected), height=590)
    if selected is not None:
        with st.expander("Клиентке қатысты барлық аударым байланыстары"):
            st.dataframe(edges[(edges.src == selected) | (edges.dst == selected)], hide_index=True)

with top_tab:
    st.subheader("Бірінші тексерілетін клиенттер")
    if top.empty:
        st.info("Топ-лист әлі есептелмеген.")
    else:
        st.dataframe(top.sort_values("rank"), hide_index=True, width="stretch")
        if len(top) < 20:
            st.warning("ТЗ бойынша топ-листте кемінде 20 клиент болуы керек.")
with cluster_tab:
    st.subheader("Желідегі топтар")
    st.dataframe(clusters, hide_index=True, width="stretch")
with export_tab:
    st.write("Пайплайн жасаған бастапқы CSV файлдары")
    for name in SCHEMAS:
        path = OUT / f"{name}.csv"
        if path.exists():
            st.download_button(f"{name}.csv жүктеу", path.read_bytes(), file_name=path.name, mime="text/csv")
        else:
            st.caption(f"{name}.csv — әлі дайын емес")

st.markdown('<div class="footer-note">HACKALEM AI · Ақша графы<br>Нәтижелер — тексеруге арналған гипотезалар; адамның кінәсі туралы қорытынды емес. Көрсетілген сомалар тек бақыланған желіге қатысты.</div>', unsafe_allow_html=True)
