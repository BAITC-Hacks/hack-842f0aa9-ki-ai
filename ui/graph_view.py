"""Render an offline, directed transaction graph."""
from html import escape
import json
import re

from pyvis.network import Network

ROLES = {
    "consolidator": ("Жинаушы", "#22c55e"),
    "transit": ("Транзит", "#38bdf8"),
    "distributor": ("Таратушы", "#f59e0b"),
    "terminal": ("Соңғы алушы", "#a78bfa"),
    "coordinator": ("Үйлестірушіге кандидат", "#fb7185"),
    "peripheral": ("Периферия", "#94a3b8"),
}


def graph_html(nodes, edges, selected=None):
    graph = Network(height="570px", width="100%", directed=True,
                    bgcolor="#0f172a", font_color="#e2e8f0", cdn_resources="in_line")
    ids = set(nodes.gid.astype(int))
    for row in nodes.to_dict("records"):
        gid = int(row["gid"])
        label, color = ROLES.get(row.get("role"), ("Рөл есептелмеген", "#64748b"))
        title = escape(f"gid: {gid} | {label} | {row.get('evidence', '')}")
        # gid may exceed JavaScript's safe integer range; preserve as text.
        display_id = str(gid) if gid == selected else "…" + str(gid)[-7:]
        graph.add_node(str(gid), label=display_id, title=title, color=color,
                       size=24 if gid == selected else 13,
                       borderWidth=4 if gid == selected else 1)
    for row in edges.itertuples(index=False):
        src, dst = int(row.src), int(row.dst)
        if src in ids and dst in ids:
            graph.add_edge(str(src), str(dst), title=f"{row.sum_kzt:,.0f} KZT", arrows="to")
    graph.set_options(json.dumps({
        "physics": {"solver": "barnesHut", "stabilization": {"iterations": 120}},
        "interaction": {"hover": True, "navigationButtons": True},
        "edges": {"color": {"color": "#64748b"}, "smooth": False},
    }))
    html = graph.generate_html()
    # PyVis adds unused Bootstrap CDN assets even with inline network assets.
    html = re.sub(r'<link\b[^>]*href="https://cdn.jsdelivr.net/[^>]*>', '', html)
    html = re.sub(r'<script\b[^>]*src="https://cdn.jsdelivr.net/[^>]*>\s*</script>', '', html)
    return html
