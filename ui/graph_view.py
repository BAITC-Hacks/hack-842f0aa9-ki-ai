"""Render an offline, directed transaction graph."""
from html import escape
import json
import re

from pyvis.network import Network

ROLES = {
    "consolidator": ("Жинаушы", "#16853b"),
    "transit": ("Транзит", "#299da0"),
    "distributor": ("Таратушы", "#d69c25"),
    "terminal": ("Соңғы алушы", "#8972b6"),
    "coordinator": ("Үйлестірушіге кандидат", "#d37069"),
    "peripheral": ("Периферия", "#718b9c"),
}


def graph_html(nodes, edges, selected=None):
    graph = Network(height="570px", width="100%", directed=True,
                    bgcolor="#fbfdfb", font_color="#45614e", cdn_resources="in_line")
    ids = set(nodes.gid.astype(int))
    for row in nodes.to_dict("records"):
        gid = int(row["gid"])
        label, color = ROLES.get(row.get("role"), ("Рөл есептелмеген", "#a7b8ad"))
        title = escape(f"gid: {gid} | {label} | {row.get('evidence', '')}")
        # gid may exceed JavaScript's safe integer range; preserve as text.
        display_id = str(gid) if gid == selected else "…" + str(gid)[-7:]
        node_color = {"background": color, "border": "#16853b" if gid == selected else color,
                      "highlight": {"background": "#d9edcf", "border": "#16853b"},
                      "hover": {"background": "#d9edcf", "border": "#16853b"}}
        graph.add_node(str(gid), label=display_id, title=title, color=node_color,
                       size=24 if gid == selected else 13,
                       borderWidth=4 if gid == selected else 1)
    for row in edges.itertuples(index=False):
        src, dst = int(row.src), int(row.dst)
        if src in ids and dst in ids:
            graph.add_edge(str(src), str(dst), title=f"{row.sum_kzt:,.0f} KZT", arrows="to")
    graph.set_options(json.dumps({
        "physics": {"solver": "barnesHut", "stabilization": {"iterations": 120}},
        "interaction": {"hover": True, "navigationButtons": True},
        "layout": {"randomSeed": 42},
        "nodes": {"shape": "dot", "font": {"size": 11, "face": "Arial"}},
        "edges": {"color": {"color": "#becfc2", "highlight": "#16853b", "hover": "#16853b"}, "smooth": False},
    }))
    html = graph.generate_html()
    # PyVis adds unused Bootstrap CDN assets even with inline network assets.
    html = re.sub(r'<link\b[^>]*href="https://cdn.jsdelivr.net/[^>]*>', '', html)
    html = re.sub(r'<script\b[^>]*src="https://cdn.jsdelivr.net/[^>]*>\s*</script>', '', html)
    canvas_style = '<style>body{margin:0;background:#fbfdfb;font-family:Arial,sans-serif}#mynetwork{border:0!important;background-image:radial-gradient(#dce7df 1px,transparent 1px)!important;background-size:20px 20px!important}div.vis-tooltip{background:white!important;border:1px solid #c8dbce!important;border-radius:10px!important;color:#23452d!important;padding:12px!important;max-width:320px;white-space:normal!important}</style>'
    return html.replace("</head>", canvas_style + "</head>")
