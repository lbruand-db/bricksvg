"""mermaid_bridge.py — Parse a Mermaid flowchart and produce graphviz JSON layout.

Supports the common flowchart / graph syntax::

    graph LR
        A["Label A"]
        subgraph ClusterName [Display Title]
            B["Label B"]
        end
        A --> B
        A -->|edge label| B

The parsed graph is converted to a DOT string and laid out by the ``dot``
engine, producing the same JSON format that ``diagram_bridge.build_ldr_scene``
expects.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import graphviz


# ---------------------------------------------------------------------------
# Style directive parser
# ---------------------------------------------------------------------------

_RE_STYLE = re.compile(
    r"^style\s+([\w\-]+)\s+(.+)$", re.IGNORECASE
)


def _parse_style(line: str) -> tuple[str, str] | None:
    """Parse ``style nodeId fill:#abc,stroke:#def,...`` → ``(nodeId, '#rrggbb')`` or None."""
    m = _RE_STYLE.match(line)
    if not m:
        return None
    node_id = m.group(1)
    props = m.group(2)
    for prop in props.split(","):
        key_val = prop.strip().split(":", 1)
        if len(key_val) == 2 and key_val[0].strip() == "fill":
            return node_id, key_val[1].strip()
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _Cluster:
    id: str
    label: str
    nodes: list[str]     = field(default_factory=list)   # direct member IDs
    children: list["_Cluster"] = field(default_factory=list)  # nested clusters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s


# Node-shape patterns: ([\w\-]+) is the ID, the second group is the label.
_SHAPE_RE = [
    re.compile(r'^([\w\-]+)\s*\(\[(.+?)\]\)$'),    # A([label])  stadium
    re.compile(r'^([\w\-]+)\s*\(\((.+?)\)\)$'),    # A((label))  circle
    re.compile(r'^([\w\-]+)\s*\[(.+?)\]$'),         # A[label]    rect
    re.compile(r'^([\w\-]+)\s*\((.+?)\)$'),         # A(label)    rounded
    re.compile(r'^([\w\-]+)\s*\{(.+?)\}$'),         # A{label}    diamond
]
_RE_BARE_ID = re.compile(r'^([\w\-]+)$')


def _parse_node_token(token: str) -> tuple[str, str]:
    """Parse ``'id[label]'`` etc. and return ``(id, label)``.

    Falls back to ``(token, token)`` for bare identifiers.
    """
    token = token.strip()
    for pat in _SHAPE_RE:
        m = pat.match(token)
        if m:
            return m.group(1), _strip_quotes(m.group(2))
    m = _RE_BARE_ID.match(token)
    if m:
        return m.group(1), m.group(1)
    return token, token


# Edge patterns — tried in order; groups (tail_idx, head_idx) vary per pattern.
_EDGE_PATTERNS: list[tuple[re.Pattern, int, int]] = [
    # A -->|label| B
    (re.compile(r'^(.+?)\s*-->\s*\|[^|]*\|\s*(.+)$'), 1, 2),
    # A -- text --> B
    (re.compile(r'^(.+?)\s*--[^>]+-->\s*(.+)$'),       1, 2),
    # A --> B
    (re.compile(r'^(.+?)\s*-->\s*(.+)$'),               1, 2),
]


def _parse_edge(line: str) -> tuple[str, str] | None:
    """Return ``(tail_id, head_id)`` or ``None`` if the line is not an edge."""
    for pat, ti, hi in _EDGE_PATTERNS:
        m = pat.match(line)
        if m:
            tail_id, _ = _parse_node_token(m.group(ti))
            head_id, _ = _parse_node_token(m.group(hi))
            return tail_id, head_id
    return None


# ---------------------------------------------------------------------------
# Mermaid parser
# ---------------------------------------------------------------------------

def _parse_mermaid(text: str) -> tuple[
    dict[str, str],             # node_id → display label
    list[tuple[str, str]],      # edges: (tail_id, head_id)
    list[_Cluster],             # top-level clusters
    list[str],                  # top-level node IDs (not in any cluster)
    dict[str, str],             # node_id → hex fill color (e.g. '#ff3620')
]:
    """Parse a Mermaid flowchart string into its structural components."""
    lines = [
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("%%")
    ]

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    cluster_stack: list[_Cluster] = []
    top_clusters: list[_Cluster] = []
    top_nodes: list[str] = []
    node_colors: dict[str, str] = {}

    for line in lines:
        # Graph / flowchart direction declaration — skip
        if re.match(r"^(?:graph|flowchart)\s", line, re.IGNORECASE):
            continue

        # Style directive — extract fill color
        style = _parse_style(line)
        if style:
            nid, hex_color = style
            node_colors[nid] = hex_color
            continue

        # Subgraph start
        m = re.match(r"^subgraph\s+(\S+?)(?:\s*\[(.+?)\])?\s*$", line)
        if m:
            sg_id    = m.group(1)
            sg_label = _strip_quotes(m.group(2)) if m.group(2) else sg_id
            cluster_stack.append(_Cluster(id=sg_id, label=sg_label))
            continue

        # Subgraph end
        if re.match(r"^end\s*$", line, re.IGNORECASE):
            if cluster_stack:
                finished = cluster_stack.pop()
                if cluster_stack:
                    cluster_stack[-1].children.append(finished)
                else:
                    top_clusters.append(finished)
            continue

        # Edge
        edge = _parse_edge(line)
        if edge:
            tail_id, head_id = edge
            edges.append((tail_id, head_id))
            for nid in (tail_id, head_id):
                if nid not in nodes:
                    nodes[nid] = nid
            continue

        # Node declaration
        node_id, label = _parse_node_token(line)
        if node_id and _RE_BARE_ID.match(node_id):
            nodes[node_id] = label
            if cluster_stack:
                if node_id not in cluster_stack[-1].nodes:
                    cluster_stack[-1].nodes.append(node_id)
            else:
                if node_id not in top_nodes:
                    top_nodes.append(node_id)

    return nodes, edges, top_clusters, top_nodes, node_colors


# ---------------------------------------------------------------------------
# DOT generation
# ---------------------------------------------------------------------------

def _dot_str(s: str) -> str:
    """Wrap *s* in double quotes for use in a DOT file."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _node_attrs(nid: str, label: str, node_colors: dict[str, str]) -> str:
    """Build DOT attribute string for a node, including fillcolor if styled."""
    attrs = [f"label={_dot_str(label)}"]
    if nid in node_colors:
        attrs.append(f'style="filled" fillcolor={_dot_str(node_colors[nid])}')
    return " ".join(attrs)


def _cluster_to_dot(
    c: _Cluster, nodes: dict[str, str], node_colors: dict[str, str], depth: int = 1,
) -> list[str]:
    pad   = "    " * depth
    lines = [f"{pad}subgraph cluster_{c.id} {{"]
    lines.append(f"{pad}    label={_dot_str(c.label)}")
    for nid in c.nodes:
        label = nodes.get(nid, nid)
        lines.append(f"{pad}    {_dot_str(nid)} [{_node_attrs(nid, label, node_colors)}]")
    for child in c.children:
        lines.extend(_cluster_to_dot(child, nodes, node_colors, depth + 1))
    lines.append(f"{pad}}}")
    return lines


def _build_dot(
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    top_clusters: list[_Cluster],
    top_nodes: list[str],
    node_colors: dict[str, str],
) -> str:
    """Return a DOT digraph string ready to be laid out by graphviz."""
    lines = ["digraph {", "    rankdir=LR"]

    # Collect node IDs declared inside clusters so we don't double-declare them
    cluster_node_ids: set[str] = set()
    def _collect_cluster_nodes(c: _Cluster) -> None:
        cluster_node_ids.update(c.nodes)
        for child in c.children:
            _collect_cluster_nodes(child)
    for c in top_clusters:
        _collect_cluster_nodes(c)

    # Emit all non-cluster nodes (top-level) with their attributes
    for nid in nodes:
        if nid not in cluster_node_ids:
            lines.append(f"    {_dot_str(nid)} [{_node_attrs(nid, nodes.get(nid, nid), node_colors)}]")

    for c in top_clusters:
        lines.extend(_cluster_to_dot(c, nodes, node_colors))

    for tail, head in edges:
        lines.append(f"    {_dot_str(tail)} -> {_dot_str(head)}")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_graph(mermaid_path: str) -> dict:
    """Read a Mermaid flowchart file and return a graphviz JSON layout dict.

    The returned dict has the same structure as the one produced by
    ``diagram_bridge.extract_graph`` and can be passed directly to
    ``diagram_bridge.build_ldr_scene``.
    """
    text = Path(mermaid_path).read_text(encoding="utf-8")
    nodes, edges, top_clusters, top_nodes, node_colors = _parse_mermaid(text)
    dot_src = _build_dot(nodes, edges, top_clusters, top_nodes, node_colors)
    src = graphviz.Source(dot_src)
    return json.loads(src.pipe(format="json", engine="dot"))
