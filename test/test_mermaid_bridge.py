"""Unit tests for ldr2svg.mermaid_bridge."""

from pathlib import Path


from ldr2svg.mermaid_bridge import (
    _parse_mermaid,
    _parse_node_token,
    _parse_edge,
    _build_dot,
    extract_graph,
)

EXAMPLE_MMD = Path(__file__).parent / "example_diagram.mmd"

_SIMPLE = """
graph LR
    A["Node A"]
    B["Node B"]
    A --> B
"""

_WITH_CLUSTER = """
graph LR
    top["Top"]
    subgraph Cluster1 [My Cluster]
        inner["Inner"]
    end
    top --> inner
"""

_NESTED = """
graph LR
    subgraph Outer
        subgraph Inner
            leaf["Leaf"]
        end
    end
"""


# ---------------------------------------------------------------------------
# _parse_node_token
# ---------------------------------------------------------------------------

class TestParseNodeToken:
    def test_rect(self):
        assert _parse_node_token('A["Label"]') == ("A", "Label")

    def test_rect_no_quotes(self):
        assert _parse_node_token("A[label]") == ("A", "label")

    def test_rounded(self):
        node_id, label = _parse_node_token("A(rounded)")
        assert node_id == "A"
        assert label == "rounded"

    def test_diamond(self):
        node_id, label = _parse_node_token("A{diamond}")
        assert node_id == "A"
        assert label == "diamond"

    def test_bare_id(self):
        assert _parse_node_token("myNode") == ("myNode", "myNode")

    def test_quoted_label_strips_quotes(self):
        node_id, label = _parse_node_token('A["hello world"]')
        assert label == "hello world"


# ---------------------------------------------------------------------------
# _parse_edge
# ---------------------------------------------------------------------------

class TestParseEdge:
    def test_plain_arrow(self):
        assert _parse_edge("A --> B") == ("A", "B")

    def test_pipe_label(self):
        assert _parse_edge("A -->|edge label| B") == ("A", "B")

    def test_text_label(self):
        assert _parse_edge("A -- text --> B") == ("A", "B")

    def test_not_an_edge(self):
        assert _parse_edge("subgraph Foo") is None

    def test_bare_node_declaration(self):
        assert _parse_edge('A["label"]') is None

    def test_inline_node_def(self):
        # A[label] --> B[label] — IDs should be extracted correctly
        result = _parse_edge('A["Label A"] --> B["Label B"]')
        assert result == ("A", "B")


# ---------------------------------------------------------------------------
# _parse_mermaid
# ---------------------------------------------------------------------------

class TestParseMermaid:
    def test_simple_nodes(self):
        nodes, edges, clusters, top_nodes, _ = _parse_mermaid(_SIMPLE)
        assert "A" in nodes
        assert "B" in nodes
        assert nodes["A"] == "Node A"
        assert nodes["B"] == "Node B"

    def test_simple_edges(self):
        nodes, edges, clusters, top_nodes, _ = _parse_mermaid(_SIMPLE)
        assert ("A", "B") in edges

    def test_no_clusters(self):
        _, _, clusters, _, _ = _parse_mermaid(_SIMPLE)
        assert clusters == []

    def test_top_nodes_not_in_cluster(self):
        _, _, _, top_nodes, _ = _parse_mermaid(_SIMPLE)
        assert "A" in top_nodes
        assert "B" in top_nodes

    def test_cluster_extracted(self):
        _, _, clusters, _, _ = _parse_mermaid(_WITH_CLUSTER)
        assert len(clusters) == 1
        assert clusters[0].id == "Cluster1"
        assert clusters[0].label == "My Cluster"

    def test_cluster_member(self):
        _, _, clusters, _, _ = _parse_mermaid(_WITH_CLUSTER)
        assert "inner" in clusters[0].nodes

    def test_top_node_not_in_cluster(self):
        _, _, clusters, top_nodes, _ = _parse_mermaid(_WITH_CLUSTER)
        assert "top" in top_nodes
        assert "top" not in clusters[0].nodes

    def test_nested_clusters(self):
        _, _, clusters, _, _ = _parse_mermaid(_NESTED)
        assert len(clusters) == 1
        outer = clusters[0]
        assert outer.id == "Outer"
        assert len(outer.children) == 1
        inner = outer.children[0]
        assert inner.id == "Inner"
        assert "leaf" in inner.nodes

    def test_example_mmd_nodes(self):
        text = EXAMPLE_MMD.read_text()
        nodes, edges, clusters, top_nodes, _ = _parse_mermaid(text)
        assert "gcp_lb" in top_nodes
        assert nodes["gcp_lb"] == "GCP LB"

    def test_example_mmd_edges(self):
        text = EXAMPLE_MMD.read_text()
        _, edges, _, _, _ = _parse_mermaid(text)
        assert ("gcp_lb", "nginx") in edges
        assert ("nginx", "myapp_ing") in edges
        assert ("myapp_ing", "myapp_pods") in edges
        assert ("myapp_pods", "myapp_db") in edges

    def test_example_mmd_clusters(self):
        text = EXAMPLE_MMD.read_text()
        _, _, clusters, _, _ = _parse_mermaid(text)
        cluster_ids = {c.id for c in clusters}
        assert "Kubernetes" in cluster_ids

    def test_example_mmd_nested_structure(self):
        text = EXAMPLE_MMD.read_text()
        _, _, clusters, _, _ = _parse_mermaid(text)
        k8s = next(c for c in clusters if c.id == "Kubernetes")
        child_ids = {ch.id for ch in k8s.children}
        assert {"Nginx", "MyApp", "MySQL"} == child_ids


# ---------------------------------------------------------------------------
# _build_dot
# ---------------------------------------------------------------------------

class TestBuildDot:
    def _dot_for(self, text):
        nodes, edges, clusters, top_nodes, node_colors = _parse_mermaid(text)
        return _build_dot(nodes, edges, clusters, top_nodes, node_colors)

    def test_starts_with_digraph(self):
        dot = self._dot_for(_SIMPLE)
        assert dot.startswith("digraph {")

    def test_contains_top_level_node(self):
        dot = self._dot_for(_SIMPLE)
        assert '"A"' in dot
        assert "Node A" in dot

    def test_contains_edge(self):
        dot = self._dot_for(_SIMPLE)
        assert '"A" -> "B"' in dot

    def test_contains_subgraph(self):
        dot = self._dot_for(_WITH_CLUSTER)
        assert "subgraph cluster_Cluster1" in dot

    def test_subgraph_has_label(self):
        dot = self._dot_for(_WITH_CLUSTER)
        assert "My Cluster" in dot

    def test_nested_subgraphs(self):
        dot = self._dot_for(_NESTED)
        assert "cluster_Outer" in dot
        assert "cluster_Inner" in dot

    def test_inner_appears_inside_outer(self):
        dot = self._dot_for(_NESTED)
        outer_pos = dot.index("cluster_Outer")
        inner_pos = dot.index("cluster_Inner")
        assert inner_pos > outer_pos


# ---------------------------------------------------------------------------
# extract_graph (integration — requires graphviz binary)
# ---------------------------------------------------------------------------

class TestExtractGraph:
    def test_returns_dict(self):
        result = extract_graph(str(EXAMPLE_MMD))
        assert isinstance(result, dict)

    def test_has_objects(self):
        result = extract_graph(str(EXAMPLE_MMD))
        assert "objects" in result

    def test_has_edges(self):
        result = extract_graph(str(EXAMPLE_MMD))
        assert "edges" in result

    def test_node_objects_have_pos(self):
        result = extract_graph(str(EXAMPLE_MMD))
        node_objs = [o for o in result.get("objects", []) if "pos" in o]
        assert len(node_objs) == 5   # gcp_lb, nginx, myapp_ing, myapp_pods, myapp_db

    def test_cluster_objects_have_nodes(self):
        result = extract_graph(str(EXAMPLE_MMD))
        cluster_objs = [o for o in result.get("objects", []) if "nodes" in o]
        assert len(cluster_objs) >= 5  # Kubernetes + Nginx + MyApp + Pods + MySQL

    def test_edge_count(self):
        result = extract_graph(str(EXAMPLE_MMD))
        assert len(result.get("edges", [])) == 4
