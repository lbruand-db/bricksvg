"""diagram_bridge.py — Extract diagrams-library graph and build LDraw scene."""

import json
import math
import runpy
import statistics
from dataclasses import dataclass

import numpy as np
import diagrams

from .parts import Piece, ldraw_rgb, register_custom_color

# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------

def extract_graph(script_path: str) -> dict:
    """Run a diagrams script, capture the dot graph, return graphviz JSON layout.

    Monkey-patches ``diagrams.Diagram`` so the context manager captures the
    dot graph without writing any output files or launching a viewer.
    """
    _captured: list = []
    _OrigDiagram = diagrams.Diagram

    class _CaptureDiagram(_OrigDiagram):
        def __exit__(self, *args):
            _captured.append(self.dot)
            # Skip super().__exit__() — no file output, no graphviz render

    setattr(diagrams, "Diagram", _CaptureDiagram)
    try:
        runpy.run_path(script_path, run_name="__main__")
    finally:
        setattr(diagrams, "Diagram", _OrigDiagram)

    if not _captured:
        raise RuntimeError(f"No Diagram context captured from {script_path!r}")

    dot = _captured[0]
    return json.loads(dot.pipe(format="json", engine="dot"))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LDraw color palette for clusters: blue, red, green, orange, light-blue, tan
_CLUSTER_PALETTE = [1, 4, 2, 25, 41, 22]

# LDraw heights in LDU (1 LDU = 0.4 mm; 1 stud = 8 mm = 20 LDU)
_PLATE_H_LDU    = 8      # height of a 1×1 plate (3024, height = 1/3 brick)
_BRICK_H_LDU    = 24     # height of a 2×2 brick (3003, height = 1 brick)
_TILE_LDU       = 20     # one stud = one 1×1 tile width
_NODE_TILE_PART      = "3068b"  # 2×2 flat tile placed on top of each node brick (no studs)
_ROUND_BRICK_PART    = "3941"   # 2×2 round brick (cylinder)
_ROUND_TILE_PART     = "14769"  # 2×2 round flat tile (cylinder cap)
_STACKED_PLATE_PART  = "3022"   # 2×2 plate — 3 stacked = 1 brick height
_STACKED_PLATE_COUNT = 5        # number of plates in a stacked-rectangle node
_PLATFORM_TILE_PART = "3024"   # 1×1 plate (with stud) for the main cluster platform
_LABEL_TILE_PART    = "3070b"  # 1×1 flat tile (no stud) — label writing surface at platform front


@dataclass
class TileExtent:
    """Axis-aligned XZ bounding box (in LDU) of a cluster platform."""
    x0: int
    x1: int
    z0: int
    z1: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider_color(image_path: str) -> int:
    """Derive a LDraw color from a diagrams provider icon path."""
    for key, color in [("k8s", 1), ("gcp", 4), ("aws", 25), ("azure", 41), ("onprem", 2)]:
        if key in image_path:
            return color
    return 7  # gray default


def _median_nn_dist(xs: list[float], ys: list[float]) -> float:
    """Median nearest-neighbor distance among a set of 2D points."""
    if len(xs) < 2:
        return 1.0
    dists = [
        min(math.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(len(xs)) if j != i)
        for i in range(len(xs))
    ]
    return statistics.median(dists)


# ---------------------------------------------------------------------------
# Scene-building steps
# ---------------------------------------------------------------------------

def _parse_objects(graph: dict) -> tuple[list[dict], list[dict]]:
    """Split graph objects into node objects and cluster subgraph objects."""
    objects = graph.get("objects", [])
    node_objs    = [o for o in objects if "pos" in o]
    cluster_objs = [o for o in objects if "nodes" in o]
    return node_objs, cluster_objs


def _compute_cluster_metadata(
    node_objs: list[dict],
    cluster_objs: list[dict],
) -> tuple[dict[int, str], dict[str, int], dict[str, int], dict[str, str | None]]:
    """Return node→cluster mapping, cluster colors, nesting depths, and parent links."""
    gvid_to_node     = {o["_gvid"]: o for o in node_objs}
    cluster_node_sets = {cl["name"]: set(cl["nodes"]) for cl in cluster_objs}

    # Each node's innermost cluster (smallest member count)
    node_cluster: dict[int, str] = {}
    for gvid in gvid_to_node:
        best_name: str | None = None
        best_size = math.inf
        for cl in cluster_objs:
            if gvid in cl["nodes"] and len(cl["nodes"]) < best_size:
                best_name = cl["name"]
                best_size = len(cl["nodes"])
        if best_name is not None:
            node_cluster[gvid] = best_name

    # Assign colors in first-appearance order
    unique_clusters = list(dict.fromkeys(
        node_cluster[gvid]
        for gvid in gvid_to_node
        if gvid in node_cluster
    ))
    cluster_color: dict[str, int] = {
        name: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)]
        for i, name in enumerate(unique_clusters)
    }

    # Nesting depth: number of clusters whose node set strictly contains this one
    cluster_depth: dict[str, int] = {
        cl["name"]: sum(
            1 for other in cluster_objs
            if cluster_node_sets[other["name"]] > cluster_node_sets[cl["name"]]
        )
        for cl in cluster_objs
    }

    # Direct parent: smallest strictly-containing cluster
    cluster_parent: dict[str, str | None] = {}
    for cl in cluster_objs:
        containing = [
            other["name"] for other in cluster_objs
            if cluster_node_sets[other["name"]] > cluster_node_sets[cl["name"]]
        ]
        cluster_parent[cl["name"]] = (
            min(containing, key=lambda n: len(cluster_node_sets[n]))
            if containing else None
        )

    return node_cluster, cluster_color, cluster_depth, cluster_parent


def _layout_positions(node_objs: list[dict]) -> dict[int, tuple[int, int]]:
    """Normalize graphviz node positions to a snapped LDraw (X, Z) grid.

    Scales so the median nearest-neighbour distance is ≥ 80 LDU (4 studs),
    then snaps to ``_TILE_LDU`` increments.
    """
    gvx = [float(o["pos"].split(",")[0]) for o in node_objs]
    gvy = [float(o["pos"].split(",")[1]) for o in node_objs]
    nn_dist = _median_nn_dist(gvx, gvy)
    scale = 80.0 / nn_dist if nn_dist > 1e-6 else 1.0

    gvid_to_ld: dict[int, tuple[int, int]] = {}
    for obj in node_objs:
        gx = float(obj["pos"].split(",")[0])
        gy = float(obj["pos"].split(",")[1])
        ldx = round(gx * scale / _TILE_LDU) * _TILE_LDU
        ldz = round(-gy * scale / _TILE_LDU) * _TILE_LDU  # negate Y: gv↑ → LDraw -Z
        gvid_to_ld[obj["_gvid"]] = (ldx, ldz)
    return gvid_to_ld


def _compute_platform_extents(
    cluster_objs: list[dict],
    gvid_to_ld: dict[int, tuple[int, int]],
    cluster_depth: dict[str, int],
    cluster_parent: dict[str, str | None],
) -> dict[str, TileExtent]:
    """Return sibling-bounded, parent-clipped XZ extents per cluster.

    Processes outermost clusters first so parent extents exist when children
    are clipped.  Additionally caps each cluster's extent at the midpoint to
    the nearest sibling node so sibling platforms never overlap.
    """
    max_depth = max(cluster_depth.values(), default=0)

    # Collect LDraw X/Z positions of member nodes per cluster
    cluster_node_xs: dict[str, list[int]] = {}
    cluster_node_zs: dict[str, list[int]] = {}
    for cl in cluster_objs:
        members = [g for g in cl["nodes"] if g in gvid_to_ld]
        if members:
            cluster_node_xs[cl["name"]] = [gvid_to_ld[g][0] for g in members]
            cluster_node_zs[cl["name"]] = [gvid_to_ld[g][1] for g in members]

    cluster_tile_extent: dict[str, TileExtent] = {}
    for d in range(max_depth + 1):
        depth_clusters = [cl for cl in cluster_objs if cluster_depth[cl["name"]] == d]
        for cl in (c for c in depth_clusters if c["name"] in cluster_node_xs):
            name = cl["name"]
            xs  = cluster_node_xs[name]
            zs  = cluster_node_zs[name]
            pad = _TILE_LDU * (max_depth - d + 2)
            x0 = math.floor((min(xs) - pad) / _TILE_LDU) * _TILE_LDU
            x1 = math.ceil ((max(xs) + pad) / _TILE_LDU) * _TILE_LDU
            z0 = math.floor((min(zs) - pad) / _TILE_LDU) * _TILE_LDU
            z1 = math.ceil ((max(zs) + pad) / _TILE_LDU) * _TILE_LDU

            # Cap at midpoint to each sibling (same parent, same depth)
            siblings = [
                sib for sib in depth_clusters
                if sib["name"] != name
                and sib["name"] in cluster_node_xs
                and cluster_parent[sib["name"]] == cluster_parent[name]
            ]
            for sib in siblings:
                sib_xs = cluster_node_xs[sib["name"]]
                sib_zs = cluster_node_zs[sib["name"]]
                if min(sib_xs) > max(xs):
                    mid = math.floor((max(xs) + min(sib_xs)) / 2 / _TILE_LDU) * _TILE_LDU
                    x1 = min(x1, mid)
                elif max(sib_xs) < min(xs):
                    mid = math.ceil((max(sib_xs) + min(xs)) / 2 / _TILE_LDU) * _TILE_LDU
                    x0 = max(x0, mid)
                if min(sib_zs) > max(zs):
                    mid = math.floor((max(zs) + min(sib_zs)) / 2 / _TILE_LDU) * _TILE_LDU
                    z1 = min(z1, mid)
                elif max(sib_zs) < min(zs):
                    mid = math.ceil((max(sib_zs) + min(zs)) / 2 / _TILE_LDU) * _TILE_LDU
                    z0 = max(z0, mid)

            # Clip to parent's already-computed extent
            parent = cluster_parent[name]
            if parent and parent in cluster_tile_extent:
                pe = cluster_tile_extent[parent]
                x0, x1 = max(x0, pe.x0), min(x1, pe.x1)
                z0, z1 = max(z0, pe.z0), min(z1, pe.z1)

            cluster_tile_extent[name] = TileExtent(x0, x1, z0, z1)

    return cluster_tile_extent


def _first_overlapping_extent(
    ldx: int,
    ldz: int,
    extents: list[TileExtent],
) -> TileExtent | None:
    """Return the first platform extent whose footprint overlaps the brick, or None."""
    return next(
        (
            ext for ext in extents
            if ldx - _TILE_LDU < ext.x1 and ldx + _TILE_LDU > ext.x0
            and ldz - _TILE_LDU < ext.z1 and ldz + _TILE_LDU > ext.z0
        ),
        None,
    )


def _displace_lone_nodes(
    gvid_to_ld: dict[int, tuple[int, int]],
    node_cluster: dict[int, str],
    cluster_tile_extent: dict[str, TileExtent],
) -> None:
    """Mutate ``gvid_to_ld`` to push lone nodes outside any overlapping cluster platform."""
    extents = list(cluster_tile_extent.values())
    for gvid in (g for g in list(gvid_to_ld) if g not in node_cluster):
        ldx, ldz = gvid_to_ld[gvid]
        ext = _first_overlapping_extent(ldx, ldz, extents)
        if ext is not None:
            dx_left  = ldx - ext.x0
            dx_right = ext.x1 - ldx
            dz_near  = ldz - ext.z0
            dz_far   = ext.z1 - ldz
            clearance = 2 * _TILE_LDU  # brick half + 1-stud gap
            if min(dx_left, dx_right) <= min(dz_near, dz_far):
                ldx = round(
                    (ext.x0 - clearance if dx_left <= dx_right else ext.x1 + clearance)
                    / _TILE_LDU
                ) * _TILE_LDU
            else:
                ldz = round(
                    (ext.z0 - clearance if dz_near <= dz_far else ext.z1 + clearance)
                    / _TILE_LDU
                ) * _TILE_LDU
            gvid_to_ld[gvid] = (ldx, ldz)


def _build_node_pieces(
    node_objs: list[dict],
    gvid_to_ld: dict[int, tuple[int, int]],
    node_cluster: dict[int, str],
    cluster_color: dict[str, int],
    cluster_depth: dict[str, int],
) -> tuple[list[Piece], list[dict], dict[str, list[Piece]], list[Piece], dict[int, float]]:
    """Build 2×2 brick pieces and metadata for each node.

    Returns
    -------
    pieces
        One :class:`Piece` per node.
    node_data
        One metadata dict per node (pos, icon_path, label, half_w).
    cluster_node_bricks
        Maps cluster name → list of its node brick pieces.
    lone_node_bricks
        Node bricks that belong to no cluster.
    """
    pieces: list[Piece] = []
    node_data: list[dict] = []
    cluster_node_bricks: dict[str, list[Piece]] = {}
    lone_node_bricks: list[Piece] = []
    gvid_to_top_y: dict[int, float] = {}

    for obj in node_objs:
        gvid = obj["_gvid"]
        ldx, ldz = gvid_to_ld[gvid]
        in_cluster = gvid in node_cluster

        fillcolor = obj.get("fillcolor")
        if fillcolor is not None and fillcolor.startswith("#"):
            h = fillcolor.lstrip("#")
            color = register_custom_color(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        elif in_cluster:
            color = cluster_color[node_cluster[gvid]]
        else:
            color = _provider_color(obj.get("image", ""))

        # Choose brick parts based on Mermaid shape
        tooltip = obj.get("tooltip", "")
        mermaid_shape = tooltip.split(":", 1)[1] if tooltip.startswith("shape:") else "rect"

        # Stacked nodes are taller than a single brick
        stacked_h = _STACKED_PLATE_COUNT * _PLATE_H_LDU
        body_h = stacked_h if mermaid_shape == "stacked" else _BRICK_H_LDU

        if in_cluster:
            depth = cluster_depth[node_cluster[gvid]]
            node_y = float(-(depth + 1) * _PLATE_H_LDU - body_h)
        else:
            node_y = float(-body_h)
        pos = np.array([float(ldx), node_y, float(ldz)])

        # Tile sits on top of the brick (more negative Y = higher in LDraw)
        tile_y   = node_y - _PLATE_H_LDU
        tile_pos = np.array([float(ldx), tile_y, float(ldz)])

        # Derive alternate color by lightening the fill ~30% toward white
        fr, fg, fb = ldraw_rgb(color)
        alt_color = register_custom_color(
            fr + (255 - fr) * 3 // 10,
            fg + (255 - fg) * 3 // 10,
            fb + (255 - fb) * 3 // 10,
        )

        node_pieces: list[Piece] = []
        if mermaid_shape == "stacked":
            # Stacked plates with alternating fill/stroke colors
            for i in range(_STACKED_PLATE_COUNT):
                plate_y = node_y + i * _PLATE_H_LDU
                plate_pos = np.array([float(ldx), plate_y, float(ldz)])
                plate_color = color if i % 2 == 0 else alt_color
                node_pieces.append(Piece(part=_STACKED_PLATE_PART, color=plate_color,
                                         pos=plate_pos, rot=np.eye(3)))
            tile = Piece(part=_NODE_TILE_PART, color=color, pos=tile_pos, rot=np.eye(3))
            node_pieces.append(tile)
        elif mermaid_shape in ("circle", "rounded", "stadium", "cylinder"):
            node_pieces.append(Piece(part=_ROUND_BRICK_PART, color=color,
                                      pos=pos, rot=np.eye(3)))
            tile = Piece(part=_ROUND_TILE_PART, color=color, pos=tile_pos, rot=np.eye(3))
            node_pieces.append(tile)
        else:
            node_pieces.append(Piece(part="3003", color=color, pos=pos, rot=np.eye(3)))
            tile = Piece(part=_NODE_TILE_PART, color=color, pos=tile_pos, rot=np.eye(3))
            node_pieces.append(tile)

        pieces.extend(node_pieces)
        node_data.append({
            "pos":       tile_pos,   # icons project onto the tile's flat top face
            "icon_path": obj.get("image") or None,
            "label":     obj.get("label", ""),
            "color":     color,      # LDraw color ID for label contrast
            "half_w":    _TILE_LDU,                       # 20 LDU = half the 2×2 footprint
            "half_h":    (_PLATE_H_LDU + _BRICK_H_LDU) // 2,  # 16 LDU = half the total height
        })
        gvid_to_top_y[gvid] = tile_y  # top face of the tile — arc arrow origin/destination

        if in_cluster:
            cluster_node_bricks.setdefault(node_cluster[gvid], []).extend(node_pieces)
        else:
            lone_node_bricks.extend(node_pieces)

    return pieces, node_data, cluster_node_bricks, lone_node_bricks, gvid_to_top_y


def _build_platform_pieces(
    cluster_objs: list[dict],
    cluster_tile_extent: dict[str, TileExtent],
    cluster_depth: dict[str, int],
    cluster_color: dict[str, int],
    exclude: dict[str, set[tuple[int, int]]] | None = None,
) -> tuple[list[Piece], dict[str, list[Piece]]]:
    """Build 1×1 plate tiles covering each cluster's XZ extent.

    Parameters
    ----------
    exclude:
        Per-cluster set of ``(x_centre, z_centre)`` integer positions to skip
        (used to leave gaps where studless label tiles will be placed instead).

    Returns
    -------
    pieces
        All platform tile pieces.
    cluster_platform_tiles
        Maps cluster name → list of its tile pieces.
    """
    pieces: list[Piece] = []
    cluster_platform_tiles: dict[str, list[Piece]] = {}

    for cl in (c for c in cluster_objs if c["name"] in cluster_tile_extent):
        name = cl["name"]
        ext  = cluster_tile_extent[name]
        plate_y = float(-(cluster_depth[name] + 1) * _PLATE_H_LDU)
        color   = cluster_color.get(name, 15)
        skip    = (exclude or {}).get(name, set())

        for x in range(ext.x0, ext.x1, _TILE_LDU):
            for z in range(ext.z0, ext.z1, _TILE_LDU):
                xc, zc = x + _TILE_LDU // 2, z + _TILE_LDU // 2
                if (xc, zc) in skip:
                    continue
                pos = np.array([float(xc), plate_y, float(zc)])
                piece = Piece(part=_PLATFORM_TILE_PART, color=color, pos=pos, rot=np.eye(3))
                pieces.append(piece)
                cluster_platform_tiles.setdefault(name, []).append(piece)

    return pieces, cluster_platform_tiles


def _build_cluster_label_tiles(
    cluster_objs: list[dict],
    cluster_tile_extent: dict[str, TileExtent],
    cluster_depth: dict[str, int],
    cluster_color: dict[str, int],
) -> tuple[list[Piece], dict[str, list[Piece]]]:
    """Place up to three 1×1 flat (no-stud) tiles at the front-centre of each cluster.

    The tiles are at the same height as the platform plates and *replace* the
    3024 studs at those positions, providing a flush writing surface.

    Returns
    -------
    pieces
        All label tile pieces.
    cluster_label_tiles
        Maps cluster name → list of its label tile pieces.
    """
    pieces: list[Piece] = []
    cluster_label_tiles: dict[str, list[Piece]] = {}

    for cl in (c for c in cluster_objs if c["name"] in cluster_tile_extent):
        name  = cl["name"]
        ext   = cluster_tile_extent[name]
        depth = cluster_depth[name]
        color = cluster_color.get(name, 15)

        # Same level as platform tiles — replaces the 3024 plate at those positions
        label_tile_y = float(-(depth + 1) * _PLATE_H_LDU)

        # X-centres of every tile column in the platform
        all_xs = [x + _TILE_LDU // 2 for x in range(ext.x0, ext.x1, _TILE_LDU)]
        if not all_xs:
            continue

        # Front row: z-centre of the last row of platform tiles
        front_z = float(ext.z1 - _TILE_LDU // 2)

        # Pick up to 3 columns closest to the platform's X-centre
        cx = (ext.x0 + ext.x1) / 2.0
        chosen_xs = sorted(all_xs, key=lambda x: abs(x - cx))[:3]

        tile_pieces: list[Piece] = []
        for tx in chosen_xs:
            pos = np.array([float(tx), label_tile_y, front_z])
            tile_pieces.append(Piece(part=_LABEL_TILE_PART, color=color, pos=pos, rot=np.eye(3)))

        pieces.extend(tile_pieces)
        cluster_label_tiles[name] = tile_pieces

    return pieces, cluster_label_tiles


def _assemble_piece_groups(
    cluster_objs: list[dict],
    cluster_depth: dict[str, int],
    cluster_node_bricks: dict[str, list[Piece]],
    cluster_platform_tiles: dict[str, list[Piece]],
    lone_node_bricks: list[Piece],
) -> list[tuple[str, list[Piece]]]:
    """Order pieces into named groups for SVG ``<g>`` elements.

    Clusters appear outermost-first (depth 0 → N) so inner tiles render on
    top.  Within each group: platform tiles first, then the node brick.
    Lone nodes are last under the name ``"lone"``.
    """
    max_depth = max(cluster_depth.values(), default=0)
    piece_groups: list[tuple[str, list[Piece]]] = []
    for d in range(max_depth + 1):
        for cl in (c for c in cluster_objs if cluster_depth[c["name"]] == d):
            name = cl["name"]
            group = cluster_platform_tiles.get(name, []) + cluster_node_bricks.get(name, [])
            if group:
                piece_groups.append((name, group))
    if lone_node_bricks:
        piece_groups.append(("lone", lone_node_bricks))
    return piece_groups


def _build_cluster_label_data(
    cluster_objs: list[dict],
    cluster_tile_extent: dict[str, TileExtent],
    cluster_depth: dict[str, int],
) -> list[dict]:
    """Return label metadata for each cluster platform.

    Each entry has ``pos`` (LDraw XYZ at the top-face centre of the three
    studless label tiles) and ``label`` (display text).
    """
    result = []
    for cl in (c for c in cluster_objs if c["name"] in cluster_tile_extent):
        name  = cl["name"]
        ext   = cluster_tile_extent[name]
        depth = cluster_depth[name]
        cx    = float(ext.x0 + ext.x1) / 2.0
        # Top face of the label tiles: one plate above the platform surface
        tile_top_y = float(-(depth + 1) * _PLATE_H_LDU - _PLATE_H_LDU)
        # Z-centre of the front row (where the 3070b tiles are placed)
        front_z = float(ext.z1 - _TILE_LDU // 2)
        result.append({
            "pos":   np.array([cx, tile_top_y, front_z]),
            "label": cl.get("label") or name,
        })
    return result


def _tile_edge_offset(dx: float, dz: float) -> tuple[float, float]:
    """XZ offset from the centre of a tile to its edge in direction (dx, dz).

    Uses the square-edge formula: distance = half_w / max(|ux|, |uz|).
    Returns (0, 0) if the direction vector is zero.
    """
    dist = math.hypot(dx, dz)
    if dist < 1e-9:
        return 0.0, 0.0
    ux, uz = dx / dist, dz / dist
    d = _TILE_LDU / max(abs(ux), abs(uz))
    return ux * d, uz * d


def _build_edge_positions(
    edges: list[dict],
    gvid_to_ld: dict[int, tuple[int, int]],
    gvid_to_top_y: dict[int, float],
) -> list[tuple]:
    """Map graphviz edges to LDraw ``(from_pos, to_pos)`` pairs.

    Each endpoint is on the edge of the node tile's top face, offset from the
    centre toward the other node, so arc arrows start and end at the tile rim.
    """
    edge_positions: list[tuple] = []
    for e in edges:
        tail_gvid = e.get("tail")
        head_gvid = e.get("head")
        if tail_gvid in gvid_to_ld and head_gvid in gvid_to_ld:
            tlx, tlz = gvid_to_ld[tail_gvid]
            hlx, hlz = gvid_to_ld[head_gvid]
            ox, oz = _tile_edge_offset(float(hlx) - float(tlx), float(hlz) - float(tlz))
            edge_positions.append((
                np.array([float(tlx) + ox, gvid_to_top_y[tail_gvid], float(tlz) + oz]),
                np.array([float(hlx) - ox, gvid_to_top_y[head_gvid], float(hlz) - oz]),
            ))
    return edge_positions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ldr_scene(
    graph: dict,
) -> tuple[list[Piece], list[tuple], list[dict], list[tuple[str, list[Piece]]], list[dict]]:
    """Parse graphviz JSON layout and build LDraw scene data.

    Parameters
    ----------
    graph:
        Dict returned by ``extract_graph()``.

    Returns
    -------
    pieces
        All :class:`~ldr2svg.parts.Piece` objects (node bricks + platform
        tiles) ready for :func:`~ldr2svg.ldr2png_svg.build_pngs_grayscale`.
    edge_positions
        ``(from_pos, to_pos)`` pairs in LDraw coordinates — used to draw
        SVG floor arrows.
    node_data
        One dict per node with keys ``pos``, ``icon_path``, ``label``, ``half_w``.
    piece_groups
        Ordered ``(group_name, pieces)`` list for SVG ``<g>`` wrapping.
    cluster_data
        One dict per cluster with keys ``pos`` (front-centre of platform
        surface) and ``label`` (display text).
    """
    node_objs, cluster_objs = _parse_objects(graph)

    node_cluster, cluster_color, cluster_depth, cluster_parent = (
        _compute_cluster_metadata(node_objs, cluster_objs)
    )

    gvid_to_ld = _layout_positions(node_objs)

    cluster_tile_extent = _compute_platform_extents(
        cluster_objs, gvid_to_ld, cluster_depth, cluster_parent
    )

    _displace_lone_nodes(gvid_to_ld, node_cluster, cluster_tile_extent)

    node_pieces, node_data, cluster_node_bricks, lone_node_bricks, gvid_to_top_y = _build_node_pieces(
        node_objs, gvid_to_ld, node_cluster, cluster_color, cluster_depth
    )

    # Build label tiles first so their positions can be excluded from the 3024 platform
    label_tile_pieces, cluster_label_tiles = _build_cluster_label_tiles(
        cluster_objs, cluster_tile_extent, cluster_depth, cluster_color
    )
    label_exclude: dict[str, set[tuple[int, int]]] = {
        name: {(int(p.pos[0]), int(p.pos[2])) for p in tiles}
        for name, tiles in cluster_label_tiles.items()
    }

    platform_pieces, cluster_platform_tiles = _build_platform_pieces(
        cluster_objs, cluster_tile_extent, cluster_depth, cluster_color,
        exclude=label_exclude,
    )
    # Merge label tiles into platform groups so they render in the platform <g>
    for name, tiles in cluster_label_tiles.items():
        cluster_platform_tiles.setdefault(name, []).extend(tiles)

    piece_groups = _assemble_piece_groups(
        cluster_objs, cluster_depth, cluster_node_bricks, cluster_platform_tiles, lone_node_bricks
    )

    edge_positions = _build_edge_positions(graph.get("edges", []), gvid_to_ld, gvid_to_top_y)

    cluster_data = _build_cluster_label_data(cluster_objs, cluster_tile_extent, cluster_depth)

    return node_pieces + platform_pieces + label_tile_pieces, edge_positions, node_data, piece_groups, cluster_data
