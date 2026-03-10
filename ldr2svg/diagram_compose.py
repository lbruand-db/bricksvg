"""diagram_compose.py — SVG composition for diagram2svg output."""

import math

import numpy as np
import svgwrite
from PIL import Image

from .compose import (
    _project_pieces, _project_piece, _canvas_bounds,
    _build_defs, _build_duotone_filters,
    _img_to_data_uri, _draw_grid,
)
from .parts import Piece, ldraw_rgb
from .projection import project_ldraw, PX_PER_MM
from .grid import _grid_params, _grid_corner_sx_sy


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def _proj_canvas(
    pos_ld: np.ndarray,
    cx,
    cy,
) -> tuple[float, float]:
    """Project a LDraw position and convert to canvas (SVG) coordinates."""
    sx, sy, _ = project_ldraw(pos_ld)
    return cx(sx * PX_PER_MM), cy(sy * PX_PER_MM)


# ---------------------------------------------------------------------------
# Isometric polygon arrows
# ---------------------------------------------------------------------------

def _arrow_polygon_3d(
    from_pos: np.ndarray,
    to_pos: np.ndarray,
    half_width: float = 4.0,
    head_length: float = 18.0,
    head_half_width: float = 10.0,
    lift_factor: float = 0.4,
    n_body: int = 16,
) -> list[np.ndarray]:
    """Return 3D polygon vertices for an arrow that arches upward (LDraw −Y).

    The centre-line follows a quadratic Bézier whose control point is lifted
    straight up (negative LDraw Y) at the XZ midpoint.  The ribbon width is
    always horizontal (XZ-perpendicular to the arrow direction) so the band
    looks like a curved strip arching over the scene.  The arrowhead is a
    flat triangle at the destination Y level.
    """
    ay = float(from_pos[1])
    by = float(to_pos[1])
    ax, az = float(from_pos[0]), float(from_pos[2])
    bx, bz = float(to_pos[0]),   float(to_pos[2])

    dx, dz = bx - ax, bz - az
    dist_xz = math.hypot(dx, dz)
    if dist_xz < 1e-6:
        return []

    ux, uz = dx / dist_xz, dz / dist_xz   # unit XZ direction
    px, pz = -uz, ux                       # horizontal perpendicular (constant)

    # Control point: XZ midpoint, lifted upward by lift_factor * dist_xz
    lift = max(30.0, lift_factor * dist_xz)
    qcx = (ax + bx) / 2
    qcy = ay - lift                        # −Y = upward in LDraw
    qcz = (az + bz) / 2

    def _bezier3d(t: float) -> tuple[float, float, float]:
        s = 1 - t
        return (
            s*s*ax + 2*s*t*qcx + t*t*bx,
            s*s*ay + 2*s*t*qcy + t*t*by,
            s*s*az + 2*s*t*qcz + t*t*bz,
        )

    # t at which the ribbon body ends and the arrowhead begins
    t_end = max(0.0, 1.0 - head_length / dist_xz) if dist_xz > head_length else 0.0

    left_pts:  list[np.ndarray] = []
    right_pts: list[np.ndarray] = []
    for i in range(n_body + 1):
        t = i / n_body * t_end
        qx, qy, qz = _bezier3d(t)
        left_pts.append( np.array([qx + px * half_width,  qy, qz + pz * half_width]))
        right_pts.append(np.array([qx - px * half_width,  qy, qz - pz * half_width]))

    # Arrowhead: flat triangle at the destination Y level
    hbx, _, hbz = _bezier3d(t_end)
    head_left  = np.array([hbx + px * head_half_width, by, hbz + pz * head_half_width])
    head_right = np.array([hbx - px * head_half_width, by, hbz - pz * head_half_width])
    tip        = np.array([bx, by, bz])

    # Full outline: left body → head_left → tip → head_right → right body (reversed)
    return left_pts + [head_left, tip, head_right] + list(reversed(right_pts))


def _make_iso_arrow(
    dwg: svgwrite.Drawing,
    from_pos: np.ndarray,
    to_pos: np.ndarray,
    cx,
    cy,
):
    """Return a projected isometric polygon for one arrow, or None if zero-length."""
    verts = _arrow_polygon_3d(from_pos, to_pos)
    if not verts:
        return None
    pts = [_proj_canvas(v, cx, cy) for v in verts]
    return dwg.polygon(
        points=[(f"{x:.1f}", f"{y:.1f}") for x, y in pts],
        fill="#888",
    )


def _draw_floor_arrows(
    dwg: svgwrite.Drawing,
    arrows: list[tuple],
    cx,
    cy,
) -> None:
    """Draw isometric polygon arrows lying on the horizontal plane."""
    grp = dwg.g(id="arrows")
    for poly in filter(None, (_make_iso_arrow(dwg, fp, tp, cx, cy) for fp, tp in arrows)):
        grp.add(poly)
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Icon overlay
# ---------------------------------------------------------------------------

def _load_icon(icon_path: str | None) -> Image.Image | None:
    """Load an icon image and return it, or None if missing, unreadable, or empty."""
    if not icon_path:
        return None
    try:
        icon = Image.open(icon_path).convert("RGBA")
    except Exception:
        return None
    W, H = icon.size
    return icon if W > 0 and H > 0 else None


def _icon_element(
    dwg: svgwrite.Drawing,
    icon: Image.Image,
    nd: dict,
    cx,
    cy,
):
    """Build the affine-transformed SVG <image> element for one node icon."""
    W, H = icon.size
    pos   = nd["pos"]
    hw    = nd["half_w"]
    ldx   = float(pos[0])
    top_y = float(pos[1])
    ldz   = float(pos[2])

    # Top-face corners in LDraw (Y = top of brick), inset by padding
    _PAD = 4.0   # LDU inset on every edge (~20 % of a 20-LDU half-width)
    hw_i = hw - _PAD
    A = np.array([ldx - hw_i, top_y, ldz - hw_i])   # back-left
    B = np.array([ldx + hw_i, top_y, ldz - hw_i])   # back-right
    D = np.array([ldx - hw_i, top_y, ldz + hw_i])   # front-left

    ax, ay = _proj_canvas(A, cx, cy)
    bx, by = _proj_canvas(B, cx, cy)
    dx, dy = _proj_canvas(D, cx, cy)

    # Affine: image (0,0)→A, (W,0)→B, (0,H)→D
    img_el = dwg.image(_img_to_data_uri(icon), insert=("0px", "0px"), size=(f"{W}px", f"{H}px"))
    img_el.attribs["preserveAspectRatio"] = "none"
    img_el.attribs["transform"] = (
        f"matrix({(bx-ax)/W:.6f},{(by-ay)/W:.6f},"
        f"{(dx-ax)/H:.6f},{(dy-ay)/H:.6f},{ax:.2f},{ay:.2f})"
    )
    return img_el


def _draw_icons(
    dwg: svgwrite.Drawing,
    node_data: list[dict],
    cx,
    cy,
) -> None:
    """Embed provider icons on the top face of each node brick.

    Each icon is affine-mapped from a W×H rectangle to the isometric top-face
    parallelogram using an SVG ``matrix(a,b,c,d,e,f)`` transform.
    """
    grp = dwg.g(id="icons")
    for nd in node_data:
        icon = _load_icon(nd.get("icon_path"))
        if icon is not None:
            grp.add(_icon_element(dwg, icon, nd, cx, cy))
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Isometric labels
# ---------------------------------------------------------------------------

def _draw_labels(
    dwg: svgwrite.Drawing,
    node_data: list[dict],
    cx,
    cy,
) -> None:
    """Draw labels on the left-hand (front/+Z) face of each node brick.

    The face transform is: scaleY(cos30°) → skewX(30°) → rotate(30° CW),
    which maps the text baseline along the isometric +X axis and character
    height straight down in screen space.
    Combined matrix: matrix(0.866025, 0.5, 0, 1, tx, ty).
    Anchor: top-centre of the front (+Z) face of the tile.
    """
    grp = dwg.g(id="labels")
    for nd in (n for n in node_data if n.get("label", "")):
        pos  = nd["pos"]
        hw   = nd["half_w"]
        hh   = nd["half_h"]
        ldx  = float(pos[0])
        ldy  = float(pos[1])
        ldz  = float(pos[2])

        # Anchor at the vertical centre of the front (+Z) face.
        # pos[1] is the tile top face; adding half_h reaches the mid-height
        # of the tile+brick assembly in LDraw Y (down = more positive).
        lx, ly = _proj_canvas(np.array([ldx, ldy + hh, ldz + hw]), cx, cy)

        # Auto-contrast: white text on dark bricks, dark text on light bricks
        color_id = nd.get("color")
        if color_id is not None:
            r, g, b = ldraw_rgb(color_id)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_fill = "#fff" if luminance < 140 else "#333"
        else:
            text_fill = "#333"

        text_el = dwg.text(
            nd["label"],
            insert=(0, 0),
            font_size="14",
            fill=text_fill,
            text_anchor="middle",
        )
        # scaleY(cos30°) → skewX(30°) → rotate(30° CW)
        text_el.attribs["transform"] = (
            f"matrix(0.866025,0.5,0,1,{lx:.1f},{ly:.1f})"
        )
        grp.add(text_el)
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Cluster labels
# ---------------------------------------------------------------------------

def _draw_cluster_labels(
    dwg: svgwrite.Drawing,
    cluster_data: list[dict],
    cx,
    cy,
) -> None:
    """Draw isometric-skewed text labels at the front edge of each cluster platform."""
    grp = dwg.g(id="cluster_labels")
    for nd in (n for n in cluster_data if n.get("label", "")):
        pos = nd["pos"]
        lx, ly = _proj_canvas(pos, cx, cy)
        # Shift f (screen-Y translation) to vertically centre the glyphs on the
        # tile. b+d=1 so adding δ to f produces a pure +δ px screen-Y shift.
        # Two contributions to the needed offset:
        #   glyph em-centre above baseline: d×0.4×fs = 0.5×0.4×fs ≈ 0.08×fs
        #   tile front face (26 px high) pushes the piece visual centre 13 px
        #   below the projected top-face anchor → ~0.42×fs at fs=40
        # Total ≈ 0.5×fs centres the text on the visible tile piece.
        _font_size = 40
        ly += _font_size * 0.5
        text_el = dwg.text(
            nd["label"],
            insert=(0, 0),
            font_size=str(_font_size),
            font_weight="bold",
            fill="#444",
            text_anchor="middle",
        )
        # Top-face isometric matrix: col1=(a,b)=+X dir, col2=(c,d)=+Z dir.
        # Glyphs lie flat on the horizontal tile surface.
        text_el.attribs["transform"] = (
            f"matrix(0.866025,0.5,-0.866025,0.5,{lx:.1f},{ly:.1f})"
        )
        grp.add(text_el)
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------

def compose_diagram_svg(
    renders: dict,
    output: str,
    arrows: list[tuple],
    node_data: list[dict],
    piece_groups: list[tuple[str, list[Piece]]] | None = None,
    cluster_data: list[dict] | None = None,
    padding: int = 60,
) -> None:
    """Compose the final isometric brick SVG.

    Layer order:
    1. Background rect
    2. Isometric floor grid
    3. Brick elements (globally sorted back-to-front across all groups)
    4. Icons on top faces
    5. Isometric labels
    6. Arrows (on top of everything so they are always visible)

    Each grayscale render is stored once per unique (part, rotation) in
    ``<defs>``.  Colour is applied per instance via a ``<filter>`` with
    ``feColorMatrix`` (duotone: white → brick colour, shadows darken
    proportionally).
    """
    pngs: list[tuple[Piece, int, int, float, float]] = [
        (piece, *img.size, ax, ay)
        for img, ax, ay, piece_list in renders.values()
        for piece in piece_list
    ]
    projected = _project_pieces(pngs)

    grid = _grid_params(renders)
    grid_corners = _grid_corner_sx_sy(*grid) if grid else []

    extra_pts = [
        (sx * PX_PER_MM, sy * PX_PER_MM)
        for from_pos, to_pos in arrows
        for p in (from_pos, to_pos)
        for sx, sy, _ in (project_ldraw(p),)
    ]

    W, H, min_x, min_y = _canvas_bounds(
        projected, padding, extra_sx_sy=grid_corners + extra_pts
    )

    def cx(val: float) -> float: return val - min_x + padding
    def cy(val: float) -> float: return val - min_y + padding

    dwg = svgwrite.Drawing(output, size=(f"{W}px", f"{H}px"))

    # 1. Background
    dwg.add(dwg.rect((0, 0), ("100%", "100%"), fill="#f8f8f0"))

    # 2. Grid
    _draw_grid(dwg, grid, cx, cy)

    # 3. Brick images in defs + placed instances (back-to-front, grouped by cluster)
    defs_m = _build_defs(dwg, renders)
    piece_proj = {
        id(piece): _project_piece(piece, iw, ih, ax, ay)
        for piece, iw, ih, ax, ay in pngs
    }
    piece_hex: dict[int, str] = {
        id(piece): "#{:02x}{:02x}{:02x}".format(*ldraw_rgb(piece.color))
        for piece, *_ in pngs
    }
    filters = _build_duotone_filters(dwg, set(piece_hex.values()))

    def _row_key(p: Piece) -> tuple:
        r = piece_proj[id(p)]
        return (-r[1], r[0])   # sort back-to-front: by -ldy then depth

    def _place_piece(grp, piece: Piece) -> None:
        row = piece_proj[id(piece)]
        sx_px, sy_px, label = row[2], row[3], row[8]
        img_id, dax, day, _, _ = defs_m[label]
        use_el = dwg.use(
            f"#{img_id}",
            insert=(f"{cx(sx_px) - dax:.1f}px", f"{cy(sy_px) - day:.1f}px"),
        )
        use_el.attribs["filter"] = f"url(#{filters[piece_hex[id(piece)]]})"
        grp.add(use_el)

    if piece_groups is not None:
        # Pass 1: platform tiles (3024 / 3070b) per cluster — each cluster in its
        # own named <g> so they can be targeted by CSS / inspected in an SVG editor.
        # Platforms are flat on the floor so no cross-cluster depth conflict.
        for group_name, group_pieces in piece_groups:
            platform_pieces = sorted(
                [p for p in group_pieces
                 if p.part in ("3024", "3070b") and id(p) in piece_proj],
                key=_row_key,
            )
            if not platform_pieces:
                continue
            grp = dwg.g(id=f"platform_{group_name}")
            for p in platform_pieces:
                _place_piece(grp, p)
            dwg.add(grp)

        # Pass 2: node bricks + tiles (non-3024/3070b) per cluster in named
        # <g> groups, sorted back-to-front within each group.
        for group_name, group_pieces in piece_groups:
            node_pieces = sorted(
                [p for p in group_pieces
                 if p.part not in ("3024", "3070b") and id(p) in piece_proj],
                key=_row_key,
            )
            if not node_pieces:
                continue
            grp = dwg.g(id=f"nodes_{group_name}")
            for p in node_pieces:
                _place_piece(grp, p)
            dwg.add(grp)
    else:
        pid_to_piece = {id(p): p for p, *_ in pngs}
        for sx_px, sy_px, _, _, _, _, label in projected:
            piece_obj = next(
                pid_to_piece[pid] for pid, row in piece_proj.items()
                if row[2] == sx_px and row[8] == label
            )
            grp = dwg.g()
            _place_piece(grp, piece_obj)
            dwg.add(grp)

    # 4. Icons on top faces
    _draw_icons(dwg, node_data, cx, cy)

    # 5. Node labels
    _draw_labels(dwg, node_data, cx, cy)

    # 6. Cluster labels
    if cluster_data:
        _draw_cluster_labels(dwg, cluster_data, cx, cy)

    # 7. Arrows
    _draw_floor_arrows(dwg, arrows, cx, cy)

    dwg.save(pretty=True)
    print(f"Saved: {output}  ({W}×{H} px, {len(projected)} pieces)")
