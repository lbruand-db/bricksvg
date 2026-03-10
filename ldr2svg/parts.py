"""parts.py — LDraw data model: colours, part definitions, scene parser."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .part_map_data import PART_MAP_DATA

# ---------------------------------------------------------------------------
# LDraw colour table
# ---------------------------------------------------------------------------
LDRAW_COLORS: dict[int, tuple[int, int, int]] = {
    0: (5, 19, 29),      1: (0, 85, 191),    2: (37, 122, 62),   3: (0, 133, 85),
    4: (201, 26, 9),     5: (200, 112, 160),  6: (88, 58, 32),    7: (155, 155, 155),
    8: (100, 100, 100),  9: (180, 210, 227),  10: (75, 151, 75),  11: (85, 165, 175),
    12: (255, 119, 0),   13: (247, 188, 213), 14: (242, 205, 55), 15: (255, 255, 255),
    17: (190, 233, 15),  19: (242, 205, 161), 22: (129, 9, 110),  25: (254, 138, 24),
    28: (173, 114, 53),  70: (92, 29, 13),    71: (150, 150, 150),72: (108, 110, 104),
    85: (84, 32, 113),   86: (100, 51, 0),    92: (181, 106, 55), 288: (0, 71, 33),
    320: (114, 14, 14),  321: (54, 172, 200), 484: (160, 82, 48),
}

_NEXT_CUSTOM_COLOR_ID = 1000


def register_custom_color(r: int, g: int, b: int) -> int:
    """Register an arbitrary RGB color and return its synthetic color ID."""
    global _NEXT_CUSTOM_COLOR_ID  # noqa: PLW0603
    for cid, rgb in LDRAW_COLORS.items():
        if rgb == (r, g, b):
            return cid
    cid = _NEXT_CUSTOM_COLOR_ID
    _NEXT_CUSTOM_COLOR_ID += 1
    LDRAW_COLORS[cid] = (r, g, b)
    return cid


def ldraw_rgb(color_id: int) -> tuple[int, int, int]:
    return LDRAW_COLORS.get(color_id, (136, 136, 136))

# ---------------------------------------------------------------------------
# LDraw part → brick.scad block() mapping
# ---------------------------------------------------------------------------
@dataclass
class PartDef:
    width: int        # studs in local X
    length: int       # studs in local Y (= -LDraw Z after axis swap)
    height: float     # 1=brick, 1/3=plate
    block_type: str   # "brick", "round", etc.

    @property
    def w_mm(self): return self.width  * 8.0
    @property
    def l_mm(self): return self.length * 8.0
    @property
    def h_mm(self): return self.height * 9.6

PART_MAP: dict[str, PartDef] = {k: PartDef(*v) for k, v in PART_MAP_DATA.items()}

# ---------------------------------------------------------------------------
# LDraw scene parser
# ---------------------------------------------------------------------------
@dataclass
class Piece:
    part: str          # e.g. "3666"
    color: int
    pos: np.ndarray    # (3,) LDraw world position (LDU)
    rot: np.ndarray    # (3, 3) LDraw rotation matrix

def _parse_ldr_line(parts: Sequence[str]) -> Piece:
    """Parse a tokenised LDraw type-1 line. Caller must ensure len >= 15 and parts[0] == '1'."""
    return Piece(
        part  = Path(" ".join(parts[14:])).stem.lower(),
        color = int(parts[1]),
        pos   = np.array([float(p) for p in parts[2:5]]),
        rot   = np.array([float(p) for p in parts[5:14]], dtype=float).reshape(3, 3),
    )

def parse_ldr(path: Path) -> list[Piece]:
    lines = path.read_text(errors="replace").splitlines()
    return [_parse_ldr_line(p) for p in (line.split() for line in lines)
            if p and p[0] == "1" and len(p) >= 15]
