"""Immutable geometry primitives. All operations return new objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

MIN_SIDE_PX: Final[int] = 2


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (top-left origin)."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(f"bbox must have positive size, got {self.width}x{self.height}")

    @property
    def center(self) -> tuple[float, float]:
        """Center point as (cx, cy)."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Corners as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        """Rounded (x, y, w, h) for OpenCV calls."""
        return (round(self.x), round(self.y), round(self.width), round(self.height))

    def scaled(self, factor: float) -> BoundingBox:
        """Return a box scaled about its own center."""
        if factor <= 0.0:
            raise ValueError(f"scale factor must be positive, got {factor}")
        cx, cy = self.center
        new_w: float = self.width * factor
        new_h: float = self.height * factor
        return BoundingBox(cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)

    def translated(self, dx: float, dy: float) -> BoundingBox:
        """Return a box shifted by (dx, dy)."""
        return BoundingBox(self.x + dx, self.y + dy, self.width, self.height)

    def moved_to_center(self, cx: float, cy: float) -> BoundingBox:
        """Return a box of the same size centered on (cx, cy)."""
        return BoundingBox(cx - self.width / 2.0, cy - self.height / 2.0, self.width, self.height)


def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    """Build a BoundingBox from corner coordinates."""
    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def clip_to_frame(box: BoundingBox, width: int, height: int) -> BoundingBox | None:
    """Clip a box to frame bounds. Returns None if nothing meaningful remains."""
    x1: float = max(0.0, box.x)
    y1: float = max(0.0, box.y)
    x2: float = min(float(width), box.x + box.width)
    y2: float = min(float(height), box.y + box.height)
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
        return None
    return from_xyxy(x1, y1, x2, y2)


def iou(left: BoundingBox, right: BoundingBox) -> float:
    """Intersection over union of two boxes, in [0, 1]."""
    ax1, ay1, ax2, ay2 = left.as_xyxy()
    bx1, by1, bx2, by2 = right.as_xyxy()
    inter_w: float = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h: float = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection: float = inter_w * inter_h
    union: float = left.area + right.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def center_distance(left: BoundingBox, right: BoundingBox) -> float:
    """Euclidean distance between box centers in pixels."""
    lx, ly = left.center
    rx, ry = right.center
    return float(((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5)
