"""Tests for geometry primitives."""

from __future__ import annotations

import pytest

from pixel_lock_lab.geometry import (
    BoundingBox,
    center_distance,
    clip_to_frame,
    from_xyxy,
    iou,
)


def test_center_and_area() -> None:
    box = BoundingBox(10.0, 20.0, 30.0, 40.0)
    assert box.center == (25.0, 40.0)
    assert box.area == 1200.0


def test_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="positive size"):
        BoundingBox(0.0, 0.0, 0.0, 10.0)


def test_scaled_preserves_center() -> None:
    box = BoundingBox(10.0, 10.0, 20.0, 20.0)
    scaled = box.scaled(2.0)
    assert scaled.center == box.center
    assert scaled.width == 40.0


def test_scaled_rejects_non_positive_factor() -> None:
    with pytest.raises(ValueError, match="positive"):
        BoundingBox(0.0, 0.0, 4.0, 4.0).scaled(0.0)


def test_moved_to_center() -> None:
    box = BoundingBox(0.0, 0.0, 10.0, 10.0).moved_to_center(50.0, 60.0)
    assert box.center == (50.0, 60.0)
    assert (box.width, box.height) == (10.0, 10.0)


def test_iou_identical_is_one() -> None:
    box = BoundingBox(5.0, 5.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_is_zero() -> None:
    left = BoundingBox(0.0, 0.0, 10.0, 10.0)
    right = BoundingBox(100.0, 100.0, 10.0, 10.0)
    assert iou(left, right) == 0.0


def test_iou_half_overlap() -> None:
    left = BoundingBox(0.0, 0.0, 10.0, 10.0)
    right = BoundingBox(5.0, 0.0, 10.0, 10.0)
    assert iou(left, right) == pytest.approx(50.0 / 150.0)


def test_clip_inside_frame_is_unchanged() -> None:
    box = BoundingBox(10.0, 10.0, 20.0, 20.0)
    assert clip_to_frame(box, 100, 100) == box


def test_clip_partially_outside() -> None:
    clipped = clip_to_frame(BoundingBox(-5.0, -5.0, 20.0, 20.0), 100, 100)
    assert clipped is not None
    assert clipped.as_xyxy() == (0.0, 0.0, 15.0, 15.0)


def test_clip_fully_outside_returns_none() -> None:
    assert clip_to_frame(BoundingBox(500.0, 500.0, 20.0, 20.0), 100, 100) is None


def test_from_xyxy_roundtrip() -> None:
    box = from_xyxy(1.0, 2.0, 11.0, 22.0)
    assert (box.width, box.height) == (10.0, 20.0)


def test_center_distance() -> None:
    left = BoundingBox(0.0, 0.0, 2.0, 2.0)
    right = BoundingBox(3.0, 4.0, 2.0, 2.0)
    assert center_distance(left, right) == pytest.approx(5.0)
