"""Shared NumPy array aliases for strict static typing across NumPy versions."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

Array = NDArray[Any]
