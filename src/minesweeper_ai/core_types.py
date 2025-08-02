import random

from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangle using (x, y, w, h) — top-left corner plus size."""
    x: int
    y: int
    w: int
    h: int

    @staticmethod
    def from_corners(x1: int, y1: int, x2: int, y2: int) -> "Rectangle":
        """Create Rectangle from two opposite corners (left-top and right-bottom)."""
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return Rectangle(x, y, w, h)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def random_point(self) -> tuple[int, int]:
        """Return a random integer point strictly inside the rectangle."""
        if self.w <= 1 or self.h <= 1:
            return self.x, self.y
        rx = random.randint(self.x + 1, max(self.x + 1, self.right - 2))
        ry = random.randint(self.y + 1, max(self.y + 1, self.bottom - 2))
        return rx, ry


class State(IntEnum):
    """Enum for the process status codes."""
    EMPTY = 0
    READY = 1
