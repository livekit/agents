from typing import Callable
from dataclasses import dataclass

# public API


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass
class PaintData:
    dirty_rects: list[Rect]
    buffer: memoryview
    width: int
    height: int


@dataclass
class BrowserOptions:
    url: str
    framerate: int
    width: int
    height: int
    paint_callback: Callable[[PaintData], None]
