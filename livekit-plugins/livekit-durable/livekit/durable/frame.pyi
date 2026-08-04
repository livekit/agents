from collections.abc import AsyncGenerator, Coroutine, Generator
from types import FrameType
from typing import Any

def get_frame_ip(frame: FrameType | Coroutine | Generator | AsyncGenerator) -> int:
    """Get instruction pointer of a generator or coroutine."""

def set_frame_ip(frame: FrameType | Coroutine | Generator | AsyncGenerator, ip: int):
    """Set instruction pointer of a generator or coroutine."""

def get_frame_sp(frame: FrameType | Coroutine | Generator | AsyncGenerator) -> int:
    """Get stack pointer of a generator or coroutine."""

def set_frame_sp(frame: FrameType | Coroutine | Generator | AsyncGenerator, sp: int):
    """Set stack pointer of a generator or coroutine."""

def get_frame_bp(frame: FrameType | Coroutine | Generator | AsyncGenerator) -> int:
    """Get block pointer of a generator or coroutine."""

def set_frame_bp(frame: FrameType | Coroutine | Generator | AsyncGenerator, bp: int):
    """Set block pointer of a generator or coroutine."""

def get_frame_stack_at(
    frame: FrameType | Coroutine | Generator | AsyncGenerator, index: int
) -> tuple[bool, Any]:
    """Get an object from a generator or coroutine's stack, as an (is_null, obj) tuple."""

def set_frame_stack_at(
    frame: FrameType | Coroutine | Generator | AsyncGenerator,
    index: int,
    unset: bool,
    value: Any,
):
    """Set or unset an object on the stack of a generator or coroutine."""

def get_frame_block_at(
    frame: FrameType | Coroutine | Generator | AsyncGenerator, index: int
) -> tuple[int, int, int]:
    """Get a block from a generator or coroutine."""

def set_frame_block_at(
    frame: FrameType | Coroutine | Generator | AsyncGenerator,
    index: int,
    value: tuple[int, int, int],
):
    """Restore a block of a generator or coroutine."""

def get_frame_state(
    frame: FrameType | Coroutine | Generator | AsyncGenerator,
) -> int:
    """Get frame state of a generator or coroutine."""

def set_frame_state(frame: FrameType | Coroutine | Generator | AsyncGenerator, state: int):
    """Set frame state of a generator or coroutine."""
