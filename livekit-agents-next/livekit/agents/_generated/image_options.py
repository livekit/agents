# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/media/images.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import Literal


@dataclass
class ResizeOptions:
    """How a frame is resized as it is encoded."""

    width: int
    height: int
    strategy: Literal['center_aspect_fit', 'center_aspect_cover', 'scale_aspect_fit', 'scale_aspect_cover', 'skew']
    """``center_aspect_fit`` letterboxes into the size, ``center_aspect_cover`` fills it
        cropping, ``scale_aspect_fit``/``scale_aspect_cover`` keep the aspect ratio inside/around it,
        ``skew`` resizes to it exactly.
    """


@dataclass
class EncodeOptions:
    """How a frame is encoded."""

    format: Literal['JPEG', 'PNG'] = "JPEG"
    resize_options: ResizeOptions | None = None
    quality: int | None = 75
    """JPEG quality, 0-100."""


ResizeOptions.__annotations__ = get_annotations(ResizeOptions, eval_str=True)
ResizeOptions.__module__ = "livekit.agents.utils.media"
EncodeOptions.__annotations__ = get_annotations(EncodeOptions, eval_str=True)
EncodeOptions.__module__ = "livekit.agents.utils.media"
