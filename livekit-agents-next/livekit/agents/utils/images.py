"""Video frames as portable images; the encoder is the core's (``_core.encode_image``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from livekit import rtc

from .. import _core

__all__ = ["EncodeOptions", "ResizeOptions", "encode"]


@dataclass
class ResizeOptions:
    """How a frame is resized as it is encoded."""

    width: int
    height: int
    strategy: Literal[
        "center_aspect_fit",
        "center_aspect_cover",
        "scale_aspect_fit",
        "scale_aspect_cover",
        "skew",
    ]
    """``center_aspect_fit`` letterboxes into the size, ``center_aspect_cover`` fills it
    cropping, ``scale_aspect_fit``/``scale_aspect_cover`` keep the aspect ratio inside/around it,
    ``skew`` resizes to it exactly."""


@dataclass
class EncodeOptions:
    """How a frame is encoded."""

    format: Literal["JPEG", "PNG"] = "JPEG"
    resize_options: ResizeOptions | None = None
    quality: int | None = 75
    """JPEG quality, 0-100."""


async def encode(frame: rtc.VideoFrame, options: EncodeOptions) -> bytes:
    """Encode a frame on an admitted worker, keeping the caller's event loop serviceable."""
    resize = options.resize_options
    return await _core.encode_image(
        frame,
        format=options.format,
        quality=options.quality,
        resize_width=resize.width if resize else None,
        resize_height=resize.height if resize else None,
        resize_strategy=resize.strategy if resize else None,
    )


class _FrameReader:
    """How the core reads a host frame it encodes: as RGBA, converted from any other type."""

    def _core_rgba(self, frame: rtc.VideoFrame) -> rtc.VideoFrame:
        if frame.type == rtc.VideoBufferType.RGBA:
            return frame
        return frame.convert(rtc.VideoBufferType.RGBA)


_core._core_install_frame_reader(_FrameReader())
