"""Video frames as portable images; the encoder is the core's (``_core.encode_image``)."""

from __future__ import annotations

from livekit import rtc
from livekit.agents._generated.image_options import (
    EncodeOptions as EncodeOptions,
    ResizeOptions as ResizeOptions,
)

from .. import _core

__all__ = ["EncodeOptions", "ResizeOptions", "encode"]


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
