import io
from typing import Literal

import pytest
from PIL import Image

from livekit import rtc
from livekit.agents.utils.images import EncodeOptions, ResizeOptions, encode

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("strategy", ["center_aspect_fit", "scale_aspect_fit"])
@pytest.mark.parametrize(
    "source_size,target_size,scaled_size",
    [
        ((1024, 1), (512, 512), (512, 1)),
        ((1, 1024), (512, 512), (1, 512)),
        ((16, 9), (1, 1), (1, 1)),
        ((9, 16), (1, 1), (1, 1)),
        ((16, 9), (8, 8), (8, 4)),
        ((9, 16), (8, 8), (4, 8)),
    ],
)
def test_encode_aspect_fit(
    strategy: Literal["center_aspect_fit", "scale_aspect_fit"],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    scaled_size: tuple[int, int],
) -> None:
    width, height = source_size
    frame = rtc.VideoFrame(
        width, height, rtc.VideoBufferType.RGBA, bytes([255, 0, 0, 255]) * (width * height)
    )
    encoded = encode(
        frame,
        EncodeOptions(
            format="PNG",
            resize_options=ResizeOptions(
                width=target_size[0], height=target_size[1], strategy=strategy
            ),
        ),
    )

    with Image.open(io.BytesIO(encoded)) as image:
        if strategy == "scale_aspect_fit":
            assert image.size == scaled_size
            assert image.getpixel((0, 0)) == (255, 0, 0)
        else:
            assert image.size == target_size
            left = (target_size[0] - scaled_size[0]) // 2
            top = (target_size[1] - scaled_size[1]) // 2
            assert image.getpixel((left, top)) == (255, 0, 0)
            if left > 0 or top > 0:
                assert image.getpixel((0, 0)) == (0, 0, 0)
