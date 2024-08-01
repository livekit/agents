# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, Optional

from livekit import rtc

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class EncodeOptions:
    format: Literal["JPEG", "PNG"] = "JPEG"
    resize_options: Optional["ResizeOptions"] = None


@dataclass
class ResizeOptions:
    width: int
    height: int
    strategy: Literal["center_aspect_fit", "center_aspect_cover", "skew"]


def import_pil():
    try:
        if "Image" not in globals():
            globals()["Image"] = import_module("PIL.Image")
    except ImportError:
        raise ImportError(
            "You haven't included the 'images' optional dependencies. Please install the 'codecs' extra by running `pip install livekit-agents[images]`"
        )


def encode(frame: rtc.VideoFrame, options: EncodeOptions):
    import_pil()
    img = _image_from_frame(frame)
    resized = _resize_image(img, options)
    buffer = io.BytesIO()
    resized.save(buffer, options.format)
    buffer.seek(0)
    return buffer.read()


def _image_from_frame(frame: rtc.VideoFrame):
    converted = frame
    if frame.type != rtc.VideoBufferType.RGBA:
        converted = frame.convert(rtc.VideoBufferType.RGBA)

    rgb_image = Image.frombytes(  # type: ignore
        "RGBA", (frame.width, frame.height), converted.data
    ).convert("RGB")
    return rgb_image


def _resize_image(image: Any, options: EncodeOptions):
    if options.resize_options is None:
        return image

    resize_opts = options.resize_options
    if resize_opts.strategy == "skew":
        return image.resize((resize_opts.width, resize_opts.height))
    elif resize_opts.strategy == "center_aspect_fit":
        result = Image.new("RGB", (resize_opts.width, resize_opts.height))  # noqa

        # Start with assuming the new image is narrower than the original
        new_width = resize_opts.width
        new_height = int(image.height * (resize_opts.width / image.width))

        # If the new image is wider than the original
        if resize_opts.width / resize_opts.height > image.width / image.height:
            new_width = resize_opts.width
            new_height = int(image.height * (resize_opts.width / image.width))

        resized = image.resize((new_width, new_height))
        Image.Image.paste(
            result,
            resized,
            (
                (resize_opts.width - new_width) // 2,
                (resize_opts.height - new_height) // 2,
            ),
        )
        return result
    elif resize_opts.strategy == "center_aspect_cover":
        result = Image.new("RGB", (resize_opts.width, resize_opts.height))  # noqa

        # Start with assuming the new image is shorter than the original
        new_height = int(image.height * (resize_opts.width / image.width))
        new_width = resize_opts.width

        # If the new image is taller than the original
        if resize_opts.height / resize_opts.width > image.height / image.width:
            new_width = int(image.width * (resize_opts.height / image.height))
            new_height = resize_opts.height

        resized = image.resize((new_width, new_height))
        Image.Image.paste(  # noqa
            result,
            resized,
            (
                (resize_opts.width - new_width) // 2,
                (resize_opts.height - new_height) // 2,
            ),
        )
        return result

    raise ValueError(f"Unknown resize strategy: {resize_opts.strategy}")
