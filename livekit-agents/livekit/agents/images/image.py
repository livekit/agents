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
from typing import Literal, Optional

import PIL.Image
from livekit import rtc


@dataclass
class EncodeOptions:
    format: Literal["JPEG", "PNG"]
    resize_options: Optional["ResizeOptions"] = None


@dataclass
class ResizeOptions:
    width: int
    height: int
    strategy: Literal["center_aspect_fit", "center_aspect_cover", "skew"]


def encode(frame: rtc.VideoFrame, options: EncodeOptions):
    img = _image_from_frame(frame)
    resized = _resize_image(img, options)
    buffer = io.BytesIO()
    resized.save(buffer, options.format)
    buffer.seek(0)
    return buffer.read()


def _image_from_frame(frame: rtc.VideoFrame):
    converted = frame.convert(rtc.VideoBufferType.RGBA)
    rgb_image = PIL.Image.frombytes(
        "RGBA", (frame.width, frame.height), converted.data
    ).convert("RGB")
    return rgb_image


def _resize_image(image: PIL.Image.Image, options: EncodeOptions):
    if options.resize_options is None:
        return image

    resize_opts = options.resize_options
    if resize_opts.strategy == "skew":
        return image.resize((resize_opts.width, resize_opts.height))
    elif resize_opts.strategy == "center_aspect_fit":
        result = PIL.Image.new("RGB", (resize_opts.width, resize_opts.height))

        # Start with assuming image is width constrained
        new_width = resize_opts.width
        new_height = int(image.height * (resize_opts.width / image.width))

        # If image is height constrained
        if image.width / image.height < resize_opts.width / resize_opts.height:
            new_width = resize_opts.width
            new_height = int(image.height * (resize_opts.width / image.width))

        resized = image.resize((new_width, new_height))
        PIL.Image.Image.paste(
            result,
            resized,
            (
                (resize_opts.width - new_width) // 2,
                (resize_opts.height - new_height) // 2,
            ),
        )
        return result
    elif resize_opts.strategy == "center_aspect_cover":
        result = PIL.Image.new("RGB", (resize_opts.width, resize_opts.height))

        # Start with assuming image is width constrained
        new_width = resize_opts.width
        new_height = int(image.height * (resize_opts.width / image.width))

        # Image is height constrained
        if image.width / image.height > resize_opts.width / resize_opts.height:
            new_width = resize_opts.width
            new_height = int(image.height * (resize_opts.width / image.width))

        resized = image.resize((new_width, new_height))
        PIL.Image.Image.paste(
            result,
            resized,
            (
                (resize_opts.width - new_width) // 2,
                (resize_opts.height - new_height) // 2,
            ),
        )
        return result

    raise ValueError(f"Unknown resize strategy: {resize_opts.strategy}")
