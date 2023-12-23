# Copyright 2023 LiveKit, Inc.
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

import aiohttp
import logging
from livekit import rtc
import asyncio
import os
import io
import msgpack
from PIL import Image
from dataclasses import dataclass

FAL_URL = "wss://110602490-sd-turbo-real-time-high-fps-msgpack.gateway.alpha.fal.ai/ws"


class SDXLPlugin:
    """Fal SDXL Plugin

    Requires FAL_KEY_ID and FAL_KEY_SECRET environment variables to be set.
    """

    @dataclass
    class Input:
        data: bytes
        width: int
        height: int
        prompt: str

    def __init__(self, *, initial_prompt: str = ""):
        if not os.getenv("FAL_KEY_ID") and os.getenv("FAL_KEY_SECRET"):
            raise ValueError(
                "The Fal plugin requires FAL_KEY_ID and FAL_KEY_SECRET environment variables to be set."
            )

        self._prompt = initial_prompt
        self._session = aiohttp.ClientSession()
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_lock = asyncio.Lock()
        self._in_flight_requests = 0
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._tasks = set()

    def start_stream(
        self, input_stream: AsyncIterable[rtc.VideoFrame]
    ) -> AsyncIterable[rtc.VideoFrame]:
        t = asyncio.create_task(self._queue_filler(input_stream))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

        t = asyncio.create_task(self._ws_sender())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

        t = asyncio.create_task(self._ws_receiver())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

        async def output_stream() -> AsyncIterable[rtc.VideoFrame]:
            while True:
                yield await self._output_queue.get()

        return output_stream()

    def update_prompt(self, prompt: str):
        self._prompt = prompt

    async def _get_connected_ws(self):
        async with self._ws_lock:
            if self._ws is not None and not self._ws.closed:
                return self._ws

            if self._ws is not None and self._ws.closed:
                self._ws = None
                logging.warning(
                    "Fal SDXL Plugin: websocket connection closed, creating a new connection"
                )

            client_id = os.environ.get("FAL_KEY_ID")
            client_secret = os.environ.get("FAL_KEY_SECRET")
            creds = f"{client_id}:{client_secret}"
            self._ws = await self._session.ws_connect(
                FAL_URL, headers={"Authorization": f"Key {creds}"}
            )
            self._in_flight_requests = 0
            return self._ws

    async def _queue_filler(self, input_stream: AsyncIterable[rtc.VideoFrame]):
        async for frame in input_stream:
            argb_frame = rtc.ArgbFrame.create(
                format=rtc.VideoFormatType.FORMAT_RGBA,
                width=frame.buffer.width,
                height=frame.buffer.height,
            )
            frame.buffer.to_argb(dst=argb_frame)
            await self._input_queue.put(
                self.Input(
                    data=argb_frame.data,
                    prompt=self._prompt,
                    width=frame.buffer.width,
                    height=frame.buffer.height,
                )
            )

    async def _ws_sender(self):
        while True:
            item = await self._input_queue.get()
            image_rgba = Image.frombytes("RGBA", (item.width, item.height), item.data)
            image_rgb = image_rgba.convert("RGB")
            jpg_img = io.BytesIO()
            image_rgb.save(jpg_img, format="JPEG")
            payload = {
                "prompt": item.prompt,
                "num_inference_steps": 3,
                "strength": 0.4,
                "image": jpg_img.getvalue(),
            }
            packed = msgpack.packb(payload)
            ws = await self._get_connected_ws()
            if self._in_flight_requests > 10:
                logging.warning(
                    f"Sending video faster than it is being processed, skipping frame. In-flight requests: {self._in_flight_requests}"
                )
                continue
            self._in_flight_requests += 1
            await ws.send_bytes(packed)

    async def _ws_receiver(self):
        while True:
            ws = await self._get_connected_ws()
            ws_message = await ws.receive()
            if ws_message.type == aiohttp.WSMsgType.TEXT:
                logging.info(f"Received text message from fal ai: {ws_message.data}")
                continue

            if ws_message.type == aiohttp.WSMsgType.CLOSED:
                logging.info("Fal AI websocket connection closed")
                continue

            if ws_message.type == aiohttp.WSMsgType.ERROR:
                logging.error("Fal AI websocket connection error")
                continue

            if ws_message.type != aiohttp.WSMsgType.BINARY:
                logging.warning(f"Received unexpected message type: {ws_message}")
                continue

            data = ws_message.data
            output = msgpack.unpackb(data)
            byte_stream = io.BytesIO(output)
            img = Image.open(byte_stream)
            img_rgba = img.convert("RGBA")
            argb_frame = rtc.ArgbFrame.create(
                rtc.VideoFormatType.FORMAT_ARGB, img.width, img.height
            )
            argb_frame.data[:] = img_rgba.tobytes()
            video_frame = rtc.VideoFrame(
                0, rtc.VideoRotation.VIDEO_ROTATION_0, argb_frame.to_i420()
            )
            self._in_flight_requests -= 1
            await self._output_queue.put(video_frame)
