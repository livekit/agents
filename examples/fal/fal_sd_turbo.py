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

import asyncio
import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Set

import aiohttp
import msgpack
from livekit import rtc
from PIL import Image

FAL_URL = "wss://fal.run/fal-ai/sd-turbo-real-time-high-fps-msgpack-a10g/ws"


class FalSDTurbo:
    """Fal SDXL Plugin

    Requires FAL_KEY_ID and FAL_KEY_SECRET environment variables to be set.
    """

    def __init__(
        self,
        *,
        key_id: Optional[str] = None,
        key_secret: Optional[str] = None,
    ):
        self._key_id = key_id or os.environ.get("FAL_KEY_ID")
        self._key_secret = key_secret or os.environ.get("FAL_KEY_SECRET")

    def stream(self) -> "SDTurboHighFPSStream":
        return SDTurboHighFPSStream(key_id=self._key_id, key_secret=self._key_secret)


def _task_done_cb(task: asyncio.Task, set: Set[asyncio.Task]) -> None:
    set.discard(task)
    if task.cancelled():
        logging.info("task cancelled: %s", task)
        return

    if task.exception():
        logging.error("task exception: %s", task, exc_info=task.exception())
        return


class SDTurboHighFPSStream:
    @dataclass
    class Input:
        data: bytes
        prompt: str
        strength: float
        width: int
        height: int

    def __init__(self, key_id: str, key_secret: str):
        self._key_id = key_id
        self._key_secret = key_secret
        self._session = aiohttp.ClientSession()
        self._input_queue = asyncio.Queue[self.Input]()
        self._output_queue = asyncio.Queue[rtc.VideoFrame]()
        self._in_flight_requests = 0
        self._tasks = set()
        self._closed = False
        self._latency = 0
        self._send_time_queue = asyncio.Queue[float]()
        self._run_task = asyncio.create_task(self._run())
        self._run_task.add_done_callback(lambda t: _task_done_cb(t, self._tasks))

    @property
    def latency(self) -> float:
        return self._latency

    def push_frame(self, frame: rtc.VideoFrame, prompt: str, strength: float) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        argb_frame = frame.convert(rtc.VideoBufferType.RGBA)
        self._input_queue.put_nowait(
            self.Input(
                data=argb_frame.data,
                prompt=prompt,
                strength=strength,
                width=frame.width,
                height=frame.height,
            )
        )

    async def aclose(self) -> None:
        self._run_task.cancel()
        await self._run_task
        self._closed = True

    async def _run(self):
        retry_count = 0
        max_retry = 32
        while True:
            try:
                ws = await self._get_connected_ws()
                retry_count = 0
                receive_task = asyncio.create_task(self._ws_receiver(ws))
                receive_task.add_done_callback(lambda t: _task_done_cb(t, self._tasks))
                in_flight_requests = 0
                while not ws.closed:
                    input = await self._input_queue.get()
                    image_rgba = Image.frombytes(
                        "RGBA", (input.width, input.height), input.data
                    )
                    image_rgb = image_rgba.convert("RGB")
                    # center crop to 512x512
                    crop_x = (image_rgb.width - 512) // 2
                    crop_y = (image_rgb.height - 512) // 2
                    image_rgb = image_rgb.crop(
                        (crop_x, crop_y, crop_x + 512, crop_y + 512)
                    )
                    jpg_img = io.BytesIO()
                    image_rgb.save(jpg_img, format="JPEG")
                    payload = {
                        "prompt": input.prompt,
                        "num_inference_steps": 3,
                        "strength": input.strength,
                        "image": jpg_img.getvalue(),
                    }
                    packed = msgpack.packb(payload)
                    if in_flight_requests > 10:
                        logging.warning(
                            f"Sending video faster than it is being processed, skipping frame. In-flight requests: {in_flight_requests}"
                        )
                        continue
                    self._in_flight_requests += 1
                    self._send_time_queue.put_nowait(time.time())
                    await ws.send_bytes(packed)
            except asyncio.CancelledError:
                if ws:
                    await ws.close()
                    if receive_task:
                        await asyncio.shield(receive_task)
                break
            except Exception as e:
                if retry_count > max_retry and max_retry > 0:
                    logging.error(f"failed to connect to FalAI: {e} - {retry_count}")
                    break

                logging.warning(f"failed to connect to FalAI: {e} - retrying")
                await asyncio.sleep(1)

    async def _ws_receiver(self, ws):
        while True:
            ws_message = await ws.receive()
            send_time = await self._send_time_queue.get()
            self._latency = time.time() - send_time
            if ws_message.type == aiohttp.WSMsgType.TEXT:
                logging.info(f"Received text message from fal ai: {ws_message.data}")
                continue

            if ws_message.type == aiohttp.WSMsgType.CLOSED:
                logging.info("Fal AI websocket connection closed")
                return

            if ws_message.type == aiohttp.WSMsgType.ERROR:
                logging.error("Fal AI websocket connection error")
                return

            if ws_message.type != aiohttp.WSMsgType.BINARY:
                logging.warning(f"Received unexpected message type: {ws_message}")
                return

            data = ws_message.data
            output = msgpack.unpackb(data)
            byte_stream = io.BytesIO(output["image"])
            img = Image.open(byte_stream)
            img_rgba = img.convert("RGBA")
            frame = rtc.VideoFrame(
                img.width, img.height, rtc.VideoBufferType.RGBA, img_rgba.tobytes()
            )
            self._in_flight_requests -= 1
            await self._output_queue.put(frame)

    async def _get_connected_ws(self):
        creds = f"{self._key_id}:{self._key_secret}"
        return await self._session.ws_connect(
            FAL_URL, headers={"Authorization": f"Key {creds}"}
        )

    async def __anext__(self) -> rtc.VideoFrame:
        if self._closed and self._output_queue.empty():
            raise StopAsyncIteration

        return await self._output_queue.get()

    def __aiter__(self) -> "SDTurboHighFPSStream":
        return self
