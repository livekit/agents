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

import asyncio
import os
from livekit import rtc
import numpy as np
from typing import AsyncIterator, Optional, AsyncIterable
import websockets.client as wsclient
import websockets.exceptions
import aiohttp
import json
import base64


class _WSWrapper:
    def __init__(self):
        self.ws: Optional[wsclient.WebSocketClientProtocol] = None
        self._voice_id = ""

    async def wait_for_connected(self):
        while self.ws is None or self.ws.closed:
            await asyncio.sleep(0.1)

    async def connect(self) -> wsclient.WebSocketClientProtocol:
        await self._get_voice_id_if_needed()
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream-input?model_id=eleven_monolingual_v1&output_format=pcm_44100&optimize_streaming_latency=2"
        self.ws = await wsclient.connect(uri)
        bos_message = {"text": " ",
                       "xi_api_key": os.environ["ELEVENLABS_API_KEY"]}
        await self.ws.send(json.dumps(bos_message))

    async def _get_voice_id_if_needed(self):
        if self._voice_id == "":
            voices_url = "https://api.elevenlabs.io/v1/voices"
            async with aiohttp.ClientSession() as session:
                async with session.get(voices_url) as resp:
                    json_resp = await resp.json()
                    voice = json_resp.get('voices', [])[0]
                    self._voice_id = voice['voice_id']


class TTSPlugin:
    """Eleven Labs TTS plugin
    """
    async def generate_speech(self, text_stream: AsyncIterator[str]) -> AsyncIterable[rtc.AudioFrame]:
        """Generate streamed speech from a stream of text

        Args:
            text_stream (AsyncIterator[str]): Iterator of text to be converted to speech

        Returns:
            AsyncIterable[rtc.AudioFrame]: Iterator of audio frames
        """
        ws = _WSWrapper()
        await ws.connect()
        result_queue = asyncio.Queue[rtc.AudioFrame]()
        asyncio.create_task(self._push_data_loop(ws, text_stream))
        asyncio.create_task(self._receive_audio_loop(ws, result_queue))

        async def iterator():
            while True:
                frame = await result_queue.get()
                if frame is None:
                    return
                yield frame

        return iterator()

    async def _close(self):
        pass

    async def _push_data_loop(self, ws_wrapper: _WSWrapper, text_stream: AsyncIterable[str]):
        await ws_wrapper.wait_for_connected()
        async for text in text_stream:
            payload = {"text": f"{text} ", "try_trigger_generation": True}
            await ws_wrapper.ws.send(json.dumps(payload))
        await ws_wrapper.ws.send(json.dumps({"text": ""}))

    async def _receive_audio_loop(self, ws_wrapper: _WSWrapper, result_queue: asyncio.Queue[rtc.AudioFrame]):
        await ws_wrapper.wait_for_connected()
        # 10ms at 44.1k * 2 bytes per sample (int16) * 1 channels
        frame_size_bytes = 441 * 2

        try:
            remainder = b''
            while True:
                response = await ws_wrapper.ws.recv()
                data = json.loads(response)

                if data['isFinal']:
                    if len(remainder) > 0:
                        if len(remainder) < frame_size_bytes:
                            remainder = remainder + b'\x00' * \
                                (frame_size_bytes - len(remainder))
                        frame = self._create_frame_from_chunk(remainder)
                        await result_queue.put(frame)
                    await ws_wrapper.ws.close()
                    await result_queue.put(None)
                    return

                if data["audio"]:
                    chunk = remainder + base64.b64decode(data["audio"])

                    if len(chunk) < frame_size_bytes:
                        remainder = chunk
                        continue
                    else:
                        remainder = chunk[-(len(chunk) % (frame_size_bytes)):]
                        chunk = chunk[:-len(remainder)]

                    for i in range(0, len(chunk), frame_size_bytes):
                        frame = self._create_frame_from_chunk(
                            chunk[i: i + frame_size_bytes])
                        await result_queue.put(frame)

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
            await result_queue.put(None)
            return

    def _create_frame_from_chunk(self, chunk: bytes):
        frame = rtc.AudioFrame.create(
            sample_rate=44100,
            num_channels=1,
            samples_per_channel=441)  # Eleven labs format
        audio_data = np.ctypeslib.as_array(frame.data)
        np.copyto(audio_data, np.frombuffer(chunk, dtype=np.int16))
        return frame
