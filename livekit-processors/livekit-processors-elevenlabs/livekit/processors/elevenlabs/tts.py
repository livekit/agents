import time
import asyncio
import os
from livekit import rtc
from livekit import agents
import numpy as np
from typing import AsyncIterator
import websockets.client as wsclient
import websockets.exceptions
import aiohttp
import json
import base64


class ElevenLabsTTS:
    def __init__(self):
        self._voice_id = ""
        self._ws: wsclient.WebSocketClientProtocol = None

    async def _connect_ws(self) -> wsclient.WebSocketClientProtocol:
        await self._get_voice_id_if_needed()
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream-input?model_id=eleven_monolingual_v1&output_format=pcm_44100&optimize_streaming_latency=2"
        ws = await wsclient.connect(uri)
        bos_message = {"text": " ",
                       "xi_api_key": os.environ["ELEVENLABS_API_KEY"]}
        await ws.send(json.dumps(bos_message))
        return ws

    async def push_text_iterator(self, text_iterator: AsyncIterator[str]) -> AsyncIterator[agents.TextToSpeechProcessor.Event]:
        print("push")
        ws = await self._connect_ws()
        print("ELE Connected")
        result_queue = asyncio.Queue()
        result_iterator = agents.utils.AsyncQueueIterator(result_queue)

        asyncio.create_task(self._push_data_loop(ws, text_iterator))
        asyncio.create_task(self._receive_audio_loop(ws, result_queue))
        return result_iterator

    async def _push_data_loop(self, ws: wsclient.WebSocketClientProtocol, text_iterator: AsyncIterator[str]):
        async for text in text_iterator:
            if text is None:
                await ws.send(json.dumps({"text": ""}))
                break

            payload = {"text": f"{text} ", "try_trigger_generation": True}
            await ws.send(json.dumps(payload))

    async def _receive_audio_loop(self, ws: wsclient.WebSocketClientProtocol, result_queue: asyncio.Queue):
        # 10ms at 44.1k * 2 bytes per sample (int16) * 1 channels
        frame_size_bytes = 441 * 2

        try:
            remainder = b''
            while True:
                response = await ws.recv()
                data = json.loads(response)

                if data['isFinal']:
                    if len(remainder) > 0:
                        if len(remainder) < frame_size_bytes:
                            remainder = remainder + b'\x00' * \
                                (frame_size_bytes - len(remainder))
                        await self._audio_source.capture_frame(self._create_frame_from_chunk(remainder))
                    await ws.close()
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
            return

    def _create_frame_from_chunk(self, chunk: bytes):
        frame = rtc.AudioFrame.create(
            sample_rate=44100, num_channels=1, samples_per_channel=441)  # Eleven labs format
        audio_data = np.ctypeslib.as_array(frame.data)
        np.copyto(audio_data, np.frombuffer(chunk, dtype=np.int16))
        return frame

    async def _get_voice_id_if_needed(self):
        if self._voice_id == "":
            voices_url = "https://api.elevenlabs.io/v1/voices"
            async with aiohttp.ClientSession() as session:
                async with session.get(voices_url) as resp:
                    json_resp = await resp.json()
                    voice = json_resp.get('voices', [])[0]
                    self._voice_id = voice['voice_id']


class ElevenLabsTTSProcessor(agents.TextToSpeechProcessor):
    def __init__(self):
        self._tts = ElevenLabsTTS()

        super().__init__(process=self._tts.push_text_iterator)
