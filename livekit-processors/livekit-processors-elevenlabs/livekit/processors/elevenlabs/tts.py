import time
import asyncio
import os
import livekit
import numpy as np
import websockets.client as wsclient
import websockets.exceptions
import aiohttp
import json
import base64

class TTS:
    def __init__(self, audio_source: livekit.AudioSource, sample_rate: int, num_channels: int):
        self._audio_source = audio_source
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._voice_id = ""
        self._ws: wsclient.WebSocketClientProtocol = None

    async def warmup(self):
        if self._ws is not None and self._ws.open:
            await self._ws.close()
            self._ws = None

        await self._get_voice_id_if_needed()
        asyncio.create_task(self._receive_audio_loop())
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream-input?model_id=eleven_monolingual_v1&output_format=pcm_44100&optimize_streaming_latency=2"
        self._ws = await wsclient.connect(uri)
        bos_message = {"text": " ", "xi_api_key": os.environ["ELEVENLABS_API_KEY"]}
        await self._ws.send(json.dumps(bos_message))

    async def push_text(self, text: str):
        text_queue = asyncio.Queue()
        await text_queue.put(text)
        await text_queue.put(None)
        await self.stream_generate_audio(text_queue=text_queue)

    async def stream_generate_audio(self, text_queue: asyncio.Queue[str]):
        await self._get_voice_id_if_needed()
        while self._ws is None or self._ws.open is False:
            await asyncio.sleep(0.1)

        while True:
            text = await text_queue.get()
            if text is None:
                await self._ws.send(json.dumps({"text": ""}))
                break

            payload = {"text": f"{text} ", "try_trigger_generation": True}
            await self._ws.send(json.dumps(payload))

    async def _receive_audio_loop(self):
        while self._ws is None or self._ws.open is False:
            await asyncio.sleep(0.1)

        frame_size_bytes = 441 * 2  # 10ms at 44.1k * 2 bytes per sample (int16) * 1 channels

        try:
            remainder = b''
            while True:
                response = await self._ws.recv()
                data = json.loads(response)

                if data['isFinal']:
                    if len(remainder) > 0:
                        if len(remainder) < frame_size_bytes:
                            remainder = remainder + b'\x00' * (frame_size_bytes - len(remainder))
                        await self._audio_source.capture_frame(self._create_frame_from_chunk(remainder))
                    await self._ws.close()
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
                        await self._audio_source.capture_frame(self._create_frame_from_chunk(chunk[i: i + frame_size_bytes]))

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
            return

    def _create_frame_from_chunk(self, chunk: bytes):
        frame = livekit.AudioFrame.create(sample_rate=44100, num_channels=1, samples_per_channel=441)  # Eleven labs format 
        audio_data = np.ctypeslib.as_array(frame.data)
        np.copyto(audio_data, np.frombuffer(chunk, dtype=np.int16))
        resampled = frame.remix_and_resample(self._sample_rate, self._num_channels)
        return resampled

    async def _get_voice_id_if_needed(self):
        if self._voice_id == "":
            voices_url = "https://api.elevenlabs.io/v1/voices"
            async with aiohttp.ClientSession() as session:
                async with session.get(voices_url) as resp:
                    json_resp = await resp.json()
                    voice = json_resp.get('voices', [])[0]
                    self._voice_id = voice['voice_id']