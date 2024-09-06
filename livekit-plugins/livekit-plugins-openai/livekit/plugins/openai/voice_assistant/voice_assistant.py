from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Literal

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils, vad
from livekit.agents.llm import _oai_api

from ..log import logger
from . import proto


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
]


@dataclass(frozen=True)
class _ImplOptions:
    voice: proto.Voices
    api_key: str
    max_tokens: int
    temperature: float


class VoiceAssistant(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        system_message: str,
        voice: proto.Voices = "echo",
        vad: vad.VAD | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        api_key: str | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
        http_session: aiohttp.ClientSession | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__()
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("api_key must be provided or set in OPENAI_API_KEY")

        self._loop = loop or asyncio.get_event_loop()
        self._http_session = http_session
        self._opts = _ImplOptions(
            voice=voice,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._system_message = system_message

        self._vad = vad
        self._fnc_ctx = fnc_ctx

        self._read_micro_task: asyncio.Task | None = None
        self._subscribed_track: rtc.RemoteAudioTrack | None = None
        self._input_audio_ch = utils.aio.Chan[rtc.AudioFrame]()

        self._linked_participant: rtc.RemoteParticipant | None = None
        self._started, self._closed = False, False

    @property
    def system_message(self) -> str:
        return self._system_message

    @system_message.setter
    def system_message(self, value: str) -> None:
        self._system_message = value

    @property
    def vad(self) -> vad.VAD | None:
        return self._vad

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, value: llm.FunctionContext | None) -> None:
        self._fnc_ctx = value

    def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    ) -> None:
        if self._started:
            raise RuntimeError("voice assistant already started")

        room.on("participant_connected", self._on_participant_connected)
        room.on("track_published", self._subscribe_to_microphone)
        room.on("track_subscribed", self._subscribe_to_microphone)

        self._room, self._participant = room, participant

        if participant is not None:
            if isinstance(participant, rtc.RemoteParticipant):
                self._link_participant(participant.identity)
            else:
                self._link_participant(participant)
        else:
            # no participant provided, try to find the first participant in the room
            for participant in self._room.remote_participants.values():
                self._link_participant(participant.identity)
                break

        self._main_atask = asyncio.create_task(self._main_task())


    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        self._audio_source = rtc.AudioSource(proto.SAMPLE_RATE, proto.NUM_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track(
            "assistant_voice", self._audio_source
        )
        self._agent_publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        await self._agent_publication.wait_for_subscription()

        try:
            headers = {"Authorization": "Bearer " + self._opts.api_key}
            ws_conn = await self._ensure_session().ws_connect(
                proto.API_URL, headers=headers
            )
        except Exception:
            logger.exception("failed to connect to OpenAI API S2S")
            return

        initial_cfg: proto.ClientMessage.SetInferenceConfig = {
            "event": "set_inference_config",
            "system_message": self._system_message,
            "turn_end_type": "server_detection",
            "voice": self._opts.voice,
            "disable_audio": False,
            "tools": {},
            "tool_choice": "auto",
            "audio_format": "pcm16",
            "temperature": self._opts.temperature,
            "max_tokens": self._opts.max_tokens,
            "transcribe_input": True,
        }

        await ws_conn.send_json(initial_cfg)

        @utils.log_exceptions(logger=logger)
        async def send_task():
            async for frame in self._input_audio_ch:
                await ws_conn.send_json(
                    {
                        "event": "add_user_audio",
                        "data": base64.b64encode(frame.data).decode("utf-8"),
                    }
                )

        @utils.log_exceptions(logger=logger)
        async def recv_task():
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise Exception("OpenAI S2S connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected OpenAI S2S message type %s", msg.type)
                    continue

                try:
                    self._handle_server_message(msg.json())
                except Exception:
                    logger.exception(
                        "failed to handle OpenAI S2S message", extra={"msg": msg}
                    )

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)


    @utils.log_exceptions(logger=logger)
    async def _playout_task(self, playout_ch: utils.aio.Chan[rtc.AudioFrame]):
        bstream = utils.audio.AudioByteStream(
            proto.SAMPLE_RATE,
            proto.NUM_CHANNELS,
            samples_per_channel=proto.OUT_FRAME_SIZE,
        )

        async for frame in playout_ch:
            for f in bstream.write(frame.data.tobytes()):
                await self._audio_source.capture_frame(f)

        for f in bstream.flush():
            await self._audio_source.capture_frame(f)

    def _handle_server_message(self, data: dict) -> None:
        event = data["event"]
        if event == "start_session":
            print(data)
        elif event == "add_content":
            type = data["type"]
            if type == "audio":
                if playout_ch is None:
                    playout_ch = utils.aio.Chan[rtc.AudioFrame]()
                    playout_atask = asyncio.create_task(_playout_task(playout_ch))

                audio_data = base64.b64decode(data["data"])

                frame = rtc.AudioFrame(
                    audio_data,
                    proto.SAMPLE_RATE,
                    proto.NUM_CHANNELS,
                    len(audio_data) // 2,
                )
                playout_ch.send_nowait(frame)
            else:
                print(data)
        elif event == "turn_finished":
            print("turn_finished")
            if playout_ch is not None:
                playout_ch.close()
                playout_ch = None

        elif event == "model_listening":
            print("cancelling")
            if playout_atask is not None:
                playout_atask.cancel()
                playout_ch = None

        elif event == "error":
            logger.error("OpenAI S2S error: %s", data["error"])
        else:
            print(data)


    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        if self._linked_participant is None:
            return

        self._link_participant(participant.identity)

    def _link_participant(self, participant_identity: str) -> None:
        self._linked_participant = self._room.remote_participants.get(
            participant_identity
        )
        if self._linked_participant is None:
            logger.error("_link_participant must be called with a valid identity")
            return

        self._subscribe_to_microphone()


    def _subscribe_to_microphone(self, *args, **kwargs) -> None:
        """Subscribe to the participant microphone if found"""

        if self._linked_participant is None:
            return

        @utils.log_exceptions(logger=logger)
        async def _read_audio_stream_task(audio_stream: rtc.AudioStream):
            bstream = utils.audio.AudioByteStream(
                proto.SAMPLE_RATE,
                proto.NUM_CHANNELS,
                samples_per_channel=proto.IN_FRAME_SIZE,
            )

            async for ev in audio_stream:
                for frame in bstream.write(ev.frame.data.tobytes()):
                    self._input_audio_ch.send_nowait(frame)

        for publication in self._linked_participant.track_publications.values():
            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                continue

            if not publication.subscribed:
                publication.set_subscribed(True)

            if (
                publication.track is not None
                and publication.track != self._subscribed_track
            ):
                self._subscribed_track = publication.track  # type: ignore
                if self._read_micro_task is not None:
                    self._read_micro_task.cancel()

                self._read_micro_task = asyncio.create_task(
                    _read_audio_stream_task(
                        rtc.AudioStream(
                            self._subscribed_track,  # type: ignore
                            sample_rate=proto.SAMPLE_RATE,
                            num_channels=proto.NUM_CHANNELS,
                        )
                    )
                )
                break


    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()

        return self._http_session
