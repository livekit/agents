from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Any, Literal, cast, Callable

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils, vad, tokenize, transcription, stt
from livekit.agents.llm import _oai_api

from ..log import logger
from . import proto, agent_playout


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
    transcription: AssistantTranscriptionOptions


@dataclass(frozen=True)
class AssistantTranscriptionOptions:
    user_transcription: bool = True
    """Whether to forward the user transcription to the client"""
    agent_transcription: bool = True
    """Whether to forward the agent transcription to the client"""
    agent_transcription_speed: float = 1.0
    """The speed at which the agent's speech transcription is forwarded to the client.
    We try to mimic the agent's speech speed by adjusting the transcription speed."""
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer()
    """The tokenizer used to split the speech into sentences.
    This is used to decide when to mark a transcript as final for the agent transcription."""
    word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
        ignore_punctuation=False
    )
    """The tokenizer used to split the speech into words.
    This is used to simulate the "interim results" of the agent transcription."""
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    """A function that takes a string (word) as input and returns a list of strings,
    representing the hyphenated parts of the word."""


class VoiceAssistant(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        system_message: str,
        voice: proto.Voices = "echo",
        vad: vad.VAD | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        transcription: AssistantTranscriptionOptions = AssistantTranscriptionOptions(),
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
            transcription=transcription,
        )
        self._system_message = system_message

        self._vad = vad
        self._fnc_ctx = fnc_ctx

        # audio input
        self._read_micro_atask: asyncio.Task | None = None
        self._subscribed_track: rtc.RemoteAudioTrack | None = None
        self._input_audio_ch = utils.aio.Chan[rtc.AudioFrame]()

        # audio output
        self._playing_handle: agent_playout.PlayoutHandle | None = None

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
        self._agent_playout = agent_playout.AgentPlayout(
            audio_source=self._audio_source
        )

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


        fncs_desc = []
        if self._fnc_ctx is not None:
            for fnc in self._fnc_ctx.ai_functions.values():
                fncs_desc.append(llm._oai_api.build_oai_function_description(fnc))


        initial_cfg: proto.ClientMessage.SetInferenceConfig = {
            "event": "set_inference_config",
            "system_message": self._system_message,
            "turn_end_type": "server_detection",
            "voice": self._opts.voice,
            "disable_audio": False,
            "tools": fncs_desc,
            "tool_choice": None,
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
                    await self._handle_server_message(ws_conn, msg.json())
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

    async def _handle_server_message(
        self, ws_conn: aiohttp.ClientWebSocketResponse, data: dict[Any, Any]
    ) -> None:
        event: proto.ServerResponse = data["event"]

        event_copy = data.copy()
        event_copy.pop("data", None)
        print(event_copy)
        if event == "start_session":
            session_data = cast(proto.ServerMessage.StartSession, data)
            logger.info(
                "OpenAI S2S session started",
                extra={
                    "session_id": session_data["session_id"],
                    "model": session_data["model"],
                    "system_fingerprint": session_data["system_fingerprint"],
                },
            )
        elif event == "add_content":
            add_content_data = cast(proto.ServerMessage.AddContent, data)

            if self._playing_handle is None or self._playing_handle.done():
                tr_fwd = transcription.TTSSegmentsForwarder(
                    room=self._room,
                    participant=self._room.local_participant,
                    speed=self._opts.transcription.agent_transcription_speed,
                    sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
                    word_tokenizer=self._opts.transcription.word_tokenizer,
                    hyphenate_word=self._opts.transcription.hyphenate_word,
                )

                self._playing_handle = self._agent_playout.play(
                    message_id=add_content_data["item_id"],
                    transcription_fwd=tr_fwd,
                )

            if add_content_data["type"] == "audio":
                self._playing_handle.push_audio(
                    base64.b64decode(add_content_data["data"])
                )
            elif add_content_data["type"] == "text":
                self._playing_handle.push_text(add_content_data["data"])

        elif event == "turn_finished":
            turn_finished_data = cast(proto.ServerMessage.TurnFinished, data)
            if turn_finished_data["reason"] not in ("interrupt", "stop"):
                logger.warning(
                    "assistant turn finished unexpectedly",
                    extra={"reason": turn_finished_data["reason"]},
                )

            if (
                self._playing_handle is not None
                and not self._playing_handle.interrupted
            ):
                self._playing_handle.end_input()

        elif event == "model_listening":
            if self._playing_handle is not None and not self._playing_handle.done():
                self._playing_handle.interrupt()
                truncate_data: proto.ClientMessage.TruncateContent = {
                    "event": "truncate_content",
                    "message_id": self._playing_handle.message_id,
                    "index": 0,  # ignored for now (see OAI docs)
                    "text_chars": self._playing_handle.text_chars,
                    "audio_samples": self._playing_handle.audio_samples,
                }
                logger.debug(
                    "truncating content",
                    extra={
                        "audio_samples": truncate_data["audio_samples"],
                        "text_chars": truncate_data["text_chars"],
                    },
                )
                await ws_conn.send_json(truncate_data)

        elif event == "error":
            logger.error("OpenAI S2S error: %s", data["error"])
        elif event == "vad_speech_started":
            self.emit("user_started_speaking")
        elif event == "vad_speech_stopped":
            self.emit("user_stopped_speaking")
        elif event == "input_transcribed":
            transcript_data = cast(proto.ServerMessage.InputTranscribed, data)
            transcript = transcript_data["transcript"]

            self._stt_forwarder.update(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language="en", text=transcript)],
                )
            )

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
                self._stt_forwarder = transcription.STTSegmentsForwarder(
                    room=self._room,
                    participant=self._linked_participant,
                    track=self._subscribed_track,
                )


                if self._read_micro_atask is not None:
                    self._read_micro_atask.cancel()

                self._read_micro_atask = asyncio.create_task(
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
