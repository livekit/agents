from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, Callable


import aiohttp
from livekit import rtc
from livekit.agents import llm, utils, vad, tokenize, transcription, stt
from livekit.agents.llm import _oai_api
from livekit.agents.llm.function_context import CalledFunction

from ..log import logger
from . import agent_playout


SAMPLE_RATE = 24000
NUM_CHANNELS = 1


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
]


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


class S2SModel(Protocol): ...


@dataclass(frozen=True)
class _ImplOptions:
    transcription: AssistantTranscriptionOptions


class OmniAssistant(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        model: S2SModel,
        vad: vad.VAD | None = None,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
        transcription: AssistantTranscriptionOptions = AssistantTranscriptionOptions(),
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__()

        self._loop = loop or asyncio.get_event_loop()

        from livekit.plugins.openai import realtime

        assert isinstance(model, realtime.RealtimeModel)

        self._model = model
        self._vad = vad
        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx

        self._opts = _ImplOptions(
            transcription=transcription,
        )

        # audio input
        self._read_micro_atask: asyncio.Task | None = None
        self._subscribed_track: rtc.RemoteAudioTrack | None = None
        self._input_audio_ch = utils.aio.Chan[rtc.AudioFrame]()

        # audio output
        self._playing_handle: agent_playout.PlayoutHandle | None = None

        # tools
        self._fnc_tasks: set[asyncio.Task] = set()

        self._linked_participant: rtc.RemoteParticipant | None = None
        self._started, self._closed = False, False

    @property
    def vad(self) -> vad.VAD | None:
        return self._vad

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._session.fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, value: llm.FunctionContext | None) -> None:
        self._session.fnc_ctx = value

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
        self._session = self._model.session(
            chat_ctx=self._chat_ctx, fnc_ctx=self._fnc_ctx
        )

        from livekit.plugins.openai import realtime

        @self._session.on("input_transcribed")
        def _input_transcribed(transcript: str):
            self._stt_forwarder.update(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language="en", text=transcript)],
                )
            )

        @self._session.on("add_message")
        def _add_message(message: realtime.PendingMessage):
            tr_fwd = transcription.TTSSegmentsForwarder(
                room=self._room,
                participant=self._room.local_participant,
                speed=self._opts.transcription.agent_transcription_speed,
                sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
                word_tokenizer=self._opts.transcription.word_tokenizer,
                hyphenate_word=self._opts.transcription.hyphenate_word,
            )

            self._playing_handle = self._agent_playout.play(
                message_id=message.id,
                transcription_fwd=tr_fwd,
                text_stream=message.text_stream,
                audio_stream=message.audio_stream,
            )

        @self._session.on("generation_canceled")
        def _generation_canceled():
            if self._playing_handle is not None and not self._playing_handle.done():
                self._playing_handle.interrupt()

                self._session.truncate_content(
                    message_id=self._playing_handle.message_id,
                    text_chars=self._playing_handle.text_chars,
                    audio_samples=self._playing_handle.audio_samples,
                )

                self._playing_handle = None

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        self._audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
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

        @utils.log_exceptions(logger=logger)
        async def send_task():
            async for frame in self._input_audio_ch:
                self._session.add_user_audio(frame)

        tasks = [
            asyncio.create_task(send_task()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

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
                SAMPLE_RATE,
                NUM_CHANNELS,
                samples_per_channel=2400,
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
                            sample_rate=SAMPLE_RATE,
                            num_channels=NUM_CHANNELS,
                        )
                    )
                )
                break

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()

        return self._http_session
