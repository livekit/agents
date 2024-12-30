from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    overload,
)

import aiohttp
from livekit import rtc
from livekit.agents import llm, stt, tokenize, transcription, utils, vad
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import MultimodalLLMMetrics

from ..log import logger
from ..types import ATTRIBUTE_AGENT_STATE, AgentState
from . import agent_playout

EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "user_speech_committed",
    "agent_speech_committed",
    "agent_speech_interrupted",
    "function_calls_collected",
    "function_calls_finished",
    "metrics_collected",
]


class _InputTranscriptionProto(Protocol):
    item_id: str
    """id of the item"""
    transcript: str
    """transcript of the input audio"""


class _ContentProto(Protocol):
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text: str
    audio: list[rtc.AudioFrame]
    text_stream: AsyncIterable[str]
    audio_stream: AsyncIterable[rtc.AudioFrame]
    content_type: Literal["text", "audio"]


class _CapabilitiesProto(Protocol):
    supports_truncate: bool


class _RealtimeAPI(Protocol):
    """Realtime API protocol"""

    @property
    def capabilities(self) -> _CapabilitiesProto: ...
    def session(
        self,
        *,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
    ) -> _RealtimeAPISession:
        """
        Create a new realtime session with the given chat and function contexts.
        """
        pass


T = TypeVar("T", bound=Callable[..., Any])


class _RealtimeAPISession(Protocol):
    async def set_chat_ctx(self, ctx: llm.ChatContext) -> None: ...
    @overload
    def on(self, event: str, callback: None = None) -> Callable[[T], T]: ...
    @overload
    def on(self, event: str, callback: T) -> T: ...
    def on(
        self, event: str, callback: Optional[T] = None
    ) -> Union[T, Callable[[T], T]]: ...

    def _push_audio(self, frame: rtc.AudioFrame) -> None: ...
    @property
    def fnc_ctx(self) -> llm.FunctionContext | None: ...
    @fnc_ctx.setter
    def fnc_ctx(self, value: llm.FunctionContext | None) -> None: ...
    def chat_ctx_copy(self) -> llm.ChatContext: ...
    def _recover_from_text_response(self, item_id: str) -> None: ...
    def _update_conversation_item_content(
        self,
        item_id: str,
        content: llm.ChatContent | list[llm.ChatContent] | None = None,
    ) -> None: ...
    def _truncate_conversation_item(
        self, item_id: str, content_index: int, audio_end_ms: int
    ) -> None: ...


@dataclass(frozen=True)
class AgentTranscriptionOptions:
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


@dataclass(frozen=True)
class _ImplOptions:
    transcription: AgentTranscriptionOptions


class MultimodalAgent(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        model: _RealtimeAPI,
        vad: vad.VAD | None = None,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
        transcription: AgentTranscriptionOptions = AgentTranscriptionOptions(),
        max_text_response_retries: int = 5,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """Create a new MultimodalAgent.

        Args:
            model: RealtimeAPI instance.
            vad: Voice Activity Detection (VAD) instance.
            chat_ctx: Chat context for the assistant.
            fnc_ctx: Function context for the assistant.
            transcription: Options for assistant transcription.
            max_text_response_retries: Maximum number of retries to recover
                from text responses to audio mode. OpenAI's realtime API has a
                chance to return text responses instead of audio if the chat
                context includes text system or assistant messages. The agent will
                attempt to recover to audio mode by deleting the text response
                and appending an empty audio message to the conversation.
            loop: Event loop to use. Default to asyncio.get_event_loop().
        """
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

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

        self._linked_participant: rtc.RemoteParticipant | None = None
        self._started, self._closed = False, False

        self._update_state_task: asyncio.Task | None = None
        self._http_session: aiohttp.ClientSession | None = None

        self._text_response_retries = 0
        self._max_text_response_retries = max_text_response_retries

    @property
    def vad(self) -> vad.VAD | None:
        return self._vad

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._session.fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, value: llm.FunctionContext | None) -> None:
        self._session.fnc_ctx = value

    def chat_ctx_copy(self) -> llm.ChatContext:
        return self._session.chat_ctx_copy()

    async def set_chat_ctx(self, ctx: llm.ChatContext) -> None:
        await self._session.set_chat_ctx(ctx)

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

        self._session = self._model.session(
            chat_ctx=self._chat_ctx, fnc_ctx=self._fnc_ctx
        )

        # Create a task to wait for initialization and start the main task
        async def _init_and_start():
            try:
                await self._session._init_sync_task
                logger.info("Session initialized with chat context")
                self._main_atask = asyncio.create_task(self._main_task())
            except Exception as e:
                logger.exception("Failed to initialize session")
                raise e

        # Schedule the initialization and start task
        asyncio.create_task(_init_and_start())

        @self._session.on("response_content_added")
        def _on_content_added(message: _ContentProto):
            tr_fwd = transcription.TTSSegmentsForwarder(
                room=self._room,
                participant=self._room.local_participant,
                speed=self._opts.transcription.agent_transcription_speed,
                sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
                word_tokenizer=self._opts.transcription.word_tokenizer,
                hyphenate_word=self._opts.transcription.hyphenate_word,
            )

            self._playing_handle = self._agent_playout.play(
                item_id=message.item_id,
                content_index=message.content_index,
                transcription_fwd=tr_fwd,
                text_stream=message.text_stream,
                audio_stream=message.audio_stream,
            )

        @self._session.on("response_content_done")
        def _response_content_done(message: _ContentProto):
            if message.content_type == "text":
                if self._text_response_retries >= self._max_text_response_retries:
                    raise RuntimeError(
                        f"The OpenAI Realtime API returned a text response "
                        f"after {self._max_text_response_retries} retries. "
                        f"Please try to reduce the number of text system or "
                        f"assistant messages in the chat context."
                    )

                self._text_response_retries += 1
                logger.warning(
                    "The OpenAI Realtime API returned a text response instead of audio. "
                    "Attempting to recover to audio mode...",
                    extra={
                        "item_id": message.item_id,
                        "text": message.text,
                        "retries": self._text_response_retries,
                    },
                )
                self._session._recover_from_text_response(message.item_id)
            else:
                self._text_response_retries = 0

        @self._session.on("input_speech_committed")
        def _input_speech_committed():
            self._stt_forwarder.update(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language="", text="")],
                )
            )

        @self._session.on("input_speech_transcription_completed")
        def _input_speech_transcription_completed(ev: _InputTranscriptionProto):
            self._stt_forwarder.update(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language="", text=ev.transcript)],
                )
            )
            user_msg = ChatMessage.create(
                text=ev.transcript, role="user", id=ev.item_id
            )

            self._session._update_conversation_item_content(
                ev.item_id, user_msg.content
            )

            self.emit("user_speech_committed", user_msg)
            logger.debug(
                "committed user speech",
                extra={"user_transcript": ev.transcript},
            )

        @self._session.on("input_speech_started")
        def _input_speech_started():
            self.emit("user_started_speaking")
            self._update_state("listening")
            if self._playing_handle is not None and not self._playing_handle.done():
                self._playing_handle.interrupt()

                if self._model.capabilities.supports_truncate:
                    self._session._truncate_conversation_item(
                        item_id=self._playing_handle.item_id,
                        content_index=self._playing_handle.content_index,
                        audio_end_ms=int(
                            self._playing_handle.audio_samples / 24000 * 1000
                        ),
                    )

        @self._session.on("input_speech_stopped")
        def _input_speech_stopped():
            self.emit("user_stopped_speaking")

        @self._session.on("function_calls_collected")
        def _function_calls_collected(fnc_call_infos: list[llm.FunctionCallInfo]):
            self.emit("function_calls_collected", fnc_call_infos)

        @self._session.on("function_calls_finished")
        def _function_calls_finished(called_fncs: list[llm.CalledFunction]):
            self.emit("function_calls_finished", called_fncs)

        @self._session.on("metrics_collected")
        def _metrics_collected(metrics: MultimodalLLMMetrics):
            self.emit("metrics_collected", metrics)

    def _update_state(self, state: AgentState, delay: float = 0.0):
        """Set the current state of the agent"""

        @utils.log_exceptions(logger=logger)
        async def _run_task(delay: float) -> None:
            await asyncio.sleep(delay)

            if self._room.isconnected():
                await self._room.local_participant.set_attributes(
                    {ATTRIBUTE_AGENT_STATE: state}
                )

        if self._update_state_task is not None:
            self._update_state_task.cancel()

        self._update_state_task = asyncio.create_task(_run_task(delay))

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        self._update_state("initializing")
        self._audio_source = rtc.AudioSource(24000, 1)
        self._agent_playout = agent_playout.AgentPlayout(
            audio_source=self._audio_source
        )

        def _on_playout_started() -> None:
            self.emit("agent_started_speaking")
            self._update_state("speaking")

        def _on_playout_stopped(interrupted: bool) -> None:
            self.emit("agent_stopped_speaking")
            self._update_state("listening")

            if self._playing_handle is not None:
                collected_text = self._playing_handle._tr_fwd.played_text
                if interrupted:
                    collected_text += "..."

                msg = ChatMessage.create(
                    text=collected_text,
                    role="assistant",
                    id=self._playing_handle.item_id,
                )
                if self._model.capabilities.supports_truncate:
                    self._session._update_conversation_item_content(
                        self._playing_handle.item_id, msg.content
                    )

                if interrupted:
                    self.emit("agent_speech_interrupted", msg)
                else:
                    self.emit("agent_speech_committed", msg)

                logger.debug(
                    "committed agent speech",
                    extra={
                        "agent_transcript": collected_text,
                        "interrupted": interrupted,
                    },
                )

        self._agent_playout.on("playout_started", _on_playout_started)
        self._agent_playout.on("playout_stopped", _on_playout_stopped)

        track = rtc.LocalAudioTrack.create_audio_track(
            "assistant_voice", self._audio_source
        )
        self._agent_publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        await self._agent_publication.wait_for_subscription()

        bstream = utils.audio.AudioByteStream(
            24000,
            1,
            samples_per_channel=2400,
        )
        async for frame in self._input_audio_ch:
            for f in bstream.write(frame.data.tobytes()):
                self._session._push_audio(f)

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

    async def _micro_task(self, track: rtc.LocalAudioTrack) -> None:
        stream_24khz = rtc.AudioStream(track, sample_rate=24000, num_channels=1)
        async for ev in stream_24khz:
            self._input_audio_ch.send_nowait(ev.frame)

    def _subscribe_to_microphone(self, *args, **kwargs) -> None:
        """Subscribe to the participant microphone if found"""

        if self._linked_participant is None:
            return

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
                    self._micro_task(self._subscribed_track)  # type: ignore
                )
                break

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()

        return self._http_session
