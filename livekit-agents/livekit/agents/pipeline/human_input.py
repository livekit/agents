from __future__ import annotations

import asyncio
from enum import Enum
from typing import Literal

from livekit import rtc

from .. import stt as speech_to_text
from .. import transcription, utils
from .. import vad as voice_activity_detection
from .log import logger

EventTypes = Literal[
    "start_of_speech",
    "vad_inference_done",
    "end_of_speech",
    "final_transcript",
    "interim_transcript",
]


class SpeechDetectionMode(str, Enum):
    VAD = "vad"
    STT = "stt"


class HumanInput(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        room: rtc.Room,
        vad: voice_activity_detection.VAD | None,
        stt: speech_to_text.STT,
        participant: rtc.RemoteParticipant,
        transcription: bool,
        speech_detection_mode: SpeechDetectionMode = SpeechDetectionMode.VAD,
    ) -> None:
        super().__init__()
        if speech_detection_mode == SpeechDetectionMode.VAD and vad is None:
            raise ValueError("VAD is required for SpeechDetectionMode.VAD")

        self._room, self._vad, self._stt, self._participant, self._transcription = (
            room,
            vad,
            stt,
            participant,
            transcription,
        )
        self._subscribed_track: rtc.RemoteAudioTrack | None = None
        self._recognize_atask: asyncio.Task[None] | None = None

        self._closed = False
        self._vad_speaking = False
        self._stt_speaking = False
        self._speech_probability = 0.0
        self._speech_detection_mode = speech_detection_mode

        self._room.on("track_published", self._subscribe_to_microphone)
        self._room.on("track_subscribed", self._subscribe_to_microphone)
        self._subscribe_to_microphone()

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("HumanInput already closed")

        self._closed = True
        self._room.off("track_published", self._subscribe_to_microphone)
        self._room.off("track_subscribed", self._subscribe_to_microphone)
        self._speaking = False

        if self._recognize_atask is not None:
            await utils.aio.gracefully_cancel(self._recognize_atask)

    @property
    def speaking(self) -> bool:
        if self._speech_detection_mode == SpeechDetectionMode.VAD:
            return self._vad_speaking
        elif self._speech_detection_mode == SpeechDetectionMode.STT:
            return self._stt_speaking
        else:
            raise ValueError(
                f"Invalid speech detection mode: {self._speech_detection_mode}"
            )

    @property
    def speaking_probability(self) -> float:
        return self._speech_probability

    def _subscribe_to_microphone(self, *args, **kwargs) -> None:
        """
        Subscribe to the participant microphone if found and not already subscribed.
        Do nothing if no track is found.
        """
        for publication in self._participant.track_publications.values():
            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                continue

            if not publication.subscribed:
                publication.set_subscribed(True)

            if (
                publication.track is not None
                and publication.track != self._subscribed_track
            ):
                self._subscribed_track = publication.track  # type: ignore
                if self._recognize_atask is not None:
                    self._recognize_atask.cancel()

                self._recognize_atask = asyncio.create_task(
                    self._recognize_task(rtc.AudioStream(self._subscribed_track))  # type: ignore
                )
                break

    @utils.log_exceptions(logger=logger)
    async def _recognize_task(self, audio_stream: rtc.AudioStream) -> None:
        """
        Receive the frames from the user audio stream and detect voice activity.
        """
        if self._speech_detection_mode == SpeechDetectionMode.VAD:
            vad_stream = self._vad.stream()
        else:
            vad_stream = None

        stt_stream = self._stt.stream()

        def _before_forward(
            fwd: transcription.STTSegmentsForwarder, transcription: rtc.Transcription
        ):
            if not self._transcription:
                transcription.segments = []

            return transcription

        stt_forwarder = transcription.STTSegmentsForwarder(
            room=self._room,
            participant=self._participant,
            track=self._subscribed_track,
            before_forward_cb=_before_forward,
        )

        async def _audio_stream_co() -> None:
            # forward the audio stream to the VAD and STT streams
            async for ev in audio_stream:
                stt_stream.push_frame(ev.frame)

                if vad_stream is not None:
                    vad_stream.push_frame(ev.frame)

        async def _vad_stream_co() -> None:
            if vad_stream is None:
                return

            async for ev in vad_stream:
                if ev.type == voice_activity_detection.VADEventType.START_OF_SPEECH:
                    self._vad_speaking = True
                    self.emit("start_of_speech", ev)
                elif ev.type == voice_activity_detection.VADEventType.INFERENCE_DONE:
                    self._speech_probability = ev.probability
                    self.emit("vad_inference_done", ev)
                elif ev.type == voice_activity_detection.VADEventType.END_OF_SPEECH:
                    self._vad_speaking = False
                    self.emit("end_of_speech", ev)

        async def _stt_stream_co() -> None:
            async for ev in stt_stream:
                stt_forwarder.update(ev)

                if ev.type == speech_to_text.SpeechEventType.FINAL_TRANSCRIPT:
                    self._stt_speaking = False
                    self.emit("final_transcript", ev)
                elif ev.type == speech_to_text.SpeechEventType.INTERIM_TRANSCRIPT:
                    self._stt_speaking = True
                    self.emit("interim_transcript", ev)

        tasks = [
            asyncio.create_task(_audio_stream_co()),
            asyncio.create_task(_vad_stream_co()),
            asyncio.create_task(_stt_stream_co()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

            await stt_forwarder.aclose()
            await stt_stream.aclose()
            if vad_stream is not None:
                await vad_stream.aclose()
