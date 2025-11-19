"""Realtime speech translation with captions and voice output.

This example demonstrates how to build a multi-user meeting where each participant
can speak in their own language, and be able to understand the other participants.

It works by:
- Each participant contains an attribute indicating theirlanguage
  (this is embedded in their access token)
- The agent keeps track of the languages needed for each participant, and creates
  translation tasks for each input audio track.
- In each translation task, we use an LLM to translate to the target language, and
  synthesize the translated audio.
- The translated audio and transcriptions are published to the room.
  - The audio track is named as "{input_track_id}-{target_language_code}"
  - The transcription is published as a text stream, with an attribute "language"
    set to the target language.
- With the above, the UI can render the right captions and audio tracks matching the
  language that the participant is speaking.
"""

import asyncio
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AgentServer,
    AutoSubscribe,
    JobContext,
    JobRequest,
    cli,
    llm,
    stt,
    tokenize,
    utils,
    voice,
)
from livekit.agents.types import (
    ATTRIBUTE_TRANSCRIPTION_FINAL,
    ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID,
    TOPIC_TRANSCRIPTION,
)
from livekit.plugins import deepgram, elevenlabs, google

load_dotenv()

logger = logging.getLogger("transcriber")


@dataclass
class Language:
    code: str
    name: str


_languages = [
    Language(code="en", name="English"),
    Language(code="de", name="German"),
    Language(code="es", name="Spanish"),
    Language(code="fr", name="French"),
    Language(code="ja", name="Japanese"),
    Language(code="zh", name="Chinese (Mandarin)"),
]

language_map: dict[str, Language] = {lang.code: lang for lang in _languages}


@dataclass
class PlayoutData:
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    transcript: str

    def __init__(self, transcript: str):
        self.audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        self.transcript = transcript


class Translator:
    """Handles real-time translation of transcribed text to target languages."""

    def __init__(
        self,
        *,
        room: rtc.Room,
        track_id: str,
        target_language: Language,
        participant_identity: str,
    ):
        """Initialize translator for a specific language."""
        self._room = room
        self._target_language = target_language
        self._track_id = track_id  # source track id
        self._participant_identity = participant_identity
        self._tasks: list[asyncio.Task] = []
        self._input_ch = utils.aio.Chan[str]()
        self._playout_ch = utils.aio.Chan[PlayoutData]()
        self._llm_stream = utils.aio.Chan[str]()

        self._llm = google.LLM()
        self._tts = elevenlabs.TTS()
        self._resampler = rtc.AudioResampler(22050, 48000)

    def start(self):
        self._tasks.append(asyncio.create_task(self._run()))
        self._tasks.append(asyncio.create_task(self._synthesize_tts()))
        self._tasks.append(asyncio.create_task(self._publish_and_playout()))

    async def aclose(self):
        self.end_input()

        await asyncio.gather(*self._tasks)
        self._tasks.clear()

    def push_sentence(self, sentence: str):
        self._input_ch.send_nowait(sentence)

    def end_input(self):
        if not self._input_ch.closed:
            self._input_ch.close()

    @utils.log_exceptions(logger=logger)
    async def _run(self):
        """Translate sentences to target language and enqueue the translated audio frames."""
        async for sentence in self._input_ch:
            if not sentence:
                continue

            context = llm.ChatContext()
            context.add_message(
                role="system",
                content=f"You are a translator for language: {self._target_language.name}. "
                f"Your only response should be the exact translation of input text in the {self._target_language.name} language.",
            )
            context.add_message(role="user", content=sentence)

            try:
                llm_stream = self._llm.chat(chat_ctx=context)
                response = ""
                async for chunk in llm_stream:
                    if not chunk.delta or not chunk.delta.content:
                        continue
                    response += chunk.delta.content
                await llm_stream.aclose()

                logger.info(f"translated to {self._target_language.name}: {response}")

                self._llm_stream.send_nowait(response)
            except Exception:
                logger.exception("Error translating sentence")

        self._llm_stream.close()

    @utils.log_exceptions(logger=logger)
    async def _synthesize_tts(self):
        async for sentence in self._llm_stream:
            stream = self._tts.synthesize(sentence)
            playout_data = PlayoutData(sentence)
            self._playout_ch.send_nowait(playout_data)
            async for frame in stream:
                frames = self._resampler.push(frame.frame)
                for f in frames:
                    playout_data.audio_ch.send_nowait(f)
            playout_data.audio_ch.close()
            await stream.aclose()

        logger.info("ending tts input")
        self._playout_ch.close()

    @utils.log_exceptions(logger=logger)
    async def _publish_and_playout(self):
        """Publish the translated audio frames to the room and play them out."""

        audio_output = voice._ParticipantAudioOutput(
            self._room,
            sample_rate=48000,
            num_channels=1,
            track_publish_options=rtc.TrackPublishOptions(),
            track_name=f"{self._track_id}-{self._target_language.code}",
        )
        await audio_output.start()
        text_output = voice._ParticipantStreamTranscriptionOutput(
            self._room,
            participant=self._participant_identity,
            is_delta_stream=True,
            attributes={
                "language": self._target_language.code,
                "translated": "true",
            },
        )
        synchronizer = voice.TranscriptSynchronizer(
            next_in_chain_audio=audio_output,
            next_in_chain_text=text_output,
        )

        async for playout_data in self._playout_ch:
            logger.info(f"translated playing out: {playout_data.transcript}")
            await synchronizer.text_output.capture_text(playout_data.transcript)
            synchronizer.text_output.flush()
            async for frame in playout_data.audio_ch:
                await synchronizer.audio_output.capture_frame(frame)
            synchronizer.audio_output.flush()

        await synchronizer.aclose()
        await audio_output.aclose()


class InputTrack:
    def __init__(
        self,
        *,
        language: str,
        track: rtc.RemoteAudioTrack,
        participant_identity: str,
        room: rtc.Room,
    ):
        self.language = language
        self.track = track
        self.participant_identity = participant_identity
        self.room = room

        self._translators: dict[str, Translator] = {}
        self._tasks: list[asyncio.Task] = []
        self._stt = deepgram.STT(language=language, model="nova-2")
        tokenizer = tokenize.blingfire.SentenceTokenizer()
        self._sentence_stream: tokenize.SentenceStream = tokenizer.stream()
        self._stt_stream = self._stt.stream(language=language)

    def start(self):
        self._tasks.append(asyncio.create_task(self._consume_input()))
        self._tasks.append(asyncio.create_task(self._tokenize_to_sentences()))
        self._tasks.append(asyncio.create_task(self._forward_to_translators()))

    async def aclose(self):
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

        for translator in self._translators.values():
            await translator.aclose()

    def set_languages(self, target_languages: list[str], *, room: rtc.Room):
        for lang in target_languages:
            if lang != self.language:
                self._add_translator(lang, room)
        for lang in list(self._translators.keys()):
            if lang != self.language and lang not in target_languages:
                self._remove_translator(lang)

    def _add_translator(self, target_language: str, room: rtc.Room):
        lang = language_map.get(target_language)
        if not lang:
            return
        if lang.code in self._translators:
            return
        logger.info(f"starting translator for {self.track.sid} - {lang.name}")
        translator = Translator(
            room=room,
            target_language=lang,
            track_id=self.track.sid,
            participant_identity=self.participant_identity,
        )
        translator.start()
        self._translators[lang.code] = translator

    def _remove_translator(self, target_language: str):
        translator = self._translators.pop(target_language, None)
        if translator:
            asyncio.create_task(translator.aclose())

    async def _forward_to_translators(self):
        """Forward transcribed sentences to each language specific translators."""

        async for ev in self._sentence_stream:
            logger.info(f"forwarding to translators: {ev.token}")
            for translator in self._translators.values():
                translator.push_sentence(ev.token)

        logger.info("ending translator input")
        for translator in self._translators.values():
            translator.end_input()

    @utils.log_exceptions(logger=logger)
    async def _tokenize_to_sentences(self):
        """tokenize STT output to sentences."""

        async def _create_text_writer(*, segment_id: str) -> rtc.TextStreamWriter:
            attributes = {
                ATTRIBUTE_TRANSCRIPTION_FINAL: "false",
                ATTRIBUTE_TRANSCRIPTION_TRACK_ID: self.track.sid,
                ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID: segment_id,
                "language": self.language,
            }

            return await self.room.local_participant.stream_text(
                topic=TOPIC_TRANSCRIPTION,
                sender_identity=self.participant_identity,
                attributes=attributes,
            )

        segment_id = utils.shortuuid("SG_")
        writer: rtc.TextStreamWriter | None = None
        async for ev in self._stt_stream:
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                logger.info(f"STT: {ev.alternatives[0].text}")
                self._sentence_stream.push_text(ev.alternatives[0].text)
                if writer is None:
                    writer = await _create_text_writer(segment_id=segment_id)
                await writer.write(ev.alternatives[0].text)
            elif ev.type == stt.SpeechEventType.END_OF_SPEECH:
                self._sentence_stream.flush()
                if writer:
                    attributes = {
                        ATTRIBUTE_TRANSCRIPTION_FINAL: "true",
                        ATTRIBUTE_TRANSCRIPTION_TRACK_ID: self.track.sid,
                    }
                    await writer.aclose(attributes=attributes)
                    writer = None
                    segment_id = utils.shortuuid("SG_")
        self._sentence_stream.end_input()

    async def _consume_input(self):
        """Transcribe the audio stream and run through STT."""

        stream = rtc.AudioStream.from_track(track=self.track)
        async for ev in stream:
            self._stt_stream.push_frame(ev.frame)

        await stream.aclose()


class RoomTranslator:
    """Translates audio tracks in a room to multiple languages, publishing audio and transcriptions back to the room."""

    def __init__(self, room: rtc.Room, *, additional_languages: list[str] | None = None):
        """Initialize the room translator.

        additional_languages: list[str] = [] - additional languages to translate to, in addition to the languages of the participants in the room.
        """
        self.room = room
        self.additional_languages = list(additional_languages) if additional_languages else []
        self.desired_languages = list(self.additional_languages)
        self.input_tracks: list[InputTrack] = []

    def start(self):
        @self.room.on("participant_connected")
        def on_participant_joined(participant: rtc.RemoteParticipant):
            self._update_languages()

        @self.room.on("participant_disconnected")
        def on_participant_left(participant: rtc.RemoteParticipant):
            self._update_languages()

        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                audio_language = participant.attributes.get("language")
                # TODO: this is test code
                if audio_language is None:
                    audio_language = "en"
                self._add_track(
                    track, language=audio_language, participant_identity=participant.identity
                )

        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self._remove_track(track)

        # first set to the current state that's already in the room
        self._update_languages()

        existing_track_ids = [input_track.track.sid for input_track in self.input_tracks]
        for participant in self.room.remote_participants.values():
            for pub in participant.track_publications.values():
                if (
                    pub.track
                    and pub.kind == rtc.TrackKind.KIND_AUDIO
                    and pub.sid not in existing_track_ids
                ):
                    on_track_subscribed(pub.track, pub, participant)

    def _update_languages(self):
        languages = set(self.additional_languages)
        for participant in self.room.remote_participants.values():
            language = participant.attributes.get("language")
            if language:
                languages.add(language)
        self.desired_languages = list(languages)
        self._reconcile_translators()

    def _add_track(self, track: rtc.RemoteAudioTrack, *, language: str, participant_identity: str):
        input_track = InputTrack(
            language=language,
            track=track,
            participant_identity=participant_identity,
            room=self.room,
        )
        input_track.start()
        self.input_tracks.append(input_track)
        self._reconcile_translators()

    def _remove_track(self, track: rtc.RemoteAudioTrack):
        input_track = next((t for t in self.input_tracks if t.track.sid == track.sid), None)
        if input_track:
            asyncio.create_task(input_track.aclose())
            self.input_tracks.remove(input_track)

    def _reconcile_translators(self):
        for track in self.input_tracks:
            track.set_languages(self.desired_languages, room=self.room)


async def request_fnc(req: JobRequest):
    await req.accept(
        name="agent",
        identity="agent",
    )


server = AgentServer()


@server.rtc_session(agent_name="translator", on_request=request_fnc)
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the translation agent service."""
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # set the current agent's state to be compatible with new sandbox
    await ctx.room.local_participant.set_attributes(
        {
            "lk.agent.state": "listening",
        }
    )
    logger.info("agent state set to listening")

    room_translator = RoomTranslator(
        ctx.room,
        # for testing, uncomment to add additional languages to translate to
        # additional_languages=[],
    )
    room_translator.start()


if __name__ == "__main__":
    cli.run_app(server)
