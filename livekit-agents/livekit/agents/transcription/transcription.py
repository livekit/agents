from typing import Callable

from livekit import rtc

from .. import tokenize
from .stt_forwarder import STTSegmentsForwarder
from .tts_forwarder import TTSOptions, TTSSegmentsForwarder


class TranscriptionManager:
    def __init__(self, room: rtc.Room):
        self._room = room

    def forward_stt_transcription(
        self, *, participant: rtc.Participant | str, track_id: str | None = None
    ) -> "STTSegmentsForwarder":
        """
        Forward STT transcription to the users. (Useful for client-side rendering)
        """
        identity = participant if isinstance(participant, str) else participant.identity
        if track_id is None:
            track_id = self._find_micro_track_id(identity)

        return STTSegmentsForwarder(
            room=self._room,
            participant_identity=identity,
            track_id=track_id,
        )

    def forward_tts_transcription(
        self,
        *,
        participant: rtc.Participant | str,
        language: str = "",
        track_id: str | None = None,
        speed: float = 4,
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
    ):
        """
        Forward TTS transcription to the users. This function tries to imitate the right timing of
        speech with the synthesized text. The first estimation is based on the speed parameter. Once
        we have received all the audio of a specific text segment, we recalculate the avg speech
        speed using the length of the text & audio.

        The word_tokenizer is used to forward the transcription word by word.
        The sentence_tokenizer is used to split the streamted text into multiple transcription
        segments.
        hyphenate_word is used to get a better approximation of how much time a word takes to say.
        """
        identity = participant if isinstance(participant, str) else participant.identity
        if track_id is None:
            track_id = self._find_micro_track_id(identity)

        return TTSSegmentsForwarder(
            TTSOptions(
                room=self._room,
                participant_identity=identity,
                track_id=track_id,
                language=language,
                speed=speed,
                word_tokenizer=word_tokenizer,
                sentence_tokenizer=sentence_tokenizer,
                hyphenate_word=hyphenate_word,
            )
        )

    def _find_micro_track_id(self, identity: str) -> str:
        p = self._room.participants_by_identity.get(identity)
        if identity == self._room.local_participant.identity:
            p = self._room.local_participant

        if p is None:
            raise ValueError(f"participant {identity} not found")

        # find first micro track
        track_id = None
        for track in p.tracks.values():
            if track.source == rtc.TrackSource.SOURCE_MICROPHONE:
                track_id = track.sid
                break

        if track_id is None:
            raise ValueError(f"participant {identity} does not have a microphone track")

        return track_id
