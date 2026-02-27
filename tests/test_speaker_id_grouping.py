"""
Set of tests to validate that speaker ID information is being processed.
The `SpeechData` class is extended to support the `text_formatted` function
which wraps the object's text with the speaker ID. The example uses the
format `[SPEAKER_ID]TEXT[/SPEAKER_ID]` for testing.
"""

import re

from livekit.agents import Language, stt


class TestSpeakerIdGrouping:
    """Test cases for speaker ID grouping functionality."""

    def _format_text(self, text, speaker_id):
        if speaker_id:
            return f"[{speaker_id}]{text}[/{speaker_id}]"
        return text

    def _process_fragments(self, fragments):
        """Helper method to process a list of (text, speaker_id) fragments."""
        result = ""
        for text, speaker_id in fragments:
            # Skip speakers to ignore
            if re.match(r"^__[A-Z0-9_]{2,}__$", speaker_id):
                continue

            # Create a SpeakerSpeechData object and get formatted text
            speech_data = stt.SpeechData(
                text=self._format_text(text, speaker_id),
                speaker_id=speaker_id,
                language=Language("en"),
                start_time=0,
                end_time=0,
                confidence=1.0,
            )

            # Add the text
            processed = speech_data.text

            # Concatenate to the result
            if processed:
                if result:
                    result += f" {processed}"
                else:
                    result = processed
        return result

    def test_single_speaker_fragment(self):
        """Test a single fragment from a single speaker."""
        fragments = [("Hello", "S1")]
        result = self._process_fragments(fragments)
        assert result == "[S1]Hello[/S1]"

    def test_single_speaker_fragments(self):
        """Test multiple consecutive fragments from a single speaker."""
        fragments = [
            ("In making reservations.", "S1"),
        ]
        result = self._process_fragments(fragments)
        assert result == "[S1]In making reservations.[/S1]"

    def test_two_speakers_simple_alternation(self):
        """Test simple alternation between two speakers."""
        fragments = [
            ("Hello!", "S1"),
            ("Hi.", "S2"),
            ("How are you?", "S1"),
            ("Good thanks!    ", "S2"),
        ]
        result = self._process_fragments(fragments)
        assert result == "[S1]Hello![/S1] [S2]Hi.[/S2] [S1]How are you?[/S1] [S2]Good thanks![/S2]"

    def test_three_speakers_rapid_switching(self):
        """Test rapid switching between three speakers."""
        fragments = [
            ("One", "S1"),
            ("Two", "S2"),
            ("Three", "S3"),
            ("Four", "S1"),
            ("Five", "S2"),
            ("Six", "S3"),
            ("Seven", "S1"),
            ("Eight", "S2"),
            ("Nine", "S3"),
        ]
        result = self._process_fragments(fragments)
        assert result == (
            "[S1]One[/S1] [S2]Two[/S2] [S3]Three[/S3] [S1]Four[/S1] "
            "[S2]Five[/S2] [S3]Six[/S3] [S1]Seven[/S1] [S2]Eight[/S2] "
            "[S3]Nine[/S3]"
        )

    def test_none_speaker_id(self):
        """Test handling fragments with None speaker_id."""
        fragments = [
            ("Hello world", None),  # No speaker ID
            ("How are you?", "S1"),
            ("Good thanks!", None),  # No speaker ID
        ]
        result = self._process_fragments(fragments)
        assert result == "Hello world [S1]How are you?[/S1] Good thanks!"

    def test_numeric_and_string_speaker_ids(self):
        """Test various speaker ID formats."""
        fragments = [
            ("One Two", "1"),
            ("Three Four", "SPEAKER_A"),
            ("Five Six", "User 123"),
        ]
        result = self._process_fragments(fragments)
        assert result == (
            "[1]One Two[/1] [SPEAKER_A]Three Four[/SPEAKER_A] [User 123]Five Six[/User 123]"
        )

    def test_ignored_speaker_ids(self):
        """Test speakers are ignored."""
        fragments = [
            ("One Two", "1"),
            ("Three Four", "__ASSISTANT__"),
            ("Five Six", "USER_123"),
            ("Seven Eight", "__IGNORE_ME__"),
            ("Nine Ten", "__But_Not_Me__"),
        ]
        result = self._process_fragments(fragments)
        assert result == (
            "[1]One Two[/1] [USER_123]Five Six[/USER_123] [__But_Not_Me__]Nine Ten[/__But_Not_Me__]"
        )
