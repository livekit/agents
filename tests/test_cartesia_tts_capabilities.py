"""Hermetic tests for cartesia's aligned_transcript capability (#6493).

``word_timestamps=True`` (the default) sets ``capabilities.aligned_transcript``
at construction, but only some model/language combinations actually return
timing data. The capability used to stay set for the ones that don't, so every
turn downstream selected the TTS-aligned transcript path and logged "no agent
transcript was returned from tts".

Which combinations are supported is the provider's business and changes over
time, so it is learned from the responses instead of being declared here.
"""

import logging

import pytest

from livekit.plugins.cartesia import TTS

pytestmark = pytest.mark.unit


class TestConstructionDoesNotPrejudge:
    """No model/language is refused up front - a stale table here would block
    combinations the provider has since added support for."""

    def test_capability_set_for_any_language(self) -> None:
        for language in ("en", "hi", "tr", None):
            t = TTS(api_key="test-key", language=language)
            assert t.capabilities.aligned_transcript is True
            assert t._opts.word_timestamps is True

    def test_capability_follows_an_explicit_opt_out(self) -> None:
        t = TTS(api_key="test-key", language="en", word_timestamps=False)
        assert t.capabilities.aligned_transcript is False


class TestLearningFromTheResponse:
    def test_clears_the_capability_and_stops_requesting(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        t = TTS(api_key="test-key", language="hi")

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
            t._word_timestamps_unavailable(model=t._opts.model, language="hi")

        # downstream must stop reading the transcript from the TTS alignment
        assert t.capabilities.aligned_transcript is False
        # and the next request must not ask for what never arrives
        assert t._opts.word_timestamps is False
        assert any("no word timestamps" in r.message for r in caplog.records)

    def test_reports_the_combination_it_learned_about(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        t = TTS(api_key="test-key", model="sonic-3", language="hi")

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
            t._word_timestamps_unavailable(model=t._opts.model, language="hi")

        record = next(r for r in caplog.records if "no word timestamps" in r.message)
        assert "sonic-3" in record.getMessage()
        assert "hi" in record.getMessage()

    def test_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        t = TTS(api_key="test-key", language="hi")

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
            t._word_timestamps_unavailable(model=t._opts.model, language="hi")
            t._word_timestamps_unavailable(model=t._opts.model, language="hi")

        assert len([r for r in caplog.records if "no word timestamps" in r.message]) == 1

    def test_is_a_no_op_when_never_requested(self, caplog: pytest.LogCaptureFixture) -> None:
        t = TTS(api_key="test-key", language="en", word_timestamps=False)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
            t._word_timestamps_unavailable(model=t._opts.model, language="en")

        assert not caplog.records

    def test_learns_about_the_combination_the_request_used(self) -> None:
        # a request is sent with a snapshot of the options; update_options may
        # have moved the instance on before the response comes back, and the
        # result belongs to the combination that was actually used
        t = TTS(api_key="test-key", model="sonic-3", language="hi")
        t.update_options(language="en")

        t._word_timestamps_unavailable(model="sonic-3", language="hi")

        # the combination in use now was never the one that came back empty
        assert t.capabilities.aligned_transcript is True
        assert t._opts.word_timestamps is True
        # and going back to it applies what was learned
        t.update_options(language="hi")
        assert t.capabilities.aligned_transcript is False


class TestLearnedPerCombination:
    """What the responses taught us is about one model/language pair, so it
    must not outlive a switch to another one."""

    def test_changing_language_re_enables(self) -> None:
        t = TTS(api_key="test-key", language="hi")
        t._word_timestamps_unavailable(model=t._opts.model, language="hi")
        assert t.capabilities.aligned_transcript is False

        t.update_options(language="en")

        assert t.capabilities.aligned_transcript is True
        assert t._opts.word_timestamps is True

    def test_changing_model_re_enables(self) -> None:
        t = TTS(api_key="test-key", model="sonic-3", language="hi")
        t._word_timestamps_unavailable(model=t._opts.model, language="hi")

        t.update_options(model="sonic-preview")

        assert t.capabilities.aligned_transcript is True

    def test_switching_back_keeps_what_was_learned(self, caplog: pytest.LogCaptureFixture) -> None:
        t = TTS(api_key="test-key", language="hi")
        t._word_timestamps_unavailable(model=t._opts.model, language="hi")
        t.update_options(language="en")

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
            t.update_options(language="hi")

        # no second round-trip needed to rediscover it, and no second warning
        assert t.capabilities.aligned_transcript is False
        assert not [r for r in caplog.records if "no word timestamps" in r.message]

    def test_an_explicit_opt_out_survives_an_update(self) -> None:
        t = TTS(api_key="test-key", language="hi", word_timestamps=False)

        t.update_options(language="en")

        assert t.capabilities.aligned_transcript is False
        assert t._opts.word_timestamps is False

    def test_unrelated_updates_leave_it_alone(self) -> None:
        t = TTS(api_key="test-key", language="hi")
        t._word_timestamps_unavailable(model=t._opts.model, language="hi")

        t.update_options(voice="some-voice")

        assert t.capabilities.aligned_transcript is False
