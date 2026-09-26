from __future__ import annotations

import pytest

from livekit.agents.observability import Tagger

pytestmark = pytest.mark.unit


def test_add_success_replaces_a_fail_outcome() -> None:
    tagger = Tagger()
    tagger.fail(reason="booking failed")

    tagger.add("lk.success")

    assert tagger.outcome == "success"
    assert tagger.tags == {"lk.success"}
    assert tagger.outcome_reason is None


def test_add_fail_replaces_a_success_outcome() -> None:
    tagger = Tagger()
    tagger.success(reason="task completed")

    tagger.add("lk.fail")

    assert tagger.outcome == "fail"
    assert tagger.tags == {"lk.fail"}
    assert tagger.outcome_reason is None


def test_add_keeps_custom_tags_out_of_the_outcome() -> None:
    tagger = Tagger()
    tagger.add("voicemail:true")
    tagger.add("language:es", metadata={"confidence": 0.9})

    assert tagger.outcome is None
    assert tagger.outcome_reason is None
    assert tagger.tags == {"voicemail:true", "language:es"}


def test_remove_outcome_tag_clears_the_reason() -> None:
    tagger = Tagger()
    tagger.success("done")

    tagger.remove("lk.success")

    assert tagger.outcome is None
    assert tagger.outcome_reason is None


def test_remove_of_an_absent_outcome_tag_keeps_the_reason() -> None:
    tagger = Tagger()
    tagger.success("done")

    tagger.remove("lk.fail")

    assert tagger.outcome == "success"
    assert tagger.outcome_reason == "done"


def test_remove_custom_tag_keeps_the_outcome() -> None:
    tagger = Tagger()
    tagger.fail(reason="user hung up")
    tagger.add("language:es")

    tagger.remove("language:es")

    assert tagger.outcome == "fail"
    assert tagger.outcome_reason == "user hung up"
    assert tagger.tags == {"lk.fail"}


def test_success_and_fail_stay_mutually_exclusive() -> None:
    tagger = Tagger()

    tagger.success(reason="first")
    assert (tagger.outcome, tagger.outcome_reason) == ("success", "first")

    tagger.fail(reason="second")
    assert (tagger.outcome, tagger.outcome_reason) == ("fail", "second")
    assert tagger.tags == {"lk.fail"}
