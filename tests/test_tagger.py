import pytest

from livekit.agents.observability import Tagger

pytestmark = pytest.mark.unit


def test_add_outcome_tag_replaces_previous_outcome() -> None:
    tagger = Tagger()
    tagger.fail(reason="booking failed")
    tagger.add("lk.success")

    assert tagger.tags == {"lk.success"}
    assert tagger.outcome == "success"
    assert tagger.outcome_reason is None

    tagger.add("lk.fail")

    assert tagger.tags == {"lk.fail"}
    assert tagger.outcome == "fail"


def test_remove_outcome_tag_clears_reason() -> None:
    tagger = Tagger()
    tagger.success(reason="done")
    tagger.remove("lk.fail")

    assert tagger.outcome == "success"
    assert tagger.outcome_reason == "done"

    tagger.remove("lk.success")

    assert tagger.tags == set()
    assert tagger.outcome is None
    assert tagger.outcome_reason is None


def test_custom_tags_do_not_affect_outcome() -> None:
    tagger = Tagger()
    tagger.fail(reason="user hung up")
    tagger.add("voicemail:true")
    tagger.remove("voicemail:true")

    assert tagger.tags == {"lk.fail"}
    assert tagger.outcome == "fail"
    assert tagger.outcome_reason == "user hung up"
