import asyncio
import json
from types import SimpleNamespace

import pytest

from livekit.plugins.anam import director_note_cue_transform, director_notes as dn

pytestmark = pytest.mark.unit


_DEFAULT = object()  # call the transform with its own default for that lever
_ME = "agent-test"  # our (local) participant identity
_ENGINE = "anam-engine"  # the engine participant, publishing on behalf of us


def _runner(*, stripped_tags=_DEFAULT, forwarded_tags=_DEFAULT):
    """Build a transform plus a coroutine that pushes one segment through it.

    The fake room has the Anam engine present (a remote participant publishing on behalf of us), so
    auto-detection resolves immediately. Returns ``(run, cues)``; ``cues`` accumulates the raw
    published payload dicts (``topic`` and ``destination_identities`` folded in) across calls — the
    transform instance is reused, so multiple ``run`` calls exercise the per-segment reset.
    """
    cues: list[dict] = []

    class _LP:
        identity = _ME

        async def publish_data(self, payload, *, topic, reliable, destination_identities):  # noqa: ANN001
            d = json.loads(payload)
            d["topic"] = topic
            d["destination_identities"] = destination_identities
            cues.append(d)

    engine = SimpleNamespace(identity=_ENGINE, attributes={dn.ATTRIBUTE_PUBLISH_ON_BEHALF: _ME})
    room = SimpleNamespace(
        local_participant=_LP(),
        remote_participants={_ENGINE: engine},
        on=lambda *a, **k: None,
        off=lambda *a, **k: None,
    )

    kw = {}
    if stripped_tags is not _DEFAULT:
        kw["stripped_tags"] = stripped_tags
    if forwarded_tags is not _DEFAULT:
        kw["forwarded_tags"] = forwarded_tags
    transform = director_note_cue_transform(room, **kw)

    async def run(*chunks: str) -> str:
        async def _text():
            for c in chunks:
                yield c

        out = [piece async for piece in transform(_text())]
        # drain the fire-and-forget publish tasks before returning
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending)
        return "".join(out)

    return run, cues


def _drive(chunks: list[str], **kw):
    """One-shot: run `chunks` through a fresh transform; return (emitted_text, published cues)."""
    run, cues = _runner(**kw)
    return asyncio.run(run(*chunks)), cues


def _tags(cues: list[dict]) -> list[str]:
    return [c["cue"]["tag"] for c in cues]


def _offsets(cues: list[dict]) -> list[int]:
    return [c["char_offset"] for c in cues]


def test_strips_tag_and_offset_at_following_text() -> None:
    emitted, cues = _drive(["[happy] Hi there"])
    assert emitted == " Hi there"
    assert _tags(cues) == ["happy"]
    assert _offsets(cues) == [0]  # the next emitted char after the stripped tag
    assert cues[0]["topic"] == "director_note_cue"
    assert cues[0]["destination_identities"] == [_ENGINE]  # auto-targeted at the engine


def test_tag_mid_text_offset() -> None:
    emitted, cues = _drive(["Hi [happy] there"])
    assert "[" not in emitted
    assert _tags(cues) == ["happy"]
    assert _offsets(cues) == [3]  # "Hi " == 3 emitted chars before the cue


def test_tag_split_across_chunks() -> None:
    emitted, cues = _drive(["Hi [hap", "py] there"])
    assert "[" not in emitted
    assert (_tags(cues), _offsets(cues)) == (["happy"], [3])


def test_consecutive_tags_share_offset() -> None:
    _, cues = _drive(["[happy][sad] hello"])
    assert _tags(cues) == ["happy", "sad"]
    assert _offsets(cues) == [0, 0]


def test_default_is_preset_cue_others_passthrough() -> None:
    # Both levers default to the presets: preset -> stripped + forwarded; everything else passes
    # through to the TTS.
    emitted, cues = _drive(["[happy] hi [whispers] bye"])
    assert "[happy]" not in emitted  # preset -> stripped + forwarded
    assert "[whispers]" in emitted  # non-preset (e.g. an ElevenLabs audio tag) left for the TTS
    assert _tags(cues) == ["happy"]


def test_forwarded_only_kept_and_forwarded() -> None:
    # Forwarded but not stripped: kept in the speech for the TTS AND sent to Anam.
    emitted, cues = _drive(
        ["[laughter] that's great"],
        stripped_tags=frozenset(),
        forwarded_tags=frozenset({"laughter"}),
    )
    assert "[laughter]" in emitted  # kept so the TTS also reacts
    assert _tags(cues) == ["laughter"]  # and forwarded to Anam


def test_forwarded_only_shifts_offset_past_kept_tag() -> None:
    # The kept tag is counted in the emitted text, so the offset points just past it.
    _, cues = _drive(
        ["[laughter] hello there"],
        stripped_tags=frozenset(),
        forwarded_tags=frozenset({"laughter"}),
    )
    assert _offsets(cues) == [len("[laughter]")]  # offset just after the kept tag


def test_cues_target_only_the_engine() -> None:
    # The engine (publish-on-behalf participant) is auto-detected from the room and every cue is
    # sent only to it — never broadcast to other participants.
    _, cues = _drive(["[happy] hi [sad] there"])
    assert _tags(cues) == ["happy", "sad"]
    assert all(c["destination_identities"] == [_ENGINE] for c in cues)


def test_both_empty_is_a_no_op() -> None:
    # Tag-preserving TTS (e.g. Cartesia sonic-3.5): nothing stripped, nothing forwarded.
    emitted, cues = _drive(
        ["hi [happy] there"], stripped_tags=frozenset(), forwarded_tags=frozenset()
    )
    assert emitted == "hi [happy] there"  # passthrough — tag left for the TTS
    assert cues == []


def test_stripped_only_removed_not_forwarded() -> None:
    # Stripped but not forwarded: gone from the speech, no cue sent.
    emitted, cues = _drive(
        ["[beep] hi"], stripped_tags=frozenset({"beep"}), forwarded_tags=frozenset()
    )
    assert emitted == " hi"
    assert cues == []


def test_stripped_none_strips_every_tag() -> None:
    emitted, cues = _drive(["[whatever] go"], stripped_tags=None, forwarded_tags=frozenset())
    assert "[" not in emitted
    assert cues == []


def test_forwarded_none_forwards_every_tag() -> None:
    emitted, cues = _drive(["[whatever] go"], stripped_tags=frozenset(), forwarded_tags=None)
    assert "[whatever]" in emitted  # not stripped -> kept for the TTS
    assert _tags(cues) == ["whatever"]  # forwarded anyway


def test_literal_bracket_with_space_passes_through() -> None:
    emitted, cues = _drive(["see [ 5 ] ref"])
    assert emitted == "see [ 5 ] ref"  # not a well-formed tag — untouched
    assert cues == []


def test_offsets_are_per_segment() -> None:
    # Each transform invocation is its own coordinate space: offsets reset, no cross-call state.
    run, cues = _runner()

    async def _go():
        await run("[happy] a")  # happy@0
        await run("xx [sad] b")  # new call resets; "xx " == 3 -> sad@3

    asyncio.run(_go())
    assert _tags(cues) == ["happy", "sad"]
    assert _offsets(cues) == [0, 3]
