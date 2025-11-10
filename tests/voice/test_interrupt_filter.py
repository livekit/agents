import os
import importlib
import contextlib
from typing import Iterator

import pytest


@contextlib.contextmanager
def env_overrides(**env: str) -> Iterator[None]:
    prev = {}
    try:
        for k, v in env.items():
            prev[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fresh_classifier(**env):
    with env_overrides(**env):
        mod = importlib.import_module("livekit.agents.voice.interrupt_filter")
        importlib.reload(mod)
        return mod.InterruptionClassifier.from_env()


@pytest.mark.parametrize(
    "transcript,confidence,agent_speaking,expected",
    [
        ("uh", 0.2, True, "ignore_filler"),
        ("umm", 0.9, True, "ignore_filler"),
        ("hmm yeah", 0.3, True, "ignore_filler"),
        ("okay stop", 0.3, True, "real_interrupt"),
        ("wait one second", 0.5, True, "real_interrupt"),
        ("umm", 0.9, False, "passive"),
        ("can you help me", 0.8, True, "real_interrupt"),
    ],
)
def test_classifier_basic(transcript, confidence, agent_speaking, expected):
    clf = fresh_classifier(
        AGENTS_IGNORED_FILLERS="uh,umm,um,hmm,haan",
        AGENTS_STOP_KEYWORDS="stop,wait,hold,hold on,pause",
        AGENTS_MIN_CONFIDENCE="0.6",
    )
    decision = clf.classify(
        transcript=transcript, confidence=confidence, agent_speaking=agent_speaking
    )
    assert decision.kind == expected


def test_classifier_env_overrides():
    # Override fillers and stop words
    clf = fresh_classifier(
        AGENTS_IGNORED_FILLERS="eh,ah",
        AGENTS_STOP_KEYWORDS="freeze",
        AGENTS_MIN_CONFIDENCE="0.7",
    )

    # Now default fillers like 'umm' are not ignored; should become real interrupt if speaking
    d1 = clf.classify(transcript="umm", confidence=0.9, agent_speaking=True)
    assert d1.kind == "real_interrupt"

    # Custom filler should be ignored during speech
    d2 = clf.classify(transcript="eh", confidence=0.9, agent_speaking=True)
    assert d2.kind == "ignore_filler"

    # Custom stop keyword should trigger interrupt
    d3 = clf.classify(transcript="please freeze now", confidence=0.2, agent_speaking=True)
    assert d3.kind == "real_interrupt"


def test_low_conf_murmur():
    clf = fresh_classifier(AGENTS_MIN_CONFIDENCE="0.8")
    # Low confidence short utterance mostly fillers should be ignored during speech
    d = clf.classify(transcript="hmm um", confidence=0.5, agent_speaking=True)
    assert d.kind == "ignore_filler"


def test_passive_mode_registers_speech():
    clf = fresh_classifier()
    # When agent not speaking, everything passes as passive
    d = clf.classify(transcript="umm", confidence=0.1, agent_speaking=False)
    assert d.kind == "passive"


def test_language_specific_sets_from_env():
    with env_overrides(
        AGENTS_IGNORED_FILLERS="um",  # default
        AGENTS_STOP_KEYWORDS="stop",
        AGENTS_IGNORED_FILLERS_HI="haan,um",
        AGENTS_STOP_KEYWORDS_HI="ruk,thamo",
    ):
        mod = importlib.import_module("livekit.agents.voice.interrupt_filter")
        importlib.reload(mod)
        clf = mod.InterruptionClassifier.from_env()

    # English default: 'umm' not in default (we set 'um'), so contentful
    d_en = clf.classify(transcript="umm", confidence=0.9, agent_speaking=True, language="en")
    assert d_en.kind == "real_interrupt"

    # Hindi: 'haan' should be treated as filler
    d_hi = clf.classify(transcript="haan", confidence=0.9, agent_speaking=True, language="hi")
    assert d_hi.kind == "ignore_filler"

    # Hindi stop word triggers interrupt
    d_hi_stop = clf.classify(
        transcript="kripya ruk jaye", confidence=0.3, agent_speaking=True, language="hi"
    )
    assert d_hi_stop.kind == "real_interrupt"


def test_runtime_hot_reload_language_overrides():
    clf = fresh_classifier(
        AGENTS_IGNORED_FILLERS="um",
        AGENTS_STOP_KEYWORDS="stop",
    )

    # Initially, 'haan' is not a filler in default or any lang
    d1 = clf.classify(transcript="haan", confidence=0.9, agent_speaking=True, language="hi")
    assert d1.kind == "real_interrupt"

    # Hot-reload: add Hindi fillers and stop keywords
    clf.update_fillers(["haan", "hmm"], language="hi")
    clf.update_stop_keywords(["ruk"], language="hi")

    d2 = clf.classify(transcript="haan", confidence=0.9, agent_speaking=True, language="hi")
    assert d2.kind == "ignore_filler"

    d3 = clf.classify(transcript="kripya ruk", confidence=0.4, agent_speaking=True, language="hi")
    assert d3.kind == "real_interrupt"
