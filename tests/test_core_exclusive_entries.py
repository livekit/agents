"""A `_core` object whose method mutates it is borrowed exclusively for the call: calls from
several Python threads are serialized by the interpreter lock, never refused."""

import threading

import pytest

from livekit.agents import _core

pytestmark = pytest.mark.unit


# Catches: a `&mut self` entry releasing the GIL for its call, so a second thread's call finds the
# object mutably borrowed and raises instead of waiting its turn.
def test_threads_calling_one_mutating_object_take_turns() -> None:
    stripper = _core.TranscriptMarkupStripper()
    errors: list[BaseException] = []

    def push() -> None:
        try:
            for _ in range(20_000):
                stripper.push("say <emphasis>this</emphasis> ")
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=push) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
