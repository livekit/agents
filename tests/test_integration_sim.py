# tests/test_integration_sim.py
import asyncio

from extensions.interrupt_handler.handler import decide_and_handle


class MockAgent:
    def __init__(self):
        self.interrupted = False

    async def interrupt_cb(self, transcript, confidence):
        self.interrupted = True


async def run_case(transcript, conf, agent_speaking):
    ag = MockAgent()
    result = await decide_and_handle(transcript, conf, agent_speaking, ag.interrupt_cb)
    return result, ag.interrupted


def test_agent_speaking_ignores_filler():
    res, interrupted = asyncio.run(run_case("uh umm", 0.8, True))
    assert res["action"] == "ignored"
    assert not interrupted


def test_agent_speaking_stops_on_command():
    res, interrupted = asyncio.run(run_case("stop", 0.9, True))
    assert res["action"] == "interrupt"
    assert interrupted


def test_agent_silent_user_speech_counts():
    res, interrupted = asyncio.run(run_case("uh umm", 0.8, False))
    assert res["action"] == "interrupt"
    assert interrupted
