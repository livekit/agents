import asyncio
import json
import os
import time
from pprint import pformat

import pytest

# Enable RunResult debug output in `run_result.py`.
os.environ.setdefault("LIVEKIT_EVALS_VERBOSE", "1")

from livekit.agents import AgentSession, inference, llm

from .mock_patient_taskgroup_agent import MockPatientTaskGroupAgent, UserData

DEBUG_LOG_PATH = "/Users/toubatbrian/Documents/agents-js/.cursor/debug-e6d38d.log"


def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "e6d38d",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _pretty_events(events: list[object]) -> str:
    rows: list[dict[str, object]] = []
    for idx, event in enumerate(events):
        row: dict[str, object] = {"index": idx, "type": getattr(event, "type", "<unknown>")}

        item = getattr(event, "item", None)
        if item is not None and hasattr(item, "model_dump"):
            row["item"] = item.model_dump(
                exclude_none=True,
                exclude_defaults=True,
                exclude={"id", "call_id", "created_at"},
            )

        old_agent = getattr(event, "old_agent", None)
        new_agent = getattr(event, "new_agent", None)
        if old_agent is not None or new_agent is not None:
            row["handoff"] = {
                "old_agent": type(old_agent).__name__ if old_agent is not None else None,
                "new_agent": type(new_agent).__name__ if new_agent is not None else None,
            }

        rows.append(row)

    return pformat(rows, width=100, compact=False, sort_dicts=False)


def _llm_model() -> llm.LLM:
    return inference.LLM(
        model="openai/gpt-4.1",
        extra_kwargs={"temperature": 0.2},
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is required")
async def test_taskgroup_handoff_visibility_with_delay() -> None:
    async def run_first_turn(
        *, startup_delay: float, delay_after_wait: float
    ) -> tuple[list[str], list[object]]:
        async with _llm_model() as model, AgentSession(llm=model, userdata=UserData()) as sess:
            await sess.start(MockPatientTaskGroupAgent())
            # Optionally give on_enter TaskGroup time to start first task.
            if startup_delay > 0:
                await asyncio.sleep(startup_delay)

            result = await sess.run(user_input="Yes, please. Also who are you?")
            if delay_after_wait > 0:
                await asyncio.sleep(delay_after_wait)

            # Accessing `expect` triggers LIVEKIT_EVALS_VERBOSE debug output.

            event_types = [event.type for event in result.events]
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H4",
                location="test_mock_patient_taskgroup_agent.py:90",
                message="run1 events captured",
                data={
                    "startup_delay": startup_delay,
                    "delay_after_wait": delay_after_wait,
                    "event_types": event_types,
                },
            )
            # endregion
            print(
                f"\nPython run #1 events (startup_delay={startup_delay}s, delay_after_wait={delay_after_wait}s):"
            )
            print(_pretty_events(list(result.events)))

            # The first run may either call a tool immediately or ask a follow-up question first.
            if "function_call" in event_types:
                result.expect.contains_function_call(name="verify_intent")
            await asyncio.sleep(0.5)

            try:
                result2 = await sess.run(
                    user_input="My name is Alice Johnson and my date of birth is 1991-04-11."
                )
                print(
                    f"\nPython run #2 events (startup_delay={startup_delay}s, delay_after_wait={delay_after_wait}s):"
                )
                print(_pretty_events(list(result2.events)))
                # region agent log
                _debug_log(
                    run_id="pre-fix",
                    hypothesis_id="H5",
                    location="test_mock_patient_taskgroup_agent.py:112",
                    message="run2 events captured",
                    data={
                        "startup_delay": startup_delay,
                        "delay_after_wait": delay_after_wait,
                        "event_types": [event.type for event in result2.events],
                    },
                )
                # endregion
            except RuntimeError as e:
                print(
                    f"\nPython run #2 raised RuntimeError "
                    f"(startup_delay={startup_delay}s, delay_after_wait={delay_after_wait}s): {e}"
                )

            return event_types, list(result.events)

    no_delay_types, no_delay_events = await run_first_turn(startup_delay=1.0, delay_after_wait=0.0)
    delayed_types, delayed_events = await run_first_turn(startup_delay=1.0, delay_after_wait=1.0)
    cold_start_types, cold_start_events = await run_first_turn(
        startup_delay=0.0, delay_after_wait=0.0
    )

    assert len(no_delay_events) > 0
    assert len(delayed_events) > 0
    assert len(cold_start_events) > 0
    assert len(cold_start_types) > 0
    assert len(no_delay_types) > 0
    assert len(delayed_types) > 0
