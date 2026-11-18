"""The conversations `examples/delegation-voice` promises, run against the example itself.

The example (`main.py`: the voice frontend and its backend on one server) runs unmodified under
a provider mode: scripted models (`scripts/delegation_voice.py`) and tone speech. It runs
against the configured LiveKit project (`LIVEKIT_URL`, key and secret): the call's room, the
endpoint routing that carries the frontend's delegation, and AgentDB are the project's.

Each test asserts what the caller hears, the backend's A2A tasks and the human messages that
reached the backend, and the business effects of its tools (each call's output, as the
backend's model read it); never model wording beyond the scripts' fixed phrases. The backend's
tools wait on real time (`asyncio.sleep`), so a conversation lasts tens of seconds; every wait
is bounded by an observation.

`--live-models` runs the first conversation with the example's own models.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import APIConnectOptions
from livekit.agents.a2a import AgentClient, AgentThreadGrant, Credentials
from livekit.agents.storage import AgentDBStorage, Thread

from .framework.app import E2E, ROOT, RunningApp
from .framework.models import live_model_environment
from .framework.providers import Providers
from .framework.voice import VoiceClient
from .scripts.delegation_voice import (
    AUDIT,
    CLARIFY,
    EXPORT,
    GREETING,
    NOT_SUBMITTED,
    PHRASES,
    SIDE_ANSWER,
)

EXAMPLE = ROOT / "examples" / "delegation-voice"
SCRIPT = "tests.e2e.scripts.delegation_voice"
JOB_ID = re.compile(r"Your job ID is (JOB-\d{4})\.")
MERGED_INTO = "livekit/mergedInto"

# The example's own words.
PRIORITY = "Should it run at normal or high priority?"


def read_back(description: str, priority: str) -> str:
    return f"Submit '{description}' at {priority} priority?"


def quarter_question(topic: str) -> str:
    return f"Which quarter should I search for '{topic}'?"


def found(topic: str, quarter: str) -> str:
    return f"3 {quarter} documents about '{topic}'; the most recent is from last week"


def submitted(description: str, priority: str) -> str:
    """The scripted backend's reply once `submit_job` ran: no job ID, which is spoken exactly."""
    return f"'{description}' is submitted at {priority} priority."


class Call:
    """The example on a call: what the caller heard, and what its models were asked."""

    def __init__(self, e2e: E2E, app: RunningApp, caller: VoiceClient, thread_id: str) -> None:
        self.e2e, self.app, self.caller, self.thread_id = e2e, app, caller, thread_id
        self.heard: list[str] = []

    async def say(self, text: str) -> None:
        assert text in PHRASES.values(), f"{text!r} has no tone"
        await self.caller.say(text)

    async def hear(self, text: str, timeout: float = 60) -> None:
        await self.hear_until(lambda: text in self.heard, f"never heard {text!r}", timeout)

    async def hear_until(self, done: Callable[[], bool], what: str, timeout: float = 60) -> None:
        try:
            async with asyncio.timeout(timeout):
                while not done():
                    try:
                        speech = await self.caller.hear()
                    except TimeoutError:
                        continue  # a quiet stretch while the backend works
                    assert speech.audio_seconds > 0, f"{speech.text!r} reached no audio"
                    self.heard.append(speech.text)
        except TimeoutError:
            raise AssertionError(f"{what}; heard {self.heard}") from None

    def job_ids(self) -> list[str]:
        return [m.group(1) for text in self.heard if (m := JOB_ID.fullmatch(text))]

    def requests(self) -> list[dict[str, Any]]:
        log = self.app.directory / "models.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def backend_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests() if "prepare_job" in r["tools"]]

    def forwarded(self) -> list[str]:
        """Every human message that reached the backend, in its conversation's order."""
        requests = self.backend_requests()
        if not requests:
            return []
        longest = max(requests, key=lambda r: len(r["items"]))
        return [
            item["text"]
            for item in longest["items"]
            if item["type"] == "message" and item["role"] == "user"
        ]

    def outputs(self, tool: str) -> list[str]:
        """The final output of each `tool` call the backend ran: its business effect."""
        seen: dict[str, str] = {}
        for request in self.backend_requests():
            calls = {
                item["call_id"]: item["name"] for item in request["items"] if item["type"] == "call"
            }
            for item in request["items"]:
                if item["type"] == "output" and calls.get(item["call_id"]) == tool:
                    seen[item["call_id"]] = item["output"]
        finals = {
            "prepare_job": " is ready at ",
            "submit_job": "submitted ",
            "look_up": " documents about ",
        }[tool]
        return [output for output in seen.values() if finals in output]

    async def tasks(self) -> dict[str, str]:
        """The backend's A2A tasks in the call's thread: each one's first request -> its state."""
        storage = AgentDBStorage()
        try:
            records = await Thread(storage, self.thread_id).partition("backend").tasks()
        finally:
            await storage.aclose()
        states: dict[str, str] = {}
        for record in sorted(records, key=lambda r: r.pos):
            first = record.data.history[0] if record.data.history else {}
            parts = first.get("parts", []) if isinstance(first, dict) else []
            states["".join(p.get("text", "") for p in parts)] = record.status
        return states

    def check_scripts(self) -> None:
        unexpected = [r for r in self.requests() if "unexpected" in r]
        assert not unexpected, f"a model got a request its script has no reply to: {unexpected[0]}"

    def write(self) -> None:
        (self.e2e.directory / "heard.json").write_text(json.dumps(self.heard, indent=2) + "\n")


async def start(e2e: E2E, *, live: bool = False) -> RunningApp:
    return await e2e.start_app(
        "main",
        voice=True,
        path=EXAMPLE,
        providers=Providers(SCRIPT, {"phrases": PHRASES}, live_models=live),
        env=live_model_environment() if live else None,
        project=True,
    )


@asynccontextmanager
async def conversation(e2e: E2E, *, live: bool = False) -> AsyncIterator[Call]:
    """The example on a call over its own durable thread, once the caller heard the greeting."""
    app = await start(e2e, live=live)
    storage = AgentDBStorage()
    thread_id = (await storage.threads.create_thread(scope="e2e-" + app.agent_name)).thread_id
    try:
        async with e2e.voice_client(app, phrases=PHRASES, thread_id=thread_id) as caller:
            call = Call(e2e, app, caller, thread_id)
            try:
                if live:
                    await call.hear_until(lambda: bool(call.heard), "never heard a greeting")
                else:
                    await call.hear(GREETING)
                # real wait: the example keeps the default 3 s echo-cancellation warmup after the
                # agent first speaks, during which the person's speech is not taken.
                await asyncio.sleep(3)
                yield call
                if not live:
                    call.check_scripts()
            finally:
                call.write()
    finally:
        await storage.threads.delete_thread(thread_id)
        await storage.aclose()


ANSWER_ON_LIVE_TASK = (
    "an answer to a backend task's question becomes a second task merged into the original "
    "(livekit/mergedInto) instead of coming back on that task, and the frontend's remote client "
    "rejects the foreign task id ('conflicting remote identity'): 'An internal error occurred'"
)
TWO_LIVE_TASKS = (
    "a second request while the first live backend task works breaks the first task's "
    "observation: its stream ends 'the A2A stream ended before an outcome' or times out, and "
    "reading it back fails 'task methods need the thread's contextId beside the task id'; "
    "then " + ANSWER_ON_LIVE_TASK
)


@pytest.mark.xfail(strict=True, reason=ANSWER_ON_LIVE_TASK)
async def test_a_job_reports_progress_asks_its_priority_reads_back_and_speaks_its_id_once(
    e2e: E2E,
) -> None:
    async with conversation(e2e) as call:
        await call.say("run the weekly export")
        await call.hear(PRIORITY)
        assert f"Update: Checking what '{EXPORT}' needs." in call.heard
        assert await call.tasks() == {"run the weekly export": "input-required"}

        await call.say("high")
        await call.hear(read_back(EXPORT, "high"))
        assert call.outputs("submit_job") == [], "nothing runs before the read-back"
        await call.say("yes")
        await call.hear(submitted(EXPORT, "high"))

        [job_id] = call.job_ids()
        assert call.outputs("submit_job") == [f"submitted '{EXPORT}' at high priority as {job_id}"]
        assert call.heard.index(f"Your job ID is {job_id}.") < call.heard.index(
            submitted(EXPORT, "high")
        )
        assert call.forwarded() == ["run the weekly export", "high", "yes"]
        assert await call.tasks() == {"run the weekly export": "completed"}


async def test_small_talk_while_a_job_works_stays_with_the_frontend_and_its_question_stays_open(
    e2e: E2E,
) -> None:
    async with conversation(e2e) as call:
        await call.say("run the weekly export")
        await call.hear(f"Update: Checking what '{EXPORT}' needs.")
        await call.say("what time is it")
        await call.hear(SIDE_ANSWER)
        assert PRIORITY not in call.heard, "the job was still preparing while the frontend talked"
        await call.hear(PRIORITY)

        heard = len(call.heard)
        await call.say("what time is it")
        await call.hear_until(
            lambda: call.heard[heard:].count(SIDE_ANSWER) == 1, "the second side question"
        )
        assert call.forwarded() == ["run the weekly export"], "small talk reached the backend"
        assert await call.tasks() == {"run the weekly export": "input-required"}


@pytest.mark.xfail(strict=True, reason=TWO_LIVE_TASKS)
async def test_an_answer_two_jobs_could_take_is_clarified_before_anything_is_sent(
    e2e: E2E,
) -> None:
    async with conversation(e2e) as call:
        await call.say("run the weekly export")
        await call.hear(f"Update: Checking what '{EXPORT}' needs.")
        await call.say("run the nightly backup")
        await call.hear_until(
            lambda: call.heard.count(PRIORITY) == 2, "both jobs asked for their priority"
        )

        await call.say("high")
        await call.hear(CLARIFY)
        assert call.forwarded() == ["run the weekly export", "run the nightly backup"]
        assert await call.tasks() == {
            "run the weekly export": "input-required",
            "run the nightly backup": "input-required",
        }

        await call.say("high for the weekly export")
        await call.hear(read_back(EXPORT, "high"))
        assert call.forwarded().count("high for the weekly export") == 1
        assert "high" not in call.forwarded()
        assert call.outputs("prepare_job") == [
            f"'{EXPORT}' is ready at high priority; submit it with submit_job now."
        ]
        assert (await call.tasks())["run the nightly backup"] == "input-required"


@pytest.mark.xfail(strict=True, reason=TWO_LIVE_TASKS)
async def test_a_partial_answer_settles_one_of_two_different_questions(e2e: E2E) -> None:
    async with conversation(e2e) as call:
        await call.say("run the weekly export")
        await call.hear(f"Update: Checking what '{EXPORT}' needs.")
        await call.say("look up the audit")
        await call.hear(quarter_question(AUDIT))
        await call.hear(PRIORITY)

        await call.say("Q3")
        await call.hear(found(AUDIT, "Q3"))
        assert call.outputs("look_up") == [found(AUDIT, "Q3")]
        assert call.outputs("prepare_job") == [], "the priority question is still open"
        assert call.forwarded().count("Q3") == 1
        assert await call.tasks() == {
            "run the weekly export": "input-required",
            "look up the audit": "completed",
        }


@pytest.mark.xfail(strict=True, reason=TWO_LIVE_TASKS)
async def test_one_sentence_answering_two_questions_reaches_each_task_once(e2e: E2E) -> None:
    async with conversation(e2e) as call:
        await call.say("run the weekly export")
        await call.hear(f"Update: Checking what '{EXPORT}' needs.")
        await call.say("look up the audit")
        await call.hear(quarter_question(AUDIT))
        await call.hear(PRIORITY)

        await call.say("high and Q3")
        await call.hear(found(AUDIT, "Q3"))
        await call.hear(read_back(EXPORT, "high"))
        assert call.forwarded().count("high and Q3") == 2, "once to each task"
        assert call.outputs("look_up") == [found(AUDIT, "Q3")]
        assert call.outputs("prepare_job") == [
            f"'{EXPORT}' is ready at high priority; submit it with submit_job now."
        ]
        assert (await call.tasks())["look up the audit"] == "completed"


@pytest.mark.parametrize("when", ["before dispatch", "after dispatch"])
@pytest.mark.xfail(strict=True, reason=ANSWER_ON_LIVE_TASK)
async def test_a_correction_replaces_the_answer_or_follows_it_as_a_correction(
    e2e: E2E, when: str
) -> None:
    async with conversation(e2e) as call:
        await call.say("run the weekly export")
        await call.hear(PRIORITY)
        if when == "before dispatch":
            # The frontend is still thinking about "high hold on" when the person corrects it.
            await call.say("high hold on")
            await call.say("actually normal")
            await call.hear(read_back(EXPORT, "normal"))
            assert "high hold on" not in call.forwarded(), "the replaced answer was never sent"
            assert call.forwarded().count("actually normal") == 1
        else:
            await call.say("high")
            await call.hear(read_back(EXPORT, "high"))
            await call.say("actually normal")
            await call.hear(read_back(EXPORT, "normal"))
            assert call.forwarded()[-2:] == ["high", "actually normal"]
        await call.say("yes")
        await call.hear(submitted(EXPORT, "normal"))
        [job_id] = call.job_ids()
        assert call.outputs("submit_job") == [
            f"submitted '{EXPORT}' at normal priority as {job_id}"
        ], "submitted once, at the corrected priority"


@pytest.mark.xfail(strict=True, reason=ANSWER_ON_LIVE_TASK)
async def test_no_to_the_read_back_submits_nothing(e2e: E2E) -> None:
    async with conversation(e2e) as call:
        await call.say("run the weekly export")
        await call.hear(PRIORITY)
        await call.say("high")
        await call.hear(read_back(EXPORT, "high"))
        await call.say("no")
        await call.hear(NOT_SUBMITTED)
        assert call.outputs("submit_job") == []
        assert call.job_ids() == []
        assert await call.tasks() == {"run the weekly export": "completed"}


async def test_text_messages_while_the_backend_works_are_merged_and_each_is_answered(
    e2e: E2E,
) -> None:
    """The `a2a` channel's batch inbox: two messages sent while a turn runs are answered by one
    next turn; the earlier one completes pointing at it."""
    app = await start(e2e)
    credentials = Credentials(app.env["LIVEKIT_API_KEY"], app.env["LIVEKIT_API_SECRET"])
    token = credentials.thread_grant_token(
        AgentThreadGrant(scope="e2e-" + app.agent_name, create=True, write=True, read=True)
    )
    # A merged message's stream is quiet until the turn that answers it ends.
    client = AgentClient(
        app.endpoint("backend"), token=token, conn_options=APIConnectOptions(read_timeout=60)
    )
    try:
        first = await client.send("look up the audit for Q1")
        first_updates = []
        async with asyncio.timeout(30):
            async for update in first:
                first_updates.append(update)
                if update.state == pb.TASK_STATE_WORKING and update.progress:
                    break  # its look_up runs: the backend is busy
        thread_id = first.thread_id

        async def send(text: str) -> list[Any]:
            reply = await client.send(text, thread_id=thread_id)
            async with asyncio.timeout(60):
                return [update async for update in reply]

        async with asyncio.timeout(60):
            budget, hiring = await asyncio.gather(
                send("look up the budget for Q2"), send("look up the hiring for Q4")
            )
            first_updates += [update async for update in first]

        assert first_updates[-1].state == pb.TASK_STATE_COMPLETED
        assert first_updates[-1].text == found("the audit", "Q1")
        merged, answered = sorted((budget, hiring), key=lambda u: MERGED_INTO in u[-1].metadata)[
            ::-1
        ]
        assert merged[-1].state == pb.TASK_STATE_COMPLETED
        assert merged[-1].metadata[MERGED_INTO] == answered[-1].task_id
        assert answered[-1].state == pb.TASK_STATE_COMPLETED
        assert found("the budget", "Q2") in answered[-1].text
        assert found("the hiring", "Q4") in answered[-1].text
        storage = AgentDBStorage()
        try:
            await storage.threads.delete_thread(thread_id)
        finally:
            await storage.aclose()
    finally:
        await client.aclose()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the frontend cannot cancel one remote task: its only tool is the backend, and a "
        "message to a working live task does not cancel it; and " + TWO_LIVE_TASKS
    ),
)
async def test_cancelling_one_task_leaves_the_other_running(e2e: E2E) -> None:
    async with conversation(e2e) as call:
        await call.say("run the weekly export")
        await call.hear(f"Update: Checking what '{EXPORT}' needs.")
        await call.say("look up the audit for Q3")
        await call.say("cancel the weekly export")
        await call.hear(found(AUDIT, "Q3"))
        tasks = await call.tasks()
        assert tasks["run the weekly export"] == "canceled"
        assert tasks["look up the audit for Q3"] == "completed"
        assert PRIORITY not in call.heard, "the cancelled job asked nothing more"
        assert call.outputs("prepare_job") == []


@pytest.mark.live_models
async def test_live_models_run_a_job_through_the_example(e2e: E2E) -> None:
    """The same example and journey with its own models: a conversation evaluation, so only the
    business outcome is asserted."""
    async with conversation(e2e, live=True) as call:
        await call.say("run the weekly export")
        await call.hear_until(lambda: len(call.heard) >= 3, "the job's question", timeout=90)
        await call.say("high")
        await call.hear_until(lambda: len(call.heard) >= 5, "the read-back", timeout=90)
        await call.say("yes")
        await call.hear_until(lambda: bool(call.job_ids()), "the job ID", timeout=90)
        assert len(call.job_ids()) == 1
