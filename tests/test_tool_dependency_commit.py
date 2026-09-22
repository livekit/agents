from __future__ import annotations

import asyncio
import gc
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import FunctionCall, RealtimeError

from .fake_io import FakeAudioOutput
from .fake_realtime import (
    FakeRealtimeModel,
    FakeRealtimeSession,
    fake_capabilities,
    generation as _generation,
)
from .tool_dependency_helpers import (
    close as _close,
    collect_terminals,
    wait as _wait,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _item_snapshot(chat_ctx: Any) -> list[tuple[str, str, str | None, str | None]]:
    return [
        (item.id, item.type, getattr(item, "call_id", None), getattr(item, "output", None))
        for item in chat_ctx.items
    ]


class _GatedRealtimeSession(FakeRealtimeSession):
    def __init__(
        self, model: _GatedRealtimeModel, *, turn_detection_disabled: bool = False
    ) -> None:
        super().__init__(model, turn_detection_disabled=turn_detection_disabled)
        self._model = model
        self.provider_updates: list[list[tuple[str, str, str | None, str | None]]] = []
        self.initial_provider_update_entered = asyncio.Event()
        self.release_initial_provider_update = asyncio.Event()
        self.initial_provider_commit_completed = asyncio.Event()
        self.final_provider_commit_completed = asyncio.Event()
        self.commit_log: list[tuple[str, str | None]] = []
        self._committed_item_ids: set[str] = set()

    def generate_reply(self, **kwargs: Any) -> asyncio.Future[Any]:
        future = super().generate_reply(**kwargs)
        index = len(self._reply_futs) - 1
        future.set_result(
            _generation(
                response_id=f"reply-{index}",
                text="acknowledged",
                audio_duration=0.01,
                function_calls=self._model.calls if index == 0 else [],
            )
        )
        return future

    async def update_chat_ctx(self, chat_ctx: Any) -> None:
        snapshot = _item_snapshot(chat_ctx)
        if not self.initial_provider_update_entered.is_set() and any(
            item[1] == "function_call_output" for item in snapshot
        ):
            self.initial_provider_update_entered.set()
            await self.release_initial_provider_update.wait()
        await super().update_chat_ctx(chat_ctx)
        self.provider_updates.append(snapshot)
        if any(item[1] == "function_call_output" for item in snapshot):
            self.initial_provider_commit_completed.set()
        for item_id, item_type, call_id, _ in snapshot:
            if item_id not in self._committed_item_ids:
                self.commit_log.append((item_type, call_id))
                self._committed_item_ids.add(item_id)
        if any(call_id == "meal_final" for _, _, call_id, _ in snapshot):
            self.final_provider_commit_completed.set()


class _GatedRealtimeModel(FakeRealtimeModel):
    def __init__(self, root_name: str = "save_room", dependent_name: str = "save_meal") -> None:
        super().__init__(capabilities=fake_capabilities(auto_tool_reply_generation=False))
        self.calls = [
            FunctionCall(call_id="meal", name=dependent_name, arguments="{}"),
            FunctionCall(call_id="root", name=root_name, arguments="{}"),
        ]

    def session(self, *, turn_detection_disabled: bool = False) -> _GatedRealtimeSession:
        session = _GatedRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        session.update_error = self.bring_up_error
        self.created_sessions.append(session)
        return session


@pytest.mark.asyncio
@pytest.mark.parametrize("close_during_commit", [False, True])
async def test_initial_provider_commit_gates_dependency_delivery(
    close_during_commit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _GatedRealtimeModel()
    session = AgentSession(llm=model)
    second_update_attempted = asyncio.Event()
    second_update_completed = asyncio.Event()
    close_task: asyncio.Task[None] | None = None
    existing_tasks = set(asyncio.all_tasks())

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext) -> str:
        return "room saved"

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        assert realtime_session is not None
        await realtime_session.initial_provider_update_entered.wait()
        await ctx.update("meal first progress")
        second_update_attempted.set()
        await ctx.update("meal second progress")
        second_update_completed.set()
        return "meal final"

    agent = Agent(instructions="booking", tools=[save_room, save_meal])
    realtime_session: _GatedRealtimeSession | None = None
    try:
        await session.start(agent)
        realtime_session = model.active_session
        assert realtime_session is not None

        reply = session.generate_reply()
        await _wait(realtime_session.initial_provider_update_entered)
        await _wait(second_update_attempted)

        # The initial provider commit is still in flight. Direct executor enqueue
        # must not complete the dependent's second update behind that commit.
        second_completed_probe = asyncio.create_task(second_update_completed.wait())
        done, _ = await asyncio.wait({second_completed_probe}, timeout=0.05)
        if not second_completed_probe.done():
            second_completed_probe.cancel()
            await asyncio.gather(second_completed_probe, return_exceptions=True)
        assert not done, "dependent second update completed before initial commit"

        if close_during_commit:
            abandon_completed = asyncio.Event()
            activity = session._activity
            assert activity is not None
            original_abandon = activity._abandon_dependency_schedulers

            async def observe_abandon() -> None:
                await original_abandon()
                abandon_completed.set()

            monkeypatch.setattr(activity, "_abandon_dependency_schedulers", observe_abandon)
            close_task = asyncio.create_task(session.aclose(), name="close_during_initial_commit")
            await _wait(abandon_completed)
            assert session._is_closing()
            assert not close_task.done()

            # The real provider call is released before measuring close completion, so a
            # close failure cannot be attributed to the deliberately blocked transport.
            realtime_session.release_initial_provider_update.set()
            await asyncio.wait_for(close_task, timeout=5)
        else:
            realtime_session.release_initial_provider_update.set()
            await _wait(second_update_completed)
            await _wait(realtime_session.final_provider_commit_completed)
            await asyncio.wait_for(session.wait_for_idle(), timeout=5)
            await asyncio.wait_for(reply.wait_for_playout(), timeout=5)

            assert realtime_session.provider_updates
            initial_update = next(
                snapshot
                for snapshot in realtime_session.provider_updates
                if any(item_type == "function_call_output" for _, item_type, _, _ in snapshot)
            )
            assert any(call_id == "meal" for _, _, call_id, _ in initial_update), (
                f"completed provider snapshots={realtime_session.provider_updates!r}; "
                f"commit_log={realtime_session.commit_log!r}"
            )
            committed_output_ids = [
                call_id
                for item_type, call_id in realtime_session.commit_log
                if item_type == "function_call_output"
            ]
            assert set(committed_output_ids[:2]) == {"root", "meal"}
            assert committed_output_ids[2:] == ["meal_update_0", "meal_update_1", "meal_final"]

            history_pairs = [
                item
                for item in session.history.items
                if item.type in ("function_call", "function_call_output")
            ]
            for call_id in ("meal_update_0", "meal_update_1", "meal_final"):
                assert sum(item.call_id == call_id for item in history_pairs) == 2
    finally:
        if realtime_session is not None:
            realtime_session.release_initial_provider_update.set()
        if close_task is not None:
            await asyncio.wait_for(close_task, timeout=5)
        else:
            await _close(session)
    assert not [
        task for task in asyncio.all_tasks() if task not in existing_tasks and not task.done()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["initial", "late", "persistent"])
async def test_provider_commit_failure_settles_dependency_delivery(
    failure: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unhandled.append(context))
    existing_tasks = set(asyncio.all_tasks())
    model = _GatedRealtimeModel("root", "meal")
    session = AgentSession(llm=model)
    session.output.audio = FakeAudioOutput()
    terminals = collect_terminals(session)
    dependent_done = asyncio.Event()
    attempts = 0
    failures = 0

    @function_tool(name="root")
    async def root(ctx: RunContext) -> str:
        return "root done"

    @function_tool(name="meal", after=("root",))
    async def meal(ctx: RunContext) -> str:
        try:
            await ctx.update("meal progress")
            return "meal final"
        finally:
            dependent_done.set()

    try:
        await session.start(Agent(instructions="booking", tools=[root, meal]))
        realtime = model.active_session
        assert isinstance(realtime, _GatedRealtimeSession)
        realtime.release_initial_provider_update.set()
        original_update = realtime.update_chat_ctx

        async def failing_update(chat_ctx: Any) -> None:
            nonlocal attempts, failures
            if any(item.type == "function_call_output" for item in chat_ctx.items):
                attempts += 1
                if (
                    failure == "persistent"
                    or (failure == "initial" and attempts == 1)
                    or (failure == "late" and attempts == 2)
                ):
                    failures += 1
                    raise RealtimeError("injected provider commit failure")
            await original_update(chat_ctx)

        monkeypatch.setattr(realtime, "update_chat_ctx", failing_update)
        speech = session.generate_reply()
        await _wait(dependent_done)
        await asyncio.wait_for(speech.wait_for_playout(), timeout=5)
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        assert failures > 0
        assert len(terminals["meal"]) == 1
        if failure == "persistent":
            assert terminals["meal"][0].status == "error"
            assert not any(item.type == "function_call_output" for item in realtime.chat_ctx.items)
        else:
            assert terminals["meal"][0].status == "done"
            assert any(
                item.type == "function_call_output" and item.call_id == "meal_final"
                for item in realtime.chat_ctx.items
            )
        if failure != "initial":
            assert "failed to deliver deferred tool output" in caplog.text
    finally:
        try:
            await _close(session)
        finally:
            gc.collect()
            await asyncio.sleep(0)
            loop.set_exception_handler(previous_handler)
    assert not unhandled
    assert not [
        task for task in asyncio.all_tasks() if task not in existing_tasks and not task.done()
    ]


@pytest.mark.asyncio
async def test_realtime_progress_dependency_waits_for_terminal_tool_result() -> None:
    existing_tasks = set(asyncio.all_tasks())
    model = _GatedRealtimeModel()
    session = AgentSession(llm=model)
    session.output.audio = FakeAudioOutput()
    terminals = collect_terminals(session)
    release_root = asyncio.Event()
    dependent_started = asyncio.Event()

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext) -> str:
        await ctx.update("room reservation is pending")
        await release_root.wait()
        return "room reservation completed"

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        assert len(terminals["root"]) == 1
        dependent_started.set()
        return "meal saved"

    try:
        await session.start(Agent(instructions="booking", tools=[save_room, save_meal]))
        realtime = model.active_session
        realtime.release_initial_provider_update.set()
        reply = session.generate_reply()
        await _wait(realtime.initial_provider_commit_completed)
        assert any(
            item.type == "function_call_output"
            and item.call_id == "root"
            and "room reservation is pending" in item.output
            for item in realtime.chat_ctx.items
        )
        assert not dependent_started.is_set()
        assert not terminals["root"]
        release_root.set()
        await _wait(realtime.final_provider_commit_completed)
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        await asyncio.wait_for(reply.wait_for_playout(), timeout=5)
        assert dependent_started.is_set()
        assert [event.status for event in terminals["root"]] == ["done"]
        assert [event.status for event in terminals["meal"]] == ["done"]
        assert any(
            item.type == "function_call_output"
            and item.call_id == "meal_final"
            and "meal saved" in item.output
            for item in realtime.chat_ctx.items
        )
    finally:
        release_root.set()
        await _close(session)
    assert not [
        task for task in asyncio.all_tasks() if task not in existing_tasks and not task.done()
    ]
