from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import FunctionCall

from .fake_realtime import (
    FakeRealtimeModel,
    FakeRealtimeSession,
    fake_capabilities,
    generation as _generation,
)
from .tool_dependency_helpers import (
    await_chain as _await_chain,
    close as _close,
    resolve_fake_realtime_replies as _resolve_fake_realtime_replies,
    wait as _wait,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _item_snapshot(chat_ctx: Any) -> list[tuple[str, str, str | None, str | None]]:
    return [
        (item.id, item.type, getattr(item, "call_id", None), getattr(item, "output", None))
        for item in chat_ctx.items
    ]


class _GatedRealtimeSession(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel, *, turn_detection_disabled: bool = False) -> None:
        super().__init__(model, turn_detection_disabled=turn_detection_disabled)
        self.provider_updates: list[list[tuple[str, str, str | None, str | None]]] = []
        self.initial_provider_update_entered = asyncio.Event()
        self.release_initial_provider_update = asyncio.Event()
        self.reply_created = asyncio.Event()
        self.final_provider_commit_completed = asyncio.Event()
        self.commit_log: list[tuple[str, str | None]] = []
        self._committed_item_ids: set[str] = set()

    def generate_reply(self, **kwargs: Any) -> asyncio.Future[Any]:
        future = super().generate_reply(**kwargs)
        self.reply_created.set()
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
        for item_id, item_type, call_id, _ in snapshot:
            if item_id not in self._committed_item_ids:
                self.commit_log.append((item_type, call_id))
                self._committed_item_ids.add(item_id)
        if any(call_id == "meal_final" for _, _, call_id, _ in snapshot):
            self.final_provider_commit_completed.set()


class _GatedRealtimeModel(FakeRealtimeModel):
    def session(self, *, turn_detection_disabled: bool = False) -> _GatedRealtimeSession:
        session = _GatedRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        session.update_error = self.bring_up_error
        self.created_sessions.append(session)
        return session


@pytest.mark.asyncio
async def test_deferred_dependency_progress_does_not_overtake_initial_commit() -> None:
    model = _GatedRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))
    session = AgentSession(llm=model)
    second_update_attempted = asyncio.Event()
    second_update_completed = asyncio.Event()
    stop_reply_resolver = asyncio.Event()
    reply_resolver: asyncio.Task[None] | None = None

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
        await _wait(realtime_session.reply_created)
        realtime_session._reply_futs[0].set_result(
            _generation(
                response_id="dependency-commit-order",
                text="checking the room",
                audio_duration=0.01,
                function_calls=[
                    FunctionCall(call_id="meal", name="save_meal", arguments="{}"),
                    FunctionCall(call_id="root", name="save_room", arguments="{}"),
                ],
            )
        )
        realtime_session.reply_created.clear()
        reply_resolver = asyncio.create_task(
            _resolve_fake_realtime_replies(realtime_session, stop_reply_resolver)
        )

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
        try:
            await _close(session)
        finally:
            stop_reply_resolver.set()
            if reply_resolver is not None:
                if realtime_session is not None:
                    realtime_session.reply_created.set()
                await asyncio.wait_for(reply_resolver, timeout=5)


@pytest.mark.asyncio
async def test_close_during_initial_provider_commit_releases_dependency_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _GatedRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))
    session = AgentSession(llm=model)
    second_update_attempted = asyncio.Event()
    second_update_completed = asyncio.Event()
    abandon_started = asyncio.Event()
    abandon_completed = asyncio.Event()
    stop_reply_resolver = asyncio.Event()
    reply_resolver: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None

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

        session.generate_reply()
        await _wait(realtime_session.reply_created)
        realtime_session._reply_futs[0].set_result(
            _generation(
                response_id="dependency-close-order",
                text="checking the room",
                audio_duration=0.01,
                function_calls=[
                    FunctionCall(call_id="meal", name="save_meal", arguments="{}"),
                    FunctionCall(call_id="root", name="save_room", arguments="{}"),
                ],
            )
        )
        realtime_session.reply_created.clear()
        reply_resolver = asyncio.create_task(
            _resolve_fake_realtime_replies(realtime_session, stop_reply_resolver)
        )

        await _wait(realtime_session.initial_provider_update_entered)
        await _wait(second_update_attempted)
        activity = session._activity
        assert activity is not None
        original_abandon = activity._abandon_dependency_schedulers

        async def observe_abandon() -> None:
            abandon_started.set()
            await original_abandon()
            abandon_completed.set()

        monkeypatch.setattr(activity, "_abandon_dependency_schedulers", observe_abandon)
        close_task = asyncio.create_task(session.aclose(), name="close_during_initial_commit")
        await _wait(abandon_started)
        assert session._is_closing()
        assert not close_task.done()
        await _wait(abandon_completed)
        assert not close_task.done()

        # The real provider call is released before measuring close completion, so a
        # close failure cannot be attributed to the deliberately blocked transport.
        realtime_session.release_initial_provider_update.set()
        await asyncio.wait_for(close_task, timeout=5)
    finally:
        if realtime_session is not None:
            realtime_session.release_initial_provider_update.set()
        if close_task is None:
            await _close(session)
        elif not close_task.done():
            await asyncio.wait_for(close_task, timeout=5)
        else:
            close_task.result()
        stop_reply_resolver.set()
        if reply_resolver is not None:
            if realtime_session is not None:
                realtime_session.reply_created.set()
            await asyncio.wait_for(reply_resolver, timeout=5)

        leaked_tasks = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and (
                task.get_name().startswith(("tool_dependency_", "tool_exec_", "func_exec_"))
                or task.get_name() in {"execute_tools_task", "tool_dependency_ready"}
            )
        ]
        assert not leaked_tasks, "close left tool tasks: " + ", ".join(
            f"{task.get_name()}: {_await_chain(task)}" for task in leaked_tasks
        )
