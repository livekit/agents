# SPDX-License-Identifier: Apache-2.0
"""Prism session adapter. The installed LiveKit core remains unchanged."""

from __future__ import annotations

import asyncio
from typing import Any, Generic, Literal, TypeVar

from opentelemetry import context as otel_context

from livekit.agents import Agent, AgentSession as CoreAgentSession
from livekit.agents.llm import AgentHandoff
from livekit.agents.log import logger
from livekit.agents.voice.agent_activity import _ReusableResources
from livekit.agents.voice.events import ConversationItemAddedEvent

from ._realtime_activity import RimeAgentActivity

Userdata_T = TypeVar("Userdata_T")


class AgentSession(CoreAgentSession[Userdata_T], Generic[Userdata_T]):
    """LiveKit session with Rime-owned typed turns, playback and recovery.

    Use this class with Prism in place of livekit.agents.AgentSession. Agent,
    AgentTask, tools, room I/O and the public session methods are unchanged.
    """

    async def _update_activity(
        self,
        agent: Agent,
        *,
        previous_activity: Literal["close", "pause"] = "close",
        new_activity: Literal["start", "resume"] = "start",
        blocked_tasks: list[asyncio.Task[Any]] | None = None,
        wait_on_enter: bool = True,
    ) -> None:
        async with self._activity_lock:
            if self._closing and new_activity == "start":
                # checked again after the drain below: closing may start while it's in flight
                logger.warning(
                    f"session is closing, skipping start activity of agent {agent.id}",
                )
                return

            # _update_activity is called directly sometimes, update for redundancy
            self._agent = agent

            if new_activity == "start":
                previous_agent = self._activity.agent if self._activity else None
                if agent._activity is not None and (
                    # allow updating the same agent that is running
                    agent is not previous_agent or previous_activity != "close"
                ):
                    raise RuntimeError("cannot start agent: an activity is already running")

                self._next_activity = RimeAgentActivity(agent, self)
            elif new_activity == "resume":
                if agent._activity is None:
                    raise RuntimeError("cannot resume agent: no existing active activity to resume")

                self._next_activity = agent._activity

            if self._root_span_context is not None:
                # restore the root span context so on_exit, on_enter, and future turns
                # are direct children of the root span, not nested under a tool call.
                otel_context.attach(self._root_span_context)

            reuse_resources: _ReusableResources | None = None
            try:
                previous_activity_v = self._activity
                if (activity := self._activity) is not None:
                    if previous_activity == "close":
                        activity._new_turns_blocked = True
                        reuse_resources = await activity.drain(new_activity=self._next_activity)
                        await activity.aclose()
                    elif previous_activity == "pause":
                        reuse_resources = await activity.pause(
                            blocked_tasks=blocked_tasks or [],
                            new_activity=self._next_activity,
                        )

                if self._closing and new_activity == "start":
                    # disallow starting a new activity when the session is closing
                    logger.warning(
                        f"session is closing, skipping {new_activity} activity of {self._next_activity.agent.id}",
                    )
                    if reuse_resources is not None:
                        await reuse_resources.cleanup()
                        reuse_resources = None
                    self._next_activity = None
                    self._activity = None
                    return

                self._activity = self._next_activity
                self._next_activity = None

                run_state = self._global_run_state
                handoff_item = AgentHandoff(
                    old_agent_id=(previous_activity_v.agent.id if previous_activity_v else None),
                    new_agent_id=self._activity.agent.id,
                )
                if run_state:
                    run_state._agent_handoff(
                        item=handoff_item,
                        old_agent=(previous_activity_v.agent if previous_activity_v else None),
                        new_agent=self._activity.agent,
                    )
                self._chat_ctx.insert(handoff_item)
                self.emit(
                    "conversation_item_added",
                    ConversationItemAddedEvent(item=handoff_item),
                )

                if new_activity == "start":
                    await self._activity.start(reuse_resources=reuse_resources)
                elif new_activity == "resume":
                    await self._activity.resume(reuse_resources=reuse_resources)
            except BaseException:
                if reuse_resources is not None:
                    await reuse_resources.cleanup()
                raise

        # move it outside the lock to allow calling _update_activity in on_enter of a new agent
        if wait_on_enter:
            assert self._activity._on_enter_task is not None
            await asyncio.shield(self._activity._on_enter_task)
