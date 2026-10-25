import logging
from collections.abc import AsyncIterator
from typing import Any

import agent

from livekit.agents import (
    AgentHandler,
    AgentServer,
    ConversationItemAddedEvent,
    JobContext,
    LiveSession,
    ToolExecutionUpdatedEvent,
    cli,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatItem
from livekit.agents.storage import AgentDBStorage
from livekit.plugins import openai

log = logging.getLogger("review.frontend")

TURN_ITEMS = ("message", "function_call", "function_call_output")


def describe(item: ChatItem) -> str:
    """One chat item as `role:'text'`, `call:name(args)` or `out:name='output'`."""
    if item.type == "message":
        return f"{item.role}:{(item.text_content or '')[:120]!r}"
    if item.type == "function_call":
        return f"call:{item.name}({item.arguments})"
    if item.type == "function_call_output":
        return f"out:{item.name}={item.output[:120]!r}"
    return item.type


support = AgentHandler()  # the worker's own lane: the call and the text door share it


@support.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = LiveSession(llm=openai.realtime.RealtimeModel(), thread=ctx.thread)

    @session.on("tool_execution_updated")
    def _tool(ev: ToolExecutionUpdatedEvent) -> None:
        update = ev.update
        if update.type == "tool_call_started":
            call = update.function_call
            log.info(f"call tool started {call.call_id}: {call.name}({call.arguments})")
        elif update.type == "tool_call_updated":
            log.info(f"call tool updated {update.call_id}: {update.message!r}")
        elif update.type == "tool_call_ended":
            log.info(
                f"call tool ended {update.call_id}: status={update.status} "
                f"message={update.message!r}"
            )

    @session.on("conversation_item_added")
    def _item(ev: ConversationItemAddedEvent) -> None:
        log.info(f"call item added: {describe(ev.item)}")
        lane = session._live_lane  # diagnostic read: the call's lane and its version
        activity = session._activity  # diagnostic read: the copy of the history the model reads
        if activity is None:
            return
        history = [i.id for i in session.history.items if i.type in TURN_ITEMS]
        in_activity = {i.id for i in activity._chat_ctx.items}  # diagnostic read
        rt = activity._rt_session  # diagnostic read: the realtime model's own context
        in_model = {i.id for i in rt.chat_ctx.items} if rt is not None else in_activity
        log.info(
            f"call lane version={lane._version if lane else None} "  # diagnostic read
            f"history={len(history)} missing from the activity: "
            f"{[i for i in history if i not in in_activity]} missing from the model: "
            f"{[i for i in history if i not in in_model]}"
        )

    await session.start(agent=agent.voice, room=ctx.room)
    log.info(f"call started: thread={ctx.thread.id if ctx.thread else None}")

    lane = session._live_lane  # diagnostic read: wrapped below to log each voice commit
    if lane is not None:
        commit_request = lane._commit_request  # diagnostic read: the private commit builder

        def logged(held: Any, **kw: Any) -> Any:
            request = commit_request(held, **kw)
            log.info(
                f"voice commit: loaded_version={held.loaded.version} "
                f"wrote={[item.id for item in request.items]} deleted={request.delete_items}"
            )
            return request

        lane._commit_request = logged  # type: ignore[method-assign]

    if not session.history.items:
        await session.generate_reply(instructions="Greet the customer in one sentence.")


@support.a2a_session(endpoint="frontend", description="An online shop's support desk.")
async def text(ctx: A2ASessionContext) -> AsyncIterator[str]:
    turn = ctx.task_id[-6:]
    log.info(f"text turn {turn}: thread={ctx.thread.id} message={ctx.message.text_content!r}")
    # the reply as the model writes it: every step's text streams to the client, and the
    # final artifact replaces it with the last step's text
    run = agent.chat.stream(ctx.message, message_history=ctx.message_history)
    async for delta in run:
        yield delta
    result = run.result
    for item in result.new_messages():
        log.info(f"text turn {turn}: added {item.id} {describe(item)}")
    log.info(f"text turn {turn}: output={result.output!r} input_required={result.input_required!r}")


server = AgentServer(storage=AgentDBStorage(), agent_name="doors")
server.add(support)

if __name__ == "__main__":
    cli.run_app(server)
