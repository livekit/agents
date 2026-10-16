import asyncio
import json
import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, llm, utils
from livekit.agents.llm import function_tool
from livekit.plugins.openai.realtime import GPTLiveDelegation, GPTLiveModel, GPTLiveSession

logger = logging.getLogger("gpt-live-client-delegation")

load_dotenv()

ORDERS = {
    "A1042": "shipped, arriving Thursday",
    "B2231": "still being packed",
}

# how many times the desk may call tools before it has to answer with what it has
MAX_DESK_ROUNDS = 4


# The desk's tools. They belong to the LLM below, not to the Agent: under client delegation the
# agent's own tool list must stay empty.
@function_tool
async def check_order_status(order_id: str) -> str:
    """Check the delivery status of an order.

    Args:
        order_id: The order reference, such as A1042.
    """
    logger.info("desk tool: checking order", extra={"lk.pii.order_id": order_id})
    status = ORDERS.get(order_id.upper())
    return f"Order {order_id} is {status}." if status else f"There is no order {order_id}."


@function_tool
async def lookup_weather(location: str) -> str:
    """Look up the current weather for a location.

    Args:
        location: The city or region to look up.
    """
    logger.info("desk tool: looking up weather", extra={"lk.pii.location": location})
    return f"The weather in {location} is 62 degrees and partly cloudy."


DESK_TOOLS = [check_order_status, lookup_weather]
DESK_TOOLS_BY_NAME = {tool.info.name: tool for tool in DESK_TOOLS}

DESK_INSTRUCTIONS = """You are the order desk for an online furniture store. You never speak to
the caller; the voice agent does. Work out what is true, use your tools, and answer in one or two
plain sentences the voice agent can say out loud. State facts, not phrasing: no greetings, no
"tell them that", no markdown. If a tool cannot answer, say so plainly."""


async def run_delegation(delegation_llm: llm.LLM, chat_ctx: llm.ChatContext) -> str:
    """Run the desk to an answer, executing whatever tools it asks for on the way."""

    async def run_tool(call: llm.FunctionToolCall) -> tuple[str, bool]:
        """Run one call, returning its output and whether it failed."""
        tool = DESK_TOOLS_BY_NAME.get(call.name)
        if tool is None:
            return f"There is no tool named {call.name}.", True
        try:
            return str(await tool(**json.loads(call.arguments))), False
        except Exception as e:
            logger.warning(
                "desk tool failed",
                extra={"lk.pii.tool_name": call.name, "error_type": type(e).__name__},
            )
            return f"{call.name} failed: {e}", True

    for _ in range(MAX_DESK_ROUNDS):
        text = ""
        calls: list[llm.FunctionToolCall] = []
        async with delegation_llm.chat(chat_ctx=chat_ctx, tools=DESK_TOOLS) as stream:
            async for chunk in stream:
                if chunk.delta is None:
                    continue
                text += chunk.delta.content or ""
                calls.extend(chunk.delta.tool_calls)

        if not calls:
            return text.strip() or "I could not work that out."

        for call in calls:
            chat_ctx.items.append(
                llm.FunctionCall(call_id=call.call_id, name=call.name, arguments=call.arguments)
            )
        for call in calls:
            output, is_error = await run_tool(call)
            chat_ctx.items.append(
                llm.FunctionCallOutput(
                    call_id=call.call_id, name=call.name, output=output, is_error=is_error
                )
            )

    return "I could not work that out in time."


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant for an online furniture store. "
                "Keep replies short and conversational. "
                "Do not use emojis, asterisks, or other special characters."
            ),
            # no tools: under client delegation the model has no way to call one
        )
        self._desk_llm = inference.LLM("openai/gpt-5.5")
        self._desk_tasks: set[asyncio.Task[None]] = set()

    async def on_enter(self) -> None:
        session = self.duplex_session
        assert isinstance(session, GPTLiveSession)
        session.on("delegation_created", self._on_delegation_created)

        self.session.generate_reply(
            instructions="Greet the caller and ask what you can help them with."
        )

    async def on_exit(self) -> None:
        await utils.aio.cancel_and_wait(*self._desk_tasks)

    def _on_delegation_created(self, delegation: GPTLiveDelegation) -> None:
        # emitted from the plugin's read loop, so the work has to run in a task of its own
        # TODO: each task knows only its own request, so a later delegation cannot supersede an
        # earlier one; the framework will hold one expert per conversation and do that (see README)
        task = asyncio.create_task(self._answer(delegation), name=f"desk:{delegation.id}")
        self._desk_tasks.add(task)
        task.add_done_callback(self._on_answer_done)

    def _on_answer_done(self, task: asyncio.Task[None]) -> None:
        self._desk_tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("delegation failed", extra={"error_type": type(error).__name__})

    async def _answer(self, delegation: GPTLiveDelegation) -> None:
        logger.info(
            "delegation requested",
            extra={
                "delegation_id": delegation.id,
                "lk.pii.pending_transcript": delegation.pending_transcript,
                "history_items": len(self.chat_ctx.items),
            },
        )

        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="system", content=DESK_INSTRUCTIONS)
        chat_ctx.items.extend(
            self.chat_ctx.copy(
                exclude_function_call=True, exclude_config_update=True, exclude_instructions=True
            ).items
        )
        if delegation.pending_transcript:
            # the turn that triggered this is not in the history yet
            chat_ctx.add_message(role="user", content=delegation.pending_transcript)

        answer = await run_delegation(self._desk_llm, chat_ctx)
        logger.info(
            "delegation answered",
            extra={"delegation_id": delegation.id, "lk.pii.answer": answer},
        )

        assert isinstance(self.duplex_session, GPTLiveSession)
        # commentary is what the model says next, in its own words, capped at 500 tokens
        self.duplex_session.append_commentary(answer, delegation_id=delegation.id)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"lk.pii.room": ctx.room.name}

    session = AgentSession(
        llm=GPTLiveModel(voice="marin", delegation="client"),
    )
    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
