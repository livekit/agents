"""A stock A2A SDK application: no LiveKit imports, extensions or shared database."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import a2a_pb2 as pb
from starlette.applications import Starlette


class Travel(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.message is not None
        text = "".join(part.text for part in context.message.parts)
        context_id = context.context_id or uuid4().hex
        if text == "hello":
            await event_queue.enqueue_event(
                pb.Message(
                    message_id=uuid4().hex,
                    context_id=context_id,
                    role=pb.Role.ROLE_AGENT,
                    parts=[pb.Part(text="Hello from the stock A2A peer.")],
                )
            )
            return

        task_id = context.task_id or uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)
        if context.current_task is None:
            await event_queue.enqueue_event(
                pb.Task(
                    id=task_id,
                    context_id=context_id,
                    status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_SUBMITTED),
                )
            )
            # Ordinary working text can contain a draft answer; only explicit progress should
            # ask the frontend to announce anything. Neither requires extension negotiation.
            await updater.update_status(
                pb.TaskState.TASK_STATE_WORKING,
                message=pb.Message(
                    message_id=uuid4().hex,
                    role=pb.Role.ROLE_AGENT,
                    parts=[pb.Part(text="A draft answer that must stay in status.")],
                ),
            )
            progress = pb.Message(
                message_id=uuid4().hex,
                role=pb.Role.ROLE_AGENT,
                parts=[pb.Part(text="Checking available dates.")],
            )
            progress.metadata.update({"https://livekit.io/a2a/ext/agents/v1/progress": True})
            await updater.update_status(pb.TaskState.TASK_STATE_WORKING, message=progress)
            await updater.requires_input(
                pb.Message(
                    message_id=uuid4().hex,
                    task_id=task_id,
                    context_id=context_id,
                    role=pb.Role.ROLE_AGENT,
                    parts=[pb.Part(text="Which day?")],
                )
            )
            return

        await updater.start_work()
        await updater.add_artifact([pb.Part(text="Booked for ")], artifact_id="confirmation")
        await updater.add_artifact(
            [pb.Part(text=f"{text}.")], artifact_id="confirmation", append=True, last_chunk=True
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.task_id is not None and context.context_id is not None
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def create_app(url: str) -> Starlette:
    import sys

    assert "livekit.agents" not in sys.modules, "the peer runs independently of LiveKit"
    card = pb.AgentCard(
        name="travel",
        description="Book a flight; ask for the day if it is missing.",
        version="1",
        supported_interfaces=[
            pb.AgentInterface(url=f"{url}/rpc", protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        capabilities=pb.AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )
    handler = DefaultRequestHandler(Travel(), InMemoryTaskStore(), card)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await handler.aclose()

    return Starlette(
        routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/rpc")],
        lifespan=lifespan,
    )
