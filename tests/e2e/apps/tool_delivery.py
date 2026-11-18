"""A service, its A2A coordinator and its voice front desk, using ordinary public APIs."""

import os
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from livekit.agents import (
    Agent,
    AgentServer,
    DelegationOptions,
    RunContext,
    SessionState,
    TextSession,
    VoiceSession,
    persisted,
)
from livekit.agents.a2a import A2ASessionContext, RemoteAgent
from livekit.agents.job import JobContext
from livekit.agents.storage import ContextStore
from livekit.agents.voice import AgentStateChangedEvent, ConversationItemAddedEvent
from tests.fakes import FakeLLM, tool_call

from ..framework.media import ToneSTT, ToneTTS
from ..framework.models import write_live_report

REQUEST = "reserve my room"
ACKNOWLEDGEMENT = "I am checking your booking."
SILENT_STATUS = "The reservation record has been locked."
EXACT = "Your booking code is R-204."
RESULT = "Reservation R-204 is confirmed."
FINAL = "Your room is reserved."
PHRASES = {500: REQUEST}

server = AgentServer()
role = os.environ.get("E2E_DELIVERY_ROLE")


@dataclass
class Conversation:
    session: SessionState = persisted(default_factory=SessionState)


def report_directory() -> Path:
    directory = Path(os.environ["E2E_ARTIFACT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    return directory


if role in ("service", "coordinator"):

    @server.a2a_session(endpoint="booking", description="Reserve a room")
    async def text(ctx: A2ASessionContext) -> object:
        app = await ctx.store.load_context(Conversation())
        routing = FakeLLM([], model="scripted-no-extra-business-call")
        if role == "service":
            model = FakeLLM([tool_call("reserve"), RESULT], model="scripted-service")
            agent = Agent(name="reservation", instructions="Reserve the room.", llm=model)

            @agent.tool
            async def reserve(run: RunContext) -> str:
                """Reserve the room in the application service."""
                async with aiohttp.ClientSession() as client:
                    for operation in ("wait", "book"):
                        async with client.post(
                            os.environ["E2E_SERVICE_URL"] + "/call",
                            json={
                                "worker": os.environ["LIVEKIT_AGENT_NAME"],
                                "thread": os.environ["E2E_DELIVERY_ID"],
                                "operation": operation,
                                "terms": "R-204",
                            },
                        ) as response:
                            response.raise_for_status()
                            await response.text()
                # Both reports are admitted before their optional readers are returned. The
                # immediate body result must not overtake either owned report.
                run.update(SILENT_STATUS, silent=True)
                run.say(EXACT)
                return RESULT

        else:
            model = FakeLLM(
                [tool_call("fulfilment", request=REQUEST), RESULT], model="scripted-coordinator"
            )
            agent = Agent(
                name="coordinator",
                instructions="Delegate the reservation and return its confirmation.",
                llm=model,
                tools=[
                    RemoteAgent(
                        os.environ["E2E_REMOTE_URL"],
                        id="fulfilment",
                        name="fulfilment",
                        description="Reserve the room.",
                    ).as_tool()
                ],
            )
        result = await TextSession(
            agent=agent, delegation=DelegationOptions(routing_model=routing)
        ).run(ctx.message, context=ctx.store.bind(app))
        assert result.status == "completed" and result.output == RESULT
        write_live_report(
            report_directory(),
            models={"frontend": model.model, "routing": routing.model},
            turns=[],
            state=app.session,
            observations={
                "result": result.output,
                "model_requests": len(model.requests),
                "routing_requests": len(routing.requests),
                "tool_results": [
                    {"id": item.id, "call_id": item.call_id, "output": item.output}
                    for item in result.new_messages()
                    if item.type == "function_call_output"
                ],
            },
        )
        return result


elif role == "voice":

    @server.rtc_session()
    async def voice(ctx: JobContext) -> None:
        store = ContextStore()
        app = await store.load_context(Conversation())
        model = FakeLLM(
            [tool_call("travel", request=REQUEST), ACKNOWLEDGEMENT, FINAL],
            model="scripted-voice",
        )
        routing = FakeLLM([], model="scripted-no-extra-business-call")
        agent = Agent(
            name="desk",
            instructions="Ask travel to reserve the room and report its confirmation.",
            llm=model,
            tools=[
                RemoteAgent(
                    os.environ["E2E_REMOTE_URL"],
                    id="travel",
                    name="travel",
                    description="Reserve the room.",
                ).as_tool()
            ],
        )
        # An explicit text-to-speech leg owns exact wording. A generative realtime model is
        # deliberately not used to prove verbatim output.
        session = VoiceSession(
            context=store.bind(app),
            delegation=DelegationOptions(routing_model=routing),
            stt=ToneSTT(PHRASES),
            tts=ToneTTS(),
            vad=None,
            turn_handling={"turn_detection": "stt"},
            aec_warmup_duration=0,
        )

        def snapshot() -> None:
            write_live_report(
                report_directory(),
                models={"frontend": model.model, "routing": routing.model},
                turns=[],
                state=app.session,
                observations={
                    "agent_state": session.agent_state,
                    "model_requests": len(model.requests),
                    "routing_requests": len(routing.requests),
                },
            )

        @session.on("conversation_item_added")
        def observed(event: ConversationItemAddedEvent) -> None:
            snapshot()

        @session.on("agent_state_changed")
        def changed(event: AgentStateChangedEvent) -> None:
            snapshot()

        await session.start(agent=agent, room=ctx.room)
