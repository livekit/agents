"""An ordinary voice frontend using separate live models and an independent A2A service."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from livekit.agents import (
    Agent,
    AgentServer,
    DelegationOptions,
    SessionState,
    VoiceSession,
    persisted,
)
from livekit.agents.a2a import RemoteAgent
from livekit.agents.inference import LLM
from livekit.agents.job import JobContext
from livekit.agents.storage import ContextStore
from livekit.agents.voice import AgentStateChangedEvent, ConversationItemAddedEvent

from ..framework.media import ToneSTT, ToneTTS
from ..framework.models import write_live_report

server = AgentServer()


@dataclass
class Conversation:
    session: SessionState = persisted(default_factory=SessionState)


@server.rtc_session()
async def voice(ctx: JobContext) -> None:
    model_ids = {
        "frontend": os.environ["E2E_FRONTEND_MODEL"],
        "routing": os.environ["E2E_ROUTING_MODEL"],
    }
    models = [
        LLM(
            model_id,
            api_key=os.environ["LIVEKIT_INFERENCE_API_KEY"],
            api_secret=os.environ["LIVEKIT_INFERENCE_API_SECRET"],
        )
        for model_id in model_ids.values()
    ]
    frontend, routing = models
    for model in models:
        ctx.add_shutdown_callback(model.aclose)
    store = ContextStore()
    app = await store.load_context(Conversation())
    agent = Agent(
        name="desk",
        llm=frontend,
        instructions=(
            "You are a conversational front desk. Speak briefly and naturally in English. travel "
            "owns flight work and follow-up questions. Delegate new flight requests immediately, "
            "even when details are missing; do not collect fields before asking travel. Present "
            "travel's actual questions. You can explain that a date is needed to check availability. "
            "Answer side questions yourself. The session forwards recognized answers to an already "
            "running travel call; do not start another call merely to repeat an answer. Explain "
            "the final confirmation only when travel confirms it, including the confirmed date."
        ),
        tools=[RemoteAgent(os.environ["E2E_PEER_URL"], name="travel").as_tool()],
    )
    phrases = {
        int(frequency): text
        for frequency, text in json.loads(os.environ["E2E_VOICE_PHRASES"]).items()
    }
    session = VoiceSession(
        context=store.bind(app),
        delegation=DelegationOptions(routing_model=routing),
        stt=ToneSTT(phrases),
        tts=ToneTTS(),
        vad=None,
        turn_handling={"turn_detection": "stt"},
        aec_warmup_duration=0,
    )

    def snapshot() -> None:
        write_live_report(
            Path(os.environ["E2E_ARTIFACT_DIR"]),
            models=model_ids,
            turns=[],
            state=app.session,
            observations={"agent_state": session.agent_state},
        )

    @session.on("conversation_item_added")
    def observed(event: ConversationItemAddedEvent) -> None:
        snapshot()

    @session.on("agent_state_changed")
    def changed(event: AgentStateChangedEvent) -> None:
        snapshot()

    await session.start(agent=agent, room=ctx.room)
