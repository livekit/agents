import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, metrics
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice import AgentStateChangedEvent, MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# This example demonstrates how to use preemptive generation to reduce latency.
# When `preemptive_generation` is enabled, the agent will generate a reply once
# the final transcript is received but before end of user turn committed.
# If the chat context or tools have changed in `on_user_turn_completed`, the preemptive
# generation will be cancelled and the new reply will be generated.

logger = logging.getLogger("preemptive-generation")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "You are curious and friendly, and have a sense of humor.",
        )

    async def on_enter(self):
        self.session.generate_reply(instructions="say hello to the user")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        # update the chat context or tools here will cancel the preemptive generation

        # # for example:
        # filler_response = "Let me think about that..."
        # self.session.say(filler_response)
        # turn_ctx.add_message(role="assistant", content=[filler_response])

        pass


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics

        metrics.log_metrics(ev.metrics)
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if (
            ev.new_state == "speaking"
            and last_eou_metrics
            and last_eou_metrics.speech_id == ev.speech_id
        ):
            logger.info(
                f"End-to-end latency: {ev.created_at - last_eou_metrics.last_speaking_time}"
            )

    await session.start(agent=MyAgent(), room=ctx.room)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
