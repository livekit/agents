import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    room_io,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
logger = logging.getLogger("interrupt-agent")

class InterruptionAwareAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant. You are currently testing your interruption handling capabilities."
            "If the user says 'Yeah', 'Ok', or 'Hmm', you should IGNORE it and keep talking."
            "If the user says 'Stop' or asks a question, you should stop and answer."
            "Speak in long sentences to allow the user to interrupt you.",
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def tell_a_long_story(self, context):
        """Called when the user asks for a story or to keep talking."""
        return "Once upon a time, in a land far away, there was a programmer writing code. " \
               "The programmer was very focused and kept typing line after line. " \
               "Suddenly, a wild bug appeared! But the programmer did not stop. " \
               "They kept typing, ignoring the distractions, just like I am ignoring your soft interruptions right now. " \
               "I will keep talking until you say something meaningful."

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    await session.start(
        agent=InterruptionAwareAgent(),
        room=ctx.room,
    )

if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(server)
