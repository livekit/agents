import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv
from stereo_audio import BargeInDetector

from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    utils,
)
from livekit.agents.stt import stt
from livekit.agents.types import NotGivenOr
from livekit.agents.voice import ModelSettings
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.plugins import silero
from livekit.plugins.cartesia import TTS
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor.",
        )
        self.should_hold_transcript: NotGivenOr[bool] = NOT_GIVEN
        self.transcript_buffer: list[stt.SpeechEvent] = []
        self.last_inference_time: NotGivenOr[float] = NOT_GIVEN
        self._stream_started_at: NotGivenOr[float] = NOT_GIVEN

    def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings):
        async def _forward():
            async for event in Agent.default.stt_node(self, audio, model_settings):

                if not utils.is_given(self.should_hold_transcript) and not utils.is_given(self.last_inference_time):
                    yield event
                    continue

                if event.type == stt.SpeechEventType.RECOGNITION_USAGE:
                    yield event
                    continue

                # during the detection window, we hold all events
                # release upon the first event that is after the last inference time
                if utils.is_given(self.last_inference_time):
                    logger.info(f"[BARGEIN] Holding transcript: {event.type}")
                    if not event.alternatives or not utils.is_given(self.last_inference_time):
                        self.transcript_buffer.append(event)
                        continue

                    if event.alternatives[0].start_time <= self.last_inference_time - self._stream_started_at:
                        self.transcript_buffer.append(event)
                        continue

                    # release any events that are after the last inference time
                    while self.transcript_buffer:
                        prev_event = self.transcript_buffer.pop()
                        if prev_event.alternatives and prev_event.alternatives[0].start_time <= self.last_inference_time - self._stream_started_at:
                            logger.info(f"[BARGEIN] Releasing event: {event.type}: {prev_event.alternatives[0].start_time}")
                            yield event
                        elif prev_event.type == stt.SpeechEventType.START_OF_SPEECH:
                            logger.info(f"[BARGEIN] Releasing event: {prev_event.type}")
                            yield prev_event
                            logger.info(f"[BARGEIN] Releasing event: {event.type}: {event.alternatives[0].start_time}")
                            yield event
                        break

                    self.transcript_buffer = []
                    self.should_hold_transcript = NOT_GIVEN
                    self.last_inference_time = NOT_GIVEN
                    continue

        return _forward()

    def skip_until(self, time: float, started_at: float):
        logger.info(f"[BARGEIN] Skipping until {time} started at {started_at}")
        self.last_inference_time = time
        self._stream_started_at = started_at

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

tasks = set()

async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4o-mini",
        tts=TTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            language="en",
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent):
        logger.info(f"[BARGEIN] Conversation item added: {ev}")

    agent = MyAgent()
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    sa = BargeInDetector(
        model_path="/Users/chenghao/Downloads/bd_best.onnx",
        enable_clipping=True,
        clipping_threshold=1e-3,
        sample_rate=16000,
    )
    sa.eavesdrop(ctx, session)


if __name__ == "__main__":
    try:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
    except Exception as e:
        logger.error(f"Failed to run app: {e}")
        raise e
