import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Literal, cast

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    TurnHandlingOptions,
    cli,
    inference,
    stt,
    utils,
    vad,
)
from livekit.plugins import silero

load_dotenv()


class Ink2WithLocalSOSAgent(Agent):
    def __init__(self, *, ink2: stt.STT, sos_vad: vad.VAD) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Keep your responses concise.",
            stt=ink2,
            # vad=None,
            # turn_handling=TurnHandlingOptions(turn_detection="stt"),
        )
        self._ink2 = ink2
        self._sos_vad = sos_vad

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the user briefly.")

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        del model_settings

        events = utils.aio.Chan[stt.SpeechEvent]()
        sos_stream = self._sos_vad.stream()
        sos_source: Literal["ink2", "vad"] | None = None
        ink2_event_received = False
        ink2_event_before_sos = False

        async with self._ink2.stream(
            conn_options=self.session.conn_options.stt_conn_options
        ) as ink2_stream:

            async def forward_audio() -> None:
                async for frame in audio:
                    sos_stream.push_frame(frame)
                    ink2_stream.push_frame(frame)

                sos_stream.end_input()
                ink2_stream.end_input()

            async def forward_sos() -> None:
                nonlocal ink2_event_before_sos, ink2_event_received, sos_source

                async for event in sos_stream:
                    if event.type == vad.VADEventType.START_OF_SPEECH and sos_source is None:
                        sos_source = "vad"
                        ink2_event_received = ink2_event_before_sos
                        ink2_event_before_sos = False
                        events.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.START_OF_SPEECH,
                                request_id="local-vad",
                                speech_start_time=time.time() - event.speech_duration,
                            )
                        )
                    elif (
                        event.type == vad.VADEventType.END_OF_SPEECH
                        and sos_source == "vad"
                        and not ink2_event_received
                    ):
                        events.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.END_OF_SPEECH,
                                request_id="local-vad",
                            )
                        )
                        sos_source = None

            async def forward_ink2() -> None:
                nonlocal ink2_event_before_sos, ink2_event_received, sos_source

                async for event in ink2_stream:
                    if event.type == stt.SpeechEventType.START_OF_SPEECH:
                        if sos_source is None:
                            sos_source = "ink2"
                            events.send_nowait(event)
                        else:
                            ink2_event_received = True
                    else:
                        if event.type != stt.SpeechEventType.RECOGNITION_USAGE:
                            if sos_source == "vad":
                                ink2_event_received = True
                            elif sos_source is None:
                                ink2_event_before_sos = True

                        events.send_nowait(event)

                    if event.type == stt.SpeechEventType.END_OF_SPEECH:
                        sos_source = None
                        ink2_event_received = False
                        ink2_event_before_sos = False

            tasks = [
                asyncio.create_task(forward_audio()),
                asyncio.create_task(forward_sos()),
                asyncio.create_task(forward_ink2()),
            ]

            async def supervise() -> None:
                try:
                    await asyncio.gather(*tasks)
                finally:
                    events.close()

            supervisor = asyncio.create_task(supervise())
            try:
                async for event in events:
                    yield event
                await supervisor
            finally:
                await utils.aio.cancel_and_wait(supervisor, *tasks)
                await sos_stream.aclose()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3.5"),
        aec_warmup_duration=None,
        # turn_handling={
        #     "turn_detection": "stt",
        # }
    )
    agent = Ink2WithLocalSOSAgent(
        ink2=inference.STT("cartesia/ink-2", language="en"),
        sos_vad=inference.VAD(),
    )
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
