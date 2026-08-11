import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)

logger = logging.getLogger("backchannel-filter")

load_dotenv()

# A starter list of English acknowledgment phrases for
# `InterruptionOptions["backchannel_filter"]`. While the agent is mid-utterance,
# an overlapping utterance made up entirely of these phrases (plus filler
# sounds like "uh", "um") neither interrupts the agent's speech nor commits a
# user turn. Utterances spoken while the agent is idle and listening are never
# filtered.
#
# Tailor this to your agent — which words are pure acknowledgments depends on
# your prompts and language. Deliberately absent: "no", "wait", "stop",
# "what", "hello", and a bare "huh" — those are real barge-ins ("uh huh" the
# backchannel still matches as a phrase).
BACKCHANNEL_PHRASES = [
    "okay",
    "ok",
    "alright",
    "got it",
    "gotcha",
    "i see",
    "makes sense",
    "sounds good",
    "thank you",
    "thanks",
    "mm hmm",
    "uh huh",
    "mhm",
    "mmhmm",
    # answer words: over agent speech these are usually acknowledgments, but
    # if your agent asks questions and keeps talking ("Does that work for
    # you? This should only take a minute...") an impatient overlapping
    # "yes" would be discarded along with the user's answer. Remove them if
    # your agent asks yes/no questions mid-speech, or use a callback (below)
    # to decide from context.
    "yes",
    "yeah",
    "yep",
    "right",
    "sure",
]

# For full control — custom languages, model-based detection,
# context-dependent rules — pass a callback instead of a list. It receives
# the transcribed text of the overlapping speech (live interims on the
# interruption path, the final transcript at turn commit) and returns True
# when it is backchannel-only. The built-in matcher is composable:
#
# from livekit.agents.voice.backchannel import is_backchannel_only
#
# def my_backchannel_filter(transcript: str) -> bool:
#     # partial=True: a trailing prefix of a phrase in a live interim
#     # ("thank" -> "thank you") defers the cut instead of committing it
#     return is_backchannel_only(transcript, BACKCHANNEL_PHRASES, partial=True)


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Keep your responses "
            "concise and to the point. Do not use emojis, asterisks, markdown, "
            "or other special characters in your responses."
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and introduce yourself")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_handling=TurnHandlingOptions(
            interruption={
                # min_words=1 lets the transcript reach the filter before an
                # acoustic-only cut fires; the filter then decides whether the
                # overlapping words are an acknowledgment or a real barge-in
                "min_words": 1,
                "backchannel_filter": BACKCHANNEL_PHRASES,
            },
        ),
    )

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
