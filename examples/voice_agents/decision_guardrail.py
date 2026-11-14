"""Check every turn with a decision model and block or steer the agent's replies.

Run with LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET set:
    python decision_guardrail.py console

The agent is an over-eager airline support rep, so each hook has something to catch:
    - "you're useless, I'll find where you live"   -> on_user_turn blocks the reply
    - "what's a good pasta recipe?"                 -> on_user_turn steers back to travel
    - "this is the third time I've called!"         -> on_user_turn asks for an apology
    - "my flight was cancelled, can I get my money back?"
                                                    -> on_agent_sentence cuts the sentence
                                                       promising a refund
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    Block,
    DecisionEvaluated,
    DecisionGuardrail,
    JobContext,
    Steer,
    cli,
    inference,
)
from livekit.agents.llm import Choice, Decision, Probability, Score

logger = logging.getLogger("decision-guardrail")

load_dotenv()

# evaluated together on each user turn, before the agent replies
USER_TURN_DECISIONS: dict[str, Decision] = {
    "abusive": Probability("The user is abusive, threatening or harassing."),
    "topic": Choice(
        "What is the user's message about?",
        options={
            "travel": "flights, bookings, baggage, refunds or the airline",
            "smalltalk": "greetings, thanks or other small talk",
            "off_topic": "anything unrelated to air travel",
        },
    ),
    "frustration": Score(
        "How frustrated does the user sound?",
        levels=["calm", "annoyed", "frustrated", "furious"],
    ),
}

# evaluated on the reply so far, each time a sentence is complete
SENTENCE_DECISIONS: dict[str, Decision] = {
    "promises_refund": Probability(
        "The last sentence promises or guarantees the customer a refund, "
        "compensation or money back."
    ),
}

# evaluated on each complete agent message
REPLY_DECISIONS: dict[str, Decision] = {
    "politeness": Score(
        "How polite is the agent's message?",
        levels=["rude", "neutral", "polite", "very polite"],
    ),
}


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Sky, a support agent for Northwind Airlines, on a voice call. "
            "Keep replies to two or three short sentences, without markdown or emojis. "
            "You are very eager to please: when a customer mentions a cancelled or delayed "
            "flight, you tend to promise them a full refund right away.",
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the caller and ask how you can help")


def _log(ev: DecisionEvaluated) -> None:
    # only the values: input_text is the conversation and may hold customer information
    values = {name: result.value for name, result in ev.response.results.items()}
    logger.info("%s: %s", ev.hook, values)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/flux-general", language="en"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    # one guardrail per session; register every check on it
    guardrail = DecisionGuardrail(session, model="openai/gpt-6-luna-decisions", timeout=2.0)

    # runs before each reply. With eager=True (the default) it re-evaluates on every final
    # transcript, so the result is usually ready when the user stops speaking
    @guardrail.on_user_turn(decisions=USER_TURN_DECISIONS)
    def check_user_turn(ev: DecisionEvaluated) -> Block | Steer | None:
        _log(ev)
        if ev.value("abusive", 0.0) > 0.8:
            return Block(say="I want to help, but I can't continue if the call stays abusive.")
        if ev.value("topic") == "off_topic":
            return Steer(
                "The user went off topic. Say briefly that you can only help with "
                "Northwind travel, then ask how you can help with their trip."
            )
        if ev.value("frustration", 0.0) >= 2:
            return Steer("The user is frustrated. Apologize first and keep the reply short.")
        return None

    # runs on each sentence before it is spoken. on_error="block" keeps a sentence back
    # if it couldn't be checked in time
    @guardrail.on_agent_sentence(decisions=SENTENCE_DECISIONS, on_error="block")
    def check_sentence(ev: DecisionEvaluated) -> Block | None:
        _log(ev)
        if ev.value("promises_refund", 0.0) > 0.6:
            return Block(
                say="Refunds depend on your fare, so let me check your booking before "
                "I promise anything."
            )
        return None

    # runs after each agent message; it can only observe
    @guardrail.on_agent_turn(decisions=REPLY_DECISIONS)
    def score_reply(ev: DecisionEvaluated) -> None:
        _log(ev)

    await session.start(agent=SupportAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
