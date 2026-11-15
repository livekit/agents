"""Knowledge-base evals for the LiveKit products agent.

This is a retrieval/knowledge-base app: the agent answers LiveKit Agents
questions from its inline instructions and everything else by calling the
`lookup_product` tool. The dominant failure mode for this kind of app is
hallucination, so these tests cover three angles:

1. Routing        - each product question triggers the correct lookup_product call.
2. Grounded facts - answers match specific facts that live in the knowledge files.
3. Anti-hallucination - questions whose answer is absent from (or contradicted by)
   the knowledge files do not get a fabricated answer.

Plus inline-knowledge coverage and a conversation-level grounding gate using the
SDK's built-in accuracy judge.

These run the agent live against its configured LLM and judge the results with a
separate LLM, so occasional variance is expected. Re-run a lone failure before
treating it as a regression.
"""

import textwrap

import pytest
from agent import Assistant

from livekit.agents import AgentSession, ChatContext, inference, llm
from livekit.agents.evals import (
    JudgeGroup,
    accuracy_judge,
    relevancy_judge,
    tool_use_judge,
)

pytestmark = pytest.mark.evals


def _judge_llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


async def _start_primed(session: AgentSession) -> None:
    """Start the agent with its opening greeting already in context.

    The agent's instructions tell it to introduce itself first, so on a cold first
    turn it sometimes greets instead of answering the question. In production the
    greeting is a separate on_enter turn before the user speaks; seeding it here
    makes the probe question below behave like a real follow-up question.
    """
    agent = Assistant()
    await session.start(agent)
    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="assistant",
        content="Hi! I'm your LiveKit assistant. What would you like to know about LiveKit?",
    )
    await agent.update_chat_ctx(chat_ctx)


# --------------------------------------------------------------------------- #
# 1. Routing: the right question reaches the right knowledge file.
# --------------------------------------------------------------------------- #

ROUTING_CASES = [
    pytest.param(
        "Can I buy a phone number directly from LiveKit?",
        "livekit-phone-numbers",
        id="phone-buy",
    ),
    pytest.param(
        "How do inbound phone calls get routed to my agent's room?",
        "livekit-phone-numbers",
        id="phone-dispatch",
    ),
    pytest.param(
        "What regions can I deploy my agent to on LiveKit Cloud?",
        "agents-on-livekit-cloud",
        id="deploy-regions",
    ),
    pytest.param(
        "What secrets get injected into my agent container at runtime?",
        "agents-on-livekit-cloud",
        id="deploy-secrets",
    ),
    pytest.param(
        "Which LLM providers can I use through LiveKit Inference?",
        "livekit-inference",
        id="inference-llms",
    ),
    pytest.param(
        "How do I see transcripts and session traces for my agent?",
        "agent-observability",
        id="observability",
    ),
    pytest.param(
        "Can I build a voice agent in my browser without writing code?",
        "agent-builder",
        id="agent-builder",
    ),
    pytest.param(
        "What security and compliance certifications does LiveKit Cloud have?",
        "platform",
        id="platform-compliance",
    ),
]


@pytest.mark.parametrize("user_input,product", ROUTING_CASES)
@pytest.mark.asyncio
async def test_routes_to_correct_product(user_input: str, product: str) -> None:
    """Each product question triggers a lookup of the correct knowledge file."""
    async with AgentSession() as session:
        await _start_primed(session)

        result = await session.run(user_input=user_input)

        # contains_* searches the whole turn, so a brief preamble before the
        # tool call won't cause a false failure. It raises if no matching call
        # exists, asserting both that a lookup happened and that it was correct.
        result.expect.contains_function_call(name="lookup_product", arguments={"product": product})


# --------------------------------------------------------------------------- #
# 2. Grounded facts: answers match what the knowledge files actually say.
# --------------------------------------------------------------------------- #

ACCURACY_CASES = [
    pytest.param(
        "Can I use LiveKit Inference if I self-host LiveKit?",
        "States that LiveKit Inference is a LiveKit Cloud feature only and is not "
        "available for self-hosted LiveKit; self-hosting requires model plugins with "
        "your own provider API keys.",
        id="inference-self-host",
    ),
    pytest.param(
        "Does LiveKit Phone Numbers support outbound calling?",
        "Indicates LiveKit Phone Numbers currently supports inbound calling only and "
        "that outbound calling is not yet available (coming soon), optionally noting "
        "outbound requires a third-party SIP provider.",
        id="phone-outbound",
    ),
    pytest.param(
        "Which countries can I get a LiveKit phone number in?",
        "States that LiveKit phone numbers are available in the United States only, "
        "both local and toll-free.",
        id="phone-country",
    ),
    pytest.param(
        "How long is my agent observability data retained?",
        "States that observability data is retained for about 30 days and that data "
        "older than 30 days is automatically deleted. Also mentioning that data is "
        "stored in the United States is optional and not required to pass.",
        id="observability-retention",
    ),
    pytest.param(
        "Does agent observability work with a fully self-hosted LiveKit deployment?",
        "Indicates agent observability does not work with entirely self-hosted "
        "deployments and requires LiveKit Cloud. Additional nuance (that self-hosted "
        "agents connecting to LiveKit Cloud media servers are supported) is optional "
        "and not required to pass.",
        id="observability-self-host",
    ),
    pytest.param(
        "How many agent session minutes does the free Build plan include?",
        "States the free Build plan includes 1000 agent session minutes.",
        id="free-plan-minutes",
    ),
    pytest.param(
        "Can I export an agent I built in Agent Builder to code?",
        "Confirms that an agent built in Agent Builder can be exported or downloaded as "
        "a complete Python project to keep iterating with the Agents SDK.",
        id="builder-export",
    ),
    pytest.param(
        "Is LiveKit Cloud HIPAA compliant?",
        "Confirms LiveKit Cloud is HIPAA compliant, optionally mentioning other "
        "compliance such as SOC 2 Type II, GDPR, and CCPA.",
        id="platform-hipaa",
    ),
]


@pytest.mark.parametrize("user_input,intent", ACCURACY_CASES)
@pytest.mark.asyncio
async def test_grounded_answer(user_input: str, intent: str) -> None:
    """The final answer reflects the specific facts in the knowledge files."""
    async with (
        _judge_llm() as judge_llm,
        AgentSession() as session,
    ):
        await _start_primed(session)

        result = await session.run(user_input=user_input)

        # Judge the final assistant message (index -1), regardless of any tool
        # calls that preceded it.
        await result.expect[-1].is_message(role="assistant").judge(judge_llm, intent=intent)


# --------------------------------------------------------------------------- #
# 3. Anti-hallucination: absent or contradicted facts are not fabricated.
#    Each intent describes the ACCEPTABLE (non-hallucinated) response, so the
#    judge passes only when the agent declines to invent an answer.
# --------------------------------------------------------------------------- #

HALLUCINATION_CASES = [
    pytest.param(
        "Exactly how many dollars per month does a LiveKit phone number cost?",
        textwrap.dedent(
            """\
            The response does NOT state a specific dollar amount for the monthly phone
            number fee. Acceptable responses say there is a small fixed monthly fee,
            direct the user to the LiveKit pricing page, or say the exact figure isn't
            known. The response fails only if it fabricates a specific price such as
            "one dollar per month" or "two dollars per month".
            """
        ),
        id="no-phone-price",
    ),
    pytest.param(
        "What is the exact price per million tokens for GPT-5 on LiveKit Inference?",
        textwrap.dedent(
            """\
            The response does NOT invent a specific per-token or per-million-token
            price. Acceptable responses say Inference pricing is usage-based and point
            the user to the LiveKit Inference pricing page, or say the exact number
            isn't known. It fails only if it states a specific fabricated dollar price.
            """
        ),
        id="no-token-price",
    ),
    pytest.param(
        "How much does the Scale plan cost per month in dollars?",
        textwrap.dedent(
            """\
            The response does NOT state a specific monthly dollar price for the Scale
            plan. Acceptable responses say Scale offers discounted or custom rates and
            higher quotas and refer the user to the pricing page or sales team. It fails
            only if it fabricates a specific price.
            """
        ),
        id="no-scale-price",
    ),
    pytest.param(
        "Can I use Anthropic's Claude models through LiveKit Inference?",
        textwrap.dedent(
            """\
            The response does NOT confidently claim that Anthropic Claude models are
            available through LiveKit Inference. Acceptable responses list the supported
            LLM providers (such as OpenAI, Google Gemini, DeepSeek, or Kimi), say Claude
            is not among the listed Inference models, or suggest checking current model
            availability. It fails if it asserts that Claude is available via LiveKit
            Inference.
            """
        ),
        id="no-claude-inference",
    ),
    pytest.param(
        "Does LiveKit offer its own built-in hosted vector database for RAG embeddings?",
        textwrap.dedent(
            """\
            The response does NOT claim LiveKit provides its own hosted vector database
            product. Acceptable responses say agents can perform RAG lookups or connect
            to external data sources, that LiveKit does not offer a hosted vector
            database itself, or that the assistant isn't sure. It fails if it fabricates
            a LiveKit-branded vector database product or its features.
            """
        ),
        id="no-vector-db",
    ),
    pytest.param(
        "Can I get a LiveKit phone number with a Berlin, Germany area code?",
        textwrap.dedent(
            """\
            The response does NOT claim that German or other non-US phone numbers are
            available from LiveKit. It should indicate LiveKit phone numbers are US-only,
            optionally suggesting a third-party SIP provider for other countries. It
            fails if it claims a Berlin or German number can be purchased through
            LiveKit.
            """
        ),
        id="no-non-us-number",
    ),
]


@pytest.mark.parametrize("user_input,intent", HALLUCINATION_CASES)
@pytest.mark.asyncio
async def test_does_not_hallucinate(user_input: str, intent: str) -> None:
    """Facts absent from or contradicted by the knowledge files aren't invented."""
    async with (
        _judge_llm() as judge_llm,
        AgentSession() as session,
    ):
        await _start_primed(session)

        result = await session.run(user_input=user_input)

        await result.expect[-1].is_message(role="assistant").judge(judge_llm, intent=intent)


# --------------------------------------------------------------------------- #
# 4. Inline knowledge: LiveKit Agents questions answered from instructions.
# --------------------------------------------------------------------------- #

INLINE_CASES = [
    pytest.param(
        "Should I build my LiveKit agent in Python or TypeScript?",
        "Indicates both Python and TypeScript are supported with the same "
        "functionality and the user should choose whichever fits their stack or team.",
        id="python-or-typescript",
    ),
    pytest.param(
        "Roughly how many developers use LiveKit Agents?",
        "Indicates a large community on the order of 250,000 developers, optionally "
        "mentioning millions of monthly downloads.",
        id="developer-count",
    ),
    pytest.param(
        "Is the TypeScript SDK supported as seriously as the Python one?",
        "Indicates LiveKit is working toward feature parity between the Python and "
        "TypeScript SDKs and that both offer the same functionality.",
        id="typescript-parity",
    ),
    pytest.param(
        "Can a coding assistant like Cursor or Claude Code help me build with LiveKit?",
        "Confirms that coding assistants can be equipped to help build with LiveKit, "
        "for example via the LiveKit coding agent starter kit (which may include a "
        "docs MCP server, an AGENTS.md file, and agent skills) or tools like Claude "
        "Code, Cursor, Codex, and Gemini. A concise yes that mentions the starter kit "
        "or equipping the assistant with LiveKit knowledge is sufficient; it need not "
        "list every detail.",
        id="coding-agents",
    ),
]


@pytest.mark.parametrize("user_input,intent", INLINE_CASES)
@pytest.mark.asyncio
async def test_inline_agents_knowledge(user_input: str, intent: str) -> None:
    """Agents SDK questions are answered accurately from the inline knowledge base."""
    async with (
        _judge_llm() as judge_llm,
        AgentSession() as session,
    ):
        await _start_primed(session)

        result = await session.run(user_input=user_input)

        await result.expect[-1].is_message(role="assistant").judge(judge_llm, intent=intent)


@pytest.mark.asyncio
async def test_inline_question_skips_lookup() -> None:
    """A question answerable from inline knowledge is answered directly, no tool call."""
    async with AgentSession() as session:
        await _start_primed(session)

        result = await session.run(
            user_input="What programming languages can I use to build a LiveKit agent?"
        )

        # The first (and only) event is the answer itself, not a product lookup.
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


# --------------------------------------------------------------------------- #
# 5. Conversation-level grounding gate.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_conversation_stays_grounded() -> None:
    """A multi-turn conversation grounds every answer in tool output, no hallucination.

    The built-in accuracy_judge specifically catches answers that aren't grounded in
    tool outputs, which is exactly the risk for a knowledge-base agent.
    """
    async with AgentSession() as session:
        await _start_primed(session)

        await session.run(user_input="What regions can I deploy my agent to on LiveKit Cloud?")
        await session.run(user_input="And how much does a phone number cost per month?")
        await session.run(user_input="Which LLM providers are available through LiveKit Inference?")

        judges = JudgeGroup(
            llm="openai/gpt-4.1-mini",
            judges=[accuracy_judge(), tool_use_judge(), relevancy_judge()],
        )
        result = await judges.evaluate(session.history)

        assert result.all_passed, "grounding judges failed: " + "; ".join(
            f"{name}={j.verdict} ({j.reasoning})" for name, j in result.judgments.items()
        )
