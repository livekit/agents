# Agent behavior is covered by the simulations in scenarios.yaml, which run full
# conversations against the agent on LiveKit Cloud (see README.md). The eval
# below is kept as an example of the in-process testing framework
# (https://docs.livekit.io/agents/start/testing/) for turn-level checks that
# don't need a live session. Uncomment it and run `uv run pytest` to use it.
#
# import textwrap
#
# import pytest
# from livekit.agents import AgentSession, inference, llm
#
# from agent import Assistant
#
#
# def _judge_llm() -> llm.LLM:
#     return inference.LLM(model="openai/gpt-4.1-mini")
#
#
# @pytest.mark.asyncio
# async def test_offers_assistance() -> None:
#     """Evaluation of the agent's friendly nature."""
#     async with (
#         _judge_llm() as judge_llm,
#         AgentSession() as session,
#     ):
#         await session.start(Assistant())
#
#         # Run an agent turn following the user's greeting
#         result = await session.run(user_input="Hello")
#
#         # Evaluate the agent's response for friendliness
#         await (
#             result.expect.next_event()
#             .is_message(role="assistant")
#             .judge(
#                 judge_llm,
#                 intent=textwrap.dedent(
#                     """\
#                     Greets the user in a friendly manner.
#
#                     Optional context that may or may not be included:
#                     - Offer of assistance with any request the user may have
#                     - Other small talk or chit chat is acceptable, so long as it is friendly and not too intrusive
#                     """
#                 ),
#             )
#         )
#
#         # Ensures there are no function calls or other unexpected events
#         result.expect.no_more_events()
