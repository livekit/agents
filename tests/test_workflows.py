import pytest

from livekit.agents import AgentSession, workflows, ToolError
from livekit.plugins import openai


@pytest.mark.asyncio
async def test_collect_email():
    async with openai.LLM(model="gpt-4o") as llm, AgentSession(llm=llm) as sess:
        await sess.start(workflows.GetEmailAgent())

        await sess.run(user_input="My email address is theo at livekit dot io?")
        result = await sess.run(user_input="Yes", output_type=workflows.GetEmailResult)
        assert result.final_output.email_address == "theo@livekit.io"

    async with openai.LLM(model="gpt-4o") as llm, AgentSession(llm=llm) as sess:
        await sess.start(workflows.GetEmailAgent())

        with pytest.raises(ToolError):
            await sess.run(user_input="I don't want to give my email address")
