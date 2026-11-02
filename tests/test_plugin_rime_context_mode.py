"""History defaults must allow tasks while retaining explicit replacement."""

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, function_tool

from .test_plugin_rime import messages, peer_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("context_mode", [None, "shared", "strict"])
async def test_context_mode_controls_omitted_history(context_mode):
    options = {} if context_mode is None else {"context_mode": context_mode}
    async with peer_model(**options) as (_, model, _):
        provider = model.session()
        original = messages("first", "second")
        await provider.update_chat_ctx(original)
        reduced = original.copy()
        reduced.items.pop(0)
        await provider.update_chat_ctx(reduced)
        expected = ["second"] if context_mode == "strict" else ["first", "second"]
        assert [item.text for item in await model._get_history()] == expected

        reduced.add_message(role="user", content="third")
        await provider.update_chat_ctx(reduced)
        assert [item.text for item in await model._get_history()] == [*expected, "third"]


@pytest.mark.parametrize("context_mode", [None, "strict"])
async def test_context_helpers_preserve_bound_tools(context_mode):
    class Assistant(Agent):
        def __init__(self):
            super().__init__(instructions="Help")

        @function_tool
        async def lookup(self) -> str:
            """Look up a value."""
            return "found"

    options = {} if context_mode is None else {"context_mode": context_mode}
    async with peer_model(**options) as (_, model, _):
        async with AgentSession(llm=model) as session:
            agent = Assistant()
            await session.start(agent)
            config = next(
                item for item in agent.chat_ctx.items if item.type == "agent_config_update"
            )
            initial = agent.chat_ctx.copy()
            initial.items.extend(messages("first", "second").items)
            await agent.update_chat_ctx(initial)
            await model.replace_chat_ctx(agent, messages("replacement"))
            assert [item.text for item in await model._get_history()] == ["replacement"]
            await model.reconcile_chat_ctx(agent)
            await model.clear_chat_ctx(agent)
            assert not await model._get_history()
            saved = next(item for item in agent.chat_ctx.items if item.id == config.id)
            assert saved.model_dump() == config.model_dump()
            assert saved._tools == config._tools


async def test_default_context_preserves_pending_tool_through_task_and_next_reply():
    completed = []

    class Task(AgentTask[str]):
        def __init__(self):
            super().__init__(instructions="Collect a detail")

        async def on_enter(self):
            self.complete("finished")

    @function_tool
    async def collect_detail() -> str:
        """Collect a detail in a sub-task."""
        result = await Task()
        completed.append(result)
        return result

    async with peer_model(playback="discard") as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Agent(instructions="Parent", tools=[collect_detail]))
            speech = session.generate_reply()
            parent = await peer.start(await peer.next())
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=parent,
                item_id="parent-item",
                call_id="parent-call",
                name="collect_detail",
                arguments="{}",
            )
            await peer.end(parent)
            continuation = await peer.next()
            assert continuation["response"]["metadata"]["prism_parent_response_id"] == parent
            response = await peer.start(continuation)
            await peer.message(response, text="Recorded")
            await peer.end(response)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert completed == ["finished"]
            assert len([item for item in peer.items if item.get("call_id") == "parent-call"]) == 1

            followup = session.generate_reply(user_input="Continue")
            response = await peer.start(await peer.next())
            await peer.message(response, text="Still here")
            await peer.end(response)
            await asyncio.wait_for(followup.wait_for_playout(), 3)
            assert any(
                item.type == "message" and item.text_content == "Still here"
                for item in session.history.items
            )
            assert peer.connections == 1
