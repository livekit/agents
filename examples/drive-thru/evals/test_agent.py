from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.voice.run_result import mock_tools
import pytest
from livekit.agents import AgentSession, llm, ChatContext
from livekit.plugins import openai

from ..drivethru_agent import DriveThruAgent, new_userdata


def _llm_model() -> llm.LLM:
    return openai.LLM(model="gpt-4o", parallel_tool_calls=False, temperature=0.45)


@pytest.mark.asyncio
async def test_item_ordering() -> None:
    userdata = await new_userdata()

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        # add big mac
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(user_input="Can I get a Big Mac, no meal?")
        result.expect.nth(0).is_function_call(
            name="order_regular_item", arguments={"item_id": "big_mac"}
        )
        fnc_out = result.expect.nth(1).is_function_call_output()
        assert fnc_out.event().item.output.startswith("The item was added")
        result.expect.nth(2).is_message(role="assistant")

        # remove item
        result = await sess.run(user_input="No actually I don't want it")
        try:
            result.expect.nth(0).is_function_call(name="list_order_items")
            result.expect.nth(1).is_function_call_output()
            result.expect.nth(2).is_function_call(name="remove_order_item")
            result.expect.nth(3).is_function_call_output()
            result.expect.nth(4).is_message(role="assistant")
        except AssertionError:
            result.expect.nth(0).is_function_call(name="remove_order_item")
            result.expect.nth(1).is_function_call_output()
            result.expect.nth(2).is_message(role="assistant")

        # order mcflurry
        result = await sess.run(user_input="Can I get a McFlurry Oreo?")
        result.expect.function_call(
            name="order_regular_item", arguments={"item_id": "sweet_mcflurry_oreo"}
        )
        result.expect.function_call_output()
        result.expect.message(role="assistant")


@pytest.mark.asyncio
async def test_meal_order() -> None:
    userdata = await new_userdata()

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        # add combo crispy, forgetting drink
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(
            user_input="Can I get a large Combo McCrispy Original with mayonnaise?"
        )
        msg_assert = result.expect.message(role="assistant")
        await msg_assert.judge(llm, intent="should prompt the user to choose a drink")

        # order the drink
        result = await sess.run(user_input="a large coca cola")
        result.expect.function_call(
            name="order_combo_meal",
            arguments={
                "meal_id": "combo_mccrispy_4a",
                "drink_id": "coca_cola",
                "drink_size": "L",
                "fries_size": "L",
                "sauce_id": "mayonnaise",
            },
        )
        result.expect.function_call_output()
        result.expect.message(role="assistant")


@pytest.mark.asyncio
async def test_failure() -> None:
    userdata = await new_userdata()

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        # simulate a tool error
        with mock_tools(
            DriveThruAgent, {"order_regular_item": lambda: RuntimeError("test failure")}
        ):
            await sess.start(DriveThruAgent(userdata=userdata))
            result = await sess.run(user_input="Can I get a large vanilla shake?")
            result.expect.function_call(
                name="order_regular_item", arguments={"item_id": "shake_vanilla", "size": "L"}
            )
            result.expect.function_call_output()
            await result.expect.message(role="assistant").judge(
                llm, intent="should inform the user that an error occurred"
            )


@pytest.mark.asyncio
async def test_unavailable_item() -> None:
    userdata = await new_userdata()

    for item in userdata.drink_items:
        if item.id == "coca_cola":
            item.available = False

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        # ask for a coke (unavailable)
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(user_input="Can I get a large coca cola?")
        try:
            await (
                result.expect.nth(0)
                .is_message(role="assistant")
                .judge(llm, intent="should inform the user that the coca cola is unavailable")
            )
        except:
            result.expect.nth(0).is_function_call(
                name="order_regular_item", arguments={"item_id": "coca_cola", "size": "L"}
            )
            result.expect.nth(1).is_function_call_output()
            await (
                result.expect.nth(2)
                .is_message(role="assistant")
                .judge(llm, intent="should inform the user that the coca cola is unavailable")
            )


@pytest.mark.asyncio
async def test_ask_for_size() -> None:
    userdata = await new_userdata()

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        await sess.start(DriveThruAgent(userdata=userdata))
        # ask for a fanta
        result = await sess.run(user_input="Can I get a fanta orange?")
        await result.expect.message(role="assistant").judge(
            llm, intent="should ask for the drink size"
        )

        # order a small fanta
        result = await sess.run(user_input="a small one")
        result.expect.function_call(
            name="order_regular_item", arguments={"item_id": "fanta_orange", "size": "S"}
        )
        result.expect.function_call_output()
        await result.expect.message(role="assistant").judge(
            llm, intent="should confirm that the fanta orange was ordered"
        )


@pytest.mark.asyncio
async def test_consecutive_order() -> None:
    userdata = await new_userdata()

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(user_input="Can I get two mayonnaise sauces?")
        result.expect.function_call(name="order_regular_item", arguments={"item_id": "mayonnaise"})
        result.expect.function_call_output()
        result.expect.function_call(name="order_regular_item", arguments={"item_id": "mayonnaise"})
        result.expect.function_call_output()
        await result.expect.message(role="assistant").judge(
            llm, intent="should confirm that two mayonnaise sauces was ordered"
        )


@pytest.mark.asyncio
async def test_conv():
    userdata = await new_userdata()

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        agent = DriveThruAgent(userdata=userdata)
        await sess.start(agent)

        # fmt: off
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="Hello, Can I get a Big Mac?")
        chat_ctx.add_message(role="assistant", content="Sure thing! Would you like that as a combo meal with fries and a drink, or just the Big Mac on its own?")
        chat_ctx.add_message(role="user", content="Yeah. With a meal")
        chat_ctx.add_message(role="assistant", content="Great! What drink would you like with your Big Mac Combo?")
        chat_ctx.add_message(role="user", content="Cook. ")
        chat_ctx.add_message(role="assistant", content="Did you mean a Coke for your drink?")
        chat_ctx.add_message(role="user", content="Yeah. ")
        chat_ctx.add_message(role="assistant", content="Alright, a Big Mac Combo with a Coke. What size would you like for your fries and drink? Medium or large?")
        chat_ctx.add_message(role="user", content="Large. ")
        chat_ctx.add_message(role="assistant", content="Got it! A Big Mac Combo with large fries and a Coke. What sauce would you like with that?")
        # fmt: on

        await agent.update_chat_ctx(chat_ctx)

        result = await sess.run(user_input="mayonnaise")
        result.expect.function_call(
            name="order_combo_meal",
            arguments={
                "meal_id": "combo_big_mac",
                "drink_id": "coca_cola",
                "drink_size": "L",
                "fries_size": "L",
                "sauce_id": "mayonnaise",
            },
        )
        result.expect.function_call_output()
        await result.expect.message(role="assistant").judge(
            llm, intent="should confirm the order of a Big Mac meal"
        )
