import pytest

from livekit.agents import Agent, AgentSession, llm
from livekit.plugins import openai

from ..restaurant_agent import Checkout, Greeter, Reservation, Takeaway, UserData


def _llm_model() -> llm.LLM:
    return openai.LLM(model="gpt-4o", parallel_tool_calls=False)


def _create_agents() -> dict[str, Agent]:
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    agents = {
        "greeter": Greeter(menu),
        "reservation": Reservation(),
        "takeaway": Takeaway(menu),
        "checkout": Checkout(menu),
    }
    return agents


@pytest.mark.asyncio
async def test_reservation() -> None:
    agents = _create_agents()
    userdata = UserData(agents=agents)

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        await sess.start(agents["greeter"])
        if sess.current_speech:
            # `generate_reply` is called in `on_enter`, wait for it to finish
            # to ensure `user_input` is created after the agent response in `on_enter`
            await sess.current_speech

        result = await sess.run(user_input="I want to make a reservation for tomorrow.")
        result.expect.maybe_message(role="assistant")
        result.expect.function_call(name="to_reservation")
        result.expect.function_call_output()
        result.expect.maybe_message(role="assistant")
        result.expect.agent_handoff(new_agent_type=Reservation)
        await result.expect.message(role="assistant").judge(
            llm, intent="must ask for the reservation time for tomorrow", verbose=True
        )

        result = await sess.run(user_input="2 pm")
        await result.expect.message(role="assistant").judge(
            llm,
            intent="must validate the reservation time 2 pm tomorrow with the user",
            verbose=True,
        )

        result = await sess.run(user_input="yes")
        result.expect.maybe_message(role="assistant")
        result.expect.function_call(name="update_reservation_time")
        result.expect.function_call_output()
        await result.expect.message(role="assistant").judge(
            llm, intent="must ask for the name of the customer", verbose=True
        )

        result = await sess.run(user_input="My name is Lone")
        await result.expect.message(role="assistant").judge(
            llm, intent="must ask user to confirm the name Lone", verbose=True
        )

        result = await sess.run(user_input="oh no, my name is Long")
        try:
            result.expect[:].contains_function_call(name="update_name")
        except AssertionError:
            result = await sess.run(user_input="yes")  # may ask for confirmation again

        result.expect.maybe_message(role="assistant")
        result.expect.function_call(name="update_name", arguments={"name": "Long"})
        result.expect.function_call_output()
        await result.expect.message(role="assistant").judge(
            llm, intent="must ask for the phone number", verbose=True
        )

        result = await sess.run(user_input="My phone number is 1234567890")
        await result.expect.message(role="assistant").judge(
            llm, intent="must ask user to confirm the phone number", verbose=True
        )

        result = await sess.run(user_input="yes")
        result.expect.maybe_message(role="assistant")
        result.expect.function_call(name="update_phone", arguments={"phone": "1234567890"})
        result.expect.function_call_output()
        await result.expect.message(role="assistant").judge(
            llm, intent="must ask user to confirm the reservation details", verbose=True
        )

        result = await sess.run(user_input="yes")
        result.expect.maybe_message(role="assistant")
        result.expect.function_call(name="confirm_reservation")
        result.expect.function_call_output()
        result.expect.maybe_message(role="assistant")
        result.expect.agent_handoff(new_agent_type=Greeter)


@pytest.mark.asyncio
async def test_takeaway() -> None:
    agents = _create_agents()
    userdata = UserData(agents=agents)

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        await sess.start(agents["greeter"])
        if sess.current_speech:
            # `generate_reply` is called in `on_enter`
            await sess.current_speech

        result = await sess.run(user_input="I want to order a pizza and a salad")
        try:
            result.expect[:].contains_function_call(name="to_takeaway")
        except AssertionError:
            result = await sess.run(user_input="okay")
        result.expect.maybe_message(role="assistant")
        result.expect.function_call(name="to_takeaway")
        result.expect.function_call_output()
        result.expect.maybe_message(role="assistant")
        result.expect.agent_handoff(new_agent_type=Takeaway)

        result = await sess.run(user_input="I want to add a coffee")
        try:
            result.expect[:].contains_function_call(name="update_order")
        except AssertionError:
            result = await sess.run(user_input="yes, that's all I want")
        result.expect.maybe_message(role="assistant")
        result.expect.function_call(name="update_order")
        result.expect.function_call_output()
        try:
            result.expect[:].contains_function_call(name="to_checkout")
        except AssertionError:
            result = await sess.run(user_input="yes, I want to checkout")

        result.expect.maybe_message(role="assistant")
        result.expect.function_call(name="to_checkout")
        result.expect.function_call_output()
        result.expect.maybe_message(role="assistant")
        result.expect.agent_handoff(new_agent_type=Checkout)
        await result.expect.message(role="assistant").judge(
            llm, intent="must tell user the expense is $17", verbose=True
        )

        # test checkout
        checkout_chat_ctx = agents["checkout"].chat_ctx.copy()

        # branch 1: user disagree with the expense
        await sess.current_agent.update_chat_ctx(checkout_chat_ctx)
        result = await sess.run(user_input="no, it should be $16")
        try:
            result.expect[:].contains_function_call(name="to_takeaway")  # back to takeaway
        except AssertionError:
            await result.expect.message(role="assistant").judge(
                llm,
                intent="must disagree with the $16 or ask user to update the order",
                verbose=True,
            )

        # branch 2: user wants to change the order
        if sess.current_agent != agents["checkout"]:
            await sess._update_activity(agents["checkout"], new_activity="start")
        await sess.current_agent.update_chat_ctx(checkout_chat_ctx)
        result = await sess.run(user_input="oh I want to add another coffee")
        try:
            result.expect[:].contains_function_call(name="to_takeaway")
        except AssertionError:
            result.expect.message(role="assistant").judge(
                llm, intent="ask user if they want to back to takeaway", verbose=True
            )
