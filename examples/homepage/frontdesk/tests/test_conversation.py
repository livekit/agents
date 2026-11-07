from livekit.agents import llm

from conversation import answered, caller_spoke_last, heard

CALL = "call_review"


def history(*items: llm.ChatItem) -> list[llm.ChatItem]:
    return list(items)


def said(role: str, text: str, *, interrupted: bool = False) -> llm.ChatMessage:
    return llm.ChatMessage(role=role, content=[text], interrupted=interrupted)


def review_call() -> list[llm.ChatItem]:
    return [
        llm.FunctionCall(call_id=CALL, name="book_visit", arguments="{}"),
        llm.FunctionCallOutput(
            call_id=CALL, name="book_visit", output="{}", is_error=False
        ),
    ]


def test_a_read_back_spoken_in_full_and_then_answered_counts():
    items = history(
        *review_call(), said("assistant", "Here's what I have..."), said("user", "Yes")
    )
    assert heard(items, CALL) and answered(items, CALL)


def test_a_read_back_not_spoken_yet_was_not_heard():
    items = history(*review_call())
    assert not heard(items, CALL) and not answered(items, CALL)


def test_a_cut_off_read_back_was_not_heard():
    items = history(
        *review_call(),
        said("assistant", "Here's what I", interrupted=True),
        said("user", "Yes"),
    )
    assert not heard(items, CALL) and not answered(items, CALL)


def test_a_heard_read_back_with_no_reply_yet_is_not_answered():
    items = history(*review_call(), said("assistant", "Here's what I have..."))
    assert heard(items, CALL) and not answered(items, CALL)


def test_what_the_agent_said_before_the_tool_ran_is_not_the_read_back():
    items = history(
        said("assistant", "One sec..."), *review_call(), said("user", "Yes")
    )
    assert not heard(items, CALL)


def test_a_line_spoken_as_is_is_found_by_its_own_id():
    opener = said("assistant", "Hi, this is Nick...")
    items = history(opener, said("user", "Yes, go ahead"))
    assert answered(items, opener.id)


def test_the_callers_goodbye_is_answered_before_hanging_up():
    goodbye = said("user", "Thanks, bye")
    assert caller_spoke_last(history(said("assistant", "Anything else?"), goodbye))
    assert not caller_spoke_last(history(goodbye, said("assistant", "Take care, bye.")))
