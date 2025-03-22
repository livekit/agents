from livekit.agents.llm import utils

# function_arguments_to_pydantic_model


def ai_function1(a: int, b: str = "default") -> None:
    """
    This is a test function
    Args:
        a: First argument
        b: Second argument
    """
    pass


def test_args_model():
    from docstring_parser import parse_from_object

    docstring = parse_from_object(ai_function1)
    print(docstring.description)

    model = utils.function_arguments_to_pydantic_model(ai_function1)
    print(model.model_json_schema())


def test_dict():
    from livekit import rtc
    from livekit.agents.llm import ChatContext, ImageContent

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="user",
        content="Hello, world!",
    )
    chat_ctx.add_message(
        role="assistant",
        content="Hello, world!",
    )
    chat_ctx.add_message(
        role="user",
        content=[
            ImageContent(
                image=rtc.VideoFrame(
                    64, 64, rtc.VideoBufferType.RGB24, b"0" * 64 * 64 * 3
                )
            )
        ],
    )
    print(chat_ctx.to_dict())
    print(chat_ctx.items)

    print(ChatContext.from_dict(chat_ctx.to_dict()).items)
