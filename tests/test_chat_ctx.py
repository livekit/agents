
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
    model = utils.function_arguments_to_pydantic_model(ai_function1)
    print(model.model_json_schema())
