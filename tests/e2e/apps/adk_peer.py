"""An ordinary ADK agent, tool and A2A server with deterministic inference."""

from collections.abc import AsyncGenerator
from urllib.parse import urlsplit

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from starlette.types import ASGIApp


def check_availability(destination: str) -> str:
    """Check whether a flight has seats for the requested destination."""
    return f"Seats are available to {destination}."


class BookingModel(BaseLlm):
    """Script the provider boundary while ADK performs its real tool and agent execution."""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        parts = llm_request.contents[-1].parts or []
        for part in parts:
            response = part.function_response
            if response is not None:
                result = response.response or {}
                yield LlmResponse(
                    content=types.Content(
                        role="model", parts=[types.Part(text=str(result["result"]))]
                    )
                )
                return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="check_availability", args={"destination": "Rome"}
                        )
                    )
                ],
            )
        )


agent = Agent(
    name="travel",
    description="Check flight availability.",
    model=BookingModel(model="scripted-adk"),
    instruction="Check flight availability and report the tool result.",
    tools=[check_availability],
)


def create_app(url: str) -> ASGIApp:
    port = urlsplit(url).port
    if port is None:
        raise ValueError("the test server needs its allocated port")
    return to_a2a(agent, host="127.0.0.1", port=port)
