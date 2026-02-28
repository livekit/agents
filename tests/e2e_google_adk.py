#!/usr/bin/env python3
"""E2E test for the Google ADK LLMAdapter.

Run this script to verify the adapter works end-to-end with a real
Google ADK agent calling Gemini. No LiveKit server required.

Prerequisites:
    pip install livekit-agents google-adk google-genai
    export GOOGLE_API_KEY=<your-key>

Usage:
    python tests/e2e_google_adk.py              # Interactive chat mode
    python tests/e2e_google_adk.py --smoke      # Single automated smoke test
    python tests/e2e_google_adk.py --tool-test  # Test with tool calling
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Add repo root to path so we can import the plugin directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "livekit-plugins",
        "livekit-plugins-google-adk",
    ),
)


def check_api_key() -> None:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        print("  export GOOGLE_API_KEY=<your-gemini-api-key>")
        sys.exit(1)


async def run_smoke_test() -> None:
    """Automated smoke test: send one message and verify a response is received."""
    from google.adk.agents import LlmAgent

    from livekit.agents.llm import ChatContext
    from livekit.plugins.google_adk import LLMAdapter

    print("=" * 60)
    print("SMOKE TEST: Google ADK LLMAdapter")
    print("=" * 60)

    agent = LlmAgent(
        name="smoke_test_agent",
        model="gemini-2.0-flash",
        instruction="You are a test assistant. Respond with exactly: SMOKE_TEST_OK",
    )

    adapter = LLMAdapter(agent)
    print(f"  Model:    {adapter.model}")
    print(f"  Provider: {adapter.provider}")

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Say SMOKE_TEST_OK")

    print("\n  Sending message: 'Say SMOKE_TEST_OK'")
    start = time.monotonic()

    full_response = ""
    chunk_count = 0
    usage_info = None

    async with adapter.chat(chat_ctx=chat_ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                full_response += chunk.delta.content
                chunk_count += 1
            if chunk.usage:
                usage_info = chunk.usage

    elapsed = time.monotonic() - start

    print(f"\n  Response: {full_response!r}")
    print(f"  Chunks:   {chunk_count}")
    print(f"  Time:     {elapsed:.2f}s")

    if usage_info:
        print(
            f"  Tokens:   prompt={usage_info.prompt_tokens}, "
            f"completion={usage_info.completion_tokens}, "
            f"total={usage_info.total_tokens}"
        )

    # Validate
    passed = True
    if not full_response.strip():
        print("\n  FAIL: Empty response")
        passed = False
    if chunk_count == 0:
        print("\n  FAIL: No chunks received")
        passed = False

    if passed:
        print("\n  PASS")
    else:
        print("\n  FAIL")
        sys.exit(1)


async def run_tool_test() -> None:
    """Test that ADK tool calling works through the adapter."""
    from google.adk.agents import LlmAgent

    from livekit.agents.llm import ChatContext
    from livekit.plugins.google_adk import LLMAdapter

    print("=" * 60)
    print("TOOL TEST: Google ADK LLMAdapter with tool calling")
    print("=" * 60)

    tool_was_called = False

    def get_weather(city: str) -> dict:
        """Returns weather information for the given city."""
        nonlocal tool_was_called
        tool_was_called = True
        print(f"  [TOOL CALLED] get_weather(city={city!r})")
        return {"city": city, "temperature": "72F", "condition": "Sunny"}

    agent = LlmAgent(
        name="tool_test_agent",
        model="gemini-2.0-flash",
        instruction=(
            "You are a weather assistant. When asked about weather, "
            "always use the get_weather tool. Keep responses concise."
        ),
        tools=[get_weather],
    )

    adapter = LLMAdapter(agent)

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="user",
        content="What's the weather in San Francisco?",
    )

    print("\n  Sending: 'What's the weather in San Francisco?'")
    start = time.monotonic()

    full_response = ""
    async with adapter.chat(chat_ctx=chat_ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                full_response += chunk.delta.content

    elapsed = time.monotonic() - start

    print(f"\n  Response: {full_response!r}")
    print(f"  Tool called: {tool_was_called}")
    print(f"  Time: {elapsed:.2f}s")

    passed = True
    if not tool_was_called:
        print("\n  FAIL: Tool was not called")
        passed = False
    if not full_response.strip():
        print("\n  FAIL: Empty response")
        passed = False

    if passed:
        print("\n  PASS")
    else:
        print("\n  FAIL")
        sys.exit(1)


async def run_session_reuse_test() -> None:
    """Test that sessions persist across multiple turns."""
    from google.adk.agents import LlmAgent

    from livekit.agents.llm import ChatContext
    from livekit.plugins.google_adk import LLMAdapter

    print("=" * 60)
    print("SESSION TEST: Multi-turn conversation with session reuse")
    print("=" * 60)

    agent = LlmAgent(
        name="session_test_agent",
        model="gemini-2.0-flash",
        instruction=(
            "You are a helpful assistant. Remember what the user tells you. "
            "Keep responses to one sentence."
        ),
    )

    adapter = LLMAdapter(agent)

    # Turn 1: Tell the agent something
    chat_ctx1 = ChatContext()
    chat_ctx1.add_message(role="user", content="My favorite color is blue. Remember that.")

    print("\n  Turn 1: 'My favorite color is blue. Remember that.'")
    response1 = ""
    async with adapter.chat(chat_ctx=chat_ctx1) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                response1 += chunk.delta.content

    print(f"  Response: {response1!r}")

    # Turn 2: Ask about what we said
    chat_ctx2 = ChatContext()
    chat_ctx2.add_message(role="user", content="What is my favorite color?")

    print("\n  Turn 2: 'What is my favorite color?'")
    response2 = ""
    async with adapter.chat(chat_ctx=chat_ctx2) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                response2 += chunk.delta.content

    print(f"  Response: {response2!r}")

    passed = True
    if "blue" not in response2.lower():
        print("\n  WARN: Response may not show session memory (expected 'blue')")
        # Not a hard fail since LLM responses can be unpredictable
    if not response1.strip() or not response2.strip():
        print("\n  FAIL: Empty response")
        passed = False

    sessions_created = len(adapter._sessions)
    print(f"\n  Sessions cached: {sessions_created}")
    if sessions_created != 1:
        print(f"  FAIL: Expected 1 cached session, got {sessions_created}")
        passed = False

    if passed:
        print("\n  PASS")
    else:
        print("\n  FAIL")
        sys.exit(1)


async def run_interactive() -> None:
    """Interactive chat loop using the ADK adapter."""
    from google.adk.agents import LlmAgent

    from livekit.agents.llm import ChatContext
    from livekit.plugins.google_adk import LLMAdapter

    print("=" * 60)
    print("INTERACTIVE: Google ADK LLMAdapter")
    print("Type 'quit' or Ctrl+C to exit")
    print("=" * 60)

    def get_weather(city: str) -> dict:
        """Returns weather information for the given city."""
        print(f"  [TOOL] get_weather(city={city!r})")
        return {"city": city, "temperature": "72F", "condition": "Sunny"}

    def get_time(timezone: str = "UTC") -> dict:
        """Returns the current time in the given timezone."""
        print(f"  [TOOL] get_time(timezone={timezone!r})")
        import datetime

        return {"time": datetime.datetime.now().isoformat(), "timezone": timezone}

    agent = LlmAgent(
        name="interactive_agent",
        model="gemini-2.0-flash",
        instruction=(
            "You are a helpful voice assistant. Keep responses concise "
            "since your output will be spoken aloud via TTS. "
            "You have access to weather and time tools."
        ),
        tools=[get_weather, get_time],
    )

    adapter = LLMAdapter(agent)
    print(f"\n  Model:    {adapter.model}")
    print(f"  Provider: {adapter.provider}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content=user_input)

        start = time.monotonic()
        full_response = ""
        usage_info = None

        try:
            async with adapter.chat(chat_ctx=chat_ctx) as stream:
                print("Agent: ", end="", flush=True)
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        print(chunk.delta.content, end="", flush=True)
                        full_response += chunk.delta.content
                    if chunk.usage:
                        usage_info = chunk.usage
        except Exception as e:
            print(f"\n  ERROR: {e}")
            continue

        elapsed = time.monotonic() - start
        stats = f"  [{elapsed:.2f}s"
        if usage_info:
            stats += f", {usage_info.total_tokens} tokens"
        stats += "]"
        print(f"\n{stats}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="E2E test for Google ADK LLMAdapter")
    parser.add_argument("--smoke", action="store_true", help="Run automated smoke test")
    parser.add_argument("--tool-test", action="store_true", help="Run tool calling test")
    parser.add_argument("--session-test", action="store_true", help="Run session reuse test")
    parser.add_argument("--all", action="store_true", help="Run all automated tests")
    args = parser.parse_args()

    check_api_key()

    if args.all:
        await run_smoke_test()
        print()
        await run_tool_test()
        print()
        await run_session_reuse_test()
    elif args.smoke:
        await run_smoke_test()
    elif args.tool_test:
        await run_tool_test()
    elif args.session_test:
        await run_session_reuse_test()
    else:
        await run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
