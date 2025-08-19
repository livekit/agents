#!/usr/bin/env python3

"""
Example demonstrating Claude's extended thinking feature with LiveKit Agents.

This example shows how to enable and configure Claude's thinking capabilities,
which allow Claude to show its internal reasoning process before delivering
the final answer.

Requirements:
- ANTHROPIC_API_KEY environment variable set
- A Claude model that supports extended thinking (claude-3-7-sonnet, claude-4-sonnet, claude-4-opus)
"""

import asyncio
import logging
import os

from livekit.agents.llm import ChatContext
from livekit.plugins import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # Ensure we have an API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("Please set the ANTHROPIC_API_KEY environment variable")
        return

    # Example 1: Using thinking with a dict configuration
    print("=== Example 1: Basic Thinking Configuration ===")
    llm_with_thinking = anthropic.LLM(
        model="claude-3-7-sonnet-20250219",  # Make sure to use a model that supports thinking
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,  # Allocate 10k tokens for thinking
        },
    )

    # Create a chat context with a complex problem
    chat_ctx = ChatContext().empty()
    chat_ctx.add_message(
        role="system", content="You are a helpful math tutor. Think step by step through problems."
    )
    chat_ctx.add_message(
        role="user",
        content="What is the derivative of x^3 + 2x^2 - 5x + 7? Please show your thinking process.",
    )

    print("Question: What is the derivative of x^3 + 2x^2 - 5x + 7?")
    print("Claude's response with thinking:")
    print("-" * 50)

    try:
        stream = llm_with_thinking.chat(chat_ctx=chat_ctx)
        full_response = ""

        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                content = chunk.delta.content
                print(content, end="", flush=True)
                full_response += content

        print("\n" + "=" * 50)

    except Exception as e:
        logger.error(f"Error during thinking example: {e}")
        return

    # Example 2: Using TypedDict style configuration
    print("\n=== Example 2: TypedDict Configuration ===")

    thinking_config: anthropic.ThinkingConfig = {"type": "enabled", "budget_tokens": 5000}

    llm_with_typed_thinking = anthropic.LLM(
        model="claude-3-7-sonnet-20250219", thinking=thinking_config
    )

    # Ask a reasoning question
    reasoning_ctx = ChatContext().empty()
    reasoning_ctx.add_message(role="system", content="You are a logical reasoning assistant.")

    reasoning_ctx.add_message(
        role="user",
        content="If all roses are flowers, and some flowers are red, can we conclude that some roses are red? Explain your reasoning.",
    )

    print("Question: Logic puzzle about roses and flowers")
    print("Claude's response with thinking:")
    print("-" * 50)

    try:
        stream = llm_with_typed_thinking.chat(chat_ctx=reasoning_ctx)

        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                content = chunk.delta.content
                print(content, end="", flush=True)

        print("\n" + "=" * 50)

    except Exception as e:
        logger.error(f"Error during reasoning example: {e}")

    # Example 3: Without thinking for comparison
    print("\n=== Example 3: Without Thinking (for comparison) ===")

    llm_no_thinking = anthropic.LLM(
        model="claude-3-7-sonnet-20250219"
        # No thinking configuration
    )

    comparison_ctx = ChatContext().empty()
    comparison_ctx.add_message(role="system", content="You are a helpful assistant.")
    comparison_ctx.add_message(role="user", content="What is 127 * 89?")

    print("Question: What is 127 * 89?")
    print("Claude's response without thinking:")
    print("-" * 50)

    try:
        stream = llm_no_thinking.chat(chat_ctx=comparison_ctx)

        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                content = chunk.delta.content
                print(content, end="", flush=True)

        print("\n" + "=" * 50)

    except Exception as e:
        logger.error(f"Error during comparison example: {e}")

    print("\nThinking configuration allows Claude to show its internal reasoning process,")
    print(
        "which can be particularly useful for complex mathematical, logical, or analytical tasks."
    )


if __name__ == "__main__":
    asyncio.run(main())
