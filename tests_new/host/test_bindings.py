"""The native core under Python concurrency and ownership: no deadlock across threads and loops,
no GIL held while native work runs, Python objects passed through the core coming back as
themselves and released afterwards, and native media objects Python owns."""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
import threading
import weakref
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import (
    Agent,
    APIConnectOptions,
    RunContext,
    VoiceSession,
    inference,
    llm,
    tts,
    utils,
)
from livekit.agents.a2a import (  # type: ignore[attr-defined]  # public through `__all__ +=`, which mypy does not follow
    AgentClient,
    DataPart,
    FilePart,
    Message,
    MessageMetadata,
    TextPart,
)
from livekit.agents.llm import Toolset
from livekit.agents.utils import media
from livekit.agents.voice.events import (
    ToolCallEnded,
    ToolCallStarted,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
)
from tests_new.fakes import FakeLLM, tool_call

from .a2a_peer import URI, Peer

AUDIO = Path(__file__).parents[1] / "fixtures" / "change-sophie.opus"


# -- threads, loops and the GIL


def test_runs_on_two_threads_do_not_wait_on_each_other() -> None:
    """Runs on two threads' event loops progress independently: each one's tool waits for the
    other's, which deadlocks if one blocks the core or holds the GIL."""
    arrived = threading.Barrier(2, timeout=10)
    outputs: dict[str, str] = {}

    def run(name: str) -> None:
        agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("meet"), f"done {name}"]))

        @agent.tool
        async def meet(ctx: RunContext) -> str:
            """Meet the other run."""
            await asyncio.to_thread(arrived.wait)
            return "met"

        result = asyncio.run(agent.run("go"))
        outputs[name] = result.output if result.status == "completed" else result.status

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert outputs == {"a": "done a", "b": "done b"}


@pytest.mark.parametrize("encoding", ["jpeg", "provider_format"])
async def test_native_image_encoding_keeps_the_event_loop_running(encoding: str) -> None:
    """Encoding a full-HD frame natively lets the loop run other tasks meanwhile."""
    frame = rtc.VideoFrame(1920, 1080, rtc.VideoBufferType.RGBA, bytes([127]) * (1920 * 1080 * 4))
    ticks = 0
    encoding_done = False

    async def tick() -> None:
        nonlocal ticks
        while not encoding_done:
            ticks += 1
            await asyncio.sleep(0)

    ticker = asyncio.create_task(tick())
    if encoding == "jpeg":
        await media.encode(frame, media.EncodeOptions())
    else:
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content=[llm.ImageContent(image=frame)])
        await chat_ctx.to_provider_format(format="openai")
    encoding_done = True
    await ticker

    assert ticks > 1


def test_a_stalled_log_handler_does_not_stall_the_job() -> None:
    """A job starts, runs and exits while the application's log handler is blocked, and its
    exit line keeps its fields once the handler resumes (a fresh interpreter: the log
    forwarding is process-wide)."""
    run = subprocess.run(
        [sys.executable, "-m", "tests_new.host.stalled_log_job"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert run.returncode == 0, run.stderr


# -- Python objects passed through the core


async def test_a_provider_tool_reaches_the_python_llm_as_the_same_object() -> None:
    """A provider tool given to an agent crosses the core and reaches the Python LLM as the
    very object the application created."""

    class CustomProviderTool(llm.ProviderTool):
        pass

    provider_tool = CustomProviderTool(id="custom_provider_tool")
    model = FakeLLM(["done"])

    await Agent(instructions="Help.", llm=model, tools=[provider_tool]).run("hello")

    [[offered]] = model.tools
    assert offered is provider_tool


async def test_a_finished_run_releases_its_python_toolset() -> None:
    """Once a request that ran an agent returns, the core holds no reference to the toolset it
    called."""

    async def handle_request() -> weakref.ref[Toolset]:
        toolset = Toolset(id="released")
        await Agent(instructions="Help.", llm=FakeLLM(["done"]), tools=[toolset]).run("hello")
        return weakref.ref(toolset)

    released = await handle_request()
    gc.collect()

    assert released() is None


async def test_tool_progress_and_completion_reach_session_listeners() -> None:
    """A tool's `ctx.update()` and its end reach `tool_execution_updated` listeners with the
    call's id."""
    agent = Agent(instructions="Check the booking.")
    updates: list[ToolExecutionUpdatedEvent] = []

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the booking."""
        await ctx.update("Checking the booking system.")
        return "Confirmed."

    model = FakeLLM([tool_call("lookup", call_id="booking-1"), "Done."])
    async with VoiceSession(llm=model, vad=None) as session:
        session.on("tool_execution_updated", updates.append)
        await session.start(agent)
        await session.generate_reply(user_input="check my booking")

    calls = [
        u.update
        for u in updates
        if isinstance(u.update, ToolCallStarted | ToolCallUpdated | ToolCallEnded)
    ]
    assert [type(update) for update in calls] == [ToolCallStarted, ToolCallUpdated, ToolCallEnded]
    progress, ended = calls[1], calls[2]
    assert isinstance(progress, ToolCallUpdated) and isinstance(ended, ToolCallEnded)
    assert (progress.call_id, progress.message) == ("booking-1", "Checking the booking system.")
    assert (ended.call_id, ended.message, ended.status) == ("booking-1", "Confirmed.", "done")


async def test_an_a2a_message_reaches_the_peer_with_its_parts_and_metadata(peer: Peer) -> None:
    """Text, data and file parts and typed metadata are sent as the A2A wire spells them."""
    client = AgentClient(peer.url, conn_options=APIConnectOptions(max_retry=0))
    message = Message(
        parts=[
            TextPart(text="Check"),
            DataPart(data={"account": 42}),
            FilePart(url="https://example.test/coverage.pdf", media_type="application/pdf"),
        ],
        message_id="stable-id",
        metadata=MessageMetadata(channel="portal", application={"trace": "abc"}),
    )

    reply = await client.send(message)
    updates = [update async for update in reply]
    await client.aclose()

    assert updates[-1].state == "completed"
    [sent] = peer.messages
    assert sent["messageId"] == "stable-id"
    assert sent["metadata"] == {URI + "/channel": "portal", "trace": "abc"}
    assert sent["parts"] == [
        {"text": "Check"},
        {"data": {"account": 42}},
        {"url": "https://example.test/coverage.pdf", "mediaType": "application/pdf"},
    ]


# -- the native inference client


# A proxy's last chunk may carry `usage` with null counts (issue #6595).
NULL_USAGE_STREAM = b"""data: {"id":"c","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"c","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"c","choices":[],"created":0,"model":"m","object":"chat.completion.chunk","usage":{"completion_tokens":null,"prompt_tokens":7,"total_tokens":null}}

data: [DONE]

"""


@pytest.fixture
async def gateway() -> AsyncIterator[tuple[str, list[web.Request]]]:
    """An inference gateway answering every completion with `NULL_USAGE_STREAM`; yields its
    base URL and the requests it received."""
    received: list[web.Request] = []

    async def complete(request: web.Request) -> web.Response:
        await request.read()
        received.append(request)
        return web.Response(body=NULL_USAGE_STREAM, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    yield f"http://127.0.0.1:{runner.addresses[0][1]}/v1", received
    await runner.cleanup()


async def test_a_completion_whose_usage_has_null_counts_completes(
    gateway: tuple[str, list[web.Request]],
) -> None:
    """A stream ending in a usage chunk with null counts completes with its text.
    https://github.com/livekit/agents/issues/6595"""
    base_url, _ = gateway
    model = inference.LLM("openai/gpt-4.1-mini", base_url=base_url, api_key="k", api_secret="s")

    result = await Agent(instructions="Help.", llm=model).run("hi")
    await model.aclose()

    assert result.status == "completed" and result.output == "ok"


async def test_a_requests_inference_class_overrides_the_models(
    gateway: tuple[str, list[web.Request]],
) -> None:
    """`inference_class` given to `chat()` is the priority sent, over the model's own."""
    base_url, received = gateway
    model = inference.LLM(
        "openai/gpt-4.1-mini",
        base_url=base_url,
        api_key="k",
        api_secret="s",
        inference_class="priority",
    )
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")

    async with model.chat(chat_ctx=chat_ctx, inference_class="low") as stream:
        await stream.collect()
    await model.aclose()

    [request] = received
    assert request.headers["x-livekit-inference-priority"] == "low"


# -- native media objects Python owns


async def test_closed_partial_decoders_leave_room_for_new_ones() -> None:
    """Decoders closed mid-stream release their native workers: after many, a new decoder
    still decodes a whole file."""
    encoded = AUDIO.read_bytes()
    for _ in range(40):
        partial = media.AudioStreamDecoder(format="audio/ogg")
        partial.push(encoded[:16])
        await partial.aclose()

    decoder = media.AudioStreamDecoder(format="audio/ogg")
    decoder.push(encoded)
    decoder.end_input()
    frames = [frame async for frame in decoder]
    await decoder.aclose()

    assert sum(frame.duration for frame in frames) > 1.0


async def test_a_compressed_synthesis_delivers_its_whole_tail_before_join_returns() -> None:
    """Encoded audio pushed to an emitter in chunks is decoded to the end, final frame
    included, by the time `join()` returns."""
    encoded = AUDIO.read_bytes()
    decoder = media.AudioStreamDecoder(sample_rate=24000, num_channels=1, format="audio/ogg")
    decoder.push(encoded)
    decoder.end_input()
    expected = b"".join([bytes(frame.data) async for frame in decoder])
    await decoder.aclose()
    channel = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test", dst_ch=channel)
    emitter.initialize(
        request_id="r", sample_rate=24000, num_channels=1, mime_type="audio/ogg", stream=False
    )

    for offset in range(0, len(encoded), 997):
        emitter.push(encoded[offset : offset + 997])
    emitter.end_input()
    await emitter.join()
    channel.close()
    events = [event async for event in channel]
    await emitter.aclose()

    assert events[-1].is_final
    assert b"".join(bytes(event.frame.data) for event in events) == expected
