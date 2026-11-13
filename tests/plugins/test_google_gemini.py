"""Gemini LLM and Gemini TTS against a local peer serving the streamGenerateContent SSE protocol."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from google.genai import types
from google.oauth2.credentials import Credentials

from livekit.agents import (
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    NotGivenOr,
    llm,
)
from livekit.plugins.google import LLM
from livekit.plugins.google.beta import GeminiTTS
from livekit.plugins.google.tools import GoogleSearch
from tests.fakes import until

if TYPE_CHECKING:
    # what LLMStream.collect() returns; the public API exports no name for it
    from livekit.agents._generated.llm import _CollectedResponse

OPTIONS = APIConnectOptions(max_retry=0)
WireJSON = Any  # a request body decoded from the wire: untyped JSON the assertions index into


class Recorded(NamedTuple):
    body: WireJSON
    headers: dict[str, str]


def answer(text: str = "Hello") -> dict[str, object]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


class GeminiPeer:
    """Gemini's REST endpoint: records each request and streams the scripted SSE payloads."""

    def __init__(self) -> None:
        self.url = ""
        self.requests: list[Recorded] = []
        self.payloads: list[dict[str, object] | str] = [answer()]
        self.status = 200
        self.delays: list[float] = []
        self.fragmented = False
        self.connections: list[asyncio.BaseTransport] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(Recorded(body=await request.json(), headers=dict(request.headers)))
        if request.transport is not None:
            self.connections.append(request.transport)
        if self.status != 200:
            error = {
                "code": self.status,
                "message": "recorded rejection",
                "status": "INVALID_ARGUMENT",
            }
            return web.json_response({"error": error}, status=self.status)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index, payload in enumerate(self.payloads):
            if index < len(self.delays):
                await asyncio.sleep(self.delays[index])  # real wait: the read timeout is under test
            event = f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"
            if self.fragmented:
                half = len(event) // 2
                await response.write(event[:half].encode())
                await response.write(event[half:].encode())
            else:
                await response.write(event.encode())
        await response.write_eof()
        return response

    @property
    def body(self) -> WireJSON:
        """The last request's JSON body."""
        return self.requests[-1].body

    def llm(
        self,
        *,
        model: str = "gemini-2.5-flash",
        thinking_config: NotGivenOr[types.ThinkingConfigOrDict] = NOT_GIVEN,
        cached_content: NotGivenOr[str] = NOT_GIVEN,
    ) -> LLM:
        return LLM(
            api_key="test",
            http_options=types.HttpOptions(base_url=self.url),
            model=model,
            thinking_config=thinking_config,
            cached_content=cached_content,
        )


@pytest.fixture
async def gemini() -> AsyncIterator[GeminiPeer]:
    peer = GeminiPeer()
    app = web.Application()
    app.router.add_post("/{path:.*}", peer.handle)
    async with TestServer(app) as server:
        peer.url = str(server.make_url(""))
        yield peer
    # the SDK's AsyncClient.__del__ schedules aclose() even on a closed client; settle it
    await asyncio.sleep(0)


async def chat(
    provider: LLM,
    chat_ctx: llm.ChatContext | None = None,
    *,
    tools: list[llm.Tool] | None = None,
    tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
    extra_kwargs: NotGivenOr[dict[str, object]] = NOT_GIVEN,
) -> _CollectedResponse:
    """One chat turn, collected; the provider is closed afterwards."""
    async with (
        provider,
        provider.chat(
            chat_ctx=chat_ctx or llm.ChatContext.empty(),
            conn_options=OPTIONS,
            tools=tools,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        ) as stream,
    ):
        return await stream.collect()


def user_says(text: str) -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content=text)
    return ctx


async def until_disconnected(peer: GeminiPeer) -> None:
    await until(
        lambda: all(c.is_closing() for c in peer.connections), "the client closed its connection"
    )


# ---- replies


async def test_a_call_written_beside_text_is_a_call_and_not_speech(gemini: GeminiPeer) -> None:
    """A part with both a function call and text yields only the call.

    https://github.com/livekit/agents/pull/4999
    """
    gemini.payloads = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                                "text": "get_weather",
                            }
                        ],
                    }
                }
            ]
        }
    ]

    response = await chat(gemini.llm())

    assert response.text == ""
    [call] = response.tool_calls
    assert (call.name, json.loads(call.arguments)) == ("get_weather", {"city": "Paris"})


async def test_thought_summaries_are_never_spoken(gemini: GeminiPeer) -> None:
    """Parts Gemini flags as thoughts stay out of the reply text.

    https://github.com/livekit/agents/pull/6428
    """
    thought = {"text": "**Calculating Refill Dates** I'm processing", "thought": True}
    gemini.payloads = [{"candidates": [{"content": {"parts": [thought]}}]}, answer("Done.")]

    assert (await chat(gemini.llm())).text == "Done."


async def test_a_call_without_a_name_is_a_non_retryable_error(gemini: GeminiPeer) -> None:
    """A nameless function call fails the turn for good instead of being retried."""
    gemini.payloads = [{"candidates": [{"content": {"parts": [{"functionCall": {"args": {}}}]}}]}]

    with pytest.raises(APIConnectionError) as failure:
        await chat(gemini.llm())

    assert not failure.value.retryable


@pytest.mark.parametrize("empty", [False, True, "thoughts"])
async def test_metadata_and_empty_chunks_do_not_replace_a_completion(
    gemini: GeminiPeer, empty: bool | str
) -> None:
    """Only real output completes a turn; an empty one fails retryably.

    https://github.com/livekit/agents/issues/4307 https://github.com/livekit/agents/issues/4706
    """
    gemini.payloads = [{}, {"candidates": [{"content": {"parts": []}}]}]
    if empty == "thoughts":
        gemini.payloads.append(
            {"candidates": [{"content": {"parts": [{"text": "secret", "thought": True}]}}]}
        )
    elif not empty:
        gemini.payloads.append(answer())
    gemini.payloads.append({"candidates": [{"finishReason": "STOP"}]})

    if empty:
        with pytest.raises(APIStatusError, match="no response generated") as failure:
            await chat(gemini.llm(), user_says("Hi"))
        assert failure.value.retryable
    else:
        assert (await chat(gemini.llm(), user_says("Hi"))).text == "Hello"
    assert len(gemini.requests) == 1


@pytest.mark.parametrize(
    "status, payload, retryable",
    [
        (200, {"candidates": [{"finishReason": "SAFETY"}]}, False),
        (200, {"promptFeedback": {"blockReason": "SAFETY"}}, False),
        (400, {}, False),
        (503, {}, True),
    ],
)
async def test_provider_rejections_are_typed_and_keep_retry_policy(
    gemini: GeminiPeer, status: int, payload: dict[str, object], retryable: bool
) -> None:
    """Blocks and HTTP rejections surface as APIStatusError with the right retry policy.

    https://github.com/livekit/agents/pull/4637 https://github.com/livekit/agents/pull/4917
    """
    gemini.payloads, gemini.status = [payload], status

    with pytest.raises(APIStatusError) as failure:
        await chat(gemini.llm())

    assert failure.value.retryable is retryable
    if status != 200:
        assert failure.value.status_code == status
        assert "recorded rejection" in str(failure.value.body)


async def test_a_499_rejection_reads_as_a_cancelled_empty_reply(gemini: GeminiPeer) -> None:
    """Gemini's 499 is the caller's own cancellation, not a provider failure."""
    gemini.status = 499

    assert (await chat(gemini.llm())).text == ""


@pytest.mark.parametrize("malformed", [False, True])
async def test_fragmented_sse_is_reassembled_and_bad_json_is_a_framework_error(
    gemini: GeminiPeer, malformed: bool
) -> None:
    """SSE events split across writes are reassembled; bad JSON never leaks JSONDecodeError.

    https://github.com/livekit/agents/issues/2683
    """
    gemini.payloads = [answer("first"), '{"candidates":' if malformed else answer(" last")]
    gemini.fragmented = True
    received: list[str] = []

    async with (
        gemini.llm() as provider,
        provider.chat(chat_ctx=llm.ChatContext.empty(), conn_options=OPTIONS) as stream,
    ):
        try:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    received.append(chunk.delta.content)
        except APIConnectionError as error:
            assert malformed
            assert not error.retryable, "a partial answer cannot be retried"
        else:
            assert not malformed

    assert received == (["first"] if malformed else ["first", " last"])


@pytest.mark.parametrize(
    "stalled",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/5972 reproduces: read_timeout does not stop a stalled Gemini stream",
            ),
        ),
    ],
)
async def test_read_timeout_bounds_silence_not_the_whole_stream(
    gemini: GeminiPeer, stalled: bool
) -> None:
    """Steady chunks outlive the silence budget; a stalled stream times out. Custom headers stay.

    https://github.com/livekit/agents/issues/5972 https://github.com/livekit/agents/issues/7987
    """
    if stalled:
        gemini.payloads, gemini.delays = [answer("0"), answer("1")], [0, 0.2]
    else:
        gemini.payloads, gemini.delays = [answer(str(i)) for i in range(5)], [0.03] * 5
    provider = LLM(
        api_key="test",
        http_options=types.HttpOptions(base_url=gemini.url, headers={"X-Regression": "kept"}),
    )
    options = APIConnectOptions(max_retry=0, connect_timeout=1, read_timeout=0.1)

    async with provider, provider.chat(chat_ctx=llm.ChatContext.empty(), conn_options=options) as s:
        if stalled:
            timed_out = False
            try:
                await s.collect()
            except APITimeoutError:
                timed_out = True
            assert timed_out, "read_timeout must stop a stalled Gemini response"
        else:
            assert (await s.collect()).text == "01234"

    headers = gemini.requests[0].headers
    assert headers["X-Regression"] == "kept"
    assert "X-Server-Timeout" not in headers
    assert "livekit-agents/" in headers["x-goog-api-client"]


async def test_usage_counts_tool_prompts_as_input_and_thoughts_as_output(
    gemini: GeminiPeer,
) -> None:
    """Gemini counts built-in tool prompts and thinking outside its totals; ours include them.

    https://github.com/livekit/agents/issues/7889
    """
    gemini.payloads = [
        answer("ok"),
        {
            "usageMetadata": {
                "promptTokenCount": 100,
                "toolUsePromptTokenCount": 40,
                "cachedContentTokenCount": 60,
                "candidatesTokenCount": 20,
                "thoughtsTokenCount": 10,
            }
        },
    ]

    usage = (await chat(gemini.llm())).usage
    assert usage is not None

    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (140, 30)
    assert (usage.output_reasoning_tokens, usage.cache_read_tokens) == (10, 60)


@pytest.mark.parametrize("model, echoed", [("gemini-2.5-flash", True), ("gemini-2.0-flash", False)])
async def test_thought_signatures_go_back_with_their_call(
    gemini: GeminiPeer, model: str, echoed: bool
) -> None:
    """Gemini 2.5+ needs a call's thought signature back on the next turn; older models do not.

    https://github.com/livekit/agents/pull/4595
    """
    signature = base64.b64encode(b"signed").decode()
    call = {
        "functionCall": {"id": "c1", "name": "lookup", "args": {}},
        "thoughtSignature": signature,
    }
    gemini.payloads = [{"candidates": [{"content": {"role": "model", "parts": [call]}}]}]
    provider = gemini.llm(model=model)
    ctx = user_says("Look it up")

    async with provider:
        async with provider.chat(chat_ctx=ctx, conn_options=OPTIONS) as stream:
            [tool_call] = (await stream.collect()).tool_calls
        gemini.payloads = [answer("Found it")]
        ctx.items.append(llm.FunctionCall(call_id="c1", name="lookup", arguments="{}"))
        ctx.items.append(
            llm.FunctionCallOutput(call_id="c1", name="lookup", output="42", is_error=False)
        )
        async with provider.chat(chat_ctx=ctx, conn_options=OPTIONS) as stream:
            await stream.collect()

    [model_turn] = [c for c in gemini.body["contents"] if c["role"] == "model"]
    assert (model_turn["parts"][0].get("thoughtSignature") == signature) == echoed


@pytest.mark.parametrize("explicit", [False, True])
async def test_vertex_authenticates_with_workload_credentials(
    gemini: GeminiPeer, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """Ambient or explicit workload credentials authenticate Vertex without key files.

    https://github.com/livekit/agents/issues/1428 https://github.com/livekit/agents/issues/2387
    """
    import google.auth

    for name in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    ambient = Credentials(token="ambient-token")  # type: ignore[no-untyped-call]  # google-auth is untyped
    monkeypatch.setattr(google.auth, "default", lambda *a, **kw: (ambient, "workload-project"))
    explicit_token = Credentials(token="explicit-token")  # type: ignore[no-untyped-call]  # google-auth is untyped
    credentials = explicit_token if explicit else None
    provider = LLM(
        vertexai=True,
        credentials=credentials,
        http_options=types.HttpOptions(base_url=gemini.url),
    )

    assert (await chat(provider)).text == "Hello"

    token = "explicit-token" if explicit else "ambient-token"
    assert gemini.requests[0].headers["Authorization"] == f"Bearer {token}"


# ---- request shape


async def test_a_conversation_without_system_messages_sends_no_system_instruction(
    gemini: GeminiPeer,
) -> None:
    """No system message means no systemInstruction in the request."""
    await chat(gemini.llm(), user_says("hello"))

    assert "systemInstruction" not in gemini.body


async def test_a_cached_prompt_is_not_sent_again(gemini: GeminiPeer) -> None:
    """With a cache attached the system prompt and raw tools stay out; Gemini 400s on them.

    https://github.com/livekit/agents/pull/5675 https://github.com/livekit/agents/pull/6426
    """
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content="system prompt that lives in cache")
    ctx.add_message(role="user", content="hi")
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.AUTO
        )
    )

    await chat(
        gemini.llm(model="gemini-3-flash-preview", cached_content="cachedContents/abc"),
        ctx,
        extra_kwargs={
            "tools": [types.Tool(google_search=types.GoogleSearch())],
            "tool_config": tool_config,
        },
    )

    assert gemini.body["cachedContent"] == "cachedContents/abc"
    assert not {"systemInstruction", "tools", "toolConfig"} & gemini.body.keys()


@llm.function_tool
async def get_weather(city: str) -> str:
    """Look up the weather."""
    return city


async def test_vertex_gemini_3_drops_the_provider_tool_beside_function_tools(
    gemini: GeminiPeer,
) -> None:
    """Vertex AI cannot mix provider and function tools, even on Gemini 3.

    https://github.com/livekit/agents/pull/6426
    """
    provider = LLM(
        model="gemini-3-flash-preview",
        vertexai=True,
        project="test-project",
        location="us-central1",
        credentials=Credentials(token="t"),  # type: ignore[no-untyped-call]  # google-auth is untyped
        http_options=types.HttpOptions(base_url=gemini.url),
    )

    await chat(provider, tools=[get_weather, GoogleSearch()])

    assert [list(tool) for tool in gemini.body["tools"]] == [["functionDeclarations"]]
    assert "includeServerSideToolInvocations" not in json.dumps(gemini.body.get("toolConfig", {}))


@pytest.mark.parametrize("model", ["gemma-4-31b-it", "models/gemma-4-31b-it"])
@pytest.mark.parametrize("level", ["MINIMAL", "HIGH"])
async def test_gemma_4_thinking_level_reaches_the_request(
    gemini: GeminiPeer, model: str, level: str
) -> None:
    """Gemma 4 takes the level it documents, under a bare or a qualified model name.

    https://github.com/livekit/agents/issues/7071
    """
    await chat(gemini.llm(model=model, thinking_config=types.ThinkingConfig(thinking_level=level)))

    assert gemini.body["generationConfig"]["thinkingConfig"]["thinking_level"] == level


@pytest.mark.parametrize("model", ["gemini-3-pro-preview", "gemma-4-31b-it"])
async def test_include_thoughts_survives_a_thinking_level(gemini: GeminiPeer, model: str) -> None:
    """Asking for a level keeps include_thoughts. https://github.com/livekit/agents/pull/7095"""
    thinking: types.ThinkingConfigDict = {
        "thinking_level": types.ThinkingLevel.HIGH,
        "include_thoughts": True,
    }

    await chat(gemini.llm(model=model, thinking_config=thinking))

    assert gemini.body["generationConfig"]["thinkingConfig"] == {
        "thinking_level": "HIGH",
        "include_thoughts": True,
    }


@pytest.mark.parametrize(
    "choice, mode, names",
    [
        ("auto", "AUTO", None),
        ("none", "NONE", None),
        ("required", "ANY", ["get_weather"]),
        ({"type": "function", "function": {"name": "get_weather"}}, "ANY", ["get_weather"]),
    ],
)
async def test_tool_choice_reaches_the_function_calling_request(
    gemini: GeminiPeer, choice: llm.ToolChoice, mode: str, names: list[str] | None
) -> None:
    """Required and named tool choices serialize to Gemini's function calling modes.

    https://github.com/livekit/agents/issues/1828
    """
    await chat(gemini.llm(), tools=[get_weather], tool_choice=choice)

    calling = gemini.body["toolConfig"]["functionCallingConfig"]
    assert (calling["mode"], calling.get("allowedFunctionNames")) == (mode, names)


@llm.function_tool
async def save_contact(fields: dict[str, str]) -> None:
    """Save the contact fields gathered in the conversation (free-form keys)."""


async def test_free_form_object_parameters_keep_their_value_schema(gemini: GeminiPeer) -> None:
    """The text API gets the full JSON schema, so a dict[str, str] keeps its string values.

    https://github.com/livekit/agents/issues/7170
    """
    await chat(gemini.llm(), tools=[save_contact])

    [declaration] = gemini.body["tools"][0]["functionDeclarations"]
    assert "parameters" not in declaration
    fields = declaration["parameters_json_schema"]["properties"]["fields"]
    assert (fields["type"], fields["additionalProperties"]) == ("object", {"type": "string"})


async def test_closing_the_llm_closes_its_connection(gemini: GeminiPeer) -> None:
    """Closing the LLM releases the HTTP client the GenAI SDK opened for it.

    https://github.com/livekit/agents/issues/2729 https://github.com/livekit/agents/pull/6686
    """
    await chat(gemini.llm(), user_says("Hi"))

    await until_disconnected(gemini)


# ---- Gemini TTS


@pytest.fixture
def gemini_tts(gemini: GeminiPeer, monkeypatch: pytest.MonkeyPatch) -> GeminiPeer:
    """The peer as Gemini's TTS endpoint, answering every request with 0.1 s of PCM."""
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", gemini.url)
    audio = {
        "inlineData": {
            "mimeType": "audio/L16;codec=pcm;rate=24000",
            "data": base64.b64encode(bytes(4800)).decode(),
        }
    }
    gemini.payloads = [{"candidates": [{"content": {"parts": [audio]}}]}]
    return gemini


async def speak(provider: GeminiTTS, text: str) -> None:
    async with provider, provider.synthesize(text, conn_options=OPTIONS) as stream:
        assert [event async for event in stream]


async def test_gemini_tts_asks_for_headerless_pcm(gemini_tts: GeminiPeer) -> None:
    """The emitter plays raw PCM, which only an explicit response_format guarantees.

    https://github.com/livekit/agents/pull/7425
    """
    await speak(GeminiTTS(api_key="test", model="gemini-3.8-flash-tts"), "Hello world")

    assert gemini_tts.body["generationConfig"]["response_format"] == {
        "audio": {"mime_type": "AUDIO_L16"}
    }


async def test_closing_gemini_tts_closes_its_connection(gemini_tts: GeminiPeer) -> None:
    """Closing the TTS releases the HTTP client the GenAI SDK opened for it.

    https://github.com/livekit/agents/issues/2666 https://github.com/livekit/agents/issues/2828
    """
    await speak(GeminiTTS(api_key="test", model="gemini-3.8-flash-tts"), "Hello world")

    await until_disconnected(gemini_tts)


async def test_gemini_tts_names_the_speaker_of_every_multi_speaker_turn(
    gemini_tts: GeminiPeer,
) -> None:
    """Each part names its speaker, even plain words. https://github.com/livekit/agents/pull/7425"""
    provider = GeminiTTS(
        api_key="test",
        model="gemini-3.8-flash-tts",
        speakers={"Sienna": "Kore", "Comanchero": "Puck"},
        speaker="Sienna",
    )

    await speak(provider, "Sienna?")

    assert gemini_tts.body["contents"] == [
        {"parts": [{"text": '"Sienna?"', "speech_metadata": {"speaker": "Sienna"}}]}
    ]
    speech = gemini_tts.body["generationConfig"]["speechConfig"]
    assert "voice_config" not in speech
    voices = speech["multi_speaker_voice_config"]["speaker_voice_configs"]
    assert [voice["speaker"] for voice in voices] == ["Sienna", "Comanchero"]


async def test_gemini_tts_gives_each_styled_sentence_its_own_part(gemini_tts: GeminiPeer) -> None:
    """Gemini takes one style per part, so styled sentences are not folded together.

    https://github.com/livekit/agents/pull/7425
    """
    await speak(
        GeminiTTS(api_key="test", model="gemini-3.8-flash-tts"),
        '<expr type="expression" label="Warm"/> Hello. '
        '<expr type="expression" label="Sad"/> <expr type="sound" label="sigh"/> Goodbye.',
    )

    assert gemini_tts.body["contents"] == [
        {
            "parts": [
                {"text": '"Hello."', "speech_metadata": {"style": "Warm"}},
                # the inline event stays in the words it belongs to
                {"text": '"<sigh> Goodbye."', "speech_metadata": {"style": "Sad"}},
            ]
        }
    ]
