"""Differential test of the call path: the same scripted call on the reference (``livekit-agents/``)
and on the copy over the Rust core (``livekit-agents-next/``), compared as a developer sees it.

Each package is installed into its own venv (the reference with uv, the copy with maturin), and
each scenario runs once per venv in a subprocess that prints a JSON transcript: the public session
events in order, the LLM requests (what the model was shown), the final ``session.history``, the
SpeechHandle outcomes, the tool calls, and what reached the room (seconds of agent audio, its text).
Ids and timestamps are dropped. The orchestrator compares the two transcripts and prints a diff.

Both stacks run on the reference tests' virtual-time loop (``tests/virtual_time.py``) with the same
scripted STT/LLM/TTS: the copy over ``_core.FakeRoom``, whose clock moves in lockstep with the
loop's, the reference over the tests' fake audio/text IO behind a ``TranscriptSynchronizer``, as its
``RoomIO`` wires it.

    python livekit-agents-next/scripts/parity.py                 # build if needed, every scenario
    python livekit-agents-next/scripts/parity.py say tool_call   # some scenarios
    python livekit-agents-next/scripts/parity.py --no-build      # skip the (incremental) rebuild
    python livekit-agents-next/scripts/parity.py --list
    python livekit-agents-next/scripts/parity.py --show say      # both transcripts, side by side

The venvs live in ``$LK_PARITY_VENVS`` (default: a directory under the system temp dir). It exits
non-zero when a scenario diverges in a way not listed in ``KNOWN``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterable, Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 24000


# Differences listed rather than fixed (see the report printed with them), as (name, why, a
# normaliser applied to both transcripts). A scenario they alone separate counts as agreeing.
KNOWN: list[tuple[str, str, Callable[[dict[str, Any], dict[str, Any]], None]]] = []


# ============================================================================ the orchestrator


def venvs_dir() -> Path:
    default = Path(tempfile.gettempdir()) / (
        "lk-parity-" + hashlib.sha1(str(REPO).encode()).hexdigest()[:8]
    )
    return Path(os.environ.get("LK_PARITY_VENVS", default))


def python_of(stack: str) -> Path:
    return venvs_dir() / stack / "bin" / "python"


def build(stack: str, *, rebuild: bool) -> None:
    venv = venvs_dir() / stack
    env = {**os.environ, "VIRTUAL_ENV": str(venv)}
    fresh = not python_of(stack).exists()
    if fresh:
        subprocess.run(["uv", "venv", "-q", "-p", "3.12", str(venv)], check=True)
    if stack == "ref":
        if fresh:
            subprocess.run(
                ["uv", "pip", "install", "-q", "-e", str(REPO / "livekit-agents"), "pytest"],
                check=True,
                env=env,
            )
        return
    if fresh:
        # the dependencies, and a first build through PEP 517
        subprocess.run(
            ["uv", "pip", "install", "-q", "-e", str(REPO / "livekit-agents-next"), "pytest"],
            check=True,
            env=env,
        )
    if rebuild or fresh:
        proc = subprocess.run(
            ["maturin", "develop", "--uv", "--release", "-q"],
            env=env,
            cwd=str(REPO / "livekit-agents-next"),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.exit(f"maturin develop failed:\n{proc.stdout}\n{proc.stderr}")


def run_stack(stack: str, scenario: str) -> dict[str, Any]:
    proc = subprocess.run(
        [str(python_of(stack)), __file__, "--run", scenario],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
        env={**os.environ, "LK_PARITY_STACK": stack},
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PARITY "):
            return json.loads(line[len("PARITY ") :])  # type: ignore[no-any-return]
    return {"crashed": proc.returncode, "stderr": proc.stderr.strip().splitlines()[-15:]}


def lines_of(transcript: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in transcript.items():
        if isinstance(value, list):
            out.append(f"{key}:")
            out.extend(f"  {json.dumps(v, sort_keys=True)}" for v in value)
        else:
            out.append(f"{key}: {json.dumps(value, sort_keys=True)}")
    return out


def _tolerate_audio(ref: dict[str, Any], nxt: dict[str, Any]) -> None:
    """The reference ends each synthesis with two 10 ms silent markers (the chunked stream's and
    the stream adapter's), the core with one: a few markers per speech are not a difference."""
    try:
        a, b = ref["room"][0], nxt["room"][0]
    except (KeyError, IndexError):
        return
    speeches = sum(1 for e in ref.get("events", []) if e[0] == "speech_created")
    # a tts_node may repeat frames, markers included
    if abs(a["audio_s"] - b["audio_s"]) <= 0.05 * speeches + 0.005:
        b["audio_s"] = a["audio_s"]


def orchestrate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scenarios", nargs="*")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(SCENARIOS))
        return 0
    names = args.scenarios or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {unknown}")
    for stack in ("ref", "next"):
        build(stack, rebuild=not args.no_build)

    diverged: list[str] = []
    for name in names:
        ref, nxt = run_stack("ref", name), run_stack("next", name)
        _tolerate_audio(ref, nxt)
        raw = lines_of(ref) != lines_of(nxt)
        known = []
        for key, _, normalise in KNOWN:
            before = (lines_of(ref), lines_of(nxt))
            normalise(ref, nxt)
            if before != (lines_of(ref), lines_of(nxt)) and before[0] != before[1]:
                known.append(key)
        a, b = lines_of(ref), lines_of(nxt)
        if args.show:
            print(f"==== {name} (ref)\n" + "\n".join(a) + f"\n==== {name} (next)\n" + "\n".join(b))
        diff = [
            d
            for d in difflib.unified_diff(a, b, "ref", "next", n=2, lineterm="")
            if not d.startswith(("---", "+++"))
        ]
        if not diff and (not raw or not known):
            print(f"{name}: same")
        elif not diff:
            print(f"{name}: same but for known differences ({', '.join(known)})")
        else:
            diverged.append(name)
            print(f"{name}: DIVERGES")
            print("\n".join(f"    {d}" for d in diff))
    print(f"\n{len(names) - len(diverged)}/{len(names)} scenarios agree")
    print("known differences:")
    for key, why, _ in KNOWN:
        print(f"  {key}: {why}")
    return 1 if diverged else 0


# ============================================================================ inside a venv
# Everything below runs on one stack: `livekit.agents` is the reference or the copy.


def scenario_main(name: str) -> None:
    sys.path.insert(0, str(REPO))
    from tests.virtual_time import _now, _perf, _VirtualTimeLoop

    with contextlib.ExitStack() as stack:
        import time
        from unittest import mock

        stack.enter_context(mock.patch.object(time, "time", _now))
        stack.enter_context(mock.patch.object(time, "perf_counter", _perf))
        with contextlib.suppress(Exception):
            from tests.virtual_time import _patched_model_clocks

            stack.enter_context(_patched_model_clocks())
        loop = _VirtualTimeLoop()
        asyncio.set_event_loop(loop)
        try:
            transcript = loop.run_until_complete(asyncio.wait_for(run_scenario(name), 120))
        finally:
            loop.close()
    print("PARITY " + json.dumps(transcript))


async def run_scenario(name: str) -> dict[str, Any]:
    call = Call()
    try:
        await SCENARIOS[name](call)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        call.log("scenario_raised", f"{type(e).__name__}: {e}")
    transcript = call.transcript()
    if call.sync is not None:
        await call.sync.aclose()
    return transcript


def _stack_name() -> str:
    return os.environ.get("LK_PARITY_STACK", "?")


def _models() -> Any:
    """The scripted providers, built on whichever `livekit.agents` this venv has."""
    from livekit import rtc
    from livekit.agents import APIConnectionError, APIConnectOptions, llm, stt, tts
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

    class Models:
        pass

    m = Models()

    class ScriptedLLM(llm.LLM):
        """Answers by `respond(texts, tools)`; every request is logged as the model saw it."""

        def __init__(self, call: Call, respond: Callable[..., list[Any]] | None = None) -> None:
            super().__init__()
            self._call = call
            self._respond = respond or default_reply

        @property
        def model(self) -> str:
            return "scripted"

        def chat(
            self, *, chat_ctx: llm.ChatContext, tools: list[Any] | None = None, **_: object
        ) -> llm.LLMStream:
            names = sorted(
                t.info.name for t in (tools or []) if hasattr(t, "info") and hasattr(t.info, "name")
            )
            seen = [item_view(i) for i in chat_ctx.items]
            self._call.llm_requests.append({"tools": names, "items": seen})
            reply = self._respond(seen, names)
            return _LLMStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools or [],
                conn_options=DEFAULT_API_CONNECT_OPTIONS,
                reply=reply,
                call=self._call,
            )

    class _LLMStream(llm.LLMStream):
        def __init__(self, *args: Any, reply: list[Any], call: Call, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._reply = reply
            self._call_ref = call

        async def _run(self) -> None:
            for part in self._reply:
                if isinstance(part, tuple):
                    _, tool, arguments = part
                    self._call_ref.counter += 1
                    delta = llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            llm.FunctionToolCall(
                                name=tool,
                                arguments=json.dumps(arguments),
                                call_id=f"call_{self._call_ref.counter}",
                            )
                        ],
                    )
                    self._event_ch.send_nowait(llm.ChatChunk(id="chunk", delta=delta))
                    continue
                for word in _words(part):
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id="chunk", delta=llm.ChoiceDelta(role="assistant", content=word)
                        )
                    )

    class FailingLLM(llm.LLM):
        @property
        def model(self) -> str:
            return "failing"

        def chat(self, *, chat_ctx: llm.ChatContext, **_: object) -> llm.LLMStream:
            return _FailingStream(
                self, chat_ctx=chat_ctx, tools=[], conn_options=DEFAULT_API_CONNECT_OPTIONS
            )

    class _FailingStream(llm.LLMStream):
        async def _run(self) -> None:
            raise APIConnectionError("scripted llm failure", retryable=False)

    class ScriptedSTT(stt.STT):
        """Hears the next scripted utterance in each burst of audio: start of speech on its first
        frame; after `after` seconds its interims, its final, and end of speech."""

        def __init__(self, call: Call) -> None:
            super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
            self.call = call

        @property
        def model(self) -> str:
            return "scripted"

        async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
            raise NotImplementedError

        def stream(self, **_: object) -> stt.RecognizeStream:
            return _Recognize(stt=self, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    class _Recognize(stt.RecognizeStream):
        async def _run(self) -> None:
            script: list[Utterance] = self._stt.call.utterances  # type: ignore[attr-defined]
            heard, speaking = 0.0, False
            async for frame in self._input_ch:
                if not isinstance(frame, rtc.AudioFrame) or not script:
                    continue
                if not speaking:
                    speaking = True
                    self._emit(stt.SpeechEventType.START_OF_SPEECH)
                heard += frame.duration
                u = script[0]
                if heard + 1e-6 < u.after:
                    continue
                script.pop(0)
                for text in u.interims:
                    self._emit(stt.SpeechEventType.INTERIM_TRANSCRIPT, text)
                if u.final is not None:
                    self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, u.final)
                self._emit(stt.SpeechEventType.END_OF_SPEECH)
                heard, speaking = 0.0, False

        def _emit(self, kind: Any, text: str | None = None) -> None:
            alternatives = []
            if text is not None:
                alternatives = [stt.SpeechData(language="en", text=text, confidence=1.0)]
            self._event_ch.send_nowait(stt.SpeechEvent(type=kind, alternatives=alternatives))

    class FailingSTT(stt.STT):
        def __init__(self) -> None:
            super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        @property
        def model(self) -> str:
            return "failing"

        async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
            raise APIConnectionError("scripted stt failure", retryable=False)

        def stream(self, **_: object) -> stt.RecognizeStream:
            return _FailingRecognize(stt=self, conn_options=APIConnectOptions(max_retry=0))

    class _FailingRecognize(stt.RecognizeStream):
        async def _run(self) -> None:
            raise APIConnectionError("scripted stt failure", retryable=False)

    class SilentTTS(tts.TTS):
        """20 ms of silence per character, one chunked synthesis per sentence."""

        def __init__(self, call: Call) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            )
            self.call = call

        @property
        def model(self) -> str:
            return "silent"

        def synthesize(
            self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
        ) -> tts.ChunkedStream:
            self.call.tts_requests.append(text)
            return _Chunked(tts=self, input_text=text, conn_options=conn_options)

    class _Chunked(tts.ChunkedStream):
        async def _run(self, output_emitter: tts.AudioEmitter) -> None:
            output_emitter.initialize(
                request_id="req",
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
            )
            output_emitter.push(b"\x00\x00" * (SAMPLE_RATE // 50 * max(1, len(self.input_text))))
            output_emitter.flush()

    class FailingTTS(tts.TTS):
        def __init__(self) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            )

        @property
        def model(self) -> str:
            return "failing"

        def synthesize(
            self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
        ) -> tts.ChunkedStream:
            return _FailingChunked(tts=self, input_text=text, conn_options=conn_options)

    class _FailingChunked(tts.ChunkedStream):
        async def _run(self, output_emitter: tts.AudioEmitter) -> None:
            raise APIConnectionError("scripted tts failure", retryable=False)

    from livekit.agents import vad

    class ScriptedVAD(vad.VAD):
        """Hears each utterance the STT hears: start of speech on its first frame, an inference
        per frame, end of speech on its last."""

        def __init__(self, call: Call) -> None:
            super().__init__(capabilities=vad.VADCapabilities(update_interval=0.02))
            self.call = call

        @property
        def model(self) -> str:
            return "scripted"

        def stream(self) -> vad.VADStream:
            return _VadStream(self)

    class _VadStream(vad.VADStream):
        async def _main_task(self) -> None:
            call: Call = self._vad.call  # type: ignore[attr-defined]
            heard, index, samples = 0.0, 0, 0
            async for frame in self._input_ch:
                if not isinstance(frame, rtc.AudioFrame) or index >= len(call.vad_bursts):
                    continue
                samples += frame.samples_per_channel
                if heard == 0.0:
                    self._emit(vad.VADEventType.START_OF_SPEECH, samples, 0.0, True)
                heard += frame.duration
                done = heard + 1e-6 >= call.vad_bursts[index]
                self._emit(vad.VADEventType.INFERENCE_DONE, samples, heard, not done)
                if done:
                    self._emit(vad.VADEventType.END_OF_SPEECH, samples, heard, False)
                    heard, index = 0.0, index + 1

        def _emit(self, kind: Any, samples: int, speech: float, speaking: bool) -> None:
            self._event_ch.send_nowait(
                vad.VADEvent(
                    type=kind,
                    samples_index=samples,
                    timestamp=0.0,
                    speech_duration=speech,
                    silence_duration=0.0,
                    probability=1.0 if speaking else 0.0,
                    speaking=speaking,
                )
            )

    m.ScriptedVAD = ScriptedVAD
    m.ScriptedLLM, m.FailingLLM = ScriptedLLM, FailingLLM
    m.ScriptedSTT, m.FailingSTT = ScriptedSTT, FailingSTT
    m.SilentTTS, m.FailingTTS = SilentTTS, FailingTTS
    return m


def _words(text: str) -> list[str]:
    parts = text.split(" ")
    return [p + (" " if i < len(parts) - 1 else "") for i, p in enumerate(parts)]


class Utterance:
    def __init__(
        self, final: str | None, *, after: float = 0.5, interims: tuple[str, ...] = ()
    ) -> None:
        self.final, self.after, self.interims = final, after, interims


COUNTING = "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve."
LONG = "This is a long answer that keeps going for a while. It has a second sentence too."


def default_reply(items: list[dict[str, Any]], tools: list[str]) -> list[Any]:
    """Reads a tool's output back, calls a tool the message names, else echoes."""
    last = items[-1] if items else {}
    if last.get("type") == "function_call_output":
        return [f"The tool said: {last.get('output')}"]
    text = last.get("text", "") if last.get("type") == "message" else ""
    for word, tool, args in (
        ("weather", "get_weather", {"city": "Paris"}),
        ("billing", "to_billing", {}),
        ("support", "to_support", {}),
    ):
        if word in text and tool in tools:
            return [("tool", tool, args)]
    if "long" in text:
        return [LONG]
    if last.get("role") != "user":
        return ["Hello, how can I help?"]
    return [f"You said: {text}"]


def item_view(item: Any) -> dict[str, Any]:
    kind = getattr(item, "type", "?")
    view: dict[str, Any] = {"type": kind}
    if kind == "message":
        view["role"] = item.role
        view["text"] = item.text_content or ""
        if getattr(item, "interrupted", False):
            view["interrupted"] = True
    elif kind == "function_call":
        view["name"] = item.name
        view["arguments"] = json.loads(item.arguments or "{}")
    elif kind == "function_call_output":
        view["name"] = item.name
        view["output"] = item.output
        if item.is_error:
            view["is_error"] = True
    elif kind == "agent_handoff":
        view["old"] = item.old_agent_id
        view["new"] = item.new_agent_id
    return view


class Call:
    """One stack's view of a scenario: the session, its room, and everything observed."""

    def __init__(self) -> None:
        self.stack = _stack_name()
        self.models = _models()
        self.events: list[list[Any]] = []
        self.metrics: list[str] = []
        self.llm_requests: list[dict[str, Any]] = []
        self.tts_requests: list[str] = []
        self.handles: list[tuple[str, Any]] = []
        self.utterances: list[Utterance] = []
        self.vad_bursts: list[float] = []
        self.counter = 0
        self.session: Any = None
        self.room: Any = None
        self.audio_out: Any = None
        self.text_out: Any = None
        self.sync: Any = None
        self.room_texts: list[str] = []
        self.extra: dict[str, Any] = {}

    def log(self, kind: str, *detail: Any) -> None:
        self.events.append([kind, *detail])

    # -- the models a scenario builds a session from

    def llm(self, respond: Callable[..., list[Any]] | None = None) -> Any:
        return self.models.ScriptedLLM(self, respond)

    def stt(self) -> Any:
        return self.models.ScriptedSTT(self)

    def vad(self) -> Any:
        return self.models.ScriptedVAD(self)

    def tts(self) -> Any:
        return self.models.SilentTTS(self)

    def session_of(self, **kwargs: Any) -> Any:
        from livekit.agents import LiveSession

        kwargs.setdefault("vad", None)
        kwargs.setdefault("aec_warmup_duration", None)
        kwargs.setdefault("user_away_timeout", None)
        th = dict(kwargs.pop("turn_handling", {}) or {})
        # "default" leaves the session's own turn detection in place
        if th.setdefault("turn_detection", "stt") == "default":
            del th["turn_detection"]
        kwargs["turn_handling"] = th
        self.session = LiveSession(**kwargs)
        self._observe(self.session)
        return self.session

    def _observe(self, session: Any) -> None:
        def on(name: str, view: Callable[[Any], list[Any]]) -> None:
            session.on(name, lambda ev: self.log(name, *view(ev)))

        on("agent_state_changed", lambda ev: [f"{ev.old_state}->{ev.new_state}"])
        on("user_state_changed", lambda ev: [f"{ev.old_state}->{ev.new_state}"])
        on(
            "user_input_transcribed",
            lambda ev: [ev.transcript, "final" if ev.is_final else "interim"],
        )
        on("conversation_item_added", lambda ev: [item_view(ev.item)])
        on("speech_created", lambda ev: [ev.source, "user" if ev.user_initiated else "agent"])
        on(
            "function_tools_executed",
            lambda ev: [
                [
                    [c.name, json.loads(c.arguments or "{}"), o.output if o else None]
                    for c, o in ev.zipped()
                ]
            ],
        )
        # emitted as each provider's stream settles, so their order against the state changes is
        # the scheduler's: counted per kind, not ordered
        session.on("metrics_collected", lambda ev: self.metrics.append(type(ev.metrics).__name__))
        on("agent_false_interruption", lambda ev: ["resumed" if ev.resumed else "not resumed"])
        on("user_transcription_timeout", lambda ev: [])
        on(
            "error",
            lambda ev: [type(ev.error).__name__, type(getattr(ev, "source", None)).__name__],
        )
        on(
            "close",
            lambda ev: [
                getattr(ev.reason, "value", str(ev.reason)),
                type(ev.error).__name__ if ev.error else None,
            ],
        )

    # -- the room

    async def start(self, agent: Any) -> None:
        if self.stack == "next":
            from livekit.agents import _core

            self.room = _core.FakeRoom(sample_rate=48000)
            self.room.join("user")
            await self.session.start(agent, room=self.room)
        else:
            from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer
            from tests.fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput

            texts = self.room_texts

            class Text(FakeTextOutput):
                """RoomIO's agent transcript: one delta stream per segment, listed from its first
                word, as the room lists the streams it opened."""

                slot: int | None = None

                async def capture_text(self, text: str) -> None:
                    await super().capture_text(text)
                    if self.slot is None:
                        self.slot = len(texts)
                        texts.append("")
                    texts[self.slot] = self._pushed_text

                def flush(self) -> None:
                    self.slot = None
                    self._pushed_text = ""

            def user_transcript(ev: Any) -> None:
                # RoomIO's user transcription is not a delta stream: one stream per transcript,
                # and a final one again when it flushes
                texts.extend([ev.transcript] * (2 if ev.is_final else 1))

            self.session.on("user_input_transcribed", user_transcript)

            class Audio(FakeAudioOutput):
                played = 0.0

                def on_playback_finished(self, *, playback_position: float, **kw: Any) -> None:
                    Audio.played += playback_position
                    super().on_playback_finished(playback_position=playback_position, **kw)

            self.audio_in = FakeAudioInput()
            self.audio_out = Audio(can_pause=True)
            self.text_out = Text()
            self.sync = sync = TranscriptSynchronizer(
                next_in_chain_audio=self.audio_out, next_in_chain_text=self.text_out
            )
            self.session.input.audio = self.audio_in
            self.session.output.audio = sync.audio_output
            self.session.output.transcription = sync.text_output
            await self.session.start(agent)

    async def settle(self, seconds: float) -> None:
        """Move both clocks forward in 20 ms steps."""
        for _ in range(round(seconds / 0.02)):
            if self.room is not None:
                self.room.advance(0.02)
            await asyncio.sleep(0.02)

    async def aclose(self) -> None:
        """`session.aclose()`, the room's clock moving meanwhile: a close may wait on timers."""
        task = asyncio.ensure_future(self.session.aclose())
        while not task.done():
            await self.settle(0.02)
        await task

    def speak(
        self, final: str | None, *, after: float = 0.5, interims: tuple[str, ...] = ()
    ) -> None:
        """The user says `final` (None: speech the STT gives no final for) for `after` seconds."""
        from livekit import rtc

        self.utterances.append(Utterance(final, after=after, interims=interims))
        self.vad_bursts.append(after)
        frames = round(after / 0.02)
        if self.room is not None:
            self.room.speak(frames=frames)
            return
        n = 48000 // 50
        for i in range(frames):
            data = bytes([(i * 7 + j) % 251 for j in range(n * 2)])
            self.audio_in.push(rtc.AudioFrame(data, 48000, 1, n))

    def handle(self, label: str, handle: Any) -> Any:
        self.handles.append((label, handle))
        return handle

    # -- the transcript

    def received(self) -> dict[str, Any]:
        if self.room is not None:
            texts = [
                text for topic, text in self.room.text_streams() if topic == "lk.transcription"
            ]
            return {
                "audio_s": round(self.room.played_bytes() / (SAMPLE_RATE * 2), 2),
                "texts": texts,
            }
        return {
            "audio_s": round(type(self.audio_out).played, 2),
            "texts": list(self.room_texts),
        }

    def transcript(self) -> dict[str, Any]:
        out: dict[str, Any] = {"events": self.events}
        out["metrics"] = [sorted(self.metrics)]
        out["llm_requests"] = self.llm_requests
        out["tts_requests"] = self.tts_requests
        if self.session is not None:
            out["history"] = [item_view(i) for i in self.session.history.items]
        out["speech_handles"] = [
            [label, "done" if h.done() else "pending", "interrupted" if h.interrupted else "-"]
            for label, h in self.handles
        ]
        if self.session is not None and (self.room is not None or self.audio_out is not None):
            out["room"] = [self.received()]
        out.update(self.extra)
        return out


# ============================================================================ the scenarios


def _agents() -> Any:
    from livekit.agents import Agent, RunContext, function_tool

    # tool annotations are resolved from this module's globals
    globals().update(Agent=Agent, RunContext=RunContext)

    class Billing(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You handle billing.")

        async def on_enter(self) -> None:
            self.session.say("Billing here.")

    class Support(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You handle support.")

        async def on_enter(self) -> None:
            self.session.generate_reply(instructions="Greet the caller.")

    class FrontDesk(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are the front desk.")

        @function_tool
        async def get_weather(self, ctx: RunContext, city: str) -> str:
            """The weather in a city."""
            return f"sunny in {city}"

        @function_tool
        async def to_billing(self, ctx: RunContext) -> None:
            """Transfer the caller to billing."""
            ctx.session.update_agent(Billing())

        @function_tool
        async def to_support(self, ctx: RunContext) -> Agent:
            """Transfer the caller to support."""
            return Support()

    return FrontDesk, Billing, Support


async def _pipeline(call: Call, *, agent: Any = None, **kwargs: Any) -> Any:
    FrontDesk, _, _ = _agents()
    kwargs.setdefault("stt", call.stt())
    kwargs.setdefault("llm", call.llm())
    kwargs.setdefault("tts", call.tts())
    session = call.session_of(**kwargs)
    await call.start(agent or FrontDesk())
    await call.settle(0.5)
    return session


async def _close(call: Call) -> None:
    await call.aclose()
    call.extra["agent"] = type(call.session.current_agent).__name__


async def sc_say(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("say", session.say("Let me say something."))
    await call.settle(2)
    await _close(call)


async def sc_generate_reply(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("instructions", session.generate_reply(instructions="Greet the user."))
    await call.settle(2)
    call.handle("user_input", session.generate_reply(user_input="hello there"))
    await call.settle(3)
    await _close(call)


async def sc_tool_call(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("reply", session.generate_reply(user_input="what's the weather like?"))
    await call.settle(4)
    await _close(call)


async def sc_user_turn(call: Call) -> None:
    await _pipeline(call)
    call.speak("my name is Theo")
    await call.settle(4)
    call.speak("what's the weather like?")
    await call.settle(5)
    await _close(call)


async def sc_user_turn_default_detection(call: Call) -> None:
    await _pipeline(call, turn_handling={"turn_detection": "default"})
    call.speak("my name is Theo")
    await call.settle(4)
    await _close(call)


async def sc_realtime_stt_default_detection(call: Call) -> None:
    from livekit.agents import Agent

    call.session_of(
        llm=_realtime_models(call)(), stt=call.stt(), turn_handling={"turn_detection": "default"}
    )
    await call.start(Agent(instructions="You are realtime."))
    await call.settle(0.5)
    call.speak("hello there")
    await call.settle(4)
    await _close(call)


async def sc_interruption(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("long", session.generate_reply(user_input="tell me something long"))
    await call.settle(1)
    call.speak("wait, stop")
    await call.settle(5)
    await _close(call)


async def sc_false_interruption(call: Call) -> None:
    """An interim pauses the speech; no final follows, so it resumes."""
    session = await _pipeline(call)
    call.handle("long", session.generate_reply(user_input="tell me something long"))
    await call.settle(1)
    call.speak(None, after=0.2, interims=("uh",))
    await call.settle(6)
    await _close(call)


async def sc_vad_user_turn(call: Call) -> None:
    await _pipeline(call, vad=call.vad(), turn_handling={"turn_detection": "vad"})
    call.speak("my name is Theo")
    await call.settle(4)
    await _close(call)


async def sc_vad_interruption(call: Call) -> None:
    session = await _pipeline(call, vad=call.vad(), turn_handling={"turn_detection": "vad"})
    call.handle("long", session.generate_reply(user_input="tell me something long"))
    await call.settle(1)
    call.speak("wait, stop", after=0.6)
    await call.settle(5)
    await _close(call)


async def sc_vad_false_interruption(call: Call) -> None:
    """Speech long enough to interrupt, with no words: the speech pauses, then resumes."""
    session = await _pipeline(call, vad=call.vad(), turn_handling={"turn_detection": "vad"})
    call.handle("long", session.generate_reply(user_input="tell me something long"))
    await call.settle(1)
    call.speak(None, after=0.6)
    await call.settle(6)
    await _close(call)


async def sc_backchannel(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("long", session.generate_reply(user_input="tell me something long"))
    await call.settle(1)
    call.speak("mhm", after=0.2)
    await call.settle(6)
    await _close(call)


async def sc_empty_transcript(call: Call) -> None:
    await _pipeline(call)
    call.speak("")
    await call.settle(3)
    call.speak("hello")
    await call.settle(3)
    await _close(call)


async def sc_handoff_update_agent(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("reply", session.generate_reply(user_input="put me through to billing"))
    await call.settle(4)
    call.handle("after", session.generate_reply(user_input="hello billing"))
    await call.settle(3)
    await _close(call)


async def sc_handoff_tool_returns_agent(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("reply", session.generate_reply(user_input="I need support"))
    await call.settle(4)
    await _close(call)


async def sc_session_run(call: Call) -> None:
    session = await _pipeline(call)
    result = session.run(user_input="what's the weather like?")
    await call.settle(4)
    call.extra["run"] = [
        ["done" if result.done() else "pending"],
        [item_view(e.item) for e in result.events],
    ]
    await _close(call)


async def sc_aclose_drain(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("long", session.generate_reply(user_input="tell me something long"))
    await call.settle(0.5)
    session.shutdown(drain=True)
    await call.settle(6)
    call.extra["agent"] = type(session.current_agent).__name__


async def sc_aclose_mid_reply(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("long", session.generate_reply(user_input="tell me something long"))
    await call.settle(0.8)
    await call.aclose()


async def sc_llm_fallback(call: Call) -> None:
    from livekit.agents import llm

    adapter = llm.FallbackAdapter([call.models.FailingLLM(), call.llm()])
    adapter.on(
        "llm_availability_changed",
        lambda ev: call.log("llm_availability_changed", ev.llm.model, ev.available),
    )
    session = await _pipeline(call, llm=adapter)
    call.handle("reply", session.generate_reply(user_input="hello there"))
    await call.settle(3)
    call.handle("again", session.generate_reply(user_input="hello again"))
    await call.settle(3)
    await _close(call)


async def sc_tts_fallback(call: Call) -> None:
    from livekit.agents import tts

    adapter = tts.FallbackAdapter([call.models.FailingTTS(), call.tts()])
    adapter.on(
        "tts_availability_changed",
        lambda ev: call.log("tts_availability_changed", ev.tts.model, ev.available),
    )
    session = await _pipeline(call, tts=adapter)
    call.handle("say", session.say("Hello from the fallback."))
    await call.settle(3)
    await _close(call)


async def sc_stt_fallback(call: Call) -> None:
    from livekit.agents import stt

    adapter = stt.FallbackAdapter([call.models.FailingSTT(), call.stt()])
    adapter.on(
        "stt_availability_changed",
        lambda ev: call.log("stt_availability_changed", ev.stt.model, ev.available),
    )
    await _pipeline(call, stt=adapter)
    await call.settle(1)
    call.speak("hello there")
    await call.settle(4)
    await _close(call)


async def sc_llm_failure(call: Call) -> None:
    session = await _pipeline(call, llm=call.models.FailingLLM())
    call.handle("reply", session.generate_reply(user_input="hello there"))
    await call.settle(3)
    await _close(call)


async def sc_tts_failure(call: Call) -> None:
    session = await _pipeline(call, tts=call.models.FailingTTS())
    call.handle("say", session.say("This will not be heard."))
    await call.settle(3)
    await _close(call)


def _realtime_models(call: Call) -> Any:
    from livekit import rtc
    from livekit.agents import llm

    class ScriptedRealtime(llm.RealtimeModel):
        def __init__(self) -> None:
            super().__init__(
                capabilities=llm.RealtimeCapabilities(
                    message_truncation=False,
                    turn_detection=False,
                    user_transcription=False,
                    auto_tool_reply_generation=False,
                    audio_output=True,
                    manual_function_calls=True,
                )
            )

        def session(self, **_: object) -> llm.RealtimeSession:
            return _Session(self)

    class _Session(llm.RealtimeSession):
        def __init__(self, model: ScriptedRealtime) -> None:
            super().__init__(model)
            self._chat_ctx = llm.ChatContext.empty()

        @property
        def chat_ctx(self) -> llm.ChatContext:
            return self._chat_ctx

        @property
        def tools(self) -> llm.ToolContext:
            return llm.ToolContext.empty()

        async def update_instructions(self, instructions: str) -> None:
            pass

        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            self._chat_ctx = chat_ctx

        async def update_tools(self, tools: list[llm.Tool]) -> None:
            pass

        def update_options(self, **_: object) -> None:
            pass

        def push_audio(self, frame: rtc.AudioFrame) -> None:
            pass

        def push_video(self, frame: rtc.VideoFrame) -> None:
            pass

        def generate_reply(self, **kw: object) -> asyncio.Future[llm.GenerationCreatedEvent]:
            call.log("realtime.generate_reply", sorted(k for k, v in kw.items() if v))
            loop = asyncio.get_running_loop()

            async def frames() -> AsyncIterable[rtc.AudioFrame]:
                for _ in range(20):  # 2 s
                    yield rtc.AudioFrame(b"\x00\x00" * 2400, SAMPLE_RATE, 1, 2400)

            async def text() -> AsyncIterable[str]:
                for w in _words("Hello from the realtime model, a longer reply."):
                    yield w

            async def one(v: Any) -> AsyncIterable[Any]:
                yield v

            async def empty() -> AsyncIterable[Any]:
                return
                yield

            modalities: asyncio.Future[list[str]] = loop.create_future()
            modalities.set_result(["audio", "text"])
            call.counter += 1
            message = llm.MessageGeneration(
                message_id=f"msg_{call.counter}",
                text_stream=text(),
                audio_stream=frames(),
                modalities=modalities,
            )
            fut: asyncio.Future[llm.GenerationCreatedEvent] = loop.create_future()
            fut.set_result(
                llm.GenerationCreatedEvent(
                    message_stream=one(message),
                    function_stream=empty(),
                    user_initiated=True,
                    response_id=f"resp_{call.counter}",
                )
            )
            return fut

        def commit_audio(self) -> None:
            call.log("realtime.commit_audio")

        def clear_audio(self) -> None:
            call.log("realtime.clear_audio")

        def interrupt(self) -> None:
            call.log("realtime.interrupt")

        def truncate(self, **kw: object) -> None:
            call.log("realtime.truncate")

        async def aclose(self) -> None:
            pass

    return ScriptedRealtime


async def sc_realtime_reply(call: Call) -> None:
    from livekit.agents import Agent

    session = call.session_of(llm=_realtime_models(call)())
    await call.start(Agent(instructions="You are realtime."))
    await call.settle(0.5)
    call.handle("reply", session.generate_reply(instructions="greet the user"))
    await call.settle(3)
    await _close(call)


async def sc_realtime_interruption(call: Call) -> None:
    from livekit.agents import Agent

    session = call.session_of(llm=_realtime_models(call)(), stt=call.stt())
    await call.start(Agent(instructions="You are realtime."))
    await call.settle(0.5)
    call.handle("reply", session.generate_reply(instructions="greet the user"))
    await call.settle(0.8)
    call.speak("wait")
    await call.settle(4)
    await _close(call)


async def sc_llm_node(call: Call) -> None:
    from livekit.agents import Agent, llm

    class Upper(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You shout.")

        async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any) -> Any:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                if isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
                    chunk.delta.content = chunk.delta.content.upper()
                yield chunk

    session = await _pipeline(call, agent=Upper())
    call.handle("reply", session.generate_reply(user_input="hello there"))
    await call.settle(3)
    await _close(call)


async def sc_llm_node_str(call: Call) -> None:
    from livekit.agents import Agent

    class Fixed(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You say a fixed line.")

        async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any) -> Any:
            return "A fixed line."

    session = await _pipeline(call, agent=Fixed())
    call.handle("reply", session.generate_reply(user_input="hello there"))
    await call.settle(3)
    await _close(call)


async def sc_tts_node(call: Call) -> None:
    from livekit.agents import Agent

    class Twice(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You echo.")

        async def tts_node(self, text: Any, model_settings: Any) -> Any:
            async for frame in Agent.default.tts_node(self, text, model_settings):
                yield frame
                yield frame

    session = await _pipeline(call, agent=Twice())
    call.handle("say", session.say("Twenty characters..."))
    await call.settle(3)
    await _close(call)


async def sc_stt_node(call: Call) -> None:
    from livekit.agents import Agent, stt

    class Redact(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You redact.")

        async def stt_node(self, audio: Any, model_settings: Any) -> Any:
            async for ev in Agent.default.stt_node(self, audio, model_settings):
                if isinstance(ev, stt.SpeechEvent) and ev.alternatives:
                    ev.alternatives[0].text = ev.alternatives[0].text.replace("Theo", "[name]")
                yield ev

    await _pipeline(call, agent=Redact())
    call.speak("my name is Theo")
    await call.settle(4)
    await _close(call)


async def sc_user_away(call: Call) -> None:
    session = await _pipeline(call, user_away_timeout=1.0)
    call.handle("say", session.say("Are you there?"))
    await call.settle(4)
    call.speak("yes I am")
    await call.settle(3)
    await _close(call)


async def sc_preemptive(call: Call) -> None:
    await _pipeline(call, turn_handling={"preemptive_generation": {"enabled": True}})
    call.speak("my name is Theo", interims=("my name",))
    await call.settle(4)
    await _close(call)


async def sc_on_user_turn_completed(call: Call) -> None:
    from livekit.agents import Agent, StopResponse

    class Gate(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You ignore 'skip'.")

        async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
            call.log("on_user_turn_completed", new_message.text_content)
            if new_message.text_content == "skip":
                raise StopResponse()

    await _pipeline(call, agent=Gate())
    call.speak("skip")
    await call.settle(3)
    call.speak("hello")
    await call.settle(3)
    await _close(call)


async def sc_say_uninterruptible(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("say", session.say(LONG, allow_interruptions=False))
    await call.settle(1)
    call.speak("stop please")
    await call.settle(6)
    await _close(call)


async def sc_interrupt_api(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("long", session.generate_reply(user_input="tell me something long"))
    await call.settle(1)
    session.interrupt()
    await call.settle(2)
    await _close(call)


async def sc_say_cut(call: Call) -> None:
    session = await _pipeline(call)
    call.handle("say", session.say(COUNTING))
    await call.settle(1.1)
    session.interrupt()
    await call.settle(2)
    await _close(call)


async def sc_live_transcript(call: Call) -> None:
    """The room's transcript while the speech plays, not only once it ended. Sampled away from
    word onsets: the room's sink reports its playhead a frame late (up to 50 ms)."""
    session = await _pipeline(call)
    call.handle("say", session.say(LONG))
    elapsed = 0.0
    for at in (0.44, 0.66, 0.94):
        await call.settle(at - elapsed)
        elapsed = at
        call.extra.setdefault("live", []).append([at, call.received()["texts"]])
    await call.settle(2)
    await _close(call)


SCENARIOS: dict[str, Callable[[Call], Any]] = {
    "say_cut": sc_say_cut,
    "live_transcript": sc_live_transcript,
    "say": sc_say,
    "generate_reply": sc_generate_reply,
    "tool_call": sc_tool_call,
    "user_turn": sc_user_turn,
    "user_turn_default_detection": sc_user_turn_default_detection,
    "realtime_stt_default_detection": sc_realtime_stt_default_detection,
    "interruption": sc_interruption,
    "false_interruption": sc_false_interruption,
    "vad_user_turn": sc_vad_user_turn,
    "vad_interruption": sc_vad_interruption,
    "vad_false_interruption": sc_vad_false_interruption,
    "backchannel": sc_backchannel,
    "empty_transcript": sc_empty_transcript,
    "handoff_update_agent": sc_handoff_update_agent,
    "handoff_tool_returns_agent": sc_handoff_tool_returns_agent,
    "session_run": sc_session_run,
    "aclose_drain": sc_aclose_drain,
    "aclose_mid_reply": sc_aclose_mid_reply,
    "llm_fallback": sc_llm_fallback,
    "tts_fallback": sc_tts_fallback,
    "stt_fallback": sc_stt_fallback,
    "llm_failure": sc_llm_failure,
    "tts_failure": sc_tts_failure,
    "realtime_reply": sc_realtime_reply,
    "realtime_interruption": sc_realtime_interruption,
    "llm_node": sc_llm_node,
    "llm_node_str": sc_llm_node_str,
    "tts_node": sc_tts_node,
    "stt_node": sc_stt_node,
    "user_away": sc_user_away,
    "preemptive": sc_preemptive,
    "on_user_turn_completed": sc_on_user_turn_completed,
    "say_uninterruptible": sc_say_uninterruptible,
    "interrupt_api": sc_interrupt_api,
}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run":
        scenario_main(sys.argv[2])
    else:
        sys.exit(orchestrate(sys.argv[1:]))
