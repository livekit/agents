"""Record what real providers send, as event kinds and relative timing: the corpus the fuzz fakes
are calibrated from and checked against (`src/corpus.rs`, `tests/conformance.rs`).

Runs against LiveKit Inference with `livekit.agents` (`livekit-agents-next/`): the realtime model
over its raw OpenAI-protocol websocket, streaming STT, streaming TTS and a streaming LLM. Nothing
the providers say is kept, only the shape of it: event types, when each arrived, audio durations,
word counts, statuses and error codes. Ids are replaced by ordinals so linkage survives.

    set -a; . creds.env; set +a      # LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    python -I rust/crates/lk-agents-test-support/corpus/record.py [realtime stt tts llm] --out <dir>

The user's side of every call is speech this script synthesises once with the inference TTS, fed
at real-time pace with silence between turns.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import inference, llm
from livekit.agents.inference._utils import create_access_token, get_default_inference_url

SR = 24000
FRAME_MS = 20
SAMPLES = SR * FRAME_MS // 1000

# What the person says. Only used to synthesise their audio; never written to the corpus.
USER_LINES = [
    "Hi, I'm calling about my order from last week.",
    "Can you hear me?",
    "No, not that one, the other one. The blue jacket.",
    "Uh, let me think. Okay, so it was last Tuesday, I think. Maybe Wednesday.",
    "My number is five five five, oh one four seven.",
    "Okay, thanks, bye.",
]
BACKCHANNEL = "Mm-hm."
INTERRUPT = "Wait, wait, stop. That's not what I asked."
AGENT_LINES = [
    "Sure, I can help with that.",
    "Let me pull up your account, one moment.",
    "I found three appointments: Tuesday at 9, Wednesday at 2:30, and Friday at 11 a.m. "
    "Which works best?",
    "Your balance is $1,204.55 as of today, and the next payment of $87.20 is due on 03/14.",
]


class Ids:
    """Real ids to ordinals per kind, so a recording keeps which event names which item."""

    def __init__(self) -> None:
        self.seen: dict[str, str] = {}
        self.count: dict[str, int] = {}

    def __call__(self, kind: str, raw: Any) -> str | None:
        if not isinstance(raw, str) or not raw:
            return None
        if raw not in self.seen:
            n = self.count.get(kind, 0)
            self.count[kind] = n + 1
            self.seen[raw] = f"{kind}{n}"
        return self.seen[raw]


TAG = ""  # `--tag`: a suffix, so a second round of recordings sits beside the first


def write(out: Path, name: str, rec: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rec["recorded"] = time.strftime("%Y-%m-%d")
    path = out / f"{name}{TAG}.json"
    path.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {path} ({len(rec.get('events', []))} events)")


# ------------------------------------------------------------------------------------------------
# TTS: also what makes the person's audio
# ------------------------------------------------------------------------------------------------


async def synthesize(tts: inference.TTS, text: str) -> tuple[np.ndarray, dict[str, Any]]:
    """One streamed synthesis: the audio, and the shape of how it arrived."""
    stream = tts.stream()
    t0 = time.perf_counter()
    stream.push_text(text)
    stream.end_input()
    chunks: list[dict[str, Any]] = []
    pcm: list[np.ndarray] = []
    async for ev in stream:
        frame = ev.frame
        data = np.frombuffer(frame.data, dtype=np.int16)
        if frame.sample_rate != SR:
            frame = frame.remix_and_resample(SR, 1)
            data = np.frombuffer(frame.data, dtype=np.int16)
        pcm.append(data)
        chunks.append(
            {
                "t": round(time.perf_counter() - t0, 4),
                "ms": round(1000 * frame.samples_per_channel / frame.sample_rate, 1),
                "final": bool(ev.is_final),
            }
        )
    await stream.aclose()
    audio = np.concatenate(pcm) if pcm else np.zeros(0, dtype=np.int16)
    return audio, {"words": len(text.split()), "chars": len(text), "chunks": chunks}


async def record_tts(
    out: Path, session: aiohttp.ClientSession, model: str
) -> dict[str, np.ndarray]:
    tts = inference.TTS(model=model, http_session=session)
    voices: dict[str, np.ndarray] = {}
    events = []
    for text in [*USER_LINES, BACKCHANNEL, INTERRUPT, *AGENT_LINES]:
        try:
            audio, shape = await synthesize(tts, text)
        except Exception as e:  # a real error mode, kept as one
            events.append({"error": type(e).__name__})
            continue
        voices[text] = audio
        events.append(shape)
    write(
        out, f"tts-{model.replace('/', '_')}", {"provider": "tts", "model": model, "events": events}
    )
    return voices


# ------------------------------------------------------------------------------------------------
# The person's line: speech and silence at real-time pace
# ------------------------------------------------------------------------------------------------


def frames(audio: np.ndarray) -> list[bytes]:
    pad = (-len(audio)) % SAMPLES
    audio = np.concatenate([audio, np.zeros(pad, dtype=np.int16)])
    return [audio[i : i + SAMPLES].tobytes() for i in range(0, len(audio), SAMPLES)]


SILENCE = np.zeros(SAMPLES, dtype=np.int16).tobytes()


class Line:
    """Feeds 20 ms frames at real-time pace: speech when asked, silence otherwise."""

    def __init__(self, sink: Any, origin: float | None = None) -> None:
        self.sink = sink
        self.queue: list[bytes] = []
        self.t0 = time.perf_counter()
        # What the speech spans are measured from: the provider's own recording origin, so a span
        # and the events about it share one clock.
        self.origin = self.t0 if origin is None else origin
        self.spans: list[dict[str, float]] = []
        self.task: asyncio.Task[None] | None = None

    def now(self) -> float:
        return round(time.perf_counter() - self.origin, 4)

    def say(self, audio: np.ndarray) -> float:
        """Queue speech after what is queued; returns its duration in seconds."""
        fs = frames(audio)
        start = self.now() + len(self.queue) * FRAME_MS / 1000
        self.queue.extend(fs)
        self.spans.append(
            {"t0": round(start, 4), "t1": round(start + len(fs) * FRAME_MS / 1000, 4)}
        )
        return len(fs) * FRAME_MS / 1000

    async def run(self) -> None:
        n = 0
        while True:
            frame = self.queue.pop(0) if self.queue else SILENCE
            await self.sink(frame)
            n += 1
            delay = self.t0 + n * FRAME_MS / 1000 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

    def start(self) -> None:
        self.task = asyncio.ensure_future(self.run())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def idle(self) -> None:
        while self.queue:
            await asyncio.sleep(0.02)


# ------------------------------------------------------------------------------------------------
# Realtime: the raw OpenAI-protocol socket through the gateway
# ------------------------------------------------------------------------------------------------


def turn_detection(kind: str) -> Any:
    if kind == "semantic_vad":
        return {
            "type": "semantic_vad",
            "eagerness": "medium",
            "create_response": True,
            "interrupt_response": True,
        }
    if kind == "server_vad":
        return {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "create_response": True,
            "interrupt_response": True,
        }
    return None


def session_update(td: Any) -> dict[str, Any]:
    """What the core's realtime machine sends on open (`session_update_frame`), model dropped for
    the gateway."""
    fmt = {"type": "audio/pcm", "rate": SR}
    return {
        "type": "session.update",
        "event_id": "session_update_0",
        "session": {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": fmt,
                    "noise_reduction": None,
                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": td,
                },
                "output": {"format": fmt, "speed": 1.0, "voice": "marin"},
            },
            "max_output_tokens": "inf",
            "tool_choice": "auto",
            "tracing": None,
            "instructions": "You are a helpful phone agent for a clothing shop. Answer in one or two "
            "sentences.",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup_order",
                    "description": "Look up the caller's order.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }
            ],
        },
    }


class Socket:
    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self.ws = ws
        self.t0 = time.perf_counter()
        self.ids = Ids()
        self.events: list[dict[str, Any]] = []
        self.client: list[dict[str, Any]] = []
        self.waiters: list[tuple[str, asyncio.Future[dict[str, Any]]]] = []
        self.last_response: str | None = None
        self.last_item: str | None = None
        self.audio_ms: dict[str, float] = {}

    def now(self) -> float:
        return round(time.perf_counter() - self.t0, 4)

    async def send(self, ev: dict[str, Any]) -> None:
        if ev["type"] != "input_audio_buffer.append":
            self.client.append({"t": self.now(), "type": ev["type"]})
        await self.ws.send_str(json.dumps(ev))

    async def append(self, pcm: bytes) -> None:
        await self.send(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
        )

    def wait(self, kind: str) -> asyncio.Future[dict[str, Any]]:
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self.waiters.append((kind, fut))
        return fut

    def shape(self, ev: dict[str, Any]) -> dict[str, Any]:
        """Everything about an event but what anybody said."""
        ty = ev.get("type", "")
        rec: dict[str, Any] = {"t": self.now(), "type": ty}
        ids = self.ids
        for key, kind in (
            ("response_id", "resp"),
            ("item_id", "item"),
            ("previous_item_id", "item"),
        ):
            if key in ev:
                rec[key] = ids(kind, ev[key])
        for key in ("output_index", "content_index", "audio_start_ms", "audio_end_ms"):
            if key in ev:
                rec[key] = ev[key]
        if ty == "response.output_audio.delta" or ty == "response.audio.delta":
            ms = len(base64.b64decode(ev.get("delta", ""))) / 2 / SR * 1000
            rec["ms"] = round(ms, 1)
        if ty.endswith("transcript.delta") or ty.endswith("text.delta"):
            rec["chars"] = len(ev.get("delta", ""))
        if ty.endswith("transcription.completed"):
            rec["words"] = len(ev.get("transcript", "").split())
            rec["usage"] = (ev.get("usage") or {}).get("type")
        if ty.endswith("function_call_arguments.done"):
            rec["name"] = ev.get("name")
        item = ev.get("item")
        if isinstance(item, dict):
            rec["item"] = {
                "id": ids("item", item.get("id")),
                "type": item.get("type"),
                "role": item.get("role"),
                "status": item.get("status"),
            }
        part = ev.get("part")
        if isinstance(part, dict):
            rec["part"] = part.get("type")
        resp = ev.get("response")
        if isinstance(resp, dict):
            rec["response"] = {
                "id": ids("resp", resp.get("id")),
                "status": resp.get("status"),
                "reason": (resp.get("status_details") or {}).get("reason"),
                "error": ((resp.get("status_details") or {}).get("error") or {}).get("code"),
                "outputs": [o.get("type") for o in resp.get("output") or []],
                "client_event_id": bool((resp.get("metadata") or {}).get("client_event_id")),
            }
        err = ev.get("error")
        if isinstance(err, dict):
            rec["error"] = {"type": err.get("type"), "code": err.get("code")}
        if ty == "rate_limits.updated":
            rec["limits"] = [r.get("name") for r in ev.get("rate_limits") or []]
        return rec

    async def pump(self) -> None:
        async for msg in self.ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                self.events.append({"t": self.now(), "type": f"ws:{msg.type.name.lower()}"})
                break
            ev = json.loads(msg.data)
            rec = self.shape(ev)
            self.events.append(rec)
            if ev.get("type") == "response.created":
                self.last_response = ev["response"]["id"]
            if ev.get("type") == "response.output_item.added":
                self.last_item = ev["item"]["id"]
            if ev.get("type") in ("response.output_audio.delta", "response.audio.delta"):
                self.audio_ms[ev["item_id"]] = self.audio_ms.get(ev["item_id"], 0) + rec["ms"]
            for kind, fut in list(self.waiters):
                if kind == ev.get("type") and not fut.done():
                    fut.set_result(ev)
                    self.waiters.remove((kind, fut))


async def until(fut: asyncio.Future[Any], timeout: float) -> Any:
    try:
        return await asyncio.wait_for(fut, timeout)
    except asyncio.TimeoutError:
        return None


async def realtime_call(
    session: aiohttp.ClientSession,
    voices: dict[str, np.ndarray],
    scenario: str,
    model: str,
    td_kind: str,
) -> dict[str, Any]:
    base = get_default_inference_url().replace("https://", "wss://").replace("http://", "ws://")
    url = f"{base}/realtime?model={model}"
    token = create_access_token(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "LiveKit Agents"}
    async with session.ws_connect(url, headers=headers, max_msg_size=0) as ws:
        sock = Socket(ws)
        pump = asyncio.ensure_future(sock.pump())
        await sock.send(session_update(turn_detection(td_kind)))
        await until(sock.wait("session.updated"), 10)
        line = Line(sock.append, origin=sock.t0)
        line.start()
        server_turns = td_kind != "none"

        async def user_turn(text: str, settle: float = 6.0) -> None:
            dur = line.say(voices[text])
            if not server_turns:
                await asyncio.sleep(dur + 0.3)
                await sock.send({"type": "input_audio_buffer.commit"})
                await sock.send({"type": "response.create", "event_id": "response_create_m"})
            done = sock.wait("response.done")
            await until(done, dur + settle + 15)
            await asyncio.sleep(0.5)

        if scenario == "turns":
            # A greeting the client asks for, then turns the server detects.
            await sock.send(
                {
                    "type": "response.create",
                    "event_id": "response_create_0",
                    "response": {"metadata": {"client_event_id": "response_create_0"}},
                }
            )
            await until(sock.wait("response.done"), 20)
            for text in USER_LINES:
                await user_turn(text)
        elif scenario == "barge_in":
            # The person talks over replies at different points: early, mid, late.
            for delay in (0.3, 0.8, 1.5, 2.5, 4.0):
                created = sock.wait("response.created")
                line.say(voices[USER_LINES[2]])
                await until(created, 20)
                # The reply audio arrives faster than real time; a listener interrupts at a point
                # in what they have heard, which a client plays from the first delta.
                await asyncio.sleep(delay)
                line.say(voices[INTERRUPT])
                await until(sock.wait("response.done"), 20)
                await until(sock.wait("response.done"), 20)
                await line.idle()
                await asyncio.sleep(1.0)
        elif scenario == "backchannel":
            created = sock.wait("response.created")
            line.say(voices[USER_LINES[3]])
            await until(created, 20)
            await asyncio.sleep(1.0)
            line.say(voices[BACKCHANNEL])
            await until(sock.wait("response.done"), 20)
            await asyncio.sleep(3.0)
        elif scenario == "client_cancel":
            # The client cancels a reply, truncates what it played, then cancels with nothing
            # active: the server's error for that.
            await sock.send({"type": "response.create", "event_id": "response_create_0"})
            delta = sock.wait("response.output_audio.delta")
            await until(delta, 20)
            await asyncio.sleep(0.4)
            await sock.send({"type": "response.cancel"})
            await until(sock.wait("response.done"), 10)
            if sock.last_item:
                played = int(min(sock.audio_ms.get(sock.last_item, 0), 400))
                await sock.send(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": sock.last_item,
                        "content_index": 0,
                        "audio_end_ms": played,
                    }
                )
                await asyncio.sleep(1.0)
                await sock.send(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": sock.last_item,
                        "content_index": 0,
                        "audio_end_ms": 999_999,
                    }
                )
            await asyncio.sleep(1.0)
            await sock.send({"type": "response.cancel"})
            await asyncio.sleep(1.0)
            await sock.send({"type": "input_audio_buffer.clear"})
            await asyncio.sleep(1.0)
        elif scenario == "tool_call":
            dur = line.say(voices[USER_LINES[0]])
            done = sock.wait("response.done")
            ev = await until(done, dur + 20)
            outputs = (ev or {}).get("response", {}).get("output") or []
            calls = [o for o in outputs if o.get("type") == "function_call"]
            for call in calls:
                await sock.send(
                    {
                        "type": "conversation.item.create",
                        "event_id": "item_create_0",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": '{"status": "shipped"}',
                        },
                    }
                )
            if calls:
                await sock.send({"type": "response.create", "event_id": "response_create_1"})
                await until(sock.wait("response.done"), 20)
            await asyncio.sleep(1.0)
        elif scenario == "manual":
            for text in USER_LINES[:2]:
                await user_turn(text)
        elif scenario == "silence":
            await asyncio.sleep(8.0)
            await user_turn(USER_LINES[1])
        await line.stop()
        await ws.close()
        await asyncio.wait_for(pump, 5)
        return {
            "provider": "realtime",
            "model": model,
            "turn_detection": td_kind,
            "scenario": scenario,
            "speech": line.spans,
            "client": sock.client,
            "events": sock.events,
        }


async def record_realtime(
    out: Path, session: aiohttp.ClientSession, voices: dict, model: str
) -> None:
    plan = [
        ("turns", "semantic_vad"),
        ("turns", "server_vad"),
        ("barge_in", "semantic_vad"),
        ("barge_in", "server_vad"),
        ("backchannel", "semantic_vad"),
        ("client_cancel", "semantic_vad"),
        ("tool_call", "semantic_vad"),
        ("manual", "none"),
        ("silence", "server_vad"),
    ]
    for scenario, td in plan:
        try:
            rec = await realtime_call(session, voices, scenario, model, td)
        except Exception as e:
            print(f"realtime {scenario}/{td} failed: {e!r}", file=sys.stderr)
            continue
        write(out, f"realtime-{model.replace('/', '_')}-{scenario}-{td}", rec)


# ------------------------------------------------------------------------------------------------
# STT
# ------------------------------------------------------------------------------------------------


async def record_stt(out: Path, session: aiohttp.ClientSession, voices: dict, model: str) -> None:
    stt = inference.STT(model=model, language="en", http_session=session)
    stream = stt.stream()
    t0 = time.perf_counter()
    events: list[dict[str, Any]] = []

    async def push(pcm: bytes) -> None:
        stream.push_frame(rtc.AudioFrame(pcm, SR, 1, SAMPLES))

    line = Line(push, origin=t0)

    async def read() -> None:
        async for ev in stream:
            alt = ev.alternatives[0] if ev.alternatives else None
            rec: dict[str, Any] = {
                "t": round(time.perf_counter() - t0, 4),
                "type": ev.type.value if hasattr(ev.type, "value") else str(ev.type),
            }
            if alt is not None:
                rec["words"] = len(alt.text.split())
                rec["start"] = round(alt.start_time, 3)
                rec["end"] = round(alt.end_time, 3)
                rec["confidence"] = round(alt.confidence, 3)
            if ev.recognition_usage is not None:
                rec["audio_s"] = round(ev.recognition_usage.audio_duration, 3)
            events.append(rec)

    reader = asyncio.ensure_future(read())
    line.start()
    await asyncio.sleep(1.0)
    for text in [*USER_LINES, BACKCHANNEL]:
        dur = line.say(voices[text])
        await asyncio.sleep(dur + 2.5)
    await asyncio.sleep(3.0)
    await line.stop()
    stream.end_input()
    try:
        await asyncio.wait_for(reader, 10)
    except asyncio.TimeoutError:
        reader.cancel()
    await stream.aclose()
    write(
        out,
        f"stt-{model.replace('/', '_')}",
        {"provider": "stt", "model": model, "speech": line.spans, "events": events},
    )


# ------------------------------------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------------------------------------


async def record_llm(out: Path, session: aiohttp.ClientSession, model: str) -> None:
    m = inference.LLM(model=model)

    @llm.function_tool
    async def lookup_order() -> str:
        """Look up the caller's order."""
        return "shipped"

    calls = []
    prompts = [
        (USER_LINES[0], []),
        (USER_LINES[3], []),
        ("Where is my order? Look it up.", [lookup_order]),
        (USER_LINES[4], [lookup_order]),
    ]
    for text, tools in prompts * 3:
        ctx = llm.ChatContext.empty()
        ctx.add_message(
            role="system", content="You are a phone agent. Answer in one or two sentences."
        )
        ctx.add_message(role="user", content=text)
        t0 = time.perf_counter()
        chunks = []
        try:
            async with m.chat(chat_ctx=ctx, tools=tools) as stream:
                async for ch in stream:
                    d = ch.delta
                    chunks.append(
                        {
                            "t": round(time.perf_counter() - t0, 4),
                            "role": bool(d and d.role),
                            "chars": len(d.content or "") if d else 0,
                            "tool_calls": len(d.tool_calls) if d and d.tool_calls else 0,
                            "usage": ch.usage is not None,
                        }
                    )
        except Exception as e:
            chunks.append({"t": round(time.perf_counter() - t0, 4), "error": type(e).__name__})
        calls.append({"tools": bool(tools), "chunks": chunks})
    write(
        out, f"llm-{model.replace('/', '_')}", {"provider": "llm", "model": model, "events": calls}
    )


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", nargs="*", default=["tts", "realtime", "stt", "llm"])
    p.add_argument("--out", required=True)
    p.add_argument("--tts-model", default="cartesia/sonic-3")
    p.add_argument("--stt-model", action="append")
    p.add_argument("--realtime-model", default="openai/gpt-realtime")
    p.add_argument("--llm-model", default="openai/gpt-4.1-mini")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    global TAG
    TAG = f"-{args.tag}" if args.tag else ""
    out = Path(args.out)
    async with aiohttp.ClientSession() as session:
        voices = await record_tts(out, session, args.tts_model)
        if "realtime" in args.what:
            await record_realtime(out, session, voices, args.realtime_model)
        if "stt" in args.what:
            for model in args.stt_model or ["deepgram/nova-3", "assemblyai/universal-streaming"]:
                try:
                    await record_stt(out, session, voices, model)
                except Exception as e:
                    print(f"stt {model} failed: {e!r}", file=sys.stderr)
        if "llm" in args.what:
            await record_llm(out, session, args.llm_model)


if __name__ == "__main__":
    asyncio.run(main())
