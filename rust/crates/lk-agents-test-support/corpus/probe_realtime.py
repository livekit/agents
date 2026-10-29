"""Probe the realtime server's edge cases and keep what it did: the ground truth the simulated
server (lk-agents-inference `tests/inference/realtime/server.rs`) must reproduce, checked by
`tests/inference/realtime/recorded.rs`.

`record.py` records ordinary calls; this drives the protocol's corners on purpose -- a cancel with
nothing active, a reply asked over an active one or before a tool's output, truncations inside and
past the audio, refused items, a session update mid-reply, barge-in with the server told not to
interrupt, manual turns, a socket closed mid-reply -- and records each session's client frames and
server events with their timing. Nothing anybody said is kept: content becomes its length, ids
become ordinals (`item0`, `resp1`, `call0`), so which event names which item survives.

    LIVEKIT_API_KEY=.. LIVEKIT_API_SECRET=.. \
        python -I rust/crates/lk-agents-test-support/corpus/probe_realtime.py --out <dir> [probe ...]

Through LiveKit Inference (`openai/gpt-realtime`) by default, as `record.py` does; `--openai`
goes to api.openai.com with `OPENAI_API_KEY`.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

# `record.py` beside this file: its socket, line and synthesis are shared (`-I` keeps the
# script's directory off the path, so it is loaded by file).
_spec = importlib.util.spec_from_file_location("record", Path(__file__).with_name("record.py"))
assert _spec and _spec.loader
rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rec)

# What the person types or says; never written out.
ASK = "Hi, can you tell me where my order is? Answer in two sentences."
ASK_TOOL = "Where is my order? Look it up."
SAY = rec.USER_LINES[0]
# Replies long enough that the person can talk over one while it is still generating: the model
# streams audio several times faster than real time, so a short reply is done before anybody
# could interrupt it.
LONG = "You are a storyteller. Whatever the caller says, tell a long story of at least twelve sentences."

INTERRUPT = rec.INTERRUPT

CONTENT_KEYS = {"text", "transcript", "output", "instructions", "arguments", "delta", "audio"}
# A server id inside an error's message: the server's own, which nothing needs.
RAW_ID = re.compile(r"\b(?:resp|item|call|event|sess)_[A-Za-z0-9]{12,}\b")
ID_KEYS = {"item_id": "item", "previous_item_id": "item", "response_id": "resp", "call_id": "call"}


class Probe(rec.Socket):
    """A socket that also keeps the client's frames, sanitised, and more of each event."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        super().__init__(ws)
        self.frames: list[dict[str, Any]] = []
        # Audio appended so far, ms: what a commit or the server's VAD has to work with.
        self.appended = 0.0
        self.last_call: str | None = None
        self.messages: list[str] = []

    def clean(self, v: Any, key: str = "") -> Any:
        """A client frame without its content, ids the server minted as their ordinals."""
        if isinstance(v, dict):
            return {k: self.clean(x, k) for k, x in v.items()}
        if isinstance(v, list):
            return [self.clean(x, key) for x in v]
        if isinstance(v, str):
            if key in CONTENT_KEYS:
                return f"<{len(v)}>"
            if key in ID_KEYS or key == "id":
                return f"@{self.ids.seen[v]}" if v in self.ids.seen else v
        return v

    async def send(self, ev: dict[str, Any]) -> None:
        if ev["type"] == "input_audio_buffer.append":
            self.appended += len(ev["audio"]) * 3 / 4 / 2 / rec.SR * 1000
        else:
            frame = self.clean(copy.deepcopy(ev))
            self.frames.append(
                {"t": self.now(), "appended_ms": round(self.appended), "frame": frame}
            )
        await super().send(ev)

    def shape(self, ev: dict[str, Any]) -> dict[str, Any]:
        out = super().shape(ev)
        if out["type"].startswith("input_audio_buffer.speech_"):
            out["appended_ms"] = round(self.appended)
        err = ev.get("error")
        if isinstance(err, dict):
            # The server's own words about the client's frame; no content of anybody's.
            out["error"].update(
                {
                    "message": RAW_ID.sub("<id>", err.get("message") or ""),
                    "param": err.get("param"),
                    "event_id": err.get("event_id"),
                }
            )
        item = ev.get("item")
        if isinstance(item, dict) and item.get("call_id"):
            out["item"]["call_id"] = self.ids("call", item.get("call_id"))
        resp = ev.get("response")
        if isinstance(resp, dict):
            details = resp.get("status_details") or {}
            out["response"]["details_type"] = details.get("type")
            usage = resp.get("usage") or {}
            if usage:
                out["response"]["usage"] = {
                    "input": usage.get("input_tokens"),
                    "output": usage.get("output_tokens"),
                    "input_text": (usage.get("input_token_details") or {}).get("text_tokens"),
                    "input_audio": (usage.get("input_token_details") or {}).get("audio_tokens"),
                }
            out["response"]["calls"] = [
                self.ids("call", o.get("call_id"))
                for o in resp.get("output") or []
                if o.get("type") == "function_call"
            ]
        if ev.get("type") == "session.updated":
            s = ev.get("session") or {}
            out["session"] = {
                "voice": ((s.get("audio") or {}).get("output") or {}).get("voice"),
                "turn_detection": (
                    ((s.get("audio") or {}).get("input") or {}).get("turn_detection") or {}
                ).get("type"),
            }
        if ev.get("type") == "response.output_item.added" and isinstance(item, dict):
            if item.get("type") == "function_call":
                self.last_call = item.get("call_id")
            if item.get("type") == "message":
                self.messages.append(item.get("id"))
        return out


def update(td: Any, **session: Any) -> dict[str, Any]:
    ev = rec.session_update(td)
    ev["session"].update(session)
    return ev


def user_text(text: str, item_id: str | None = None, previous: str | None = None) -> dict:
    item: dict[str, Any] = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }
    if item_id:
        item["id"] = item_id
    ev: dict[str, Any] = {"type": "conversation.item.create", "item": item}
    if previous:
        ev["previous_item_id"] = previous
    return ev


def create(event_id: str, **response: Any) -> dict[str, Any]:
    response.setdefault("metadata", {"client_event_id": event_id})
    return {"type": "response.create", "event_id": event_id, "response": response}


async def settle(s: Probe, seconds: float = 1.5) -> None:
    await asyncio.sleep(seconds)


async def done(s: Probe, timeout: float = 20) -> dict | None:
    return await rec.until(s.wait("response.done"), timeout)


# ------------------------------------------------------------------------------------------------
# The probes: each one session, given the open socket and the person's line
# ------------------------------------------------------------------------------------------------


async def refusals(s: Probe, line: Any, voices: dict) -> None:
    """Every frame a server might refuse, on an idle session, one at a time."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    frames = [
        {"type": "response.cancel", "event_id": "cancel_idle"},
        {"type": "response.cancel", "event_id": "cancel_named", "response_id": "resp_nope"},
        {"type": "input_audio_buffer.commit", "event_id": "commit_empty"},
        {"type": "input_audio_buffer.clear", "event_id": "clear_empty"},
        {"type": "conversation.item.retrieve", "event_id": "retrieve_nope", "item_id": "nope"},
        {"type": "conversation.item.delete", "event_id": "delete_nope", "item_id": "nope"},
        {
            "type": "conversation.item.truncate",
            "event_id": "truncate_nope",
            "item_id": "nope",
            "content_index": 0,
            "audio_end_ms": 0,
        },
        {**user_text(ASK, previous="nope"), "event_id": "create_after_nope"},
        {
            "type": "conversation.item.create",
            "event_id": "output_nope",
            "item": {"type": "function_call_output", "call_id": "call_nope", "output": "{}"},
        },
        {**user_text(ASK, item_id="probe_a"), "event_id": "create_a"},
        {**user_text(ASK, item_id="probe_a"), "event_id": "create_a_again"},
        {
            "type": "conversation.item.truncate",
            "event_id": "truncate_user",
            "item_id": "probe_a",
            "content_index": 0,
            "audio_end_ms": 0,
        },
        {
            "type": "conversation.item.create",
            "event_id": "create_assistant_audio",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_audio", "transcript": "hi"}],
            },
        },
        {
            "type": "conversation.item.create",
            "event_id": "create_assistant_text",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        },
        {"type": "conversation.item.retrieve", "event_id": "retrieve_a", "item_id": "probe_a"},
        {"type": "conversation.item.delete", "event_id": "delete_a", "item_id": "probe_a"},
        {"type": "conversation.item.delete", "event_id": "delete_a_again", "item_id": "probe_a"},
        {"type": "output_audio_buffer.clear", "event_id": "output_clear"},
        {"type": "no.such.event", "event_id": "unknown"},
    ]
    for f in frames:
        await s.send(f)
        await asyncio.sleep(0.6)
    # A commit of a sliver of audio, then of enough.
    await s.append(b"\x00\x00" * (rec.SR // 50))
    await s.send({"type": "input_audio_buffer.commit", "event_id": "commit_20ms"})
    await asyncio.sleep(0.6)
    await s.append(b"\x00\x00" * (rec.SR // 2))
    await s.send({"type": "input_audio_buffer.commit", "event_id": "commit_500ms"})
    await settle(s)


async def create_over_active(s: Probe, line: Any, voices: dict) -> None:
    """A reply asked while one is generating; a cancel once it is done."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    await s.send(user_text(ASK))
    await s.send(create("create_a"))
    await s.send(create("create_b"))
    await done(s)
    await settle(s)
    await s.send({"type": "response.cancel", "event_id": "cancel_after_done"})
    await settle(s)


async def cancel_mid_audio(s: Probe, line: Any, voices: dict) -> None:
    """Cancel while the audio streams: what still arrives, and a cancel naming the done one."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    await s.send(user_text(ASK))
    await s.send(create("create_a"))
    created = await rec.until(s.wait("response.created"), 20)
    await rec.until(s.wait("response.output_audio.delta"), 20)
    await asyncio.sleep(0.3)
    await s.send({"type": "response.cancel", "event_id": "cancel_a"})
    await done(s)
    await settle(s)
    rid = (created or {}).get("response", {}).get("id")
    if rid:
        await s.send({"type": "response.cancel", "event_id": "cancel_done", "response_id": rid})
    await settle(s)


async def cancel_then_create(s: Probe, line: Any, voices: dict) -> None:
    """A cancel and a new reply in one breath, as a client interrupting to answer anew does:
    whether the server takes the reply while the cancelled one is still winding down."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    await s.send(user_text(ASK))
    await s.send(create("create_a"))
    await rec.until(s.wait("response.output_audio.delta"), 20)
    await s.send({"type": "response.cancel", "event_id": "cancel_a"})
    await s.send(create("create_b"))
    await done(s, 30)
    await done(s, 30)
    await settle(s)


async def truncate(s: Probe, line: Any, voices: dict) -> None:
    """Truncations inside, at and past a reply's audio, and what the next reply reads."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    await s.send(user_text(ASK))
    await s.send(create("create_a"))
    await done(s, 30)
    await settle(s, 1.0)
    item = s.messages[-1] if s.messages else "nope"
    audio = int(s.audio_ms.get(item, 0))
    # Longest first: a truncation shortens what the server holds, so each later one is judged
    # against the one before it.
    for n, end in enumerate([audio + 1000, audio + 1, audio, 500, 0, 0]):
        await s.send(
            {
                "type": "conversation.item.truncate",
                "event_id": f"truncate_{n}",
                "item_id": item,
                "content_index": 0,
                "audio_end_ms": end,
            }
        )
        await asyncio.sleep(0.6)
    await s.send(create("create_b"))
    await done(s, 30)
    await settle(s)


async def tool_early(s: Probe, line: Any, voices: dict) -> None:
    """A tool call answered late: a reply asked before its output, then after."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    await s.send(user_text(ASK_TOOL))
    await s.send(create("create_call", tool_choice="required"))
    await done(s, 30)
    await settle(s, 0.5)
    call = s.last_call
    await s.send(create("create_early"))
    await done(s, 30)
    await settle(s, 0.5)
    if call:
        await s.send(
            {
                "type": "conversation.item.create",
                "event_id": "output",
                "item": {
                    "type": "function_call_output",
                    "call_id": call,
                    "output": '{"status": "shipped"}',
                },
            }
        )
        await settle(s, 0.5)
        await s.send(
            {
                "type": "conversation.item.create",
                "event_id": "output_again",
                "item": {
                    "type": "function_call_output",
                    "call_id": call,
                    "output": '{"status": "shipped"}',
                },
            }
        )
        await settle(s, 0.5)
    await s.send(create("create_after"))
    await done(s, 30)
    await settle(s)


async def tool_in_one_go(s: Probe, line: Any, voices: dict) -> None:
    """A reply asked in the same breath as the output, after it and before it."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    for order in ("output_first", "create_first"):
        await s.send(user_text(ASK_TOOL))
        await s.send(create(f"call_{order}", tool_choice="required"))
        await done(s, 30)
        call = s.last_call
        output = {
            "type": "conversation.item.create",
            "event_id": f"output_{order}",
            "item": {"type": "function_call_output", "call_id": call, "output": "{}"},
        }
        reply = create(f"reply_{order}")
        for f in [output, reply] if order == "output_first" else [reply, output]:
            await s.send(f)
        await done(s, 30)
        await settle(s)


async def session_mid_reply(s: Probe, line: Any, voices: dict) -> None:
    """Instructions and the voice changed while a reply streams, and the voice after."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    await s.send(user_text(ASK))
    await s.send(create("create_a"))
    await rec.until(s.wait("response.output_audio.delta"), 20)
    await s.send(
        {
            "type": "session.update",
            "event_id": "instructions_mid",
            "session": {"type": "realtime", "instructions": "Be brief."},
        }
    )
    await s.send(
        {
            "type": "session.update",
            "event_id": "voice_mid",
            "session": {"type": "realtime", "audio": {"output": {"voice": "alloy"}}},
        }
    )
    await done(s, 30)
    await settle(s)
    await s.send(
        {
            "type": "session.update",
            "event_id": "voice_after",
            "session": {"type": "realtime", "audio": {"output": {"voice": "ash"}}},
        }
    )
    await settle(s)
    await s.send(create("create_b"))
    await done(s, 30)
    await settle(s)


async def no_interrupt(s: Probe, line: Any, voices: dict) -> None:
    """Server VAD told neither to interrupt nor to reply: the person over a reply, then a turn
    the client answers."""
    td = rec.turn_detection("server_vad") | {"interrupt_response": False}
    await s.send(update(td, instructions=LONG))
    await rec.until(s.wait("session.updated"), 10)
    line.start()
    created = s.wait("response.created")
    line.say(voices[SAY])
    await rec.until(created, 20)
    await asyncio.sleep(0.3)
    line.say(voices[INTERRUPT])
    await done(s, 30)
    await line.idle()
    await settle(s, 3.0)
    await s.send(
        {
            "type": "session.update",
            "event_id": "no_reply",
            "session": {
                "type": "realtime",
                "audio": {
                    "input": {
                        "turn_detection": rec.turn_detection("server_vad")
                        | {"create_response": False}
                    }
                },
            },
        }
    )
    await settle(s, 0.5)
    line.say(voices[SAY])
    await rec.until(s.wait("input_audio_buffer.committed"), 20)
    await settle(s, 2.0)
    await s.send(create("create_manual"))
    await done(s, 20)
    await settle(s)
    await line.stop()


async def manual_turns(s: Probe, line: Any, voices: dict) -> None:
    """Turn detection off: speech is only committed when the client says, and nobody replies
    until asked; a commit mid-reply, and a clear."""
    await s.send(update(None, instructions=LONG))
    await rec.until(s.wait("session.updated"), 10)
    line.start()
    dur = line.say(voices[SAY])
    await asyncio.sleep(dur + 0.3)
    await s.send({"type": "input_audio_buffer.commit", "event_id": "commit_a"})
    await settle(s, 2.0)
    await s.send(create("create_a"))
    await rec.until(s.wait("response.output_audio.delta"), 20)
    await s.append(b"\x00\x00" * (rec.SR // 2))
    await s.send({"type": "input_audio_buffer.commit", "event_id": "commit_mid_reply"})
    await done(s, 30)
    await settle(s, 1.0)
    await line.idle()
    line.say(voices[INTERRUPT])
    await asyncio.sleep(0.5)
    await s.send({"type": "input_audio_buffer.clear", "event_id": "clear_mid_speech"})
    await settle(s, 2.0)
    await s.send({"type": "input_audio_buffer.commit", "event_id": "commit_after_clear"})
    await settle(s)
    await line.stop()


async def semantic_barge_in(s: Probe, line: Any, voices: dict) -> None:
    """Semantic VAD: the person over a reply, with the server told to interrupt."""
    await barge_in(s, line, voices, "semantic_vad")


async def server_barge_in(s: Probe, line: Any, voices: dict) -> None:
    """Server VAD: the person over a reply still generating, with the server told to
    interrupt."""
    await barge_in(s, line, voices, "server_vad")


async def barge_in(s: Probe, line: Any, voices: dict, td: str) -> None:
    await s.send(update(rec.turn_detection(td), instructions=LONG))
    await rec.until(s.wait("session.updated"), 10)
    line.start()
    created = s.wait("response.created")
    line.say(voices[SAY])
    await rec.until(created, 20)
    await asyncio.sleep(0.3)
    line.say(voices[INTERRUPT])
    await done(s, 20)
    await done(s, 20)
    await line.idle()
    await settle(s, 2.0)
    await line.stop()


async def replies(s: Probe, line: Any, voices: dict) -> None:
    """Replies back to back: how often the rate limits are restated."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    for n in range(4):
        await s.send(user_text(ASK))
        await s.send(create(f"create_{n}"))
        await done(s, 30)
        await settle(s, 0.5)


async def close_mid_reply(s: Probe, line: Any, voices: dict) -> None:
    """The client closes the socket while a reply streams."""
    await s.send(update(None))
    await rec.until(s.wait("session.updated"), 10)
    await s.send(user_text(ASK))
    await s.send(create("create_a"))
    await rec.until(s.wait("response.output_audio.delta"), 20)
    await s.ws.close()


PROBES = {
    f.__name__: f
    for f in (
        refusals,
        create_over_active,
        cancel_mid_audio,
        cancel_then_create,
        truncate,
        tool_early,
        tool_in_one_go,
        session_mid_reply,
        no_interrupt,
        manual_turns,
        semantic_barge_in,
        server_barge_in,
        replies,
        close_mid_reply,
    )
}


async def probe(session: aiohttp.ClientSession, name: str, voices: dict, openai: bool) -> dict:
    if openai:
        url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        model = "gpt-realtime"
    else:
        base = rec.get_default_inference_url().replace("https://", "wss://")
        model = "openai/gpt-realtime"
        url = f"{base}/realtime?model={model}"
        token = rec.create_access_token(
            os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
        )
        headers = {"Authorization": f"Bearer {token}", "User-Agent": "LiveKit Agents"}
    async with session.ws_connect(url, headers=headers, max_msg_size=0) as ws:
        s = Probe(ws)
        pump = asyncio.ensure_future(s.pump())
        line = rec.Line(s.append, origin=s.t0)
        try:
            await PROBES[name](s, line, voices)
        finally:
            await line.stop()
            if not ws.closed:
                await ws.close()
            try:
                await asyncio.wait_for(pump, 5)
            except Exception:
                pass
        return {
            "provider": "realtime",
            "model": model,
            "via": "openai" if openai else "livekit-inference",
            "probe": name,
            "doc": (PROBES[name].__doc__ or "").strip(),
            "speech": line.spans,
            "frames": s.frames,
            "events": s.events,
        }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("probes", nargs="*", default=list(PROBES))
    p.add_argument("--out", required=True)
    p.add_argument("--openai", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    async with aiohttp.ClientSession() as session:
        tts = rec.inference.TTS(model="cartesia/sonic-3", http_session=session)
        voices = {t: (await rec.synthesize(tts, t))[0] for t in (SAY, INTERRUPT)}
        for name in args.probes:
            try:
                result = await probe(session, name, voices, args.openai)
            except Exception as e:
                print(f"probe {name} failed: {e!r}", file=sys.stderr)
                continue
            result["recorded"] = time.strftime("%Y-%m-%d")
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"probe-{name}.json"
            path.write_text(json.dumps(result, indent=1) + "\n")
            print(f"wrote {path} ({len(result['events'])} events)")


if __name__ == "__main__":
    asyncio.run(main())
