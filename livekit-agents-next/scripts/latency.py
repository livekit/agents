"""Latency benchmark: the same scripted call on two stacks, interleaved, per-stage median and p90.

Each stack runs a worker serving one benchmark agent (LiveKit inference models, identical on both
sides). The caller joins a fresh room per call, dispatches the agents in turn (A/B/A/B...), speaks
the same pre-synthesized lines, and measures when it hears the reply. Each agent sends the
metrics of every chat item it adds back to the caller on the ``lk.latency`` text topic.

Run against a project with LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET set:

    # one worker per stack, each from its own environment
    <reference venv>/bin/python livekit-agents-next/scripts/latency.py agent --name bench-ref
    <next venv>/bin/python livekit-agents-next/scripts/latency.py agent --name bench-next

    # the caller, from either environment, once both workers registered
    python livekit-agents-next/scripts/latency.py bench --agents bench-ref,bench-next --turns 30

Stages, per agent reply:
    end_of_turn       user stopped speaking -> turn decided (user item ``end_of_turn_delay``)
    llm_ttft          LLM request -> first token (``llm_node_ttft``)
    tts_ttfb          TTS request -> first audio (``tts_node_ttfb``)
    first_audio_out   user stopped speaking -> agent audio out (``e2e_latency``)
    caller_heard      caller's audio played out -> first audible agent frame, at the caller
    llm_tps           LLM output tokens per second over the streaming window (``llm_node_tps``)

With two agents, each stage also shows B - A at p50 and p90 with a 95% bootstrap interval. The
gateway's time to first token is heavy-tailed: at ~40 replies a p90 gap of 0.2 s is still noise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from typing import Any

import aiohttp
import numpy as np

from livekit import api, rtc
from livekit.agents import Agent, AgentServer, JobContext, VoiceSession, cli, inference
from livekit.agents.utils import shortuuid

TOPIC = "lk.latency"
SAMPLE_RATE = 24000
LOUD = 400  # int16 RMS above which an agent frame is audible
STT, LLM, TTS = "deepgram/nova-3", "openai/gpt-4.1-mini", "cartesia/sonic-3"
LINES = [
    "Hi, I'd like to check the status of my order please.",
    "It's order number one two three four.",
    "Great, and when will it arrive?",
    "Can you send me a receipt by email?",
    "My email is jane at example dot com.",
    "Thanks, that's all for today.",
]
STAGES = {
    "end_of_turn": ("user", "end_of_turn_delay"),
    "llm_ttft": ("assistant", "llm_node_ttft"),
    "tts_ttfb": ("assistant", "tts_node_ttfb"),
    "first_audio_out": ("assistant", "e2e_latency"),
    "caller_heard": ("caller", "heard"),
    "llm_tps": ("assistant", "llm_node_tps"),
}


# ------------------------------------------------------------------------------------- agent


async def entry(ctx: JobContext) -> None:
    session = VoiceSession(stt=STT, llm=LLM, tts=TTS)
    sends: set[asyncio.Task[Any]] = set()

    def on_item(ev: Any) -> None:
        metrics = dict(getattr(ev.item, "metrics", None) or {})
        role = getattr(ev.item, "role", None)
        if role in ("user", "assistant") and metrics:
            payload = json.dumps({"role": role, "metrics": metrics})
            task = asyncio.ensure_future(ctx.room.local_participant.send_text(payload, topic=TOPIC))
            sends.add(task)
            task.add_done_callback(sends.discard)

    session.on("conversation_item_added", on_item)
    agent = Agent(instructions="You are a shop's phone support. Answer in one short sentence.")
    await session.start(agent=agent, room=ctx.room)


def run_agent(name: str) -> None:
    # the scripted calls are the only load: a busy machine must not turn them away
    server = AgentServer(agent_name=name, port=0, load_fnc=lambda: 0.0)
    server.rtc_session()(entry)
    sys.argv = [sys.argv[0], "start"]
    cli.run_app(server)


# ------------------------------------------------------------------------------------ caller


async def synthesize(lines: list[str]) -> list[list[rtc.AudioFrame]]:
    async with aiohttp.ClientSession() as http:
        tts = inference.TTS(TTS, http_session=http)
        out = []
        for line in lines:
            out.append([audio.frame async for audio in tts.synthesize(line)])
        return out


async def one_call(lkapi: api.LiveKitAPI, agent: str, speech: list[list[rtc.AudioFrame]]) -> dict:
    room_name = f"latency-{agent}-{shortuuid()}"
    await lkapi.room.create_room(api.CreateRoomRequest(name=room_name, empty_timeout=30))
    token = (
        api.AccessToken()
        .with_identity("caller")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    tasks: set[asyncio.Future[Any]] = set()

    def spawn(coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    state = {"agent": "initializing", "loud_at": 0.0, "first_loud": None, "since": None}
    talking = asyncio.Event()
    items: list[dict] = []

    def on_attrs(changed: dict, participant: rtc.Participant) -> None:
        if "lk.agent.state" in changed:
            state["agent"] = changed["lk.agent.state"]

    async def read_metrics(reader: rtc.TextStreamReader, _: str) -> None:
        if talking.is_set():  # the greeting is not a reply
            items.append(json.loads(await reader.read_all()))

    async def monitor(track: rtc.Track) -> None:
        async for ev in rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1):
            data = np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32)
            if data.size and math.sqrt(float(np.mean(data**2))) > LOUD:
                state["loud_at"] = time.time()
                if state["since"] is not None and state["first_loud"] is None:
                    state["first_loud"] = state["loud_at"]

    def on_track(track: rtc.Track, *_: Any) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            spawn(monitor(track))

    room.on("participant_attributes_changed", on_attrs)
    room.on("track_subscribed", on_track)
    room.register_text_stream_handler(TOPIC, lambda reader, who: spawn(read_metrics(reader, who)))
    await room.connect(os.environ["LIVEKIT_URL"], token)
    source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=200)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(agent_name=agent, room=room_name)
    )

    async def settled(timeout: float) -> bool:
        """The agent is listening and has been quiet for 1.5 s."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if state["agent"] == "listening" and time.time() - state["loud_at"] > 1.5:
                return True
            await asyncio.sleep(0.1)
        return False

    heard: list[float | None] = []
    try:
        await asyncio.sleep(2.0)
        await settled(30)
        talking.set()
        for frames in speech:
            state["since"], state["first_loud"] = None, None
            for frame in frames:
                await source.capture_frame(frame)
            await source.wait_for_playout()
            state["since"] = time.time()
            while state["first_loud"] is None and time.time() - state["since"] < 15:
                await asyncio.sleep(0.02)
            first = state["first_loud"]
            heard.append(None if first is None else first - state["since"])
            await settled(30)
        await asyncio.sleep(1.0)
    finally:
        await room.disconnect()
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
    return {"heard": heard, "items": items}


def stage_values(calls: list[dict], role: str, key: str) -> list[float]:
    if role == "caller":
        return [v for c in calls for v in c["heard"] if v is not None]
    return [
        float(i["metrics"][key])
        for c in calls
        for i in c["items"]
        if i["role"] == role and i["metrics"].get(key) is not None
    ]


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]


def diff_ci(a: list[float], b: list[float], stat: Any) -> str:
    """B - A of ``stat``, with its 95% bootstrap interval: a gap inside it is noise."""
    rng = np.random.default_rng(0)
    boot = [
        stat(list(rng.choice(b, len(b)))) - stat(list(rng.choice(a, len(a)))) for _ in range(2000)
    ]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return f"{stat(b) - stat(a):+.3f} [{lo:+.3f},{hi:+.3f}]"


async def run_bench(agents: list[str], turns: int, out: str | None) -> None:
    speech = await synthesize(LINES)
    calls_each = math.ceil(turns / len(LINES))
    results: dict[str, list[dict]] = {a: [] for a in agents}
    lkapi = api.LiveKitAPI()
    try:
        for n in range(calls_each):
            for agent in agents:
                call = await one_call(lkapi, agent, speech)
                results[agent].append(call)
                heard = sum(v is not None for v in call["heard"])
                print(f"call {n + 1}/{calls_each} {agent}: {heard}/{len(LINES)} replies heard")
    finally:
        await lkapi.aclose()

    head = f"{'stage':<16}" + "".join(f"{a + ' p50':>18}{a + ' p90':>18}{'n':>5}" for a in agents)
    if len(agents) == 2:
        head += f"{'B-A p50 [95% CI]':>28}{'B-A p90 [95% CI]':>28}"
    print("\n" + head)
    for stage, (role, key) in STAGES.items():
        row = f"{stage:<16}"
        columns = [stage_values(results[agent], role, key) for agent in agents]
        for values in columns:
            if values:
                row += f"{statistics.median(values):>18.3f}{p90(values):>18.3f}{len(values):>5}"
            else:
                row += f"{'-':>18}{'-':>18}{0:>5}"
        if len(agents) == 2 and all(columns):
            row += f"{diff_ci(*columns, statistics.median):>28}{diff_ci(*columns, p90):>28}"
        print(row)
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="role", required=True)
    a = sub.add_parser("agent", help="serve the benchmark agent from this environment")
    a.add_argument("--name", required=True, help="the agent name the bench dispatches")
    b = sub.add_parser("bench", help="run the interleaved calls and print the table")
    b.add_argument("--agents", required=True, help="comma separated agent names, A,B")
    b.add_argument("--turns", type=int, default=30, help="replies per agent, at least")
    b.add_argument("--out", help="write the raw per-call results here as JSON")
    args = p.parse_args()
    if args.role == "agent":
        run_agent(args.name)
    else:
        asyncio.run(run_bench(args.agents.split(","), args.turns, args.out))


if __name__ == "__main__":
    main()
