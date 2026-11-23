"""Run real dispatched AgentSessions and a browser client against a test LiveKit project."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from aiohttp import web
from playwright.async_api import async_playwright

from livekit import api

HERE = Path(__file__).resolve().parent


def write_mark(path: Path, name: str, **fields):
    with path.open("a") as f:
        f.write(
            json.dumps({"kind": "mark", "name": name, "at_ms": time.time() * 1000, **fields}) + "\n"
        )


async def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    traces = output / "traces"
    traces.mkdir(exist_ok=True)
    worker_name = "startup-trace-" + uuid.uuid4().hex[:12]
    env = dict(
        os.environ,
        STARTUP_TRACE_DIR=str(traces),
        STARTUP_AGENT_NAME=worker_name,
        STARTUP_IDLE_PROCESSES=str(args.idle_processes),
        PYTHONUNBUFFERED="1",
    )
    worker_log = (output / "worker.log").open("w")
    vite_log = (output / "browser-server.log").open("w")
    worker = subprocess.Popen(
        [sys.executable, str(HERE / "agent.py")],
        env=env,
        stdout=worker_log,
        stderr=subprocess.STDOUT,
    )
    vite = subprocess.Popen(
        ["node", str(HERE / "web/server.mjs")], env=env, stdout=vite_log, stderr=subprocess.STDOUT
    )
    jobs = {}
    lk = api.LiveKitAPI()
    app = web.Application()

    async def token(request):
        run_id = request.match_info["run_id"]
        scenario = jobs[run_id]
        write_mark(traces / f"{run_id}.jsonl", "token_request", run_id=run_id)
        jwt = (
            api.AccessToken()
            .with_identity("user-" + run_id)
            .with_grants(api.VideoGrants(room_join=True, room=scenario["room"], can_publish=True))
            .with_room_config(
                api.RoomConfiguration(
                    agents=[
                        api.RoomAgentDispatch(agent_name=worker_name, metadata=json.dumps(scenario))
                    ]
                )
            )
            .to_jwt()
        )
        response = web.json_response({"url": os.environ["LIVEKIT_URL"], "token": jwt})
        response.headers["Access-Control-Allow-Origin"] = "http://127.0.0.1:18791"
        return response

    app.router.add_get("/token/{run_id}", token)
    app_runner = web.AppRunner(app)
    await app_runner.setup()
    await web.TCPSite(app_runner, "127.0.0.1", 18790).start()
    samples = []
    try:
        deadline = time.monotonic() + 90
        while True:
            if worker.poll() is not None:
                raise RuntimeError(
                    f"Agent worker exited ({worker.returncode}); inspect {output / 'worker.log'}"
                )
            ready = (
                (traces / "worker.jsonl").read_text() if (traces / "worker.jsonl").exists() else ""
            )
            if '"worker_registered"' in ready and (
                args.idle_processes == 0 or '"process_prepared"' in ready
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("worker registration/prewarm timeout")
            await asyncio.sleep(0.1)
        print("Worker registered; browser-driven dispatch begins", flush=True)
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                channel="chrome",
                headless=True,
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                ],
            )
            manifest = {
                "agents_source_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "browser_version": browser.version,
                "livekit_url": os.environ["LIVEKIT_URL"],
                "idle_processes": args.idle_processes,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "livekit-agents",
                        "livekit",
                        "livekit-api",
                        "livekit-local-inference",
                        "livekit-plugins-openai",
                        "playwright",
                    ]
                },
                "source_sha256": {
                    str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in HERE.rglob("*")
                    if path.is_file()
                    and path.suffix in [".py", ".js", ".mjs", ".html"]
                    and "reports" not in path.parts
                    and ".vite" not in path.parts
                },
                "method": "loaded browser UI; real token-triggered Cloud dispatch; synthetic tone in controlled cases; no artificial network relay",
                "audio_input": False,
                "requested_single_pc": True,
                "warmup_sample": -1,
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            cases = []
            if args.mode in ("controlled", "all"):
                cases += [
                    ("controlled-fast", "controlled", 0, 0),
                    ("controlled-activity-1200", "controlled", 1200, 0),
                    ("controlled-pcm-1200", "controlled", 0, 1200),
                ]
            if args.mode in ("realtime", "all"):
                cases += [("openai-realtime", "realtime", 0, 0)]
            if args.case:
                cases = [case for case in cases if case[0] == args.case]
            for name, mode, activity_delay, pcm_delay in cases:
                for sample in range(-1, args.repetitions):
                    profiles = (
                        ["example-order", "automatic"]
                        if sample % 2 == 0
                        else ["automatic", "example-order"]
                    )
                    if args.profile == "all":
                        order = ["example-order", "parallel-output", "automatic"]
                        offset = sample % len(order)
                        profiles = order[offset:] + order[:offset]
                    elif args.profile != "both":
                        profiles = [args.profile]
                    for profile in profiles:
                        run_id = uuid.uuid4().hex[:12]
                        scenario = {
                            "run_id": run_id,
                            "room": f"agent-startup-trace-{run_id}",
                            "case": name,
                            "mode": mode,
                            "sample": sample,
                            "profile": profile,
                            "activity_delay_ms": activity_delay,
                            "pcm_delay_ms": pcm_delay,
                        }
                        jobs[run_id] = scenario
                        artifact = output / run_id
                        artifact.mkdir()
                        context = await browser.new_context(
                            viewport={"width": 1100, "height": 760},
                            record_video_dir=str(artifact) if sample == 0 else None,
                        )
                        page = await context.new_page()
                        errors = []
                        page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                        try:
                            await page.goto(f"http://127.0.0.1:18791/?run={run_id}")
                            await page.wait_for_function("window.probeReady === true")
                            await page.click("#start")
                            await page.wait_for_function(
                                "window.result !== undefined", timeout=60000
                            )
                            result = await page.evaluate("window.result")
                            audio = result.pop("audioRecording", None)
                            if audio:
                                (artifact / "received.webm").write_bytes(base64.b64decode(audio))
                            result["pageErrors"] = errors
                            result["scenario"] = scenario
                            if (
                                result.get("errors")
                                or errors
                                or "first_rendered_audio" not in result["marks"]
                            ):
                                raise RuntimeError(
                                    "Browser did not receive/render audio: "
                                    + str(result.get("errors") or errors)
                                )
                            (artifact / "client.json").write_text(json.dumps(result, indent=2))
                            for marker, at_ms in result["marks"].items():
                                with (traces / f"{run_id}.jsonl").open("a") as f:
                                    f.write(
                                        json.dumps(
                                            {
                                                "kind": "mark",
                                                "name": "client_" + marker,
                                                "at_ms": at_ms,
                                                "run_id": run_id,
                                            }
                                        )
                                        + "\n"
                                    )
                            await page.screenshot(path=str(artifact / "client.png"))
                            samples.append(result)
                            t0 = result["marks"]["click"]
                            print(
                                f"{name} {profile} #{sample}: connected={result['marks']['room_connected'] - t0:.0f}ms audio={result['marks']['first_rendered_audio'] - t0:.0f}ms",
                                flush=True,
                            )
                        except BaseException:
                            print(
                                "BROWSER_DEBUG",
                                await page.evaluate("window.traceDebug && window.traceDebug()"),
                                flush=True,
                            )
                            await page.screenshot(path=str(artifact / "failure.png"))
                            raise
                        finally:
                            with contextlib.suppress(Exception):
                                await page.evaluate("window.cleanup && window.cleanup()")
                            with contextlib.suppress(Exception):
                                await lk.room.delete_room(
                                    api.DeleteRoomRequest(room=scenario["room"])
                                )
                            await context.close()
                            await asyncio.sleep(0.25)
                        (output / "clients.json").write_text(json.dumps(samples, indent=2))
            await browser.close()
    finally:
        await app_runner.cleanup()
        await lk.aclose()
        for proc in (worker, vite):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in (worker, vite):
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), 20)
            except asyncio.TimeoutError:
                proc.kill()
                await asyncio.to_thread(proc.wait)
        worker_log.close()
        vite_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["controlled", "realtime", "all"], default="controlled")
    parser.add_argument(
        "--profile",
        choices=["example-order", "parallel-output", "automatic", "both", "all"],
        default="both",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--idle-processes", type=int, default=1)
    parser.add_argument("--output", default="/tmp/agent-startup-trace")
    parser.add_argument("--provider-env", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.provider_env:
        from dotenv import dotenv_values

        configured = dotenv_values(args.provider_env)
        for key in ("OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT"):
            if configured.get(key):
                os.environ[key] = configured[key]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
