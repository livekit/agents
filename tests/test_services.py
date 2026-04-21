"""Standalone end-to-end test for 60db TTS, STT, and LLM services.

Talks directly to the raw WebSocket/HTTP APIs, bypassing LiveKit wrappers.

Usage:
    python test_services.py
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import io
import json
import os
import sys
import wave
from pathlib import Path

# Fix Windows console encoding for emoji/unicode in LLM output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import httpx
import websockets
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(Path(".env.local"))

API_KEY = os.getenv("SIXTY_DB_API_KEY", "")
TTS_URL = os.getenv("SIXTY_DB_TTS_URL", "wss://api.60db.ai/ws/tts")
STT_URL = os.getenv("SIXTY_DB_STT_URL", "wss://api.60db.ai/ws/stt")
LLM_URL = os.getenv("SIXTY_DB_LLM_URL", "https://api.60db.ai/v1/chat/completions")

REF_DIR = Path("Refrencefile")
TTS_OUTPUT = REF_DIR / "tts_output.wav"
TEST_JSON = REF_DIR / "test_new.json"

BANNER_WIDTH = 60


def _banner(label: str, passed: bool) -> None:
    tag = "[PASS]" if passed else "[FAIL]"
    line = f" {tag} {label} ".center(BANNER_WIDTH, "=")
    print(f"\n{line}\n")


# ---------------------------------------------------------------------------
# 1. TTS test
# ---------------------------------------------------------------------------
async def test_tts() -> bool:
    print("\n" + " TTS TEST ".center(BANNER_WIDTH, "-"))
    text = "My name is nitish and I am a developer"
    context_id = "test-ctx-001"

    try:
        url = f"{TTS_URL}?apiKey={API_KEY}"
        print(f"  Connecting to {TTS_URL} ...")

        async with websockets.connect(
            url, ping_interval=30, ping_timeout=10, max_size=10 * 1024 * 1024, open_timeout=15
        ) as ws:
            # 1) connection_established (server may send a "connecting" message first)
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if msg.get("connection_established"):
                    break
                print(f"  Server: {msg}")
            print("  Connected.")

            # 2) create_context
            await ws.send(json.dumps({
                "create_context": {
                    "context_id": context_id,
                    "voice_id": "fbb75ed2-975a-40c7-9e06-38e30524a9a1",
                    "audio_config": {
                        "audio_encoding": "LINEAR16",
                        "sample_rate_hertz": 16000,
                    },
                }
            }))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            assert msg.get("context_created"), f"Expected context_created, got {msg}"
            print("  Context created.")

            # 3) send text + flush
            await ws.send(json.dumps({
                "send_text": {"context_id": context_id, "text": text}
            }))
            await ws.send(json.dumps({
                "flush_context": {"context_id": context_id}
            }))

            # 4) collect audio chunks
            audio_data = bytearray()
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if "audio_chunk" in msg:
                    b64 = msg["audio_chunk"].get("audioContent", "")
                    if b64:
                        audio_data.extend(base64.b64decode(b64))
                if msg.get("flush_completed"):
                    break

            # 5) close context
            await ws.send(json.dumps({
                "close_context": {"context_id": context_id}
            }))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if msg.get("context_closed"):
                print("  Context closed.")

        # 6) save WAV
        REF_DIR.mkdir(parents=True, exist_ok=True)
        with wave.open(str(TTS_OUTPUT), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(16000)
            wf.writeframes(bytes(audio_data))

        size_kb = TTS_OUTPUT.stat().st_size / 1024
        print(f"  Saved {TTS_OUTPUT} ({size_kb:.1f} KB, {len(audio_data)} bytes PCM)")
        _banner("TTS", True)
        return True

    except Exception as exc:
        print(f"  ERROR: {exc}")
        _banner("TTS", False)
        return False


# ---------------------------------------------------------------------------
# 2. STT test
# ---------------------------------------------------------------------------
async def test_stt() -> bool:
    print("\n" + " STT TEST ".center(BANNER_WIDTH, "-"))

    if not TTS_OUTPUT.exists():
        print(f"  ERROR: {TTS_OUTPUT} not found — run TTS test first")
        _banner("STT", False)
        return False

    # Read the WAV file back as raw PCM
    if not TTS_OUTPUT.exists():
        print(f"  ERROR: {TTS_OUTPUT} not found — run TTS test first")
        _banner("STT", False)
        return False

    with wave.open(str(TTS_OUTPUT), "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono WAV"
        assert wf.getsampwidth() == 2, "Expected 16-bit WAV"
        input_rate = wf.getframerate()
        raw_pcm = wf.readframes(wf.getnframes())

    print(f"  Read {TTS_OUTPUT}: {input_rate} Hz, {len(raw_pcm)} bytes PCM")

    # Resample 16000 → 8000 Hz
    target_rate = 8000
    resampled, _ = audioop.ratecv(raw_pcm, 2, 1, input_rate, target_rate, None)
    # Convert LINEAR16 → mulaw
    mulaw_data = audioop.lin2ulaw(resampled, 2)

    print(f"  Converted to mulaw @ {target_rate} Hz ({len(mulaw_data)} bytes)")

    url = f"{STT_URL}?apiKey={API_KEY}"
    MAX_RETRIES = 3

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Connecting to {STT_URL} (attempt {attempt}/{MAX_RETRIES}) ...")

            async with websockets.connect(
                url, ping_interval=30, ping_timeout=10, max_size=10 * 1024 * 1024, open_timeout=15
            ) as ws:
                # Wait for connection_established (server may send "connecting" first)
                got_error = False
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    if msg.get("connection_established") or msg.get("type") == "connection_established":
                        break
                    print(f"  Server: {msg}")
                    if msg.get("type") == "error":
                        print(f"  Server error during handshake: {msg.get('error')}")
                        got_error = True
                        break
                if got_error:
                    if attempt < MAX_RETRIES:
                        print(f"  Retrying in 2s ...")
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise RuntimeError(f"Server error after {MAX_RETRIES} attempts")

                print("  Connected.")

                # Send start
                await ws.send(json.dumps({
                    "type": "start",
                    "languages": ["en"],
                    "config": {
                        "encoding": "mulaw",
                        "sample_rate": 8000,
                        "continuous_mode": True,
                    },
                }))

                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                assert msg.get("type") == "connected", f"Expected connected, got {msg}"
                print("  STT session started.")

                # Stream audio in chunks
                CHUNK_SIZE = 4096
                for i in range(0, len(mulaw_data), CHUNK_SIZE):
                    await ws.send(mulaw_data[i : i + CHUNK_SIZE])

                # Send stop
                await ws.send(json.dumps({"type": "stop"}))
                print("  Audio sent, waiting for transcription ...")

                # Collect transcriptions
                final_transcript = ""
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    msg_type = msg.get("type", "")

                    if msg_type == "transcription":
                        text = msg.get("text", "")
                        is_final = msg.get("is_final", False)
                        if is_final:
                            final_transcript = text
                        print(f"  Transcription ({'final' if is_final else 'partial'}): {text}")

                    if msg_type == "session_stopped":
                        print(f"  Session stopped.")
                        break

                    if msg_type == "error":
                        print(f"  Server error: {msg.get('error')}")
                        break

            print(f"  Final transcript: {final_transcript!r}")
            _banner("STT", True)
            return True

        except Exception as exc:
            print(f"  ERROR: {exc}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in 2s ...")
                await asyncio.sleep(2)
            else:
                _banner("STT", False)
                return False

    _banner("STT", False)
    return False


# ---------------------------------------------------------------------------
# 3. LLM test
# ---------------------------------------------------------------------------
async def test_llm() -> bool:
    print("\n" + " LLM TEST ".center(BANNER_WIDTH, "-"))

    if not TEST_JSON.exists():
        print(f"  ERROR: {TEST_JSON} not found")
        _banner("LLM", False)
        return False

    try:
        with open(TEST_JSON, "r", encoding="utf-8") as f:
            context_data = json.load(f)

        system_content = (
            "You are a helpful AI assistant. Use the following context to answer the user's question.\n\n"
            f"Context:\n{json.dumps(context_data, indent=2)[:4000]}"
        )
        user_question = "How do we save electricity?"

        body = {
            "model": "qcall/slm-3b-int4",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_question},
            ],
        }

        print(f"  POST {LLM_URL}")
        print(f"  Question: {user_question}")

        full_response = ""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
        ) as client:
            async with client.stream("POST", LLM_URL, json=body) as resp:
                resp.raise_for_status()
                print("  Streaming response:")
                print("  ---")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    print(json.dumps(chunk, indent=2))
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_response += content
                print("\n  ---")

        print(f"  Full response ({len(full_response)} chars)")
        _banner("LLM", True)
        return True

    except Exception as exc:
        print(f"  ERROR: {exc}")
        _banner("LLM", False)
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def main() -> None:
    if not API_KEY:
        print("ERROR: SIXTY_DB_API_KEY not set in .env.local")
        return

    results = {}
    results["TTS"] = await test_tts()
    results["STT"] = await test_stt()
    results["LLM"] = await test_llm()

    print("\n" + " SUMMARY ".center(BANNER_WIDTH, "="))
    for name, passed in results.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  {name}: {tag}")
    print("=" * BANNER_WIDTH)


if __name__ == "__main__":
    asyncio.run(main())
