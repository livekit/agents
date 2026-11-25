"""Per-stage latency benchmark for the local voice stack.

Measures each stage directly against its endpoint — the same numbers the
agent's metrics_collected event reports, but reproducible standalone:

    TTS  — time to first audio byte, and total synthesis time
    STT  — full transcription latency for the TTS-generated utterance
    LLM  — time to first streamed token (TTFT), and tokens/sec

Run after all three servers are up:  python bench.py
"""

import io
import json
import statistics
import time
import urllib.request

STT_URL = "http://localhost:8000/v1/audio/transcriptions"
LLM_URL = "http://localhost:11434/v1/chat/completions"
TTS_URL = "http://localhost:8880/v1/audio/speech"

STT_MODEL = "Systran/faster-whisper-small"
LLM_MODEL = "llama3.2:3b"
TTS_MODEL = "kokoro"
TTS_VOICE = "af_heart"

RUNS = 5
UTTERANCE = "What's the weather looking like in Hartford today?"


def bench_tts() -> tuple[list[float], list[float], bytes]:
    first_byte, total, audio = [], [], b""
    for _ in range(RUNS):
        req = urllib.request.Request(
            TTS_URL,
            data=json.dumps(
                {
                    "model": TTS_MODEL,
                    "voice": TTS_VOICE,
                    "input": UTTERANCE,
                    "response_format": "wav",
                    "stream": True,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as resp:
            chunk = resp.read(4096)
            first_byte.append(time.perf_counter() - t0)
            buf = io.BytesIO()
            buf.write(chunk)
            while chunk := resp.read(65536):
                buf.write(chunk)
            total.append(time.perf_counter() - t0)
            audio = buf.getvalue()
    return first_byte, total, audio


def bench_stt(audio_wav: bytes) -> list[float]:
    boundary = "benchboundary"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{STT_MODEL}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="u.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + audio_wav
        + f"\r\n--{boundary}--\r\n".encode()
    )
    times = []
    for _ in range(RUNS):
        req = urllib.request.Request(
            STT_URL,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as resp:
            out = json.load(resp)
        times.append(time.perf_counter() - t0)
    print(f"  STT transcript: {out.get('text', '?')!r}")
    return times


def bench_llm() -> tuple[list[float], list[float]]:
    ttft, tps = [], []
    for _ in range(RUNS):
        req = urllib.request.Request(
            LLM_URL,
            data=json.dumps(
                {
                    "model": LLM_MODEL,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": "You are a concise voice assistant."},
                        {"role": "user", "content": UTTERANCE},
                    ],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        first, ntok = None, 0
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if delta:
                    ntok += 1
                    if first is None:
                        first = time.perf_counter() - t0
        elapsed = time.perf_counter() - t0
        ttft.append(first if first is not None else elapsed)
        tps.append(ntok / elapsed if elapsed else 0.0)
    return ttft, tps


def report(name: str, values: list[float], unit: str = "ms") -> None:
    scale = 1000 if unit == "ms" else 1
    vals = [v * scale for v in values]
    print(
        f"  {name}: median {statistics.median(vals):.0f}{unit}  "
        f"(min {min(vals):.0f}, max {max(vals):.0f}, n={len(vals)})"
    )


def warmup() -> None:
    """One throwaway call per endpoint so measurements reflect the warm path.

    Cold starts are real (model load into VRAM: ~20s for faster-whisper-small,
    a few seconds for the LLM) but they happen once per server lifetime —
    report them separately, don't let them pollute the steady-state numbers.
    """
    global RUNS
    runs, RUNS = RUNS, 1
    try:
        _, _, audio = bench_tts()
        bench_stt(audio)
        bench_llm()
    finally:
        RUNS = runs


if __name__ == "__main__":
    print("Warming up all three endpoints (cold-start excluded from numbers)...")
    warmup()
    print(f"\nBenchmarking local stack ({RUNS} runs per stage)\n")

    print("[TTS] kokoro")
    fb, total, audio = bench_tts()
    report("first audio byte", fb)
    report("full synthesis", total)
    print(f"  audio size: {len(audio) // 1024} KiB\n")

    print(f"[STT] {STT_MODEL}")
    report("full transcription", bench_stt(audio))
    print()

    print(f"[LLM] {LLM_MODEL}")
    ttft, tps = bench_llm()
    report("time to first token", ttft)
    print(f"  throughput: median {statistics.median(tps):.0f} tok/s")
