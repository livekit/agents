"""Explicit-opt-in, bounded Microsoft AI speech smoke test without LiveKit Cloud.

Confirm the plugin README's STT protocol or Azure Speech TTS endpoint before
using --run-live. No audio or transcripts are saved or printed. The STT fixture
and expected-text file must be user-approved inputs. --tts alone needs no STT access.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import stat
import time
import wave
from contextlib import AsyncExitStack
from pathlib import Path

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, stt, utils
from livekit.plugins import microsoft_ai

TTS_TEXT = "Hello, this is a Microsoft AI voice test."
CONNECT_OPTIONS = APIConnectOptions(max_retry=0, timeout=10.0)
MAX_DURATION = 5.0


def _check_env_file_permissions(path: Path) -> None:
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("The selected dotenv file must have permissions 0600; no requests sent")


def _read_fixture(path: Path) -> bytes:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("The approved WAV fixture must be at most 1 MiB")
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != 16000
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError("The approved fixture must be PCM16 mono WAV at 16000 Hz")
        if not 0 < audio.getnframes() <= int(MAX_DURATION * 16000):
            raise ValueError("The approved speech fixture must be between 0 and 5 seconds")
        pcm = audio.readframes(audio.getnframes())
        if len(pcm) != audio.getnframes() * 2:
            raise ValueError("The approved WAV fixture is truncated")
        return pcm


def _read_expected(path: Path) -> str:
    if path.stat().st_size > 4096:
        raise ValueError("The expected-text file must be at most 4096 bytes")
    expected = path.read_text(encoding="utf-8").strip()
    if not _normalize(expected) or len(expected) > 256:
        raise ValueError("The expected transcript must contain words and at most 256 characters")
    return expected


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


async def _check_stt(provider: microsoft_ai.STT, pcm: bytes, expected: str) -> None:
    async with provider.stream(conn_options=CONNECT_OPTIONS) as stream:

        async def send() -> None:
            for offset in range(0, len(pcm), 640):
                chunk = pcm[offset : offset + 640]
                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2,
                )
                stream.push_frame(frame)
                await asyncio.sleep(frame.duration)
            stream.end_input()

        async def receive() -> list[str]:
            return [
                event.alternatives[0].text
                async for event in stream
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
            ]

        sender = asyncio.create_task(send())
        receiver = asyncio.create_task(receive())
        try:
            await asyncio.gather(sender, receiver)
            transcripts = receiver.result()
        finally:
            await utils.aio.cancel_and_wait(sender, receiver)
    if len(transcripts) != 1 or _normalize(transcripts[0]) != _normalize(expected):
        raise ValueError("STT did not match the complete expected transcript, including its tail")
    print("STT: one finalized item matched the expected words; transcript not printed.")


async def _check_tts(provider: microsoft_ai.TTS) -> None:
    started = time.perf_counter()
    async with provider.synthesize(TTS_TEXT, conn_options=CONNECT_OPTIONS) as stream:
        samples = 0
        finals = 0
        frames = 0
        async for event in stream:
            if (
                event.frame.sample_rate != provider.sample_rate
                or event.frame.num_channels != 1
                or event.frame.data.itemsize != 2
            ):
                raise ValueError("TTS output has an unexpected audio format")
            samples += event.frame.samples_per_channel
            finals += int(event.is_final)
            frames += 1
        if samples == 0 or finals != 1:
            raise ValueError("TTS did not return exactly one nonempty final audio segment")
    elapsed = time.perf_counter() - started
    print(
        f"TTS: {frames} PCM16 mono frames at {provider.sample_rate} Hz; "
        f"audio_duration={samples / provider.sample_rate:.3f}s; "
        f"elapsed={elapsed:.3f}s (client synthesis call through stream closure, not model TTFA). "
        "Audio not saved or played."
    )


async def _run(
    *, pcm: bytes | None, expected: str | None, check_tts: bool, env_file: Path | None
) -> None:
    async with AsyncExitStack() as stack:
        speech_to_text = None
        text_to_speech = None
        # Preflight every selected service before opening a socket or issuing HTTP.
        if pcm is not None and expected is not None:
            speech_to_text = microsoft_ai.STT(vad=None, env_file=env_file)
            await stack.enter_async_context(speech_to_text)
        if check_tts:
            text_to_speech = microsoft_ai.TTS(
                env_file=env_file, request_timeout=20.0, max_audio_bytes=1024 * 1024
            )
            await stack.enter_async_context(text_to_speech)
        if speech_to_text is not None and pcm is not None and expected is not None:
            await _check_stt(speech_to_text, pcm, expected)
        if text_to_speech is not None:
            await _check_tts(text_to_speech)
    print("All selected providers and transports closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Confirm endpoint contracts/auth and approve sending the selected fixture/fixed text",
    )
    parser.add_argument(
        "--stt-wav", type=Path, help="Approved PCM16 mono 16kHz speech WAV, at most 5 seconds"
    )
    parser.add_argument(
        "--expected-text-file", type=Path, help="Approved expected transcript (not logged)"
    )
    parser.add_argument(
        "--tts", action="store_true", help=f"Send exactly one TTS request for: {TTS_TEXT!r}"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Explicit local dotenv file; alternatively set MICROSOFT_AI_ENV_FILE",
    )
    args = parser.parse_args()
    if not args.run_live:
        parser.error("No requests sent. Explicit --run-live approval is required")
    if not args.tts and args.stt_wav is None:
        parser.error("Select --tts and/or --stt-wav")
    if (args.stt_wav is None) != (args.expected_text_file is None):
        parser.error("--stt-wav and --expected-text-file must be supplied together")
    if os.environ.get("LK_DUMP_TTS", "0") != "0":
        parser.error("Unset LK_DUMP_TTS: smoke tests must not write audio captures")

    try:
        env_path = args.env_file or os.environ.get("MICROSOFT_AI_ENV_FILE")
        if env_path is not None:
            _check_env_file_permissions(Path(env_path))
        pcm = _read_fixture(args.stt_wav) if args.stt_wav is not None else None
        expected = (
            _read_expected(args.expected_text_file) if args.expected_text_file is not None else None
        )
        asyncio.run(
            asyncio.wait_for(
                _run(pcm=pcm, expected=expected, check_tts=args.tts, env_file=args.env_file),
                timeout=50.0,
            )
        )
    except (APIError, OSError, ValueError, wave.Error, EOFError, asyncio.TimeoutError) as error:
        # Filesystem/transport errors may contain private paths or endpoint information.
        # Provider APIError messages are deliberately sanitized by the plugin.
        detail = str(error) if isinstance(error, (APIError, ValueError)) else type(error).__name__
        parser.exit(1, f"Smoke test failed: {detail}\n")


if __name__ == "__main__":
    main()
