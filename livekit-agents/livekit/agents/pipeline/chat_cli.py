from __future__ import annotations

import asyncio
import sys
import termios
import threading
import time
import tty
from typing import Literal

import click
import numpy as np
import sounddevice as sd
from livekit import rtc

from ..log import logger
from ..utils import aio, log_exceptions
from . import io
from .pipeline_agent import PipelineAgent

MAX_AUDIO_BAR = 30
INPUT_DB_MIN = -70.0
INPUT_DB_MAX = 0.0
FPS = 20


def _esc(*codes: int) -> str:
    return "\033[" + ";".join(str(c) for c in codes) + "m"


def _normalize_db(amplitude_db: float, db_min: float, db_max: float) -> float:
    amplitude_db = max(db_min, min(amplitude_db, db_max))
    return (amplitude_db - db_min) / (db_max - db_min)


class _TextSink(io.TextSink):
    def __init__(self, cli: "ChatCLI") -> None:
        self._cli = cli
        self._capturing = False

    async def capture_text(self, text: str) -> None:
        if not self._capturing:
            self._capturing = True
            sys.stdout.write("\r")
            sys.stdout.flush()
            click.echo(_esc(36), nl=False)

        click.echo(text, nl=False)

    def flush(self) -> None:
        if self._capturing:
            click.echo(_esc(0))
            self._capturing = False


class _AudioSink(io.AudioSink):
    def __init__(self, cli: "ChatCLI") -> None:
        super().__init__(sample_rate=24000)
        self._cli = cli
        self._capturing = False
        self._pushed_duration: float = 0.0
        self._capture_start: float = 0.0
        self._dispatch_handle: asyncio.TimerHandle | None = None

        self._flush_complete = asyncio.Event()
        self._flush_complete.set()

        self._output_buf = bytearray()
        self._output_lock = threading.Lock()

    @property
    def lock(self) -> threading.Lock:
        return self._output_lock

    @property
    def audio_buffer(self) -> bytearray:
        return self._output_buf

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        await self._flush_complete.wait()

        if not self._capturing:
            self._capturing = True
            self._pushed_duration = 0.0
            self._capture_start = time.monotonic()

        self._pushed_duration += frame.duration
        with self._output_lock:
            self._output_buf += frame.data

    def flush(self) -> None:
        super().flush()
        if self._capturing:
            self._flush_complete.clear()
            self._capturing = False
            to_wait = max(0.0, self._pushed_duration - (time.monotonic() - self._capture_start))
            self._dispatch_handle = self._cli._loop.call_later(
                to_wait, self._dispatch_playback_finished
            )

    def clear_buffer(self) -> None:
        self._capturing = False

        with self._output_lock:
            self._output_buf.clear()

        if self._pushed_duration > 0.0:
            if self._dispatch_handle is not None:
                self._dispatch_handle.cancel()

            self._flush_complete.set()
            self._pushed_duration = 0.0
            played_duration = min(time.monotonic() - self._capture_start, self._pushed_duration)
            self.on_playback_finished(
                playback_position=played_duration,
                interrupted=played_duration + 1.0 < self._pushed_duration,
            )

    def _dispatch_playback_finished(self) -> None:
        self.on_playback_finished(playback_position=self._pushed_duration, interrupted=False)
        self._flush_complete.set()
        self._pushed_duration = 0.0


class ChatCLI:
    def __init__(
        self,
        agent: PipelineAgent,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._agent = agent
        self._done_fut = asyncio.Future()
        self._micro_db = INPUT_DB_MIN

        self._audio_input_ch = aio.Chan[rtc.AudioFrame](loop=self._loop)

        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        self._cli_mode: Literal["text", "audio"] = "audio"

        self._text_input_buf = []

        self._text_sink = _TextSink(self)
        self._audio_sink = _AudioSink(self)

    def _print_welcome(self):
        print(_esc(34) + "=" * 50 + _esc(0))
        print(_esc(34) + "     Livekit Agents - ChatCLI" + _esc(0))
        print(_esc(34) + "=" * 50 + _esc(0))
        print("Press [Ctrl+B] to toggle between Text/Audio mode, [Q] to quit.\n")

    async def run(self) -> None:
        self._print_welcome()

        fd = sys.stdin.fileno()
        stdin_ch = aio.Chan[str](loop=self._loop)

        def _on_input():
            try:
                ch = sys.stdin.read(1)
                stdin_ch.send_nowait(ch)
            except Exception:
                stdin_ch.close()

        self._loop.add_reader(fd, _on_input)
        old_settings = termios.tcgetattr(fd)

        self._update_microphone(enable=True)
        self._update_speaker(enable=True)

        try:
            tty.setcbreak(fd)
            input_cli_task = asyncio.create_task(self._input_cli_task(stdin_ch))
            input_cli_task.add_done_callback(lambda _: self._done_fut.set_result(None))

            render_cli_task = asyncio.create_task(self._render_cli_task())

            await self._done_fut
            await aio.cancel_and_wait(render_cli_task)

            self._update_microphone(enable=False)
            self._update_speaker(enable=False)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self._loop.remove_reader(fd)

    def _update_microphone(self, *, enable: bool) -> None:
        input_device, _ = sd.default.device
        if input_device is not None and enable:
            device_info = sd.query_devices(input_device)
            assert isinstance(device_info, dict)

            self._input_device_name: str = device_info.get("name", "Microphone")
            self._input_stream = sd.InputStream(
                callback=self._sd_input_callback,
                dtype="int16",
                channels=1,
                device=input_device,
                samplerate=24000,
            )
            self._input_stream.start()
            self._agent.input.audio = self._audio_input_ch
        elif self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
            self._agent.input.audio = None

    def _update_speaker(self, *, enable: bool) -> None:
        _, output_device = sd.default.device
        if output_device is not None and enable:
            self._output_stream = sd.OutputStream(
                callback=self._sd_output_callback,
                dtype="int16",
                channels=1,
                device=output_device,
                samplerate=24000,
                blocksize=4800,
            )
            self._output_stream.start()
            self._agent.output.audio = self._audio_sink
        elif self._output_stream is not None:
            self._output_stream.close()
            self._output_stream = None
            self._agent.output.audio = None

    def _update_text_output(self, *, enable: bool) -> None:
        if enable:
            self._agent.output.text = self._text_sink
        else:
            self._agent.output.text = None
            self._text_input_buf = []

    def _sd_output_callback(self, outdata: np.ndarray, frames: int, *_) -> None:
        with self._audio_sink.lock:
            bytes_needed = frames * 2
            if len(self._audio_sink.audio_buffer) < bytes_needed:
                available_bytes = len(self._audio_sink.audio_buffer)
                outdata[: available_bytes // 2, 0] = np.frombuffer(
                    self._audio_sink.audio_buffer,
                    dtype=np.int16,
                    count=available_bytes // 2,
                )

                outdata[available_bytes // 2 :, 0] = 0
                del self._audio_sink.audio_buffer[:available_bytes]
            else:
                chunk = self._audio_sink.audio_buffer[:bytes_needed]
                outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16, count=frames)
                del self._audio_sink.audio_buffer[:bytes_needed]

    def _sd_input_callback(self, indata: np.ndarray, frame_count: int, *_) -> None:
        rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
        max_int16 = np.iinfo(np.int16).max
        self._micro_db = 20.0 * np.log10(rms / max_int16 + 1e-6)

        frame = rtc.AudioFrame(
            data=indata.tobytes(),
            samples_per_channel=frame_count,
            sample_rate=24000,
            num_channels=1,
        )
        self._loop.call_soon_threadsafe(self._audio_input_ch.send_nowait, frame)

    @log_exceptions(logger=logger)
    async def _input_cli_task(self, in_ch: aio.Chan[str]) -> None:
        while True:
            char = await in_ch.recv()
            if char is None:
                break

            if char == "\x02":  # Ctrl+B
                if self._cli_mode == "audio":
                    self._cli_mode = "text"
                    self._update_text_output(enable=True)
                    self._update_microphone(enable=False)
                    self._update_speaker(enable=False)
                    click.echo("\nSwitched to Text Input Mode.", nl=False)
                else:
                    self._cli_mode = "audio"
                    self._update_text_output(enable=False)
                    self._update_microphone(enable=True)
                    self._update_speaker(enable=True)
                    self._text_input_buf = []
                    click.echo("\nSwitched to Audio Input Mode.", nl=False)

            if self._cli_mode == "text":  # Read input
                if char in ("\r", "\n"):
                    text = "".join(self._text_input_buf)
                    if text:
                        self._text_input_buf = []
                        self._agent.generate_reply(user_input=text)
                        click.echo("\n", nl=False)
                elif char == "\x7f":  # Backspace
                    if self._text_input_buf:
                        self._text_input_buf.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif char.isprintable():
                    self._text_input_buf.append(char)
                    click.echo(char, nl=False)
                    sys.stdout.flush()

    async def _render_cli_task(self) -> None:
        next_frame = time.perf_counter()
        while True:
            next_frame += 1 / FPS
            if self._cli_mode == "audio":
                self._print_audio_mode()
            elif self._cli_mode == "text" and not self._text_sink._capturing:
                self._print_text_mode()

            await asyncio.sleep(max(0, next_frame - time.perf_counter()))

    def _print_audio_mode(self):
        amplitude_db = _normalize_db(self._micro_db, db_min=INPUT_DB_MIN, db_max=INPUT_DB_MAX)
        nb_bar = round(amplitude_db * MAX_AUDIO_BAR)

        color_code = 31 if amplitude_db > 0.75 else 33 if amplitude_db > 0.5 else 32
        bar = "#" * nb_bar + "-" * (MAX_AUDIO_BAR - nb_bar)
        sys.stdout.write(
            f"\r[Audio] {self._input_device_name[-20:]} [{self._micro_db:6.2f} dBFS] {_esc(color_code)}[{bar}]{_esc(0)}"
        )
        sys.stdout.flush()

    def _print_text_mode(self):
        sys.stdout.write("\r")
        sys.stdout.flush()
        prompt = "Enter your message: "
        sys.stdout.write(f"[Text {prompt}{''.join(self._text_input_buf)}")
        sys.stdout.flush()
