from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import enum
import hashlib
import json
import logging
import os
import pathlib
import re
import signal
import sys
import textwrap
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from inspect import istraceback
from types import FrameType
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Optional, Union

import numpy as np
import typer
from rich.columns import Columns
from rich.console import Console, ConsoleRenderable, Group, RenderableType
from rich.live import Live
from rich.segment import Segment
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text
from rich.theme import Theme

from livekit import rtc

from .._exceptions import CLIError
from ..job import JobExecutorType
from ..log import logger
from ..plugin import Plugin
from ..utils import aio
from ..voice import AgentSession, io
from ..voice.run_result import RunEvent
from ..worker import AgentServer, WorkerOptions
from . import proto

# from .discover import get_import_data
from .readchar import key, readkey

TRACE_LOG_LEVEL = 5

if TYPE_CHECKING:
    import sounddevice as sd  # type: ignore

HANDLED_SIGNALS = (
    signal.SIGINT,  # Unix signal 2. Sent by Ctrl+C.
    signal.SIGTERM,
)


class _ToggleMode(Exception):
    pass


class _ExitCli(Exception):
    pass


# from https://github.com/encode/uvicorn/blob/c1144fd4f130388cffc05ee17b08747ce8c1be11/uvicorn/importer.py#L9C1-L34C20
# def import_from_string(import_str: Any) -> Any:
#     if not isinstance(import_str, str):
#         return import_str

#     module_str, _, attrs_str = import_str.partition(":")
#     if not module_str or not attrs_str:
#         message = 'Import string "{import_str}" must be in format "<module>:<attribute>".'
#         raise RuntimeError(message.format(import_str=import_str))

#     try:
#         module = importlib.import_module(module_str)
#     except ModuleNotFoundError as exc:
#         if exc.name != module_str:
#             raise exc from None
#         message = 'Could not import module "{module_str}".'
#         raise RuntimeError(message.format(module_str=module_str)) from None

#     instance = module
#     try:
#         for attr_str in attrs_str.split("."):
#             instance = getattr(instance, attr_str)
#     except AttributeError:
#         message = 'Attribute "{attrs_str}" not found in module "{module_str}".'
#         raise RuntimeError(message.format(attrs_str=attrs_str, module_str=module_str)) from None

#     return instance


ConsoleMode = Literal["text", "audio"]

SAMPLE_RATE = 24000


class ConsoleAudioInput(io.AudioInput):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(label="Console")
        self._loop = loop
        self._audio_ch: aio.Chan[rtc.AudioFrame] = aio.Chan()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._audio_ch.send_nowait(frame)

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._audio_ch.__anext__()


class ConsoleAudioOutput(io.AudioOutput):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(
            label="Console",
            next_in_chain=None,
            sample_rate=SAMPLE_RATE,
            capabilities=io.AudioOutputCapabilities(pause=False),  # TODO(theomonnom): support pause
        )
        self._loop = loop

        self._capturing = False
        self._pushed_duration: float = 0.0
        self._capture_start: float = 0.0
        self._dispatch_handle: asyncio.TimerHandle | None = None

        self._flush_complete = asyncio.Event()
        self._flush_complete.set()

        self._output_buf = bytearray()
        self._audio_lock = threading.Lock()

    @property
    def audio_lock(self) -> threading.Lock:
        return self._audio_lock

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
        with self._audio_lock:
            self._output_buf += frame.data  # TODO: optimize

    def flush(self) -> None:
        super().flush()
        if self._capturing:
            self._flush_complete.clear()
            self._capturing = False
            to_wait = max(0.0, self._pushed_duration - (time.monotonic() - self._capture_start))

            def _dispatch_playback_finished() -> None:
                self.on_playback_finished(
                    playback_position=self._pushed_duration, interrupted=False
                )
                self._flush_complete.set()
                self._pushed_duration = 0.0

            self._dispatch_handle = self._loop.call_later(to_wait, _dispatch_playback_finished)

    def clear_buffer(self) -> None:
        self._capturing = False

        with self._audio_lock:
            self._output_buf.clear()

        if self._pushed_duration > 0.0:
            if self._dispatch_handle is not None:
                self._dispatch_handle.cancel()

            self._flush_complete.set()
            played_duration = min(time.monotonic() - self._capture_start, self._pushed_duration)
            self.on_playback_finished(
                playback_position=played_duration,
                interrupted=played_duration + 1.0 < self._pushed_duration,
            )
            self._pushed_duration = 0.0


class AgentsConsole:
    _instance: AgentsConsole | None = None
    _console_directory = "console-recordings"

    @classmethod
    def get_instance(cls) -> AgentsConsole:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        theme: dict[str, str | Style] = {
            "tag": "black on #1fd5f9",
            "label": "#8f83ff",
            "error": "red",
            "lk-fg": "#1fd5f9",
            "log.name": Style.null(),
            "log.extra": Style(dim=True),
            "logging.level.notset": Style(dim=True),
            "logging.level.debug": Style(color="cyan"),
            "logging.level.info": Style(color="green"),
            "logging.level.warning": Style(color="yellow"),
            "logging.level.dev": Style(color="blue"),
            "logging.level.error": Style(color="red", bold=True),
            "logging.level.critical": Style(color="red", bold=True, reverse=True),
        }
        self.tag_width = 11
        self.console = Console(theme=Theme(theme))

        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=True,
        )

        self._input_delay = 0.0
        self._input_name: str | None = None
        self._input_stream: sd.InputStream | None = None

        self._output_delay = 0.0
        self._output_name: str | None = None
        self._output_stream: sd.OutputStream | None = None

        self._input_lock = threading.Lock()
        self._input_levels = np.zeros(14, dtype=np.float32)

        self._console_mode: ConsoleMode = "audio"

        self._lock = threading.Lock()
        self._io_acquired = False
        self._io_acquired_event = threading.Event()

        self._enabled = False
        self._record = False

        self._text_mode_log_filter = TextModeLogFilter()
        self._log_handler = RichLoggingHandler(self)

        self._session_directory = pathlib.Path(
            self._console_directory, f"session-{datetime.datetime.now().strftime('%m-%d-%H%M%S')}"
        )

    def acquire_io(self, *, loop: asyncio.AbstractEventLoop, session: AgentSession) -> None:
        with self._lock:
            if self._io_acquired:
                raise RuntimeError("the ConsoleIO was already acquired by another session")

            if asyncio.get_running_loop() != loop:
                raise RuntimeError(
                    "the ConsoleIO must be acquired in the same asyncio loop as the session"
                )

            self._io_acquired = True
            self._io_loop = loop
            self._io_context = contextvars.copy_context()
            self._io_audio_input = ConsoleAudioInput(loop)
            self._io_audio_output = ConsoleAudioOutput(loop)
            self._io_acquired_event.set()
            self._io_session = session

        self._update_sess_io(
            session, self.console_mode, self._io_audio_input, self._io_audio_output
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    @property
    def record(self) -> bool:
        return self._record

    @record.setter
    def record(self, val: bool) -> None:
        self._record = val

    @property
    def session_directory(self) -> pathlib.Path:
        return self._session_directory

    @property
    def io_acquired(self) -> bool:
        with self._lock:
            return self._io_acquired

    @property
    def io_session(self) -> AgentSession:
        if not self._io_acquired:
            raise RuntimeError("AgentsConsole is not acquired")

        return self._io_session

    @property
    def io_loop(self) -> asyncio.AbstractEventLoop:
        if not self._io_acquired:
            raise RuntimeError("AgentsConsole is not acquired")

        return self._io_loop

    @property
    def io_context(self) -> contextvars.Context:
        if not self._io_acquired:
            raise RuntimeError("AgentsConsole is not acquired")

        return self._io_context

    def wait_for_io_acquisition(self) -> None:
        self._io_acquired_event.wait()

    @property
    def input_name(self) -> str | None:
        return self._input_name

    @property
    def output_name(self) -> str | None:
        return self._output_name

    @property
    def console_mode(self) -> ConsoleMode:
        return self._console_mode

    @console_mode.setter
    def console_mode(self, mode: ConsoleMode) -> None:
        with self._lock:
            self._console_mode = mode

            if not self._io_acquired:
                return

            self.io_loop.call_soon_threadsafe(
                self._update_sess_io,
                self.io_session,
                mode,
                self._io_audio_input,
                self._io_audio_output,
            )

    def _update_sess_io(
        self,
        sess: AgentSession,
        mode: ConsoleMode,
        audio_input: ConsoleAudioInput,
        audio_output: ConsoleAudioOutput,
    ) -> None:
        if asyncio.get_running_loop() != self.io_loop:
            raise RuntimeError("_update_sess_io must be executed on the io_loop")

        with self._lock:
            if not self._io_acquired:
                return

            if self._io_session != sess or self._console_mode != mode:
                return

            if mode == "text":
                sess.input.audio = None
                sess.output.audio = None
                self._log_handler.addFilter(self._text_mode_log_filter)
            else:
                sess.input.audio = audio_input
                sess.output.audio = audio_output
                self._log_handler.removeFilter(self._text_mode_log_filter)

    def print(
        self, child: RenderableType, *, tag: str = "", tag_style: Style | None = None
    ) -> None:
        self.console.print(self._render_tag(child, tag=tag, tag_style=tag_style))

    def _render_tag(
        self,
        child: RenderableType,
        *,
        tag: str = "",
        tag_width: int | None = None,
        tag_style: Style | None = None,
    ) -> ConsoleRenderable:
        if tag:
            tag = f" {tag} "

        tag_width = tag_width or self.tag_width
        table = Table.grid(
            Column(width=tag_width + 2, no_wrap=True),
            Column(no_wrap=False, overflow="fold"),
            padding=(0, 0, 0, 0),
            collapse_padding=True,
            pad_edge=False,
        )

        left_padding = tag_width - len(tag)
        left_padding = max(0, left_padding)

        style = tag_style or self.console.get_style("tag")
        tag_segments = [Segment(tag, style=style)]

        left = [Segment(" " * left_padding), *tag_segments]
        table.add_row(Group(*left), Group(child))  # type: ignore
        return table

    def set_microphone_enabled(self, enable: bool, *, device: int | str | None = None) -> None:
        if self._input_stream:
            self._input_stream.close()
            self._input_stream = self._input_name = None

        if not enable:
            return

        import sounddevice as sd

        if device is None:
            device, _ = sd.default.device

        try:
            device_info = sd.query_devices(device, kind="input")
        except Exception:
            raise CLIError(
                "Unable to access the microphone. \n"
                "Please ensure a microphone is connected and recognized by your system. "
                "To see available input devices, run: lk-agents console --list-devices"
            ) from None

        assert isinstance(device_info, dict), "device_info is dict"

        self._input_name = device_info.get("name", "Unnamed microphone")
        self._input_stream = sd.InputStream(
            callback=self._sd_input_callback,
            dtype="int16",
            channels=1,
            device=device,
            samplerate=24000,
            blocksize=2400,
        )
        self._input_stream.start()

    def set_speaker_enabled(self, enable: bool, *, device: int | str | None = None) -> None:
        if self._output_stream:
            self._output_stream.close()
            self._output_stream = self._output_name = None

        if not enable:
            return

        import sounddevice as sd

        if device is None:
            _, device = sd.default.device

        try:
            device_info = sd.query_devices(device, kind="output")
        except Exception:
            raise CLIError(
                "Unable to access the speaker. \n"
                "Please ensure a speaker is connected and recognized by your system. "
                "To see available output devices, run: lk-agents console --list-devices"
            ) from None

        assert isinstance(device_info, dict), "device_info is dict"

        self._output_name = device_info.get("name", "Unnamed speaker")
        self._output_stream = sd.OutputStream(
            callback=self._sd_output_callback,
            dtype="int16",
            channels=1,
            device=device,
            samplerate=24000,
            blocksize=2400,
        )
        self._output_stream.start()

    def _validate_device_or_raise(
        self, *, input_device: str | None, output_device: str | None
    ) -> None:
        import sounddevice as sd

        try:
            if input_device:
                sd.query_devices(input_device, kind="input")
        except Exception:
            raise CLIError(
                "Unable to access the microphone. \n"
                "Please ensure a microphone is connected and recognized by your system. "
                "To see available input devices, run: lk-agents console --list-devices"
            ) from None

        try:
            if output_device:
                sd.query_devices(output_device, kind="output")
        except Exception:
            raise CLIError(
                "Unable to access the speaker. \n"
                "Please ensure a speaker is connected and recognized by your system. "
                "To see available output devices, run: lk-agents console --list-devices"
            ) from None

    def _sd_input_callback(self, indata: np.ndarray, frame_count: int, time: Any, *_: Any) -> None:
        self._input_delay = time.currentTime - time.inputBufferAdcTime
        total_delay = self._output_delay + self._input_delay

        try:
            self._apm.set_stream_delay_ms(int(total_delay * 1000))
        except RuntimeError:
            pass  # setting stream delay in console mode fails often, so we silently continue

        sr = 24000
        x = indata[:, 0].astype(np.float32) / 32768.0
        n = x.size
        x *= np.hanning(n).astype(np.float32)

        X = np.fft.rfft(x, n=n)
        mag = np.abs(X).astype(np.float32) * (2.0 / n)
        mag[0] *= 0.5
        mag[-1] *= 1.0 - 0.5 * float(n % 2 == 0)

        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        nb = len(self._input_levels)
        edges = np.geomspace(20.0, (sr * 0.5) * 0.96, nb + 1).astype(np.float32)
        b = np.clip(np.digitize(freqs, edges) - 1, 0, nb - 1)

        p = (mag * mag).astype(np.float32)
        sump = np.bincount(b, weights=p, minlength=nb)
        cnts = np.maximum(np.bincount(b, minlength=nb), 1)
        pmean = sump / cnts

        db = 10.0 * np.log10(pmean + 1e-12)
        floor_db, hot_db = -70.0, -20
        lev = np.clip(((db - floor_db) / (hot_db - floor_db)).astype(np.float32), 0.0, 1.0)
        lev = np.maximum(lev**0.75 - 0.02, 0.0)
        peak = float(lev.max())
        lev *= np.clip(0.95 / (peak + 1e-6), 0.0, 3.0)
        lev = np.clip(lev, 0.0, 1.0)

        decay = float(np.exp(-(n / sr) / 0.1))
        with self._input_lock:
            prev = self._input_levels.astype(np.float32)
            self._input_levels = np.maximum(lev, prev * decay)

        if not self._io_acquired:
            return

        FRAME_SAMPLES = 240  # 10ms at 24000 Hz
        num_frames = frame_count // FRAME_SAMPLES

        for i in range(num_frames):
            start = i * FRAME_SAMPLES
            end = start + FRAME_SAMPLES
            capture_chunk = indata[start:end]

            frame = rtc.AudioFrame(
                data=capture_chunk.tobytes(),
                samples_per_channel=FRAME_SAMPLES,
                sample_rate=24000,
                num_channels=1,
            )
            self._apm.process_stream(frame)

            in_data_aec = np.frombuffer(frame.data, dtype=np.int16)
            rms = np.sqrt(np.mean(in_data_aec.astype(np.float32) ** 2))
            max_int16 = np.iinfo(np.int16).max
            self._micro_db = 20.0 * np.log10(rms / max_int16 + 1e-6)

            self._io_loop.call_soon_threadsafe(self._io_audio_input.push_frame, frame)

    def _sd_output_callback(self, outdata: np.ndarray, frames: int, time: Any, *_: Any) -> None:
        if not self.io_acquired:
            outdata[:] = 0
            return

        self._output_delay = time.outputBufferDacTime - time.currentTime

        FRAME_SAMPLES = 240
        with self._io_audio_output.audio_lock:
            bytes_needed = frames * 2
            if len(self._io_audio_output.audio_buffer) < bytes_needed:
                available_bytes = len(self._io_audio_output.audio_buffer)
                outdata[: available_bytes // 2, 0] = np.frombuffer(
                    self._io_audio_output.audio_buffer, dtype=np.int16, count=available_bytes // 2
                )
                outdata[available_bytes // 2 :, 0] = 0
                del self._io_audio_output.audio_buffer[:available_bytes]  # TODO: optimize
            else:
                chunk = self._io_audio_output.audio_buffer[:bytes_needed]
                outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16, count=frames)
                del self._io_audio_output.audio_buffer[:bytes_needed]

        num_chunks = frames // FRAME_SAMPLES
        for i in range(num_chunks):
            start = i * FRAME_SAMPLES
            end = start + FRAME_SAMPLES
            render_chunk = outdata[start:end, 0]
            render_frame_for_aec = rtc.AudioFrame(
                data=render_chunk.tobytes(),
                samples_per_channel=FRAME_SAMPLES,
                sample_rate=24000,
                num_channels=1,
            )
            self._apm.process_reverse_stream(render_frame_for_aec)


class FrequencyVisualizer:
    def __init__(self, agents_console: AgentsConsole, *, label: str = "Unlabeled microphone"):
        self.label = label
        self.height_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        self.c = agents_console

    def update(self) -> None:
        with self.c._input_lock:
            lv = list(self.c._input_levels)
            self._levels_idx = [max(0, min(7, int(round(v * 7)))) for v in lv]

    def __rich__(self) -> RenderableType:
        label = f"   {self.label}  "
        table = Table.grid(
            Column(width=len(label), no_wrap=True),
            Column(no_wrap=True, overflow="fold"),
            Column(),
            padding=(0, 0, 0, 0),
            collapse_padding=True,
            pad_edge=False,
        )

        style = self.c.console.get_style("label")
        label_seg = Text(label, style=style)

        bar = "".join(f" {self.height_chars[i]}" for i in self._levels_idx)
        table.add_row(
            Group(label_seg),
            Group(bar),
            Text.from_markup("  [bold]<Ctrl+T>[/bold] Text Mode", style="dim"),
        )
        return table


class JsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime.date, datetime.datetime, datetime.time)):
            return o.isoformat()
        elif istraceback(o):
            return "".join(traceback.format_tb(o)).strip()
        elif type(o) is Exception or isinstance(o, Exception) or type(o) is type:
            return str(o)

        # extra values are formatted as str() if the encoder raises TypeError
        try:
            return super().default(o)
        except TypeError:
            try:
                return str(o)
            except Exception:
                return None


class JsonFormatter(logging.Formatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record and serializes to json"""
        message_dict: dict[str, Any] = {}
        message_dict["level"] = record.levelname
        message_dict["name"] = record.name
        message_dict["message"] = record.getMessage()

        if record.exc_info and not message_dict.get("exc_info"):
            message_dict["exc_info"] = self.formatException(record.exc_info)
        if not message_dict.get("exc_info") and record.exc_text:
            message_dict["exc_info"] = record.exc_text
        if record.stack_info and not message_dict.get("stack_info"):
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        log_record: dict[str, Any] = OrderedDict()
        log_record.update(message_dict)
        _merge_record_extra(record, log_record)

        log_record["timestamp"] = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        )

        return json.dumps(log_record, cls=JsonEncoder, ensure_ascii=False)


# skip default LogRecord attributes
# http://docs.python.org/library/logging.html#logrecord-attributes
_RESERVED_ATTRS: tuple[str, ...] = (
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
)


def _merge_record_extra(record: logging.LogRecord, target: dict[Any, Any]) -> None:
    for k, v in record.__dict__.items():
        if k not in _RESERVED_ATTRS and not (hasattr(k, "startswith") and k.startswith("_")):
            target[k] = v


class RichLoggingHandler(logging.Handler):
    def __init__(self, agents_console: AgentsConsole):
        super().__init__()
        self.c = agents_console

        # used to avoid rendering two same time
        self._last_time: Text | None = None

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        # traceback = None
        # if record.exc_info and record.exc_info != (None, None, None):
        #     exc_type, exc_value, exc_traceback = record.exc_info
        #     assert exc_type is not None
        #     assert exc_value is not None
        #     traceback = rich_traceback.Traceback.from_exception(
        #         exc_type,
        #         exc_value,
        #         exc_traceback,
        #         width=None,
        #         code_width=88,
        #         extra_lines=3,
        #         theme=None,
        #         word_wrap=True,
        #         show_locals=False,
        #         locals_max_length=10,
        #         locals_max_string=80,
        #         suppress=(),
        #         max_frames=100,
        #     )
        #     message = record.getMessage()
        #     if self.formatter:
        #         record.message = record.getMessage()
        #         formatter = self.formatter
        #         if hasattr(formatter, "usesTime") and formatter.usesTime():
        #             record.asctime = formatter.formatTime(record, formatter.datefmt)
        #         message = formatter.formatMessage(record)

        output = Table.grid(padding=(0, 1))
        output.add_column(style="log.time")
        output.add_column(style="log.level", width=6)
        output.add_column(style="log.name", width=16)
        output.add_column(ratio=1, style="log.message", overflow="fold")
        output.add_column(style="log.extra")
        row: list[RenderableType] = []

        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.datetime.fromtimestamp(record.created)

        log_time = log_time or self.c.console.get_datetime()

        log_time_display = (
            Text(log_time.strftime(time_format))
            if time_format
            else Text(log_time.strftime("%H:%M:%S.%f")[:-3])
        )

        if log_time_display == self._last_time:
            row.append(Text(" " * len(log_time_display)))
        else:
            row.append(log_time_display)
            self._last_time = log_time_display

        row.append(
            Text.styled(record.levelname.ljust(8), f"logging.level.{record.levelname.lower()}")
        )
        row.append(Text(record.name))

        text_msg = Text(message)
        # row.append(Renderables([text_msg] if not traceback else [text_msg, traceback]))
        row.append(text_msg)

        extra: dict[Any, Any] = {}
        _merge_record_extra(record, extra)
        row.append(json.dumps(extra, cls=JsonEncoder, ensure_ascii=False) if extra else " ")

        output.add_row(*row)
        output = self.c._render_tag(output, tag_width=2)  # type: ignore

        try:
            self.c.console.print(output)
        except Exception:
            self.handleError(record)


# noisy loggers are set to warn by default
NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "openai",
    "watchfiles",
    "anthropic",
    "websockets.client",
    "aiohttp.access",
    "livekit",
    "botocore",
    "aiobotocore",
    "urllib3.connectionpool",
    "mcp.client",
]


def _silence_noisy_loggers() -> None:
    for noisy_logger in NOISY_LOGGERS:
        logger = logging.getLogger(noisy_logger)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.WARN)


def _configure_logger(c: AgentsConsole | None, log_level: int | str) -> None:
    logging.addLevelName(TRACE_LOG_LEVEL, "TRACE")

    root = logging.getLogger()
    if c:
        root.addHandler(c._log_handler)
    else:
        handler = logging.StreamHandler(sys.stdout)
        root.addHandler(handler)
        handler.setFormatter(JsonFormatter())

    root.setLevel(log_level)

    _silence_noisy_loggers()

    from ..log import logger

    if logger.level == logging.NOTSET:
        logger.setLevel(log_level)

    from ..plugin import Plugin

    def _configure_plugin_logger(plugin: Plugin) -> None:
        if plugin.logger is not None and plugin.logger.level == logging.NOTSET:
            plugin.logger.setLevel(log_level)

    for plugin in Plugin.registered_plugins:
        _configure_plugin_logger(plugin)

    Plugin.emitter.on("plugin_registered", _configure_plugin_logger)


class TextModeLogFilter(logging.Filter):
    # We don't want to remove the DEBUG logs from the agents codebase since they're useful. But we now have duplicate content when using
    # the text mode, so we use logging.Filter
    _patterns = [
        re.compile(r"\bexecuting tool\b", re.IGNORECASE),
        re.compile(r"\btools execution completed\b", re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "livekit.agents":
            return True

        msg = record.getMessage()
        return not any(rx.search(msg) for rx in self._patterns)


def _print_audio_devices() -> None:
    import sounddevice as sd

    console = Console()
    devices = sd.query_devices()
    default_input, default_output = sd.default.device

    table = Table(show_header=True, show_lines=False, box=None)
    table.add_column("ID", style="#1fd5f9", justify="right")
    table.add_column("Type", style="bold", justify="center")
    table.add_column("Name", style="bold")
    table.add_column("Default", justify="center")

    for idx, dev in enumerate(devices):
        name = dev["name"]
        has_input = dev["max_input_channels"] > 0
        has_output = dev["max_output_channels"] > 0

        if has_input:
            default = Text("yes", style="#23de6b") if idx == default_input else ""
            table.add_row(str(idx), Text("Input", style="#6c7a89"), name, default)

        if has_output:
            default = Text("yes", style="#23de6b") if idx == default_output else ""
            table.add_row(str(idx), Text("Output", style="#6c7a89"), name, default)

    console.print(table)


def prompt(
    message: str | Text, *, console: Console, key_read_cb: Callable[[str], Any] | None = None
) -> str:
    buffer: list[str] = []

    def render_prompt() -> Text:
        return Text.assemble(message, ("".join(buffer), "bold blue"))

    with Live(render_prompt(), console=console, transient=True) as live:
        console.show_cursor(True)
        while True:
            ch = readkey()

            if key_read_cb:
                key_read_cb(ch)

            if ch == key.ENTER:
                break

            if ch == key.BACKSPACE:
                if buffer:
                    buffer.pop()
                    live.update(render_prompt())
                    live.refresh()

                continue

            if len(ch) == 1 and ch.isprintable():
                buffer.append(ch)
                live.update(render_prompt())
                live.refresh()

        live.update(render_prompt())

    return "".join(buffer)


UpdateFn = Callable[[Optional[Union[str, Text]]], None]


@contextmanager
def live_status(
    console: Console,
    text: str | Text,
    *,
    spinner: str = "line",
    spinner_style: str = "bold blue",
    refresh_per_second: int = 12,
    transient: bool = True,
) -> Iterator[UpdateFn]:
    msg: Text = text if isinstance(text, Text) else Text(str(text))
    spin = Spinner(spinner, style=spinner_style)

    def _render() -> Columns:
        return Columns([msg, spin], expand=False, equal=False, padding=(0, 1))

    with Live(
        _render(), console=console, refresh_per_second=refresh_per_second, transient=transient
    ) as live:

        def update(new_text: str | Text | None = None) -> None:
            nonlocal msg
            if new_text is not None:
                msg = new_text if isinstance(new_text, Text) else Text(str(new_text))
                live.update(_render())

        yield update


def _text_mode(c: AgentsConsole) -> None:
    def _key_read(ch: str) -> None:
        if ch == key.CTRL_T:
            raise _ToggleMode()

    while True:
        try:
            text = prompt(
                Text.from_markup("  [bold]User input[/bold]: "),
                console=c.console,
                key_read_cb=_key_read,
            )
        except KeyboardInterrupt:
            break

        if not text.strip():
            c.console.bell()
            continue

        def _generate_with_context(text: str, result_fut: asyncio.Future[list[RunEvent]]) -> None:
            async def _generate(text: str) -> list[RunEvent]:
                sess = await c.io_session.run(user_input=text)  # type: ignore
                return sess.events.copy()

            def _done_callback(task: asyncio.Task[list[RunEvent]]) -> None:
                if exception := task.exception():
                    result_fut.set_exception(exception)
                else:
                    result_fut.set_result(task.result())

            task = asyncio.create_task(_generate(text))
            task.add_done_callback(_done_callback)

        h: asyncio.Future[list[RunEvent]] = asyncio.Future()
        c.io_loop.call_soon_threadsafe(_generate_with_context, text, h, context=c.io_context)

        # h = asyncio.run_coroutine_threadsafe(_generate(text), loop=c.io_loop)
        c.print(text, tag="You")

        with live_status(c.console, Text.from_markup("   [bold]Generating...[/bold]")):
            while not h.done():
                time.sleep(0.1)

        for event in h.result():
            _print_run_event(c, event)


AGENT_PALETTE: list[str] = ["#1FD5F9", "#09C338", "#1F5DF9", "#BA1FF9", "#F9AE1F", "#FA4C39"]


def _agent_style(name: str) -> Style:
    h = hashlib.blake2b(name.encode("utf-8"), digest_size=2).digest()
    idx = int.from_bytes(h, "big") % len(AGENT_PALETTE)
    return Style(color=AGENT_PALETTE[idx], bold=True)


def _truncate_text(text: str, max_lines: int = 2, width: int = 80) -> str:
    wrapped = textwrap.wrap(text, width=width)

    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)

    head_count = max_lines - 2
    head = wrapped[:head_count]
    tail = wrapped[-1:]

    return "\n".join(head + ["..."] + tail)


def _print_run_event(c: AgentsConsole, event: RunEvent) -> None:
    if event.type == "function_call":
        c.print(
            Text.from_markup(
                f"[bold]{event.item.name}[/bold] [dim](arguments: {event.item.arguments})[/dim]"
            ),
            tag="Tool",
            tag_style=Style.parse("black on #6E9DFE"),
        )
    elif event.type == "function_call_output":
        truncated_output = _truncate_text(event.item.output)
        c.print(
            Text.from_markup(f"Tool output: [dim]{event.item.name}\n {truncated_output}[/dim]"),
            tag="Tool",
            tag_style=Style.parse("black on #6E9DFE"),
        )
    elif event.type == "agent_handoff":
        old_agent = event.old_agent
        new_agent = event.new_agent

        old_style = _agent_style(old_agent.__class__.__name__)
        new_style = _agent_style(new_agent.__class__.__name__)
        c.print(
            Text.assemble(
                Text(f"{old_agent.__class__.__name__}", style=old_style),
                Text.from_markup(" [dim]->[/dim] "),
                Text(f"{new_agent.__class__.__name__}", style=new_style),
            ),
            tag="Handoff",
            tag_style=Style.parse("black on #6E9DFE"),
        )

    elif event.type == "message":
        if event.item.text_content:
            c.print(event.item.text_content, tag="Agent", tag_style=Style.parse("black on #B11FF9"))
    else:
        logger.warning(f"unknown RunEvent type {event.type}")


def _audio_mode(c: AgentsConsole, *, input_device: str | None, output_device: str | None) -> None:
    ctrl_t_e = threading.Event()

    def _listen_for_toggle() -> None:
        while not ctrl_t_e.is_set():
            ch = readkey()
            if ch == key.CTRL_T:
                ctrl_t_e.set()
                break

    listener = threading.Thread(target=_listen_for_toggle, daemon=True)
    listener.start()

    c.set_microphone_enabled(True, device=input_device)
    c.set_speaker_enabled(True, device=output_device)

    visualizer = FrequencyVisualizer(c, label=c.input_name or "unknown")
    visualizer.update()

    with Live(visualizer, console=c.console, refresh_per_second=12, transient=True):
        while not ctrl_t_e.is_set():
            visualizer.update()
            time.sleep(0.05)

    c.set_microphone_enabled(False)
    c.set_speaker_enabled(False)

    if ctrl_t_e.is_set():
        raise _ToggleMode()


class _ConsoleWorker:
    def __init__(self, *, server: AgentServer, shutdown_cb: Callable) -> None:
        self._loop = asyncio.new_event_loop()
        self._server = server
        self._shutdown_cb = shutdown_cb
        self._lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker_thread)
        self._thread.start()

    def join(self) -> None:
        self._thread.join()

    def shutdown(self) -> None:
        with self._lock:
            asyncio.run_coroutine_threadsafe(self._server.aclose(), self._loop)

    def _worker_thread(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def _async_main() -> None:
            with self._lock:
                if self._closed:
                    self._shutdown_cb()
                    return

            self._server._job_executor_type = JobExecutorType.THREAD  # TODO: better setter

            @self._server.once("worker_started")
            def _simulate_job() -> None:
                asyncio.run_coroutine_threadsafe(
                    self._server.simulate_job(
                        "console-room", agent_identity="console", fake_job=True
                    ),
                    self._loop,
                )

            await self._server.run(devmode=True, unregistered=True)
            self._shutdown_cb()

        self._loop.run_until_complete(_async_main())


def _run_console(
    *,
    server: AgentServer,
    input_device: str | None,
    output_device: str | None,
    mode: ConsoleMode,
    record: bool,
) -> None:
    c = AgentsConsole.get_instance()
    c.console_mode = mode
    c.enabled = True
    c.record = record

    _configure_logger(c, logging.DEBUG)
    c.print("Starting console mode 🚀", tag="Agents")

    if c.record:
        c.print(
            f"Session recording will be saved to {c.session_directory}",
            tag="Recording",
            tag_style=Style.parse("black on red"),
        )

    c.print(" ")
    # c.print(
    #     "Searching for package file structure from directories with [blue]__init__.py[/blue] files"
    # )
    try:
        # import_data = get_import_data(path=path)
        # c.print(f"Importing from {import_data.module_data.extra_sys_path}")
        # c.print(" ")

        c._validate_device_or_raise(input_device=input_device, output_device=output_device)

        exit_triggered = False

        def _on_worker_shutdown() -> None:
            try:
                signal.raise_signal(signal.SIGTERM)
            except Exception:
                try:
                    signal.raise_signal(signal.SIGINT)
                except Exception:
                    pass

        def _handle_exit(sig: int, frame: FrameType | None) -> None:
            nonlocal exit_triggered
            if not exit_triggered:
                exit_triggered = True
                raise _ExitCli()

            console_worker.shutdown()

        for sig in HANDLED_SIGNALS:
            signal.signal(sig, _handle_exit)

        console_worker = _ConsoleWorker(server=server, shutdown_cb=_on_worker_shutdown)
        console_worker.start()

        # TODO: wait for a session request the agents console context before showing any of the mode
        try:
            c.wait_for_io_acquisition()

            while True:
                try:
                    if c.console_mode == "text":
                        _text_mode(c)
                    elif c.console_mode == "audio":
                        _audio_mode(c, input_device=input_device, output_device=output_device)

                except _ToggleMode:
                    c.console_mode = "audio" if c.console_mode == "text" else "text"

        except _ExitCli:
            pass
        finally:
            console_worker.shutdown()
            console_worker.join()

    except CLIError as e:
        c.print(" ")
        c.print(f"[error]{e}")
        c.print(" ")
        raise typer.Exit(code=1) from None


def _run_worker(server: AgentServer, args: proto.CliArgs, jupyter: bool = False) -> None:
    c: AgentsConsole | None = None
    if args.devmode:
        c = AgentsConsole.get_instance()  # colored logs

    exit_triggered = False

    if not jupyter:

        def _handle_exit(sig: int, frame: FrameType | None) -> None:
            nonlocal exit_triggered
            if not exit_triggered:
                exit_triggered = True
                raise _ExitCli()

        for sig in HANDLED_SIGNALS:
            signal.signal(sig, _handle_exit)

    _configure_logger(c, args.log_level)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.slow_callback_duration = 0.1  # 100ms

    async def _worker_run(worker: AgentServer) -> None:
        try:
            await server.run(devmode=args.devmode, unregistered=jupyter)
        except Exception:
            logger.exception("worker failed")

    watch_client = None
    if args.reload:
        from .watcher import WatchClient

        watch_client = WatchClient(server, args, loop=loop)
        watch_client.start()

    try:
        main_task = loop.create_task(_worker_run(server), name="worker_main_task_cli")
        try:
            loop.run_until_complete(main_task)
        except _ExitCli:
            pass

        try:
            exit_triggered = False  # allow a new _ExitCLI raise
            if not args.devmode:
                loop.run_until_complete(server.drain())

            loop.run_until_complete(server.aclose())

            if watch_client:
                loop.run_until_complete(watch_client.aclose())
        except _ExitCli:
            if not jupyter:
                logger.warning("exiting forcefully")
                import os

                os._exit(1)  # TODO(theomonnom): add aclose(force=True) in worker
    finally:
        if jupyter:
            loop.close()  # close can only be called from the main thread
            return  # noqa: B012

        with contextlib.suppress(_ExitCli):
            try:
                tasks = asyncio.all_tasks(loop)
                for task in tasks:
                    task.cancel()

                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
                loop.close()


class LogLevel(str, enum.Enum):
    trace = "TRACE"
    debug = "DEBUG"
    info = "INFO"
    warn = "WARN"
    error = "ERROR"
    critical = "CRITICAL"


def _build_cli(server: AgentServer) -> typer.Typer:
    app = typer.Typer(rich_markup_mode="rich")

    @app.command()
    def console(
        *,
        input_device: Annotated[
            Optional[str],  # noqa: UP007, required for python 3.9
            typer.Option(
                help="Numeric input device ID or input device name substring(s)",
            ),
        ] = None,
        output_device: Annotated[
            Optional[str],  # noqa: UP007
            typer.Option(
                help="Numeric output device ID or output device name substring(s)",
            ),
        ] = None,
        list_devices: Annotated[
            bool,
            typer.Option(
                help="List all available input and output audio devices.",
            ),
        ] = False,
        text: Annotated[
            bool,
            typer.Option(help="Whether to start the console in text mode"),
        ] = False,
        record: Annotated[bool, typer.Option(help="Whether to record the AgentSession")] = False,
    ) -> None:
        """
        Run a [bold]LiveKit Agents[/bold] in [yellow]console[/yellow] mode.
        """
        if list_devices:
            _print_audio_devices()
            raise typer.Exit()

        if input_device and input_device.isdigit():
            input_device = int(input_device)  # type: ignore

        if output_device and output_device.isdigit():
            output_device = int(output_device)  # type: ignore

        _run_console(
            server=server,
            input_device=input_device,
            output_device=output_device,
            mode="text" if text else "audio",
            record=record,
        )

    @app.command()
    def start(
        *,
        log_level: Annotated[
            LogLevel,
            typer.Option(help="Set the log level", case_sensitive=False),
        ] = LogLevel.info,
        url: Annotated[
            Optional[str],  # noqa: UP007
            typer.Option(
                help="The WebSocket URL of your LiveKit server or Cloud project.",
                envvar="LIVEKIT_URL",
            ),
        ] = None,
        api_key: Annotated[
            Optional[str],  # noqa: UP007
            typer.Option(
                help="API key for authenticating with your LiveKit server or Cloud project.",
                envvar="LIVEKIT_API_KEY",
            ),
        ] = None,
        api_secret: Annotated[
            Optional[str],  # noqa: UP007
            typer.Option(
                help="API secret for authenticating with your LiveKit server or Cloud project.",
                envvar="LIVEKIT_API_SECRET",
            ),
        ] = None,
        drain_timeout: Annotated[
            Optional[int],  # noqa: UP007
            typer.Option(
                help="Time in seconds to wait for jobs to finish before shutting down.",
            ),
        ] = None,
    ) -> None:
        if drain_timeout is not None:
            server.update_options(drain_timeout=drain_timeout)

        _run_worker(
            server=server,
            args=proto.CliArgs(
                log_level=log_level.value, url=url, api_key=api_key, api_secret=api_secret
            ),
        )

    @app.command()
    def dev(
        *,
        log_level: Annotated[
            LogLevel,
            typer.Option(help="Set the log level", case_sensitive=False),
        ] = LogLevel.debug,
        reload: Annotated[
            bool,
            typer.Option(help="Enable auto-reload of the server when (code) files change."),
        ] = True,
        url: Annotated[
            Optional[str],  # noqa: UP007
            typer.Option(
                help="The WebSocket URL of your LiveKit server or Cloud project.",
                envvar="LIVEKIT_URL",
            ),
        ] = None,
        api_key: Annotated[
            Optional[str],  # noqa: UP007
            typer.Option(
                help="API key for authenticating with your LiveKit server or Cloud project.",
                envvar="LIVEKIT_API_KEY",
            ),
        ] = None,
        api_secret: Annotated[
            Optional[str],  # noqa: UP007
            typer.Option(
                help="API secret for authenticating with your LiveKit server or Cloud project.",
                envvar="LIVEKIT_API_SECRET",
            ),
        ] = None,
    ) -> None:
        args = proto.CliArgs(
            log_level=log_level.value,
            url=url,
            api_key=api_key,
            api_secret=api_secret,
            devmode=True,
            reload=reload,
        )

        c = AgentsConsole.get_instance()
        _configure_logger(c, log_level.value)

        term_program = os.environ.get("TERM_PROGRAM")

        if term_program == "iTerm.app" and args.reload:
            c.print("[error]Auto-reload is not supported on the iTerm2 terminal, disabling...")
            args.reload = False

        if not args.reload:
            _run_worker(server=server, args=args)
            return

        from .watcher import WatchServer

        main_file = pathlib.Path(sys.argv[0]).parent

        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)

        watch_server = WatchServer(_run_worker, server, main_file, args, loop=loop)

        def _handle_exit(sig: int, frame: FrameType | None) -> None:
            asyncio.run_coroutine_threadsafe(watch_server.aclose(), loop=loop)

        for sig in HANDLED_SIGNALS:
            signal.signal(sig, _handle_exit)

        async def _run_loop() -> None:
            await watch_server.run()

        try:
            loop = asyncio.get_event_loop()
            asyncio.set_event_loop(loop)

            loop.run_until_complete(_run_loop())
        except _ExitCli:
            raise typer.Exit() from None

    @app.command()
    def download_files() -> None:
        c = AgentsConsole.get_instance()
        c.enabled = True

        _configure_logger(c, logging.DEBUG)

        try:
            # import_data = get_import_data(path=path)
            # c.print(f"Importing from {import_data.module_data.extra_sys_path}")
            # c.print(" ")

            for plugin in Plugin.registered_plugins:
                logger.info(f"Downloading files for {plugin.package}")
                plugin.download_files()
                logger.info(f"Finished downloading files for {plugin.package}")

        except CLIError as e:
            c.print(" ")
            c.print(f"[error]{e}")
            c.print(" ")
            raise typer.Exit(code=1) from None

    return app


def run_app(server: AgentServer | WorkerOptions) -> None:
    if isinstance(server, WorkerOptions):
        server = AgentServer.from_server_options(server)

    _build_cli(server)()
