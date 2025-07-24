from __future__ import annotations

import asyncio
import sys
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import click
import numpy as np

from livekit import rtc

from ..cli import cli
from ..job import get_job_context
from ..log import logger
from ..utils import aio, log_exceptions
from . import io
from .agent_session import AgentSession
from .recorder_io import RecorderIO
from .transcription import TranscriptSynchronizer

if TYPE_CHECKING:
    import sounddevice as sd  # type: ignore

MAX_AUDIO_BAR = 30
INPUT_DB_MIN = -70.0
INPUT_DB_MAX = 0.0
FPS = 16


AEC_RING_BUFFER_SIZE = 24000 * 4


def _esc(*codes: int) -> str:
    return "\033[" + ";".join(str(c) for c in codes) + "m"


def _normalize_db(amplitude_db: float, db_min: float, db_max: float) -> float:
    amplitude_db = max(db_min, min(amplitude_db, db_max))
    return (amplitude_db - db_min) / (db_max - db_min)


class _AudioInput(io.AudioInput):
    def __init__(self, cli: ChatCLI) -> None:
        super().__init__(label="ChatCLI")
        self._cli = cli

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._cli._audio_input_ch.__anext__()


class _TextOutput(io.TextOutput):
    def __init__(self, cli: ChatCLI) -> None:
        super().__init__(label="ChatCLI", next_in_chain=None)
        self._cli = cli
        self._capturing = False
        self._enabled = True

    async def capture_text(self, text: str) -> None:
        if not self._enabled:
            return

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

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not self._enabled:
            self._capturing = False


class _AudioOutput(io.AudioOutput):
    def __init__(self, cli: ChatCLI) -> None:
        super().__init__(label="ChatCLI", next_in_chain=None, sample_rate=24000)
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
            played_duration = min(time.monotonic() - self._capture_start, self._pushed_duration)
            self.on_playback_finished(
                playback_position=played_duration,
                interrupted=played_duration + 1.0 < self._pushed_duration,
            )
            self._pushed_duration = 0.0

    def _dispatch_playback_finished(self) -> None:
        self.on_playback_finished(playback_position=self._pushed_duration, interrupted=False)
        self._flush_complete.set()
        self._pushed_duration = 0.0


class ChatCLI:
    def __init__(
        self,
        agent_session: AgentSession,
        *,
        sync_transcription: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._session = agent_session
        self._done_fut = asyncio.Future[None]()
        self._micro_db = INPUT_DB_MIN

        self._audio_input_ch = aio.Chan[rtc.AudioFrame](loop=self._loop)

        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        self._cli_mode: Literal["text", "audio"] = "audio"

        self._text_input_buf: list[str] = []

        self._text_sink = _TextOutput(self)
        self._audio_sink = _AudioOutput(self)
        self._transcript_syncer: TranscriptSynchronizer | None = None
        if sync_transcription:
            self._transcript_syncer = TranscriptSynchronizer(
                next_in_chain_audio=self._audio_sink,
                next_in_chain_text=self._text_sink,
            )

        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=True,
        )

        self._output_delay = 0.0
        self._input_delay = 0.0

        self._main_atask: asyncio.Task[None] | None = None

        self._input_audio: io.AudioInput = _AudioInput(self)
        self._output_audio: io.AudioOutput = (
            self._transcript_syncer.audio_output if self._transcript_syncer else self._audio_sink
        )

        self._recorder_io: RecorderIO | None = None
        if cli.CLI_ARGUMENTS is not None and cli.CLI_ARGUMENTS.record:
            self._recorder_io = RecorderIO(agent_session=agent_session)
            self._input_audio = self._recorder_io.record_input(self._input_audio)
            self._output_audio = self._recorder_io.record_output(self._output_audio)

    async def start(self) -> None:
        if self._recorder_io:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"console_{timestamp}.ogg"
            await self._recorder_io.start(output_path=filename)

            try:
                job_ctx = get_job_context()
                job_ctx.add_shutdown_callback(self._recorder_io.aclose)
            except RuntimeError:
                pass  # ignore

        if self._transcript_syncer:
            self._update_text_output(enable=True, stdout_enable=False)

        self._update_microphone(enable=True)
        self._update_speaker(enable=True)
        self._main_atask = asyncio.create_task(self._main_task(), name="_main_task")

    @log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        stdin_ch = aio.Chan[str](loop=self._loop)

        if sys.platform == "win32":
            import msvcrt

            async def win_reader():
                while True:
                    ch = await self._loop.run_in_executor(None, msvcrt.getch)

                    if ch == b"\x03":  # Ctrl+C on Windows
                        break

                    try:
                        ch = ch.decode("utf-8")
                    except Exception:
                        pass
                    await stdin_ch.send(ch)

            self._win_read_task = asyncio.create_task(win_reader())
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            def on_input() -> None:
                try:
                    ch = sys.stdin.read(1)
                    stdin_ch.send_nowait(ch)
                except Exception:
                    stdin_ch.close()

            self._loop.add_reader(fd, on_input)

        try:
            input_cli_task = asyncio.create_task(self._input_cli_task(stdin_ch))
            input_cli_task.add_done_callback(lambda _: self._done_fut.set_result(None))
            render_cli_task = asyncio.create_task(self._render_cli_task())

            await self._done_fut
            await aio.cancel_and_wait(render_cli_task)

            self._update_microphone(enable=False)
            self._update_speaker(enable=False)
        finally:
            if sys.platform != "win32":
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                self._loop.remove_reader(fd)

    def _update_microphone(self, *, enable: bool) -> None:
        import sounddevice as sd

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
                blocksize=2400,
            )
            self._input_stream.start()
            self._session.input.audio = self._input_audio
        elif self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
            self._session.input.audio = None

    def _update_speaker(self, *, enable: bool) -> None:
        import sounddevice as sd

        _, output_device = sd.default.device
        if output_device is not None and enable:
            self._output_stream = sd.OutputStream(
                callback=self._sd_output_callback,
                dtype="int16",
                channels=1,
                device=output_device,
                samplerate=24000,
                blocksize=2400,  # 100ms
            )
            self._output_stream.start()
            self._session.output.audio = self._output_audio
        elif self._output_stream is not None:
            self._output_stream.close()
            self._output_stream = None
            self._session.output.audio = None

    def _update_text_output(self, *, enable: bool, stdout_enable: bool) -> None:
        if enable:
            self._session.output.transcription = (
                self._transcript_syncer.text_output if self._transcript_syncer else self._text_sink
            )
            self._text_sink.set_enabled(stdout_enable)
        else:
            self._session.output.transcription = None
            self._text_input_buf = []

    def _sd_output_callback(self, outdata: np.ndarray, frames: int, time, *_) -> None:  # type: ignore
        self._output_delay = time.outputBufferDacTime - time.currentTime

        FRAME_SAMPLES = 240
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

    def _sd_input_callback(self, indata: np.ndarray, frame_count: int, time, *_) -> None:  # type: ignore
        self._input_delay = time.currentTime - time.inputBufferAdcTime
        total_delay = self._output_delay + self._input_delay

        try:
            self._apm.set_stream_delay_ms(int(total_delay * 1000))
        except RuntimeError:
            pass  # setting stream delay in console mode fails often, so we silently continue

        FRAME_SAMPLES = 240  # 10ms at 24000 Hz
        num_frames = frame_count // FRAME_SAMPLES

        for i in range(num_frames):
            start = i * FRAME_SAMPLES
            end = start + FRAME_SAMPLES
            capture_chunk = indata[start:end]

            capture_frame_for_aec = rtc.AudioFrame(
                data=capture_chunk.tobytes(),
                samples_per_channel=FRAME_SAMPLES,
                sample_rate=24000,
                num_channels=1,
            )
            self._apm.process_stream(capture_frame_for_aec)

            in_data_aec = np.frombuffer(capture_frame_for_aec.data, dtype=np.int16)
            rms = np.sqrt(np.mean(in_data_aec.astype(np.float32) ** 2))
            max_int16 = np.iinfo(np.int16).max
            self._micro_db = 20.0 * np.log10(rms / max_int16 + 1e-6)

            self._loop.call_soon_threadsafe(self._audio_input_ch.send_nowait, capture_frame_for_aec)

    @log_exceptions(logger=logger)
    async def _input_cli_task(self, in_ch: aio.Chan[str]) -> None:
        while True:
            char = await in_ch.recv()
            if char is None:
                break

            if char == "\x02":  # Ctrl+B
                if self._cli_mode == "audio":
                    self._cli_mode = "text"
                    self._update_text_output(enable=True, stdout_enable=True)
                    self._update_microphone(enable=False)
                    self._update_speaker(enable=False)
                    click.echo("\nSwitched to Text Input Mode.", nl=False)
                else:
                    self._cli_mode = "audio"
                    self._update_text_output(enable=True, stdout_enable=False)
                    self._update_microphone(enable=True)
                    self._update_speaker(enable=True)
                    self._text_input_buf = []
                    click.echo("\nSwitched to Audio Input Mode.", nl=False)

            if self._cli_mode == "text":  # Read input
                if char in ("\r", "\n"):
                    text = "".join(self._text_input_buf)
                    if text:
                        self._text_input_buf = []
                        self._session.interrupt()
                        self._session.generate_reply(user_input=text)
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

    def _print_audio_mode(self) -> None:
        amplitude_db = _normalize_db(self._micro_db, db_min=INPUT_DB_MIN, db_max=INPUT_DB_MAX)
        nb_bar = round(amplitude_db * MAX_AUDIO_BAR)

        color_code = 31 if amplitude_db > 0.75 else 33 if amplitude_db > 0.5 else 32
        bar = "#" * nb_bar + "-" * (MAX_AUDIO_BAR - nb_bar)
        sys.stdout.write(
            f"\r[Audio] {self._input_device_name[-20:]} [{self._micro_db:6.2f} dBFS] {_esc(color_code)}[{bar}]{_esc(0)}"  # noqa: E501
        )
        sys.stdout.flush()

    def _print_text_mode(self) -> None:
        sys.stdout.write("\r")
        sys.stdout.flush()
        prompt = "Enter your message: "
        sys.stdout.write(f"[Text {prompt}{''.join(self._text_input_buf)}")
        sys.stdout.flush()
