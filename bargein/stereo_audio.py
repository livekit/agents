from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import onnxruntime as ort

from livekit import rtc
from livekit.agents import NOT_GIVEN, io, utils
from livekit.agents.job import JobContext
from livekit.agents.log import logger
from livekit.agents.types import NotGivenOr
from livekit.agents.utils.misc import is_given
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.events import AgentStateChangedEvent, UserStateChangedEvent

WRITE_INTERVAL = 0.1


def load_onnx_model(model_path):
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    return sess


class BargeInDetectorONNX:
    def __init__(self, model_path, enable_clipping=True, clipping_threshold=1e-3):
        self.sess = load_onnx_model(model_path)
        self.enable_clipping = enable_clipping
        self.clipping_threshold = clipping_threshold

    def predict_prob(self, waveform, with_timestamps=False):
        # wavform is (num_channels, num_samples) at 16_000 Hz
        # the first channel is assistent speech, the second is user speech
        waveform = np.array(waveform, dtype=np.float32)
        probs = self.sess.run(None, {"waveform": waveform})[0]

        if with_timestamps:
            timestamps = self.get_frame_timestamps(probs.shape[0])
            return (probs, timestamps)
        return probs

    def get_frame_timestamps(self, num_frames):
        # gets the END time of each frame
        hop_sz, win_sz, sr = 400, 400, 16_000
        i = np.arange(num_frames)
        return (i * hop_sz + win_sz) / sr

    def predict(self, wavform, threshold=0.75, min_frames=2):
        # if prob > threshold for at least `min_frames` consecutive frames, call it a barge-in
        # - note: each frame is 25ms
        if self.enable_clipping:
            wavform[1, np.abs(wavform[1]) < self.clipping_threshold] = 0.0
        if wavform.shape[1] == 0:
            return False
        probs = self.predict_prob(wavform) > threshold
        running_true_counts = np.convolve(probs.astype(int), np.ones(min_frames), mode="valid")
        return np.any(running_true_counts >= min_frames)


class BargeInDetector:
    def __init__(
        self,
        *,
        model_path: str = "/Users/chenghao/Downloads/bd_best.onnx",
        enable_clipping: bool = True,
        clipping_threshold: float = 1e-3,
        sample_rate: int = 16000,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._in_record: SyncedAudioInput | None = None
        self._out_record: SyncedAudioOutput | None = None

        self._in_q: queue.Queue[list[rtc.AudioFrame] | None] = queue.Queue()
        self._out_q: queue.Queue[list[rtc.AudioFrame] | None] = queue.Queue()
        self._session = None
        self._sample_rate = sample_rate
        self._started = False
        self._loop = loop or asyncio.get_event_loop()
        self._lock = asyncio.Lock()
        self._close_fut: asyncio.Future[None] = self._loop.create_future()

        self._model = BargeInDetectorONNX(
            model_path, enable_clipping=enable_clipping, clipping_threshold=clipping_threshold
        )
        # State variables
        self._user_speaking = False
        self._agent_speaking = False
        self._last_user_speech_started_at: NotGivenOr[float] = NOT_GIVEN
        self._last_agent_speech_ended_at: NotGivenOr[float] = NOT_GIVEN
        self._conditions_met = asyncio.Event()
        self._prev_interrupt_by_audio_activity = None
        self._barged_in = NOT_GIVEN
        self._allowed_interruptions = NOT_GIVEN
        self._capacity = self._sample_rate * 8
        self._stereo_buf = np.zeros((2, self._capacity), dtype=np.float32)  # 8s, 2ch
        self._silence_samples = self._capacity
        self._entered = False

        # TODO: this is only used for debugging, remove it later
        self._history_buffer = np.zeros((2, 0), dtype=np.float32)
        self._agent_speech_started_at: NotGivenOr[float] = NOT_GIVEN

        self._main_atask = None
        for folder in ["recordings/barge_in", "recordings/not_barge_in"]:
            if not Path(folder).exists():
                Path(folder).mkdir(parents=True, exist_ok=True)

    def eavesdrop(self, ctx: JobContext, session: AgentSession) -> None:
        if self._main_atask is not None:
            self._main_atask.cancel()
            self._main_atask = None

        self._session = session
        self._register_custom_hooks()
        self._main_atask = asyncio.create_task(self.main_task(ctx))

    async def main_task(self, ctx: JobContext):
        while self._session.input.audio is None or self._session.output.audio is None:
            await asyncio.sleep(0.05)

        try:
            self._session.input.audio = self.record_input(self._session.input.audio)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"RecorderIO record_input failed: {e}")
        try:
            self._session.output.audio = self.record_output(self._session.output.audio)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"RecorderIO record_output failed: {e}")

        try:
            await self.start()
            try:
                ctx.add_shutdown_callback(self.aclose)
            except RuntimeError:
                pass
        except Exception as e:
            logger.error(f"Failed to start RecorderIO: {e}")

    @utils.log_exceptions(logger=logger)
    def disable_interruption(self) -> None:
        logger.info("[BARGEIN] Temporarily disabling interruption by audio activity")

        def noop(*args, **kwargs):
            pass

        self._prev_interrupt_by_audio_activity = (
            self._session._activity._interrupt_by_audio_activity
        )
        self._session._activity._interrupt_by_audio_activity = noop

    @utils.log_exceptions(logger=logger)
    def enable_interruptiton(self) -> None:
        if self._prev_interrupt_by_audio_activity is not None:
            logger.info("[BARGEIN] Enabling interruption by audio activity")
            self._session._activity._interrupt_by_audio_activity = (
                self._prev_interrupt_by_audio_activity
            )
            self._prev_interrupt_by_audio_activity = None

    @utils.log_exceptions(logger=logger)
    def barge_in(self) -> None:
        logger.info("[BARGEIN] Barging in")
        self.enable_interruptiton()
        self._session._agent.skip_until(
            self._last_user_speech_started_at, self._in_record._started_at
        )
        if is_given(self._allowed_interruptions) and self._allowed_interruptions:
            self._session._activity._interrupt_by_audio_activity()

        self.on_exit()

    def _register_custom_hooks(self) -> None:
        self._session.on("user_state_changed", self._on_user_state_changed)
        self._session.on("agent_state_changed", self._on_agent_state_changed)

    @utils.log_exceptions(logger=logger)
    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        logger.info(f"[BARGEIN] agent state changed: {ev.new_state}")
        try:
            self._agent_speaking = ev.new_state == "speaking"
            if ev.new_state == "speaking":
                logger.info(f"[BARGEIN] agent speaking started at {ev.created_at}")
                self._agent_speech_started_at = ev.created_at
                self._out_record.reset(self._agent_speech_started_at)
                self.on_enter()
            elif ev.old_state == "speaking":
                self._agent_speech_started_at = NOT_GIVEN
                self._last_agent_speech_ended_at = ev.created_at
                self.on_exit()
        except Exception as e:
            logger.error(f"Failed to update agent state: {e}")

    @utils.log_exceptions(logger=logger)
    def _on_user_state_changed(self, ev: UserStateChangedEvent) -> None:
        logger.info(f"[BARGEIN] user state changed: {ev.new_state}")
        self._user_speaking = ev.new_state == "speaking"
        if self._user_speaking and self._agent_speaking:
            self._last_user_speech_started_at = ev.created_at
            self.resume_inference()
            return

        self.pause_inference()

    def resume_inference(self) -> None:
        if self._agent_speaking:
            logger.info("[BARGEIN] Resuming inference")
        self._conditions_met.set()

    def pause_inference(self) -> None:
        if self._agent_speaking:
            logger.info("[BARGEIN] Pausing inference")
        self._conditions_met.clear()
        self._barged_in = NOT_GIVEN

    def on_enter(self) -> None:
        logger.info("[BARGEIN] Entering barge in detection mode")
        logger.info(
            f"[BARGEIN] Allowed interruptions: {self._session.current_speech.allow_interruptions}"
        )

        self.disable_interruption()
        self._barged_in = NOT_GIVEN
        self._allowed_interruptions = self._session.current_speech.allow_interruptions
        self._session._agent.should_hold_transcript = True
        self._session._agent.last_inference_time = NOT_GIVEN
        self._session._agent._stream_started_at = NOT_GIVEN
        self._history_buffer = np.zeros((2, 0), dtype=np.float32)
        self._stereo_buf = np.zeros((2, self._capacity), dtype=np.float32)
        self._entered = True

    def on_exit(self) -> None:
        if self._entered:
            self.enable_interruptiton()
            # If a barge-in was detected, barge_in should have been called before this
            # Otherwise, skip until the end of the agent's speech
            if not is_given(self._barged_in) or (is_given(self._barged_in) and not self._barged_in):
                self._session._agent.skip_until(
                    self._last_agent_speech_ended_at, self._in_record._started_at
                )

            self._barged_in = NOT_GIVEN
            self._conditions_met.clear()
            self._history_buffer = np.zeros((2, 0), dtype=np.float32)
            self._stereo_buf = np.zeros((2, self._capacity), dtype=np.float32)
            self._silence_samples = self._capacity
            self._entered = False

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return

            if not self._in_record or not self._out_record:
                raise RuntimeError(
                    "RecorderIO not properly initialized: both `record_input()` and `record_output()` "
                    "must be called before starting the recorder."
                )

            self._started = True
            self._close_fut = self._loop.create_future()
            self._forward_atask = asyncio.create_task(self._forward_task())

            thread = threading.Thread(target=self._encode_thread, daemon=True)
            thread.start()

    async def aclose(self) -> None:
        async with self._lock:
            if not self._started:
                return

            self._in_q.put_nowait(None)
            self._out_q.put_nowait(None)
            await asyncio.shield(self._close_fut)
            self._started = False
            if self._main_atask is not None:
                self._main_atask.cancel()
                self._main_atask = None

    def record_input(self, audio_input: io.AudioInput) -> SyncedAudioInput:
        self._in_record = SyncedAudioInput(recording_io=self, source=audio_input)
        return self._in_record

    def record_output(self, audio_output: io.AudioOutput) -> SyncedAudioOutput:
        self._out_record = SyncedAudioOutput(
            recording_io=self,
            audio_output=audio_output,
        )
        return self._out_record

    @property
    def recording(self) -> bool:
        return self._started

    async def _forward_task(self) -> None:
        assert self._in_record is not None
        assert self._out_record is not None

        while True:
            await asyncio.sleep(WRITE_INTERVAL)
            duration = time.time() - self._out_record.started_at - self._out_record._taken
            # user speech is taken aggressively as they come
            input_buf = self._in_record.take_buf()
            # agent speech is taken lazily as it plays in real-time
            output_buf = self._out_record.take_buf(duration)
            # logger.info(f"[BARGEIN] input_buf: {len(input_buf)} output_buf: {len(output_buf)}")
            self._in_q.put_nowait(input_buf)
            self._out_q.put_nowait(output_buf)

    @utils.log_exceptions(logger=logger)
    def _encode_thread(self) -> None:
        INV_INT16 = 1.0 / 32768.0

        in_resampler: rtc.AudioResampler | None = None
        out_resampler: rtc.AudioResampler | None = None
        input_frames_history: list[rtc.AudioFrame] = []

        def remix_and_resample(frames: list[rtc.AudioFrame], channel_idx: int) -> int:
            total_output_samples = sum(f.samples_per_channel for f in frames)
            dest = self._stereo_buf[channel_idx]

            if total_output_samples == 0:
                return 0

            if total_output_samples > self._capacity:
                pos = 0
            else:
                pos = self._capacity - total_output_samples
                # shift the buffer to the left to make room
                dest[:pos] = dest[-pos:]

            end = len(dest)
            written = 0
            for f in frames[::-1]:
                if len(f.data) == 0:
                    continue
                count = f.samples_per_channel * f.num_channels
                arr_i16 = np.frombuffer(f.data, dtype=np.int16, count=count).reshape(
                    -1, f.num_channels
                )
                slice_start = max(end - f.samples_per_channel, pos)
                arr_start = max(0, pos - (end - f.samples_per_channel))
                arr_size = len(arr_i16) - arr_start
                if arr_size <= 0:
                    break
                slice_ = dest[slice_start:end]
                np.sum(arr_i16[arr_start:, :], axis=1, dtype=np.float32, out=slice_)
                slice_ *= INV_INT16 / f.num_channels
                end -= arr_size
                written += arr_size

            return written

        while True:
            input_buf = self._in_q.get()
            output_buf = self._out_q.get()

            if input_buf is None or output_buf is None:
                break

            # lazy creation of the resamplers
            if in_resampler is None and len(input_buf):
                input_rate, num_channels = input_buf[0].sample_rate, input_buf[0].num_channels
                in_resampler = rtc.AudioResampler(
                    input_rate=input_rate,
                    output_rate=self._sample_rate,
                    num_channels=num_channels,
                )

            if out_resampler is None and len(output_buf):
                input_rate, num_channels = output_buf[0].sample_rate, output_buf[0].num_channels
                out_resampler = rtc.AudioResampler(
                    input_rate=input_rate,
                    output_rate=self._sample_rate,
                    num_channels=num_channels,
                )

            input_resampled = []
            for frame in input_buf:
                assert in_resampler is not None
                input_resampled.extend(in_resampler.push(frame))

            output_resampled = []
            for frame in output_buf:
                assert out_resampler is not None
                if len(frame.data) <= 0:
                    continue
                for f in out_resampler.push(frame):
                    if len(f.data) > 0:
                        output_resampled.append(f)

            input_frames_history.extend(input_resampled)
            _ = remix_and_resample(input_resampled, 0)
            len_right = remix_and_resample(output_resampled, 1)
            self._silence_samples -= len_right
            self._silence_samples = max(0, self._silence_samples)

            if self._conditions_met.is_set():
                # trim the agent silence
                inp = self._stereo_buf[[1, 0], self._silence_samples :]
                barge_in_detected = self._model.predict(inp)
                self._barged_in = (
                    barge_in_detected
                    if not is_given(self._barged_in)
                    else (self._barged_in or barge_in_detected)
                )
                logger.info(f"[BARGEIN] Barge in detected: {barge_in_detected}")
                if barge_in_detected:
                    logger.info(
                        f"Barge in detected: {self._user_speaking=} and {self._agent_speaking=}"
                    )
                    self._loop.call_soon_threadsafe(self.barge_in)
                    self.save_input_for_debug(inp, prefix="recordings/barge_in")
                    self._in_record.write_to_file("recordings/barge_in/input.wav")
                    self.save_frames_to_file(input_frames_history, "recordings/barge_in/input_frames.pkl")
                else:
                    self.save_input_for_debug(inp, prefix="recordings/not_barge_in")

        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._close_fut.set_result, None)

    def save_input_for_debug(self, data, prefix: str = "input") -> None:
        import soundfile

        file_name = f"{prefix}/{time.strftime('%Y-%m-%d_%H-%M-%S')}.wav"
        soundfile.write(file_name, data.T, self._sample_rate)
        soundfile.write(file_name.replace(".wav", "_user_speech.wav"), data[1], self._sample_rate)
        soundfile.write(file_name.replace(".wav", "_agent_speech.wav"), data[0], self._sample_rate)

    def save_frames_to_file(self, frames: list[rtc.AudioFrame], filename: str) -> None:
        import pickle

        with open(filename, "wb") as f:
            pickle.dump([
                f.to_wav_bytes()
                for f in frames
            ], f)



class SyncedAudioInput(io.AudioInput):
    def __init__(self, *, recording_io: BargeInDetector, source: io.AudioInput) -> None:
        super().__init__(label="SyncedAudioInput", source=source)
        self.__audio_input = source
        self.__recording_io = recording_io
        self.__acc_frames: deque[rtc.AudioFrame] = deque()
        self.__acc_history: list[rtc.AudioFrame] = []
        self._started_at: float | None = None

    def take_buf(self) -> list[rtc.AudioFrame]:
        frames = self.__acc_frames
        self.__acc_frames = []
        return frames

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self.__audio_input.__anext__()
        if self._started_at is None:
            self._started_at = time.time()
            logger.info(f"[BARGEIN] User speech sampling rate: {frame.sample_rate}")

        if self.__recording_io.recording:
            self.__acc_frames.append(frame)
            self.__acc_history.append(frame)

        return frame

    def on_attached(self) -> None:
        self.__audio_input.on_attached()

    def on_detached(self) -> None:
        self.__audio_input.on_detached()

    def write_to_file(self, filename: str) -> None:
        import soundfile

        with open(filename, "wb") as f:
            ff = rtc.combine_audio_frames(self.__acc_history)
            arr_i16 = np.frombuffer(ff.data, dtype=np.int16).reshape(-1, ff.num_channels)
            arr_i16 = arr_i16.astype(np.float32) / 32768.0
            f.write(ff.to_wav_bytes())
            soundfile.write(filename.replace(".wav", "_float32.wav"), arr_i16, ff._sample_rate)




class SyncedAudioOutput(io.AudioOutput):
    def __init__(
        self,
        *,
        recording_io: BargeInDetector,
        audio_output: io.AudioOutput | None = None,
    ) -> None:
        super().__init__(
            label="SyncedAudioOutput",
            next_in_chain=audio_output,
            sample_rate=None,
            capabilities=io.AudioOutputCapabilities(pause=True),  # depends on the next_in_chain
        )
        self.__recording_io = recording_io
        self.__acc_frames: deque[rtc.AudioFrame] = deque()
        self._started_at: float = time.time()
        self._taken = 0.0
        self._sampling_rate = None

    def reset(self, started_at: float | None = None) -> None:
        self._started_at = started_at or time.time()
        self._taken = 0.0
        self.__acc_frames = deque()

    @property
    def started_at(self) -> float:
        return self._started_at

    @started_at.setter
    def started_at(self, value: float) -> None:
        self._started_at = value

    @utils.log_exceptions(logger=logger)
    def take_buf(self, duration: float) -> list[rtc.AudioFrame]:
        """
        Take agent audio frames from the buffer, up to the current playback position.
        """
        # TODO: add speed factor to the duration @chenghao
        # duration = time.time() - self.started_at - self._taken
        buf = []
        acc_dur = 0.0
        while self.__acc_frames:
            frame = self.__acc_frames.popleft()
            if frame.duration + acc_dur > duration:
                duration_needed = duration - acc_dur
                samples_needed = int(duration_needed * frame.sample_rate) * frame.num_channels
                left, right = self._split_frame(frame, samples_needed)
                acc_dur += left.duration
                buf.append(left)
                self._taken += left.duration
                if right is not None and len(right.data) > 0:
                    self.__acc_frames.appendleft(right)
                break

            acc_dur += frame.duration
            buf.append(frame)
            self._taken += frame.duration

        return buf

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )

        if not self.__recording_io.recording:
            return

        self.reset()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self._sampling_rate is None:
            self._sampling_rate = frame.sample_rate
            logger.info(f"[BARGEIN] Agent speech sampling rate: {frame.sample_rate}")

        if self.__recording_io.recording:
            self.__acc_frames.append(frame)

        if self.next_in_chain:
            await self.next_in_chain.capture_frame(frame)

    def flush(self) -> None:
        super().flush()

        if self.next_in_chain:
            self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()

    @staticmethod
    def _split_frame(
        frame: rtc.AudioFrame, position: int
    ) -> tuple[rtc.AudioFrame, rtc.AudioFrame | None]:
        position -= position % frame.num_channels
        if len(frame.data) <= position:
            return frame, None

        return rtc.AudioFrame(
            data=frame.data[:position],
            num_channels=frame.num_channels,
            samples_per_channel=position // frame.num_channels,
            sample_rate=frame.sample_rate,
        ), rtc.AudioFrame(
            data=frame.data[position:],
            num_channels=frame.num_channels,
            samples_per_channel=len(frame.data[position:]) // frame.num_channels,
            sample_rate=frame.sample_rate,
        )
