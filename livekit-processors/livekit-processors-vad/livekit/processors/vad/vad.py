import asyncio
from dataclasses import dataclass
from livekit import rtc
from livekit import agents
import torch
import numpy as np

VAD_SAMPLE_RATE = 16000


class VAD:
    """Class for Voice Activity Detection (VAD)
    """

    @dataclass
    class Event:
        type: str
        frames: [rtc.AudioFrame]

    def __init__(self, *, silence_threshold_ms: int):
        self._silence_threshold_ms = silence_threshold_ms
        self._window_buffer = np.zeros(512, dtype=np.float32)
        self._window_buffer_scratch = np.zeros(512, dtype=np.float32)
        (self._model, _) = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self._copy_int16 = np.zeros(1, dtype=np.int16)
        self._copy_float32 = np.zeros(1, dtype=np.float32)
        self._silent_frame_count_since_voice = 0
        self._voice_frames = []
        self._frame_queue = []
        self._talking_state = False

    async def push_frame(self, frame: rtc.AudioFrame):
        self._frame_queue.append(frame)

        # run inference every 30ms
        if len(self._frame_queue) * 10 < 30:
            return None

        frame_queue_copy = self._frame_queue.copy()
        self._frame_queue = []

        talking_detected = await asyncio.get_event_loop().run_in_executor(None, self._process_frames, frame_queue_copy)
        if self._talking_state:
            if talking_detected:
                self._silent_frame_count_since_voice = 0
                self._voice_frames.extend(frame_queue_copy)
            else:
                self._silent_frame_count_since_voice += len(frame_queue_copy)
                if self._silent_frame_count_since_voice * 10 > self._silence_threshold_ms:
                    self._talking_state = False
                    voice_frames_copy = self._voice_frames.copy()
                    self._voice_frames = []
                    return VAD.Event(type="voice_finished", frames=voice_frames_copy)
        else:
            if talking_detected:
                self._talking_state = True
                self._voice_frames.extend(frame_queue_copy)
                return VAD.Event(type="voice_started", frames=self._voice_frames)
            else:
                # if silent and not in the talking state, do nothing
                pass

        return None

    def _process_frames(self, frames: [rtc.AudioFrame]) -> bool:
        resampled = [f.remix_and_resample(VAD_SAMPLE_RATE, 1) for f in frames]
        buffer_count = 0
        for f in resampled:
            buffer_count += len(f.data)

        assert buffer_count <= 512, "Buffer count should be less than the input to the VAD model"

        if self._copy_int16.shape[0] < buffer_count:
            self._copy_int16 = np.zeros(buffer_count, dtype=np.int16)
            self._copy_float32 = np.zeros(buffer_count, dtype=np.float32)

        for i, f in enumerate(resampled):
            self._copy_int16[i * len(f.data): (i + 1) *
                             len(f.data)] = np.ctypeslib.as_array(f.data)
            self._copy_float32 = self._copy_int16.astype(
                np.float32, copy=False) / 32768.0

        self._window_buffer_scratch[:-
                                    buffer_count] = self._window_buffer[buffer_count:]
        self._window_buffer[-buffer_count:] = self._copy_float32
        self._window_buffer[:-
                            buffer_count] = self._window_buffer_scratch[:-buffer_count]
        tensor = torch.from_numpy(self._window_buffer)
        speech_prob = self._model(tensor, VAD_SAMPLE_RATE).item()
        return speech_prob > 0.5


class VADProcessor(agents.Processor):
    def __init__(self, silence_threshold_ms=250):
        self.vad = VAD(silence_threshold_ms=silence_threshold_ms)
        super().__init__(process=self.vad.push_frame)
