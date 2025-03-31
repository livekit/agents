import asyncio
import logging
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from pathlib import Path
from typing import Optional, Union

import numpy as np

from livekit import rtc
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
    VideoGenerator,
)

sys.path.insert(0, str(Path(__file__).parent))
from wave_viz import WaveformVisualizer

logger = logging.getLogger("avatar-example")


class AudioWaveGenerator(VideoGenerator):
    def __init__(self, options: AvatarOptions):
        self._options = options
        self._audio_queue: asyncio.Queue[Union[rtc.AudioFrame, AudioSegmentEnd]] = asyncio.Queue()

        self._audio_resampler: Optional[rtc.AudioResampler] = None

        # NOTE: Audio frames and video frames have different frequencies,
        #       so we accumulate audio samples in a buffer before
        #       generating corresponding video frames
        self._audio_buffer = np.zeros((0, self._options.audio_channels), dtype=np.int16)
        self._audio_samples_per_frame = int(
            self._options.audio_sample_rate / self._options.video_fps
        )
        self._av_sync: Optional[rtc.AVSynchronizer] = None

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Process and queue audio frames, handling resampling if needed.

        Args:
            frame: Either an AudioFrame to process or AudioFlushSentinel to flush
        """
        if isinstance(frame, AudioSegmentEnd):
            if self._audio_resampler is not None:
                # flush the resampler and queue any remaining frames
                for resampled_frame in self._audio_resampler.flush():
                    await self._audio_queue.put(resampled_frame)
            await self._audio_queue.put(frame)
            return

        # initialize resampler if needed
        needs_resampling = (
            frame.sample_rate != self._options.audio_sample_rate
            or frame.num_channels != self._options.audio_channels
        )

        if needs_resampling and self._audio_resampler is None:
            self._audio_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=self._options.audio_sample_rate,
                num_channels=self._options.audio_channels,
            )

        if self._audio_resampler is not None:
            for resampled_frame in self._audio_resampler.push(frame):
                await self._audio_queue.put(resampled_frame)
        else:
            # no resampling needed, queue directly
            await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._reset_audio_buffer()

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream_impl()

    async def _stream_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        # initialize background canvas
        background = np.zeros(
            (self._options.video_height, self._options.video_width, 4),
            dtype=np.uint8,
        )
        background.fill(255)

        wave_visualizer = WaveformVisualizer(sample_rate=self._options.audio_sample_rate)

        def _generate_idle_frame() -> rtc.VideoFrame:
            idle_frame = background.copy()
            fps = self._av_sync.actual_fps if self._av_sync else None
            wave_visualizer.draw(
                idle_frame,
                audio_samples=np.zeros((1, self._options.audio_channels)),
                fps=fps,
            )
            return self._np_to_video_frame(idle_frame)

        def _generate_active_frames(
            audio_frame: rtc.AudioFrame | AudioSegmentEnd,
        ) -> Generator[tuple[rtc.VideoFrame, rtc.AudioFrame], None, None]:
            samples_per_frame = self._audio_samples_per_frame

            if isinstance(audio_frame, rtc.AudioFrame):
                audio_samples = np.frombuffer(audio_frame.data, dtype=np.int16).reshape(
                    -1, audio_frame.num_channels
                )  # (n_samples, n_channels)
            else:
                # fill the buffer with zeros if the buffer is not multiple of samples_per_frame
                n_fill_samples = (
                    (samples_per_frame - len(self._audio_buffer) % samples_per_frame)
                    if len(self._audio_buffer) > 0
                    else 0
                )
                audio_samples = np.zeros(
                    [n_fill_samples, self._audio_buffer.shape[1]],
                    dtype=self._audio_buffer.dtype,
                )
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_samples], axis=0)

            # generate video frames with audio in buffer
            while len(self._audio_buffer) >= samples_per_frame:
                sub_samples = self._audio_buffer[:samples_per_frame]
                self._audio_buffer = self._audio_buffer[samples_per_frame:]

                canvas = background.copy()
                fps = self._av_sync.actual_fps if self._av_sync else None
                wave_visualizer.draw(canvas, sub_samples, fps=fps)
                video_frame = self._np_to_video_frame(canvas)
                sub_audio_frame = rtc.AudioFrame(
                    data=sub_samples.tobytes(),
                    sample_rate=self._options.audio_sample_rate,
                    num_channels=sub_samples.shape[1],
                    samples_per_channel=sub_samples.shape[0],
                )
                yield video_frame, sub_audio_frame

        while True:
            try:
                # timeout has to be shorter than the frame interval to avoid starvation
                frame = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5 / self._options.video_fps
                )
            except asyncio.TimeoutError:
                # generate frame without audio (e.g. silence state)
                if self._av_sync and self._av_sync._video_queue.qsize() > 1:
                    # skip if there are already video frames in the queue
                    continue
                video_frame = _generate_idle_frame()
                yield video_frame
                await asyncio.sleep(0)
                continue

            for video_frame, audio_frame in _generate_active_frames(frame):
                yield video_frame
                yield audio_frame

            if isinstance(frame, AudioSegmentEnd):
                yield frame
                self._reset_audio_buffer()

    def set_av_sync(self, av_sync: rtc.AVSynchronizer | None) -> None:
        self._av_sync = av_sync

    def _reset_audio_buffer(self) -> None:
        self._audio_buffer = np.zeros((0, self._options.audio_channels), dtype=np.int16)

    def _np_to_video_frame(self, image: np.ndarray) -> rtc.VideoFrame:
        return rtc.VideoFrame(
            width=image.shape[1],
            height=image.shape[0],
            type=rtc.VideoBufferType.RGBA,
            data=image.tobytes(),
        )


async def main(room: rtc.Room):
    """Main application logic for the avatar worker"""
    runner: AvatarRunner | None = None
    stop_event = asyncio.Event()

    try:
        # Initialize and start worker
        avatar_options = AvatarOptions(
            video_width=1280,
            video_height=720,
            video_fps=30,
            audio_sample_rate=24000,
            audio_channels=1,
        )
        video_gen = AudioWaveGenerator(avatar_options)
        runner = AvatarRunner(
            room,
            audio_recv=DataStreamAudioReceiver(room),
            video_gen=video_gen,
            options=avatar_options,
        )
        video_gen.set_av_sync(runner.av_sync)
        await runner.start()

        # Set up disconnect handler
        async def handle_disconnect(participant: rtc.RemoteParticipant):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                logging.info("Agent %s disconnected, stopping worker...", participant.identity)
                stop_event.set()

        room.on(
            "participant_disconnected",
            lambda p: asyncio.create_task(handle_disconnect(p)),
        )
        room.on("disconnected", lambda _: stop_event.set())

        # Wait until stopped
        await stop_event.wait()

    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise
    finally:
        if runner:
            await runner.aclose()


async def run_service(url: str, token: str):
    """Run the avatar worker service"""
    room = rtc.Room()
    try:
        # Connect to LiveKit room
        logging.info("Connecting to %s", url)
        await room.connect(url, token)
        logging.info("Connected to room %s", room.name)

        # Run main application logic
        await main(room)
    except rtc.ConnectError as e:
        logging.error("Failed to connect to room: %s", e)
        raise
    finally:
        await room.disconnect()


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    def parse_args():
        """Parse command line arguments"""
        parser = ArgumentParser()
        parser.add_argument("--url", required=True, help="LiveKit server URL")
        parser.add_argument("--token", required=True, help="Token for joining room")
        parser.add_argument("--room", help="Room name")
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Log level",
        )
        return parser.parse_args()

    def setup_logging(room: Optional[str], level: str):
        """Set up logging configuration"""
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        if room:
            log_format = f"[{room}] {log_format}"

        logging.basicConfig(level=getattr(logging, level.upper()), format=log_format)

    args = parse_args()
    setup_logging(args.room, args.log_level)
    try:
        asyncio.run(run_service(args.url, args.token))
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)
    finally:
        logging.info("Shutting down...")
