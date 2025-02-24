import asyncio
import logging
import sys
from pathlib import Path
from typing import AsyncIterator, Optional, Union

from livekit import rtc
from livekit.agents.pipeline.avatar import AvatarRunner, MediaOptions
from livekit.agents.pipeline.avatar import VideoGenerator
from livekit.agents.pipeline.datastream_io import AudioFlushSentinel
from simli import SimliClient, SimliConfig

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger("avatar-example")
logging.getLogger("livekit.rtc").setLevel(logging.WARNING)

class SimliVideoGenerator(VideoGenerator):
    def __init__(self, media_options: MediaOptions):
        self.media_options = media_options
        self._audio_queue: asyncio.Queue[Union[rtc.AudioFrame, AudioFlushSentinel]] = (
            asyncio.Queue()
        )

        self._audio_resampler: Optional[rtc.AudioResampler] = None

        self.audioReceiverTask: asyncio.Task = None
        self.audioBytesQueue: bytearray = bytearray()
        self.audioBufferMutex: asyncio.Lock = asyncio.Lock()

        self.videoReceiverTask: asyncio.Task = None
        self.videoQueue: asyncio.Queue[rtc.VideoFrame] = asyncio.Queue()
        self._av_sync: Optional[rtc.AVSynchronizer] = None
        self.clearBufferTasks: list[asyncio.Task] = []

    def set_av_sync(self, av_sync: rtc.AVSynchronizer | None) -> None:
        self._av_sync = av_sync

    async def push_audio(self, frame: rtc.AudioFrame | AudioFlushSentinel) -> None:
        # resample audio frame if necessary
        # print(frame)
        if isinstance(frame, rtc.AudioFrame):
            if self._audio_resampler is None and (
                frame.sample_rate != 16000 or frame.num_channels != 1
            ):
                self._audio_resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=16000,
                    num_channels=1,
                )
            if self._audio_resampler is not None:
                for resampled_frame in self._audio_resampler.push(frame):
                    await self.simliClient.send(
                        resampled_frame.to_wav_bytes()[44:]
                    )  # Remove wav header
                return

        elif self._audio_resampler is not None:
            # flush the resampler
            for resampled_frame in self._audio_resampler.flush():
                await self.simliClient.send(
                    resampled_frame.to_wav_bytes()[44:]
                )  # Remove wav header
            return

        # elif isinstance(frame, AudioFlushSentinel):
        #     self.clear_buffer()

    def clear_buffer(self):
        # asking for feedback: our built in clear buffer function is async, got a better idea?
        return self.simliClient.clearBuffer()

    async def _generate_frames(self):
        # 1/30 * 48000 * 2 * 2
        # 30FPS * audio_sample_rate * audio_channel_count * 2bytes per sample
        while self.simliClient.run:
            while len(self.audioBytesQueue) < 6400:
                # wait 0.1ms to get next frame
                await asyncio.sleep(0.0001)

            audioChunk: bytearray = None
            async with self.audioBufferMutex:
                audioChunk = self.audioBytesQueue[:6400]
                self.audioBytesQueue = self.audioBytesQueue[6400:]
            lkAudioFrame = rtc.AudioFrame(audioChunk, 48000, 2, 1600)
            lkVideoFrame = await self.videoQueue.get()
            yield (lkVideoFrame, lkAudioFrame)

    async def stream(
        self,
    ) -> AsyncIterator[
        tuple[rtc.VideoFrame, Optional[rtc.AudioFrame]] | AudioFlushSentinel
    ]:
        try:
            self.audioReceiverTask = asyncio.create_task(self.getAudioFrames())
            self.videoReceiverTask = asyncio.create_task(self.getVideoFrames())

            async for video_frame, audio_frame in self._generate_frames():
                yield video_frame, audio_frame
                # print(video_frame, audio_frame)
        finally:
            await asyncio.gather(*self.clearBufferTasks)

    async def initSimli(self, api_key: str, face_id: str):
        self.config = SimliConfig(
            api_key,
            face_id,
            maxIdleTime=100,
        )
        self.simliClient = SimliClient(
            self.config,
            # simliURL="http://127.0.0.1:8892",  # used for debugging on simli servers or to connect to a relay server
        )
        await self.simliClient.Initialize()

    async def getAudioFrames(self):
        async for audioFrame in self.simliClient.getAudioStreamIterator():
            async with self.audioBufferMutex:
                self.audioBytesQueue.extend(audioFrame.to_ndarray().tobytes())

    async def getVideoFrames(self):
        async for videoFrame in self.simliClient.getVideoStreamIterator("yuva420p"):
            await self.videoQueue.put(
                rtc.VideoFrame(
                    videoFrame.width,
                    videoFrame.height,
                    rtc.VideoBufferType.I420A,
                    videoFrame.to_ndarray().tobytes(),
                )
            )


async def main(room: rtc.Room, apikey, faceid):
    """Main application logic for the avatar worker"""
    worker = None
    stop_event = asyncio.Event()

    try:
        # Initialize and start worker
        media_options = MediaOptions(
            video_width=512,
            video_height=512,
            video_fps=30,
            audio_sample_rate=48000,
            audio_channels=2,
        )
        video_generator = SimliVideoGenerator(media_options)
        await video_generator.initSimli(apikey, faceid)
        await video_generator.simliClient.sendSilence()
        worker = AvatarRunner(
            room, video_generator=video_generator, media_options=media_options
        )
        video_generator.set_av_sync(worker.av_sync)
        await worker.start()

        # Set up disconnect handler
        async def handle_disconnect(participant: rtc.RemoteParticipant):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                logging.info(
                    "Agent %s disconnected, stopping worker...", participant.identity
                )
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
        if worker:
            await worker.aclose()


async def run_service(url: str, token: str, apikey, faceid):
    """Run the avatar worker service"""
    room = rtc.Room()
    try:
        # Connect to LiveKit room
        logging.info("Connecting to %s", url)
        await room.connect(url, token)
        logging.info("Connected to room %s", room.name)

        # Run main application logic
        await main(room, apikey, faceid)
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
            "--simli-api-key",
            required=True,
            help="Simli API key, get it from https://app.simli.com",
        )
        parser.add_argument(
            "--simli-face-id",
            default="tmp9i8bbq7c",
            help="Simli face id, create your own at https://app.simli.com or get a premade one from https://docs.simli.com/api-reference/available-faces",
        )
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
        asyncio.run(
            run_service(args.url, args.token, args.simli_api_key, args.simli_face_id)
        )
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)
    finally:
        logging.info("Shutting down...")
