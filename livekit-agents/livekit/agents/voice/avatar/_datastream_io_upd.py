import asyncio
import time
import uuid
from collections import deque

from livekit import rtc

from ...log import logger
from ...metrics import VideoAvatarMetrics
from ._datastream_io import DataStreamAudioOutput


class LatencyAudioOutput(DataStreamAudioOutput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audio_send_times = (
            deque()
        )  # deque of {'send_ts': float, 'ingest_ts': float | None, 'duration': float}
        self._total_audio_sent = 0
        self._lock = asyncio.Lock()

    async def capture_frame(self, frame: rtc.AudioFrame):
        async with self._lock:
            send_ts = time.time()
            ingest_ts = frame.userdata.get("ingest_ts")

            # add to this audio chunk with its timestamps
            self._audio_send_times.append(
                {"send_ts": send_ts, "ingest_ts": ingest_ts, "duration": frame.duration}
            )
            self._total_audio_sent += 1

            logger.debug(
                f"Audio pushed: total={self._total_audio_sent}, pending={len(self._audio_send_times)}"
            )

        await super().capture_frame(frame)

    async def pop_audio_timing(self):
        """Pop the oldest audio timing for latency calculation"""
        async with self._lock:
            # print(f'dequeue   {self._audio_send_times}')
            if self._audio_send_times:
                timm = self._audio_send_times.popleft()
                print("TIMM. ", timm)
                return timm
            return None


def attach_video_latency_listener(room: rtc.Room, audio_output: LatencyAudioOutput, avatar_session):
    @room.on("track_subscribed")
    def subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(
            f"Track subscribed: kind={publication.kind}, participant={participant.identity}"
        )

        if publication.kind != rtc.TrackKind.KIND_VIDEO:
            return

        if participant.identity != avatar_session._avatar_participant_identity:
            logger.info(f"Skipping non-avatar participant: {participant.identity}")
            return

        if isinstance(track, rtc.RemoteVideoTrack):
            logger.info("Setting up video frame listener for avatar")

            async def listen_video_frames():
                try:
                    video_stream = rtc.VideoStream(track)
                    logger.info("Video stream created, waiting for frames...")

                    async for _frame in video_stream:
                        timing = (
                            await audio_output.pop_audio_timing()
                        )  # get the matching audio chunk
                        if timing:
                            recv_ts = time.time()  # when video frame arrives
                            metrics = VideoAvatarMetrics(
                                event_id=str(uuid.uuid4()),
                                timestamp=time.time(),
                                audio_sent_ts=timing["send_ts"],
                                ws_received_ts=timing.get("ingest_ts") or timing["send_ts"],
                                video_received_ts=recv_ts,
                                full_latency=recv_ts - timing["send_ts"],
                                server_latency=(timing.get("ingest_ts") or timing["send_ts"])
                                - timing["send_ts"],
                                video_pipeline_latency=recv_ts
                                - (timing.get("ingest_ts") or timing["send_ts"]),
                            )
                            avatar_session.emit("metrics_collected", metrics)
                        else:
                            # idle animation
                            # logger.debug(f"Video frame {frame_count} received (no pending audio)")
                            pass

                    logger.info("Video stream ended")
                except Exception as e:
                    logger.error(f"Error in video frame listener: {e}", exc_info=True)

            task = asyncio.create_task(listen_video_frames())
            avatar_session._latency_tasks.add(task)
            task.add_done_callback(avatar_session._latency_tasks.discard)
            logger.info("Video frame listener task created")
