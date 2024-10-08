import asyncio
import logging
import multiprocessing

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, JobProcess, vad
from livekit.plugins import silero
import rerun as rr
import rerun.blueprint as rrb


load_dotenv()

logger = logging.getLogger("vad-worker")
logger.setLevel(logging.INFO)


def _prewarm_job(proc: JobProcess):
    logger.info("loading VAD model")
    proc.userdata["vad"] = silero.VAD.load()


async def _vad_task(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info(f"starting vad for {participant.identity}")
    preloaded_vad: silero.VAD = ctx.proc.userdata["vad"]
    audio_stream = rtc.AudioStream.from_participant(
        participant=participant, track_source=rtc.TrackSource.SOURCE_MICROPHONE
    )

    vad_stream = preloaded_vad.stream()

    async def _read_vad_stream():
        rr.init("voice-activity-detection", spawn=True)

        ts_origin = f"vad-{participant.identity}"
        rr.send_blueprint(
            rrb.TimeSeriesView(
                origin=ts_origin,
                time_ranges=rrb.VisibleTimeRange(
                    "time",
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-7.5),
                    end=rrb.TimeRangeBoundary.cursor_relative(),
                ),
                axis_y=rrb.ScalarAxis(range=(0.0, 1.0), zoom_lock=True),
            ),
        )

        async for event in vad_stream:
            rr.set_time_seconds("time", event.timestamp)
            if event.type == vad.VADEventType.INFERENCE_DONE:
                rr.log(ts_origin, rr.Scalar(event.probability))
            elif event.type == vad.VADEventType.START_OF_SPEECH:
                pass
            elif event.type == vad.VADEventType.END_OF_SPEECH:
                pass



    read_vad_task = asyncio.create_task(_read_vad_stream())

    async for ev in audio_stream:
        vad_stream.push_frame(ev.frame)

    vad_stream.end_input()
    await read_vad_task


async def entrypoint(ctx: JobContext):
    logger.info("waiting for participants..")
    ctx.add_participant_entrypoint(entrypoint_fnc=_vad_task)
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info("connected to the room")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(prewarm_fnc=_prewarm_job, entrypoint_fnc=entrypoint))
