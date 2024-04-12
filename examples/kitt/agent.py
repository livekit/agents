import asyncio
import json

from inference_job import EventType, InferenceJob
from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from livekit.plugins.deepgram import STT
from state_manager import StateManager

PROMPT = "You are KITT, a friendly voice assistant powered by LiveKit.  \
          Conversation should be personable, and be sure to ask follow up questions. \
          If your response is a question, please append a question mark symbol to the end of it.\
          Don't respond with more than a few sentences."
INTRO = "Hello, I am KITT, a friendly voice assistant powered by LiveKit Agents. \
        You can find my source code in the top right of this screen if you're curious how I work. \
        Feel free to ask me anything — I'm here to help! Just start talking or type in the chat."
SIP_INTRO = "Hello, I am KITT, a friendly voice assistant powered by LiveKit Agents. \
             Feel free to ask me anything — I'm here to help! Just start talking."


async def entrypoint(job: JobContext):
    # LiveKit Entities
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    # Plugins
    stt = STT()
    stt_stream = stt.stream()

    # Agent state
    state = StateManager(job.room, PROMPT)
    inference_task: asyncio.Task | None = None
    current_transcription = ""

    audio_stream_future = asyncio.Future[rtc.AudioStream]()

    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream_future.set_result(rtc.AudioStream(track))

    def on_data(dp: rtc.DataPacket):
        nonlocal current_transcription
        print("Data received: ", dp)
        # Ignore if the agent is speaking
        if state.agent_speaking:
            return
        if dp.topic != "lk-chat-topic":
            return
        payload = json.loads(dp.data)
        message = payload["message"]
        current_transcription = message
        asyncio.create_task(start_new_inference())

    for participant in job.room.participants.values():
        for track_pub in participant.tracks.values():
            # This track is not yet subscribed, when it is subscribed it will
            # call the on_track_subscribed callback
            if track_pub.track is None:
                continue
            audio_stream_future.set_result(rtc.AudioStream(track_pub.track))

    job.room.on("track_subscribed", on_track_subscribed)
    job.room.on("data_received", on_data)

    # Wait for user audio
    audio_stream = await audio_stream_future

    # Publish agent mic after waiting for user audio (simple way to avoid subscribing to self)
    await job.room.local_participant.publish_track(track, options)

    async def start_new_inference(force_text: str | None = None):
        nonlocal current_transcription

        state.agent_thinking = True
        job = InferenceJob(
            transcription=current_transcription,
            audio_source=source,
            chat_history=state.chat_history,
            force_text_response=force_text,
        )

        try:
            agent_done_thinking = False
            agent_has_spoken = False
            comitted_agent = False

            def commit_agent_text_if_needed():
                nonlocal agent_has_spoken, agent_done_thinking, comitted_agent
                if agent_done_thinking and agent_has_spoken and not comitted_agent:
                    comitted_agent = True
                    state.commit_agent_response(job.current_response)

            async for e in job:
                # Allow cancellation
                if e.type == EventType.AGENT_RESPONSE:
                    if e.finished_generating:
                        state.agent_thinking = False
                        agent_done_thinking = True
                        commit_agent_text_if_needed()
                elif e.type == EventType.AGENT_SPEAKING:
                    state.agent_speaking = e.speaking
                    if e.speaking:
                        agent_has_spoken = True
                        # Only commit user text for real transcriptions
                        if not force_text:
                            state.commit_user_transcription(job.transcription)
                        commit_agent_text_if_needed()
                        current_transcription = ""
        except asyncio.CancelledError:
            await job.acancel()

    async def audio_stream_task():
        async for audio_frame_event in audio_stream:
            stt_stream.push_frame(audio_frame_event.frame)

    async def stt_stream_task():
        nonlocal current_transcription, inference_task
        async for stt_event in stt_stream:
            # We eagerly try to run inference to keep the latency as low as possible.
            # If we get a new transcript, we update the working text, cancel in-flight inference,
            # and run new inference.
            if stt_event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                delta = stt_event.alternatives[0].text
                # Do nothing
                if delta == "":
                    continue
                current_transcription += " " + delta
                # Cancel in-flight inference
                if inference_task:
                    inference_task.cancel()
                    await inference_task
                # Start new inference
                inference_task = asyncio.create_task(start_new_inference())

    try:
        sip = job.room.name.startswith("sip")
        intro_text = SIP_INTRO if sip else INTRO
        inference_task = asyncio.create_task(start_new_inference(force_text=intro_text))
        async with asyncio.TaskGroup() as tg:
            tg.create_task(audio_stream_task())
            tg.create_task(stt_stream_task())
    except BaseExceptionGroup as e:
        for exc in e.exceptions:
            print("Exception: ", exc)
    except Exception as e:
        print("Exception: ", e)


async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint, auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
