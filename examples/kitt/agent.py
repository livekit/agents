import asyncio
import logging
from typing import List

import dotenv
from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import ChatMessage, ChatRole, LLMStream
from livekit.plugins.deepgram import STT
from livekit.plugins.elevenlabs import TTS
from livekit.plugins.openai import LLM

dotenv.load_dotenv()


async def entrypoint(job: JobContext):
    # LiveKit Entities
    chat_manager = rtc.ChatManager(job.room)
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    # Plugins
    tts = TTS()
    llm = LLM()
    stt = STT()
    stt_stream = stt.stream()

    # State
    committed_messages: List[agents.llm.ChatMessage] = [
        ChatMessage(role=ChatRole.SYSTEM, text="You are a friendly assistant.")
    ]
    llm_stream_queue = asyncio.Queue[LLMStream]()
    latest_llm_stream: LLMStream = None
    tts_stream_queue = asyncio.Queue[agents.tts.SynthesizeStream]()
    latest_tts_stream = tts.stream()
    working_text = ""
    agent_speaking = False

    audio_stream_future = asyncio.Future[rtc.AudioStream]()

    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream_future.set_result(rtc.AudioStream(track))

    def on_chat_message(message: rtc.ChatMessage):
        print("Got chat message: ", message)

    # IMPORTANT: Subscribe to audio tracks immediately. If you don't,
    # you are introducing a race condition where the track_publication
    # already exists before you subscribe to it and therefore the track_subscribed
    # event will never be fired.
    job.room.on("track_subscribed", on_track_subscribed)
    chat_manager.on_message(on_chat_message)

    # Publish agent mic

    await job.room.local_participant.publish_track(track, options)

    # Wait for user audio
    audio_stream = await audio_stream_future

    async def audio_stream_task():
        async for audio_frame_event in audio_stream:
            # Ignore user input while the agent is speaking
            if agent_speaking:
                continue
            stt_stream.push_frame(audio_frame_event.frame)

    async def stt_stream_task():
        nonlocal working_text, latest_llm_stream, latest_tts_stream
        async for stt_event in stt_stream:
            print("Got STT event:", stt_event)
            # We eagerly try to run inference to keep the latency as low as possible.
            # If we get a new transcript, we update the working text, cancel in-flight inference,
            # and run new inference.
            if stt_event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                delta = stt_event.alternatives[0].text
                # Do nothing
                if delta == "":
                    continue
                working_text += delta
                print("Working text:", working_text)
                working_message = ChatMessage(role=ChatRole.USER, text=working_text)
                chat_context = agents.llm.ChatContext(
                    messages=committed_messages + [working_message]
                )
                # Cancel exisiting LLM inference
                if latest_llm_stream is not None:
                    await latest_llm_stream.aclose(wait=False)
                latest_llm_stream = await llm.chat(history=chat_context)
                # Cancel existing tts inference
                await latest_tts_stream.aclose(wait=False)
                # Pre-warm the tts
                latest_tts_stream = tts.stream()
                await llm_stream_queue.put(latest_llm_stream)

    async def chat_stream_task():
        while True:
            llm_stream = await llm_stream_queue.get()
            if llm_stream is None:
                break
            await tts_stream_queue.put(latest_tts_stream)
            async for chunk in llm_stream:
                print("Got chunk:", chunk)
                content = chunk.choices[0].delta.content
                latest_tts_stream.push_text(content)
            await latest_tts_stream.flush()

    async def tts_stream_task():
        nonlocal agent_speaking, working_text, committed_messages
        while True:
            tts_stream = await tts_stream_queue.get()
            if tts_stream is None:
                break
            agent_speaking = True
            committed = False
            async for event in tts_stream:
                if event.type == agents.tts.SynthesisEventType.AUDIO:
                    print("Got audio from 11labs")
                    # As soon as the agent speaks, we commit the working text
                    # and the agent's speech to the chat.

                    if not committed:
                        # committed_messages += [
                        #     ChatMessage(role=ChatRole.USER, text=working_text),
                        #     ChatMessage(role=ChatRole.ASSISTANT, text=event.audio.text),
                        # ]
                        committed = True
                        working_text = ""
                    await source.capture_frame(event.audio.data)
                elif event.type == agents.tts.SynthesisEventType.FINISHED:
                    break
            await tts_stream.aclose()
            agent_speaking = False

    async def end_session_task():
        # Flush llm_queue
        await asyncio.sleep(120)
        await audio_stream.aclose()
        await stt_stream.aclose(wait=False)
        while True:
            try:
                llm_stream_queue.get_nowait()
            except asyncio.QueueEmpty:
                llm_stream_queue.put_nowait(None)
                break

        while True:
            try:
                tts_stream_queue.get_nowait()
            except asyncio.QueueEmpty:
                tts_stream_queue.put_nowait(None)
                break

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(audio_stream_task())
            tg.create_task(stt_stream_task())
            tg.create_task(chat_stream_task())
            tg.create_task(tts_stream_task())
            tg.create_task(end_session_task())
    except BaseExceptionGroup as e:
        for exc in e.exceptions:
            print("Exception: ", exc)
    except Exception as e:
        print("Exception: ", e)

    print("NEIL got here")


async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint, auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
