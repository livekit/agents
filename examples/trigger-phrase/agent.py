import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
    transcription,
    tokenize,
    tts,
    llm,
)
from livekit.plugins import deepgram, openai, silero

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

word_tokenizer_without_punctuation: tokenize.WordTokenizer = (
    tokenize.basic.WordTokenizer(ignore_punctuation=True)
)

trigger_phrase = "Hi Bob!"
trigger_phrase_words = word_tokenizer_without_punctuation.tokenize(text=trigger_phrase)


async def _respond_to_user(
    stt_stream: stt.SpeechStream,
    stt_forwarder: transcription.STTSegmentsForwarder,
    tts: tts.TTS,
    agent_audio_source: rtc.AudioSource,
    local_llm: llm.LLM,
    llm_stream: llm.LLMStream,
):
    async for ev in stt_stream:
        stt_forwarder.update(ev)
        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:

            new_transcribed_text_words = word_tokenizer_without_punctuation.tokenize(
                text=ev.alternatives[0].text
            )

            for i in range(len(trigger_phrase_words)):
                if (
                    len(new_transcribed_text_words) >= len(trigger_phrase_words)
                    and trigger_phrase_words[i].lower()
                    != new_transcribed_text_words[i].lower()
                ):
                    # ignore user speech by not sending it to LLM
                    break
                elif i == len(trigger_phrase_words) - 1:
                    # trigger phrase is validated
                    new_chat_context = llm_stream.chat_ctx.append(
                        text=ev.alternatives[0].text
                    )
                    llm_stream = local_llm.chat(chat_ctx=new_chat_context)
                    llm_reply = ""
                    async for chunk in llm_stream:
                        content = chunk.choices[0].delta.content
                        if content is None:
                            continue
                        llm_reply = llm_reply + content
                    msg = llm.ChatMessage.create(text=llm_reply, role="assistant")
                    llm_stream.chat_ctx.messages.append(msg)

                    async for output in tts.synthesize(llm_reply):
                        await agent_audio_source.capture_frame(output.frame)


async def entrypoint(ctx: JobContext):
    logger.info("starting trigger-word agent example")

    vad = silero.VAD.load(
        min_speech_duration=0.01,
        min_silence_duration=0.5,
    )
    stt_local = stt.StreamAdapter(stt=deepgram.STT(), vad=vad)
    stt_stream = stt_local.stream()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    first_participant = await ctx.wait_for_participant()

    # publish agent track
    tts = openai.TTS(model="tts-1", voice="nova")
    agent_audio_source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    agent_track = rtc.LocalAudioTrack.create_audio_track(
        "agent-mic", agent_audio_source
    )
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await ctx.room.local_participant.publish_track(agent_track, options)

    # setup LLM
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            f"You are {trigger_phrase}, a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )
    local_llm = openai.LLM()
    llm_stream = local_llm.chat(chat_ctx=initial_ctx)

    async def subscribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track
        )
        asyncio.create_task(
            _respond_to_user(
                stt_stream=stt_stream,
                stt_forwarder=stt_forwarder,
                tts=tts,
                agent_audio_source=agent_audio_source,
                local_llm=local_llm,
                llm_stream=llm_stream,
            )
        )

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if (
            track.kind == rtc.TrackKind.KIND_AUDIO
            and participant.identity == first_participant.identity
        ):
            asyncio.create_task(subscribe_track(participant, track))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
