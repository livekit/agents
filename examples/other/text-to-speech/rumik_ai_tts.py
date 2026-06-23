import asyncio
import logging
import os
import wave
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli
from livekit.plugins import rumik_ai

load_dotenv()

logger = logging.getLogger("rumik-ai-tts-demo")
logger.setLevel(logging.INFO)

server = AgentServer()

DEFAULT_MUGA_TEXTS = [
    "[happy] Hey, main Mira hoon. Aaj ka din kaisa gaya tumhara?",
    "[neutral] Kabhi kabhi bas kisi ko sunana hota hai. Tum batao, abhi mind mein kya chal raha hai?",
    "[sad] <sigh> Haan, ye thoda heavy lag raha hai. Tum is baat ko kab se carry kar rahe ho?",
]

# Mulberry speaks pure English and Hindi in Devanagari (English words stay in Latin
# script) -- not the Romanized Hinglish that Muga uses.
DEFAULT_MULBERRY_TEXTS = [
    "Hey, I'm Mira. How was your day today?",
    "कभी कभी बस किसी को सुनाना होता है। तुम बताओ, अभी mind में क्या चल रहा है?",
    "हाँ, ये थोड़ा heavy लग रहा है। तुम इस बात को कब से carry कर रहे हो?",
]


def _recording_path() -> Path:
    output_dir = Path(os.getenv("RUMIK_AI_RECORDING_DIR", "recordings"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_dir / f"rumik-ai-tts-{timestamp}.wav"


def _record_texts(model: str) -> list[str]:
    raw_texts = os.getenv("RUMIK_AI_RECORD_TEXTS")
    if not raw_texts:
        if model == "mulberry":
            return DEFAULT_MULBERRY_TEXTS
        return DEFAULT_MUGA_TEXTS
    return [text.strip() for text in raw_texts.split("||") if text.strip()]


def _text_chunks(text: str) -> list[str]:
    words_per_chunk = int(os.getenv("RUMIK_AI_STREAM_WORDS_PER_CHUNK", "3"))
    words = text.split()
    return [
        " ".join(words[i : i + words_per_chunk]) + " "
        for i in range(0, len(words), words_per_chunk)
    ]


def _rumik_tts() -> rumik_ai.TTS:
    model = os.getenv("RUMIK_AI_MODEL", "muga")

    # Point at a custom Rumik gateway (e.g. staging) via RUMIK_GATEWAY_URL; otherwise
    # the plugin's default base URL is used.
    gateway_url = os.getenv("RUMIK_GATEWAY_URL")
    common: dict[str, str] = {"base_url": gateway_url} if gateway_url else {}

    if model == "mulberry":
        # Only pass speaker when set -- the plugin rejects None as an invalid speaker.
        speaker = os.getenv("RUMIK_AI_SPEAKER")
        if speaker:
            common["speaker"] = speaker
        return rumik_ai.TTS(
            model="mulberry",
            description=os.getenv(
                "RUMIK_AI_DESCRIPTION",
                "warm, gentle friend with a calm, natural listening style",
            ),
            f0_up_key=float(os.getenv("RUMIK_AI_F0_UP_KEY", "0")),
            **common,
        )

    return rumik_ai.TTS(
        model="muga",
        tone=os.getenv("RUMIK_AI_TONE") or None,
        **common,
    )


@server.rtc_session()
async def entrypoint(job: JobContext) -> None:
    logger.info("starting rumik-ai TTS recording example")

    tts = _rumik_tts()
    output_path = _recording_path()

    # This example records raw rumik-ai TTS output; it does not run STT or an LLM.
    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("rumik-ai-tts", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await job.room.local_participant.publish_track(track, options)
    if os.getenv("RUMIK_AI_WAIT_FOR_SUBSCRIBER", "").lower() in {"1", "true", "yes"}:
        await publication.wait_for_subscription()

    texts = _record_texts(tts.model)
    logger.info("recording %d utterance(s) to %s", len(texts), output_path)

    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(tts.num_channels)
        wav.setsampwidth(2)
        wav.setframerate(tts.sample_rate)

        for text in texts:
            logger.info('streaming "%s"', text)

            async with tts.stream() as stream:

                async def playback_and_record() -> None:
                    async for output in stream:
                        await source.capture_frame(output.frame)
                        wav.writeframes(bytes(output.frame.data))

                playback_task = asyncio.create_task(playback_and_record())

                for chunk in _text_chunks(text):
                    logger.info('pushing text chunk "%s"', chunk.strip())
                    stream.push_text(chunk)

                stream.flush()
                stream.end_input()
                await playback_task

            await asyncio.sleep(0.4)

    logger.info("recorded rumik-ai TTS output to %s", output_path)
    await asyncio.sleep(1)


if __name__ == "__main__":
    cli.run_app(server)
