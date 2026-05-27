"""Cartesia STT final transcript on flush use-case: WER example.

If you're building your first voice agent, try examples/other/cartesia.py

When configured to "emit_on_flush", Cartesia STT only emits a
:attr:`~livekit.agents.stt.SpeechEventType.FINAL_TRANSCRIPT` when *you* call
:meth:`~livekit.agents.stt.RecognizeStream.flush`.

It never emits ``START_OF_SPEECH`` / ``END_OF_SPEECH``.

That makes it a good fit for offline evaluation:
you push known audio, call ``flush()`` at the segment boundaries you control,
and score the final transcripts.

This script is fully self-contained: it synthesizes the reference audio with
``cartesia.TTS`` (so the TTS input text doubles as the WER reference), feeds it to
``cartesia.STT(final_transcript_mode="emit_on_flush")``, flushes once per segment, and prints the
word error rate.

Run with ``CARTESIA_API_KEY`` set:

    uv run examples/other/cartesia_final_transcript_on_flush_eval.py
"""

from __future__ import annotations

import asyncio
import logging
import re

import aiohttp
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import stt
from livekit.plugins import cartesia

# Each entry is flushed as its own segment. In a real eval these would be utterances
# from your dataset, each paired with a ground-truth transcript.
REFERENCE_SEGMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    " Cartesia STT transcribes speech with low latency.",
]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level WER via Levenshtein edit distance, after light normalization.

    Kept dependency-free so the example runs with only the plugins installed. For serious
    evaluation prefer a maintained library such as ``jiwer``.
    """

    def normalize(text: str) -> list[str]:
        text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
        return text.split()

    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    prev_row = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr_row = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,  # insertion
                prev_row[j] + 1,  # deletion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row
    return prev_row[len(hyp)] / len(ref)


async def transcribe_segment(
    logger: logging.Logger, stream: stt.RecognizeStream, audio: rtc.AudioFrame
) -> str:
    """Push one segment, flush, and return the resulting final transcript."""
    stream.push_frame(audio)
    # emit_on_flush emits exactly one FINAL_TRANSCRIPT per flush().
    stream.flush()
    async for ev in stream:
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            logger.debug("interim: %s", ev.alternatives[0].text)
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            return ev.alternatives[0].text
    return ""


async def main() -> None:
    load_dotenv()

    logger = logging.getLogger("cartesia-final-transcript-on-flush-eval")

    logging.basicConfig(level=logging.INFO)

    async with aiohttp.ClientSession() as http_session:
        tts = cartesia.TTS(model="sonic-3.5", http_session=http_session)
        speech_to_text = cartesia.STT(
            model="ink-2",
            final_transcript_mode="emit_on_flush",
            http_session=http_session,
        )

        stream = speech_to_text.stream()
        hypotheses: list[str] = []
        try:
            for segment_text in REFERENCE_SEGMENTS:
                # Generate the reference audio with TTS so the example needs no audio file.
                audio = await tts.synthesize(segment_text).collect()
                # RecognizeStream.push_frame resamples to the STT sample rate automatically.
                hypothesis = await transcribe_segment(logger, stream, audio)
                logger.info("segment final: %s", hypothesis)
                hypotheses.append(hypothesis)
        finally:
            await stream.aclose()

    # do not add or remove spaces when joining!
    # Cartesia's API expects transcript chunks to be joined with no extra formatting
    reference = "".join(REFERENCE_SEGMENTS)
    hypothesis = "".join(hypotheses)
    logger.info("reference:  %s", reference)
    logger.info("hypothesis: %s", hypothesis)
    logger.info("WER: %.2f%%", word_error_rate(reference, hypothesis) * 100)


if __name__ == "__main__":
    asyncio.run(main())
