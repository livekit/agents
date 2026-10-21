"""
Synthesize speech on the LiveKit agent, then align visemes on AiTwin server.

Pipeline:
  OpenAI TTS → PCM
  → POST {AITWIN_API_BASE}/avatar_alignVisemes  (server G2P / alignment)
  → audio + visemes for browser AiTwin pushSpeechChunk

Optional HEADTTS_URL skips OpenAI TTS and uses HeadTTS audio+timings
(HeadTTS already returns visemes; no local G2P).
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests
from openai import AsyncOpenAI

from aitwin_auth import mint_aitwin_auth_token

logger = logging.getLogger("aitwin-livekit-demo.speech")

# OpenAI `pcm` = raw s16le mono @ 24 kHz for gpt-4o-mini-tts / tts-1.
OPENAI_PCM_SAMPLE_RATE = 24_000
DEFAULT_PCM_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class SpeechBundle:
    text: str
    audio_b64: str
    sample_rate: int
    visemes: list[dict[str, Any]]
    words: list[dict[str, Any]]


def _pcm_duration_sec(pcm: bytes, sample_rate: int) -> float:
    if sample_rate <= 0 or not pcm:
        return 0.0
    return len(pcm) / 2 / sample_rate


def align_visemes_on_aitwin(
    pcm: bytes,
    *,
    sample_rate: int,
    language: str = "en",
    text: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Call AiTwin server G2P/alignment:
      POST /avatar_alignVisemes
      { pcm: base64 s16le, sample_rate, language, text? }
    Returns (visemes, words, sample_rate).
    """
    api_base = (os.getenv("AITWIN_API_BASE") or "https://ai.aitwin.me").rstrip("/")
    url = f"{api_base}/avatar_alignVisemes"
    token = mint_aitwin_auth_token()

    payload: dict[str, Any] = {
        "pcm": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": sample_rate,
        "language": language,
    }
    cleaned = (text or "").strip()
    if cleaned:
        payload["text"] = cleaned
    if (os.getenv("AITWIN_ALIGN_G2P_ONLY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        payload["g2pOnly"] = True
    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json=payload,
        timeout=60,
    )
    if resp.status_code == 401:
        # Token may have been revoked — mint once and retry.
        token = mint_aitwin_auth_token(force=True)
        resp = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json=payload,
            timeout=60,
        )

    if not resp.ok:
        raise RuntimeError(
            f"avatar_alignVisemes failed ({resp.status_code}): {resp.text[:300]}"
        )

    data = resp.json() if resp.content else {}
    visemes = list(data.get("visemes") or [])
    words = list(data.get("words") or [])
    out_rate = int(data.get("sample_rate") or sample_rate)
    if not visemes:
        raise RuntimeError("avatar_alignVisemes returned no visemes")
    logger.info(
        "aligned via AiTwin server: %d visemes, %d words, sr=%d, dur=%.2fs",
        len(visemes),
        len(words),
        out_rate,
        _pcm_duration_sec(pcm, sample_rate),
    )
    return visemes, words, out_rate


async def synthesize_openai_pcm(
    text: str,
    *,
    model: str,
    voice: str,
) -> tuple[bytes, int]:
    client = AsyncOpenAI()
    resp = await client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="pcm",
    )
    if hasattr(resp, "aread"):
        pcm = await resp.aread()
    else:
        raw = getattr(resp, "content", b"")
        pcm = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
    return bytes(pcm), OPENAI_PCM_SAMPLE_RATE


def synthesize_headtts(
    text: str,
    *,
    base_url: str,
    voice: str,
    language: str = "en-us",
) -> SpeechBundle:
    """Optional path: HeadTTS returns audio + timed visemes (no AiTwin align)."""
    url = f"{base_url.rstrip('/')}/v1/synthesize"
    payload = {
        "input": text,
        "voice": voice,
        "language": language,
        "audioEncoding": "pcm",
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    audio_b64 = data.get("audio") or ""
    if not audio_b64:
        raise RuntimeError("HeadTTS response missing audio")

    pcm = base64.b64decode(audio_b64)
    sample_rate = int(
        data.get("sampleRate") or data.get("sample_rate") or DEFAULT_PCM_SAMPLE_RATE
    )

    raw_visemes = data.get("visemes") or []
    vtimes = data.get("vtimes") or []
    vdurations = data.get("vdurations") or []
    visemes: list[dict[str, Any]] = []
    for i, symbol in enumerate(raw_visemes):
        start_ms = float(vtimes[i]) if i < len(vtimes) else 0.0
        dur_ms = float(vdurations[i]) if i < len(vdurations) else 40.0
        visemes.append(
            {
                "symbol": str(symbol),
                "start": round(start_ms / 1000.0, 4),
                "duration": round(max(dur_ms, 1.0) / 1000.0, 4),
            }
        )

    raw_words = data.get("words") or []
    wtimes = data.get("wtimes") or []
    wdurations = data.get("wdurations") or []
    words: list[dict[str, Any]] = []
    for i, word in enumerate(raw_words):
        start_ms = float(wtimes[i]) if i < len(wtimes) else 0.0
        dur_ms = float(wdurations[i]) if i < len(wdurations) else 80.0
        words.append(
            {
                "word": str(word).strip(),
                "start": round(start_ms / 1000.0, 4),
                "duration": round(max(dur_ms, 1.0) / 1000.0, 4),
            }
        )

    if not visemes:
        # Fall through to AiTwin server alignment on the HeadTTS PCM.
        visemes, words, sample_rate = align_visemes_on_aitwin(
            pcm,
            sample_rate=sample_rate,
            language=(os.getenv("AITWIN_ALIGN_LANGUAGE") or "en").strip(),
            text=text,
        )
        audio_b64 = base64.b64encode(pcm).decode("ascii")

    return SpeechBundle(
        text=text,
        audio_b64=audio_b64,
        sample_rate=sample_rate,
        visemes=visemes,
        words=words,
    )


async def synthesize_speech(text: str) -> SpeechBundle:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty text")

    headtts_url = (os.getenv("HEADTTS_URL") or "").strip()
    if headtts_url:
        voice = (os.getenv("HEADTTS_VOICE") or "af_bella").strip()
        language = (os.getenv("HEADTTS_LANGUAGE") or "en-us").strip()
        logger.info("synthesizing via HeadTTS (%s)", headtts_url)
        return synthesize_headtts(
            cleaned,
            base_url=headtts_url,
            voice=voice,
            language=language,
        )

    model = (os.getenv("TTS_MODEL") or "gpt-4o-mini-tts").strip()
    voice = (os.getenv("TTS_VOICE") or "alloy").strip()
    align_lang = (os.getenv("AITWIN_ALIGN_LANGUAGE") or "en").strip()

    logger.info("OpenAI TTS model=%s voice=%s -> AiTwin /avatar_alignVisemes", model, voice)
    pcm, sample_rate = await synthesize_openai_pcm(cleaned, model=model, voice=voice)
    visemes, words, out_rate = align_visemes_on_aitwin(
        pcm,
        sample_rate=sample_rate,
        language=align_lang,
        text=cleaned,
    )
    return SpeechBundle(
        text=cleaned,
        audio_b64=base64.b64encode(pcm).decode("ascii"),
        sample_rate=out_rate,
        visemes=visemes,
        words=words,
    )
