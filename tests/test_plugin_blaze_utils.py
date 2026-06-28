"""Unit tests for Blaze plugin shared utilities."""

from __future__ import annotations

import struct

import pytest

from livekit.plugins.blaze._utils import apply_normalization_rules, convert_pcm_to_wav

pytestmark = pytest.mark.unit


def test_convert_pcm_to_wav_produces_valid_header() -> None:
    pcm = b"\x01\x00" * 100
    wav = convert_pcm_to_wav(pcm, sample_rate=16000, channels=1, bits_per_sample=16)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert wav[36:40] == b"data"
    assert len(wav) == 44 + len(pcm)


def test_convert_pcm_to_wav_embeds_sample_rate() -> None:
    pcm = b"\x00\x00" * 8
    wav = convert_pcm_to_wav(pcm, sample_rate=24000, channels=2, bits_per_sample=16)

    sample_rate = struct.unpack_from("<I", wav, 24)[0]
    channels = struct.unpack_from("<H", wav, 22)[0]
    assert sample_rate == 24000
    assert channels == 2


def test_apply_normalization_rules_no_rules_returns_original() -> None:
    assert apply_normalization_rules("Hello API", None) == "Hello API"
    assert apply_normalization_rules("Hello API", {}) == "Hello API"


def test_apply_normalization_rules_applies_replacements() -> None:
    rules = {"API": "A P I", "USD": "đô la"}
    assert apply_normalization_rules("USD API", rules) == "đô la A P I"


def test_apply_normalization_rules_prefers_longer_patterns_first() -> None:
    rules = {"$": "dollar", "USD": "đô la Mỹ"}
    assert apply_normalization_rules("USD price is $5", rules) == "đô la Mỹ price is dollar5"


def test_apply_normalization_rules_skips_empty_patterns() -> None:
    rules = {"": "ignored", "API": "A P I"}
    assert apply_normalization_rules("API", rules) == "A P I"
