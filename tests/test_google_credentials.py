"""Hermetic tests for the google plugin's direct ``credentials`` parameter (#6586).

Uses ``AnonymousCredentials`` — a real ``google.auth.credentials.Credentials``
object that requires no files, environment, or network — to verify the object
is passed through to the underlying clients, mirroring what Workload Identity
Federation setups do with in-memory credentials.
"""

import pytest
from google.auth.credentials import AnonymousCredentials

from livekit.plugins.google import STT, TTS

pytestmark = pytest.mark.unit


class TestSTTCredentials:
    def test_constructs_without_adc(self) -> None:
        # before #6586 there was no way to construct STT without ADC,
        # credentials_info, or a credentials file on disk
        creds = AnonymousCredentials()
        stt_instance = STT(credentials=creds)
        assert stt_instance._credentials is creds

    async def test_client_uses_supplied_credentials(self) -> None:
        creds = AnonymousCredentials()
        stt_instance = STT(credentials=creds)
        client = await stt_instance._create_client(timeout=1.0)
        assert client.transport._credentials is creds

    async def test_credentials_take_precedence_over_file(self, tmp_path) -> None:
        creds = AnonymousCredentials()
        # the file path is never read when a credentials object is supplied
        stt_instance = STT(credentials=creds, credentials_file=str(tmp_path / "missing.json"))
        client = await stt_instance._create_client(timeout=1.0)
        assert client.transport._credentials is creds


class TestTTSCredentials:
    def test_client_uses_supplied_credentials(self) -> None:
        creds = AnonymousCredentials()
        tts_instance = TTS(credentials=creds)
        client = tts_instance._ensure_client()
        assert client.transport._credentials is creds

    def test_credentials_take_precedence_over_file(self, tmp_path) -> None:
        creds = AnonymousCredentials()
        tts_instance = TTS(credentials=creds, credentials_file=str(tmp_path / "missing.json"))
        client = tts_instance._ensure_client()
        assert client.transport._credentials is creds
