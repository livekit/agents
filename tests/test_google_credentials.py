"""Hermetic tests for the google plugin's direct ``credentials`` parameter (#6586).

Uses ``AnonymousCredentials`` — a real ``google.auth.credentials.Credentials``
object that requires no files, environment, or network — to verify the object
is passed through to the underlying clients, mirroring what Workload Identity
Federation setups do with in-memory credentials.
"""

import pytest
from google.auth.credentials import AnonymousCredentials

from livekit.agents import APIConnectionError
from livekit.plugins.google import STT, TTS

pytestmark = pytest.mark.unit


class _ProjectCredentials(AnonymousCredentials):
    """In-memory credentials that carry a project, like WIF impersonated creds."""

    project_id = "test-project-123"


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

    async def test_recognizer_uses_project_from_credentials(self) -> None:
        # credentials carrying a project must not fall back to ADC
        creds = _ProjectCredentials()
        stt_instance = STT(credentials=creds)
        client = await stt_instance._create_client(timeout=1.0)
        assert stt_instance._project_id == "test-project-123"
        recognizer = stt_instance._get_recognizer(client)
        assert recognizer == "projects/test-project-123/locations/global/recognizers/_"

    async def test_recognizer_uses_explicit_project(self) -> None:
        creds = AnonymousCredentials()
        stt_instance = STT(credentials=creds, project="explicit-project")
        client = await stt_instance._create_client(timeout=1.0)

        assert stt_instance._get_recognizer(client) == (
            "projects/explicit-project/locations/global/recognizers/_"
        )

    async def test_clear_error_when_project_unresolvable(self, monkeypatch) -> None:
        # no project on the credentials and no ADC available: the error must
        # say what is wrong instead of a confusing "default credentials not
        # found" at transcription time (Devin finding on #6618)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        creds = AnonymousCredentials()
        stt_instance = STT(credentials=creds)
        client = await stt_instance._create_client(timeout=1.0)
        with pytest.raises(APIConnectionError, match="could not determine the GCP project id"):
            stt_instance._get_recognizer(client)


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
