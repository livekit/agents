"""Hermetic tests for the google plugin's direct ``credentials`` parameter (#6586).

Uses ``AnonymousCredentials``, a real ``google.auth.credentials.Credentials``
object that requires no files, environment, or network, to verify how the GCP
project is resolved for in-memory credentials, as Workload Identity Federation
setups supply them.
"""

import pytest
from google.auth.credentials import AnonymousCredentials

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.google import STT

pytestmark = pytest.mark.unit


class _ProjectCredentials(AnonymousCredentials):
    """In-memory credentials that carry a project, like WIF impersonated creds."""

    project_id = "test-project-123"


class TestSTTCredentials:
    async def test_recognizer_uses_project_from_credentials(self) -> None:
        # credentials carrying a project must not fall back to ADC
        creds = _ProjectCredentials()
        stt_instance = STT(credentials=creds)
        client = await stt_instance._create_client(conn_options=APIConnectOptions())
        assert stt_instance._project_id == "test-project-123"
        recognizer = stt_instance._get_recognizer(client)
        assert recognizer == "projects/test-project-123/locations/global/recognizers/_"

    async def test_recognizer_uses_explicit_project(self) -> None:
        creds = AnonymousCredentials()
        stt_instance = STT(credentials=creds, project="explicit-project")
        client = await stt_instance._create_client(conn_options=APIConnectOptions())

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
        client = await stt_instance._create_client(conn_options=APIConnectOptions())
        with pytest.raises(APIConnectionError, match="could not determine the GCP project id"):
            stt_instance._get_recognizer(client)
