"""Telemetry uses the shared CA context and cannot weaken TLS or follow redirects."""

from __future__ import annotations

import datetime
import ssl
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from livekit.agents import _core
from livekit.agents.telemetry import _destination
from livekit.agents.utils import http_context

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@contextmanager
def https_server(cert: Path, key: Path, *, redirect: bool = False):
    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests_seen.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(307 if redirect else 200)
            if redirect:
                self.send_header("Location", "/must-not-receive-private-content")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"https://localhost:{server.server_port}/export", requests_seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def issue_certificate(tmp_path: Path, name: str, ca_key, ca_cert, *, expired: bool = False):
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=2))
        .not_valid_after(now + datetime.timedelta(days=-1 if expired else 1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(name)]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "server.pem", tmp_path / "server.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def create_ca():
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test CA")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


@pytest.mark.parametrize("case", ["trusted", "untrusted", "wrong_hostname", "expired", "redirect"])
def test_telemetry_enforces_shared_trust_and_never_follows_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    ca_key, ca_cert = create_ca()
    bundle = tmp_path / "ca.pem"
    bundle.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    # These Requests-specific settings must not broaden our selected trust roots.
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/must-not-override-shared-ca.pem")
    monkeypatch.setattr(_destination, "upload_gate", _core.UploadGate())
    if case == "untrusted":
        ca_key, ca_cert = create_ca()
    cert, key = issue_certificate(
        tmp_path,
        "wrong.example" if case == "wrong_hostname" else "localhost",
        ca_key,
        ca_cert,
        expired=case == "expired",
    )
    http_context._ssl_context_for.cache_clear()
    try:
        with https_server(cert, key, redirect=case == "redirect") as (url, seen):
            with _destination._AuthRefreshingSession(
                lambda: {"Authorization": "Bearer test"}
            ) as session:
                session.verify = False
                monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
                context = http_context._create_ssl_context()
                adapter = session.get_adapter(url)
                assert adapter._ssl_context is context
                roots_before = context.cert_store_stats()
                if case in ("untrusted", "wrong_hostname", "expired"):
                    with pytest.raises(requests.exceptions.SSLError):
                        session.post(url, data=b"private", verify=False, timeout=2)
                    assert seen == []
                else:
                    response = session.post(
                        url, data=b"private", verify=False, allow_redirects=True, timeout=2
                    )
                    assert response.status_code == (307 if case == "redirect" else 200)
                    assert seen == ["/export"]
                assert context.cert_store_stats() == roots_before
                assert context.verify_mode == ssl.CERT_REQUIRED
                assert context.check_hostname
    finally:
        http_context._ssl_context_for.cache_clear()


@pytest.mark.parametrize("contents", [None, "", "invalid certificate data"])
def test_invalid_explicit_ca_bundle_never_falls_back_to_default_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str | None
) -> None:
    bundle = tmp_path / "ca.pem"
    if contents is not None:
        bundle.write_text(contents)
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    http_context._ssl_context_for.cache_clear()
    try:
        with pytest.raises((OSError, ssl.SSLError)):
            _destination._AuthRefreshingSession(lambda: {})
    finally:
        http_context._ssl_context_for.cache_clear()
