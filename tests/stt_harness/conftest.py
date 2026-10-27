"""A missed endpoint override must fail before contacting a real provider."""

import ipaddress
import socket

import pytest


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    original = socket.socket.connect

    def connect(sock, address):
        if isinstance(address, tuple):
            assert ipaddress.ip_address(address[0]).is_loopback, "non-local STT test connection"
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
