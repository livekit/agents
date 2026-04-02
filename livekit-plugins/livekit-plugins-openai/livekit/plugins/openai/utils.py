from __future__ import annotations

import ipaddress
import os
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

AsyncAzureADTokenProvider = Callable[[], str | Awaitable[str]]


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    parsed_url = urlparse(base_url)
    hostname = parsed_url.hostname
    if parsed_url.scheme.lower() != "https" or not hostname:
        raise ValueError("OPENAI base URL must be a valid https URL")

    allowed_hosts = {"api.openai.com"}
    allowed_hosts.update(
        host.strip().lower()
        for host in os.getenv("OPENAI_ALLOWED_BASE_URL_HOSTS", "").split(",")
        if host.strip()
    )

    normalized_hostname = hostname.lower()
    if normalized_hostname not in allowed_hosts:
        raise ValueError("OPENAI base URL host is not allowed")

    allow_private_hosts = os.getenv("OPENAI_ALLOW_PRIVATE_BASE_URL", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not allow_private_hosts:
        if normalized_hostname in {"localhost", "localhost.localdomain"}:
            raise ValueError("OPENAI base URL host is not allowed")
        try:
            ip_addr = ipaddress.ip_address(normalized_hostname)
        except ValueError:
            pass
        else:
            if (
                ip_addr.is_private
                or ip_addr.is_loopback
                or ip_addr.is_link_local
                or ip_addr.is_multicast
                or ip_addr.is_reserved
                or ip_addr.is_unspecified
            ):
                raise ValueError("OPENAI base URL host is not allowed")

    return base_url


__all__ = ["get_base_url", "AsyncAzureADTokenProvider"]
