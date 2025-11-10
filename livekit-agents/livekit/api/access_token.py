"""Shim: livekit.api.access_token

Provides a minimal Claims class used by tests/imports.
If runtime needs additional fields/methods, we will extend this file.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class Claims:
    """Very small placeholder for token claims - stores anything passed."""
    issuer: Optional[str] = None
    subject: Optional[str] = None
    exp: Optional[int] = None
    nbf: Optional[int] = None
    iat: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        # copy common fields if present, store rest in extra
        self.issuer = kwargs.pop("iss", kwargs.pop("issuer", None))
        self.subject = kwargs.pop("sub", kwargs.pop("subject", None))
        self.exp = kwargs.pop("exp", None)
        self.nbf = kwargs.pop("nbf", None)
        self.iat = kwargs.pop("iat", None)
        self.extra = kwargs
