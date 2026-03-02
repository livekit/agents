from __future__ import annotations

import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

__all__ = ["encrypt", "decrypt"]


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte key from a passphrase using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt(data: bytes, passphrase: str) -> bytes:
    """
    Encrypt bytes using AES-256-GCM with a passphrase.

    Args:
        data: The bytes to encrypt
        passphrase: A string passphrase of any length

    Returns:
        Encrypted bytes (salt + nonce + ciphertext)
    """
    salt = os.urandom(16)
    nonce = os.urandom(12)  # 96-bit nonce for GCM

    key = _derive_key(passphrase, salt)

    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, None)

    # Concatenate: salt (16) + nonce (12) + ciphertext
    return salt + nonce + ciphertext


def decrypt(encrypted: bytes, passphrase: str) -> bytes:
    """
    Decrypt data that was encrypted with encrypt().

    Args:
        encrypted: Bytes containing salt + nonce + ciphertext
        passphrase: The same passphrase used for encryption

    Returns:
        The decrypted bytes

    Raises:
        ValueError: If decryption fails (wrong passphrase or corrupted data)
    """
    if len(encrypted) < 28:  # 16 (salt) + 12 (nonce) = 28 minimum
        raise ValueError("Encrypted data too short")

    try:
        salt = encrypted[:16]
        nonce = encrypted[16:28]
        ciphertext = encrypted[28:]

        key = _derive_key(passphrase, salt)
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as e:
        raise ValueError(f"Failed to decrypt: {e}") from e
