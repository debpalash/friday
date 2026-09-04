"""AES-256-CTR primitive shared by Friday's local encrypted stores.

This reproduces ``openssl enc -aes-256-ctr -K <key> -iv <iv>`` exactly: the
16-byte IV is the initial 128-bit big-endian counter block and the whole
block increments, so ciphertext produced by the previous OpenSSL command-line
path decrypts unchanged. CTR encryption and decryption are the same
operation.
"""

from __future__ import annotations

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def aes256_ctr(key: bytes, iv: bytes, data: bytes) -> bytes:
    if len(key) != 32:
        raise ValueError("AES-256 key must contain exactly 32 bytes")
    if len(iv) != 16:
        raise ValueError("AES-CTR IV must contain exactly 16 bytes")
    encryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
    return encryptor.update(bytes(data)) + encryptor.finalize()


__all__ = ["aes256_ctr"]
