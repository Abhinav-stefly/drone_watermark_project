"""
Payload = the secret message hidden inside every drone image.

Layout (26 bytes = 208 bits):
    drone_id   2 bytes   which drone took the picture
    timestamp  4 bytes   capture time (unix seconds)
    nonce      4 bytes   random number, used to catch replay attacks
    phash      8 bytes   64-bit "fingerprint" of the image content
    tag        8 bytes   HMAC-SHA256 over all fields above (needs the secret key)

Only someone who knows the secret key can create a valid tag, so forged
images cannot carry a valid watermark.
"""
import hashlib
import hmac
import struct
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.fft import dctn

PAYLOAD_BYTES = 26
PAYLOAD_BITS = PAYLOAD_BYTES * 8


def perceptual_hash(img_bgr: np.ndarray) -> bytes:
    """64-bit DCT hash. Stays almost the same after JPEG, mild noise, etc."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    low = dctn(small, norm="ortho")[:8, :8].flatten()
    bits = (low > np.median(low)).astype(np.uint8)
    return np.packbits(bits).tobytes()


def hamming(a: bytes, b: bytes) -> int:
    return int(np.unpackbits(np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).sum())


def _tag(mac_key: bytes, body: bytes) -> bytes:
    return hmac.new(mac_key, body, hashlib.sha256).digest()[:8]


@dataclass
class Payload:
    drone_id: int
    timestamp: int
    nonce: int
    phash: bytes
    tag: bytes = b""

    def body(self) -> bytes:
        return struct.pack(">HII", self.drone_id, self.timestamp, self.nonce) + self.phash

    def to_bits(self, mac_key: bytes) -> np.ndarray:
        raw = self.body() + _tag(mac_key, self.body())
        return np.unpackbits(np.frombuffer(raw, np.uint8))

    @staticmethod
    def from_bits(bits: np.ndarray) -> "Payload":
        raw = np.packbits(bits.astype(np.uint8)).tobytes()
        drone_id, ts, nonce = struct.unpack(">HII", raw[:10])
        return Payload(drone_id, ts, nonce, raw[10:18], raw[18:26])

    def tag_is_valid(self, mac_key: bytes) -> bool:
        return hmac.compare_digest(self.tag, _tag(mac_key, self.body()))
