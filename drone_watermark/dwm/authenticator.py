"""
High-level API:   sign()  = embed a signed watermark
                  verify() = extract it, check it, report tampering
"""
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import label, uniform_filter

from .payload import PAYLOAD_BITS, Payload, hamming, perceptual_hash
from .watermark import Watermarker

# Status codes returned by verify()
AUTHENTIC = "AUTHENTIC"                      # valid watermark, image content unchanged
AUTHENTIC_MODIFIED = "AUTHENTIC_BUT_MODIFIED"  # valid watermark, but content was changed
REPLAY = "REPLAY_DETECTED"                   # valid watermark, but this image was already received
STALE = "STALE_OR_FUTURE_TIMESTAMP"          # valid watermark, but timestamp is outside allowed window
NOT_AUTHENTIC = "NOT_AUTHENTIC"              # no valid watermark found


class ReplayStore:
    """Remembers (drone_id, nonce) pairs already accepted. Memory-only, or JSON file if a path is given."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.seen = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self.seen = json.load(f)

    def check_and_add(self, drone_id: int, nonce: int) -> bool:
        """Returns True if this is a NEW message, False if it was seen before."""
        key = f"{drone_id}:{nonce}"
        if key in self.seen:
            return False
        self.seen[key] = time.time()
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self.seen, f)
        return True


@dataclass
class VerificationResult:
    status: str
    tag_valid: bool
    drone_id: Optional[int] = None
    timestamp: Optional[int] = None
    nonce: Optional[int] = None
    hash_distance: Optional[int] = None
    block_error_rate: Optional[float] = None
    tamper_fraction: Optional[float] = None
    tamper_map: Optional[np.ndarray] = field(default=None, repr=False)
    notes: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == AUTHENTIC


class DroneImageAuthenticator:
    def __init__(self, master_key: bytes, delta: float = 32.0, hash_tolerance: int = 12, cluster_blocks: int = 30,
                 max_age_seconds: Optional[float] = None, replay_store: Optional[ReplayStore] = None):
        # Two separate keys are derived from the master key (good practice).
        self.mac_key = hashlib.sha256(master_key + b"|mac").digest()
        self.wm = Watermarker(hashlib.sha256(master_key + b"|embed").digest(), delta=delta)
        self.hash_tolerance = hash_tolerance
        self.cluster_blocks = cluster_blocks
        self.max_age = max_age_seconds
        self.replay = replay_store if replay_store is not None else ReplayStore()

    # ------------------------------------------------------------------ sign
    def sign(self, img_bgr, drone_id: int, timestamp: Optional[int] = None, nonce: Optional[int] = None):
        """Embed the watermark. Returns (watermarked_image, Payload)."""
        timestamp = int(time.time()) if timestamp is None else int(timestamp)
        nonce = secrets.randbits(32) if nonce is None else int(nonce)
        payload = Payload(drone_id, timestamp, nonce, perceptual_hash(img_bgr))
        bits = payload.to_bits(self.mac_key)
        need = PAYLOAD_BITS
        if self.wm.capacity_blocks(*img_bgr.shape[:2]) < need * 3:
            raise ValueError("Image too small to carry the watermark with enough redundancy.")
        return self.wm.embed(img_bgr, bits), payload

    # ---------------------------------------------------------------- verify
    def verify(self, img_bgr, check_replay: bool = True, now: Optional[float] = None) -> VerificationResult:
        bits, block_bits, score, bit_of_block, grid = self.wm.extract(img_bgr, PAYLOAD_BITS)
        payload = Payload.from_bits(bits)

        # block-level agreement with the decoded payload (used for tamper localisation)
        expected = bits[bit_of_block]
        block_err = (block_bits != expected).astype(float)
        ber_blocks = float(block_err.mean())

        if not payload.tag_is_valid(self.mac_key):
            return VerificationResult(NOT_AUTHENTIC, False, block_error_rate=ber_blocks,
                                      notes="No valid watermark (tag mismatch).")

        # --- valid origin. Now check freshness, replay and integrity.
        res = VerificationResult(AUTHENTIC, True, payload.drone_id, payload.timestamp, payload.nonce,
                                 block_error_rate=ber_blocks)

        now = time.time() if now is None else now
        if self.max_age is not None and not (-300 <= now - payload.timestamp <= self.max_age):
            res.status, res.notes = STALE, "Timestamp outside the allowed time window."
            return res

        res.hash_distance = hamming(payload.phash, perceptual_hash(img_bgr))

        # Tamper map: share of wrong blocks in every 3x3 neighbourhood. Normal processing
        # (JPEG, noise) spreads errors thinly; an edited region gives a dense cluster.
        err_map = uniform_filter(block_err.reshape(grid), size=3, mode="nearest")
        mask = err_map > max(0.5, 3 * ber_blocks)
        labels, n = label(mask)
        biggest = max((int((labels == i).sum()) for i in range(1, n + 1)), default=0)
        res.tamper_map = err_map
        res.tamper_fraction = float(mask.mean())

        if res.hash_distance > self.hash_tolerance or biggest >= self.cluster_blocks:
            res.status = AUTHENTIC_MODIFIED
            res.notes = (f"Watermark is valid but the content differs from what the drone signed "
                         f"(hash distance {res.hash_distance}, largest damaged cluster {biggest} blocks).")
            return res

        if check_replay and not self.replay.check_and_add(payload.drone_id, payload.nonce):
            res.status, res.notes = REPLAY, "This (drone_id, nonce) was already accepted earlier."
        return res
