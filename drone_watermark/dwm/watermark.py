"""
Core watermarking engine: DWT + DCT + keyed QIM.

How embedding works (in plain words)
 1. Take the brightness (Y) channel of the image.
 2. Apply a 1-level Haar DWT. Keep the LL (low-frequency) band - it survives
    JPEG, noise and blur much better than the detail bands.
 3. Cut the LL band into 4x4 blocks and apply a DCT to every block.
 4. Hide ONE bit per block by nudging two mid-frequency DCT coefficients to
    the nearest point of a "bit-0 grid" or "bit-1 grid" (QIM, quantization
    index modulation). The grids are shifted by a secret random "dither", so
    without the key the grids (and the hidden bits) cannot be found.
 5. The payload is repeated many times over the blocks. Which block carries
    which bit is decided by a secret random permutation.
 6. Inverse DCT -> inverse DWT -> back to a normal image.

Extraction is "blind": the original image is NOT needed. For each block we
measure which grid (0 or 1) the coefficients are closer to, add up the votes
of all repetitions of each payload bit, and take the majority.
"""
import hashlib

import cv2
import numpy as np
import pywt
from scipy.fft import dctn, idctn

BLOCK = 4
COEFFS = ((1, 2), (2, 1))  # mid-frequency positions inside each 4x4 DCT block


class Watermarker:
    def __init__(self, embed_key: bytes, delta: float = 32.0, wavelet: str = "haar"):
        self.embed_key = embed_key
        self.delta = float(delta)
        self.wavelet = wavelet

    # ---------- helpers ----------
    def _rng(self, tag: bytes) -> np.random.Generator:
        seed = int.from_bytes(hashlib.sha256(self.embed_key + tag).digest()[:8], "big")
        return np.random.default_rng(seed)

    def _layout(self, n_blocks: int, n_bits: int):
        """Secret permutation + secret dither + which payload bit each block carries."""
        perm = self._rng(b"perm").permutation(n_blocks)
        dither = self._rng(b"dither").uniform(0, self.delta, size=(n_blocks, len(COEFFS)))
        bit_of_block = np.empty(n_blocks, dtype=np.int64)
        bit_of_block[perm] = np.arange(n_blocks) % n_bits
        return bit_of_block, dither

    def _usable_size(self, h, w):
        step = 2 * BLOCK  # 1-level DWT halves the size, then 4x4 blocks
        return h - h % step, w - w % step

    @staticmethod
    def _to_blocks(ll):
        h, w = ll.shape
        b = ll.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).swapaxes(1, 2)
        return b.reshape(-1, BLOCK, BLOCK)

    @staticmethod
    def _from_blocks(blocks, h, w):
        b = blocks.reshape(h // BLOCK, w // BLOCK, BLOCK, BLOCK).swapaxes(1, 2)
        return b.reshape(h, w)

    def _distances(self, c, dither, bit):
        """Distance of coefficient c to the lattice of the given bit value."""
        d = self.delta
        off = dither + bit * d / 2.0
        q = np.round((c - off) / d) * d + off
        return q, np.abs(c - q)

    def _decompose(self, img_bgr):
        ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y = ycc[..., 0].astype(np.float64)
        h, w = self._usable_size(*y.shape)
        ll, (lh, hl, hh) = pywt.dwt2(y[:h, :w], self.wavelet)
        return ycc, (h, w), ll, (lh, hl, hh)

    # ---------- embed ----------
    def embed(self, img_bgr: np.ndarray, bits: np.ndarray) -> np.ndarray:
        ycc, (h, w), ll, details = self._decompose(img_bgr)
        blocks = dctn(self._to_blocks(ll), axes=(1, 2), norm="ortho")
        n = blocks.shape[0]
        bit_of_block, dither = self._layout(n, len(bits))
        block_bits = bits[bit_of_block]
        for k, (r, c) in enumerate(COEFFS):
            q0, _ = self._distances(blocks[:, r, c], dither[:, k], 0)
            q1, _ = self._distances(blocks[:, r, c], dither[:, k], 1)
            blocks[:, r, c] = np.where(block_bits == 1, q1, q0)
        ll_new = self._from_blocks(idctn(blocks, axes=(1, 2), norm="ortho"), *ll.shape)
        y_new = pywt.idwt2((ll_new, details), self.wavelet)
        out = ycc.copy()
        out[:h, :w, 0] = np.clip(np.rint(y_new), 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

    # ---------- extract ----------
    def extract(self, img_bgr: np.ndarray, n_bits: int):
        """Returns (payload_bits, per_block_hard_bits, per_block_confidence, ll_shape)."""
        _, _, ll, _ = self._decompose(img_bgr)
        blocks = dctn(self._to_blocks(ll), axes=(1, 2), norm="ortho")
        n = blocks.shape[0]
        bit_of_block, dither = self._layout(n, n_bits)
        score = np.zeros(n)  # > 0 means "looks like bit 1"
        for k, (r, c) in enumerate(COEFFS):
            _, d0 = self._distances(blocks[:, r, c], dither[:, k], 0)
            _, d1 = self._distances(blocks[:, r, c], dither[:, k], 1)
            score += (d0 - d1) / (self.delta / 2.0)
        vote = np.bincount(bit_of_block, weights=score, minlength=n_bits)
        payload_bits = (vote > 0).astype(np.uint8)
        return payload_bits, (score > 0).astype(np.uint8), score, bit_of_block, (ll.shape[0] // BLOCK, ll.shape[1] // BLOCK)

    def capacity_blocks(self, h, w):
        h, w = self._usable_size(h, w)
        return (h // 2 // BLOCK) * (w // 2 // BLOCK)
