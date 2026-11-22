"""Quality and error metrics."""
import numpy as np
from skimage.metrics import structural_similarity


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def ber(a: np.ndarray, b: np.ndarray) -> float:
    """Bit error rate between two bit arrays."""
    return float(np.mean(a != b))
