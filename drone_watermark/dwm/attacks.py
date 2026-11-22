"""
Attacks used for testing.
 - Signal-processing attacks (robustness): JPEG, noise, rotation, crop, scaling, blur.
 - Malicious attacks (security): splice, copy-move, watermark removal, transplant.
All functions take and return uint8 BGR images of the same size.
"""
import cv2
import numpy as np


def jpeg(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def gaussian_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def salt_pepper(img, amount, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < amount / 2] = 0
    out[m > 1 - amount / 2] = 255
    return out


def rotate(img, angle):
    """Rotate by `angle` degrees and rotate back (typical 'rotation attack' test)."""
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    fwd = cv2.warpAffine(img, cv2.getRotationMatrix2D(c, angle, 1.0), (w, h), flags=cv2.INTER_LINEAR)
    return cv2.warpAffine(fwd, cv2.getRotationMatrix2D(c, -angle, 1.0), (w, h), flags=cv2.INTER_LINEAR)


def crop_out(img, fraction):
    """Cut away a frame of the image (blacked out), keep the centre. fraction = share of area removed."""
    h, w = img.shape[:2]
    keep = np.sqrt(1 - fraction)
    kh, kw = int(h * keep), int(w * keep)
    y0, x0 = (h - kh) // 2, (w - kw) // 2
    out = np.zeros_like(img)
    out[y0:y0 + kh, x0:x0 + kw] = img[y0:y0 + kh, x0:x0 + kw]
    return out


def scale(img, factor):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


# ----------------- malicious attacks -----------------
def splice(img, other, box):
    """Paste a region from a different image (object insertion / removal). box = (y0, y1, x0, x1)."""
    y0, y1, x0, x1 = box
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.resize(other, (img.shape[1], img.shape[0]))[y0:y1, x0:x1]
    return out


def copy_move(img, src, dst, size):
    """Copy a patch from one place to another inside the same image."""
    out = img.copy()
    (sy, sx), (dy, dx), (ph, pw) = src, dst, size
    out[dy:dy + ph, dx:dx + pw] = img[sy:sy + ph, sx:sx + pw]
    return out


def remove_watermark_denoise(img):
    """Try to wipe the watermark: strong denoise + re-quantise."""
    den = cv2.fastNlMeansDenoisingColored(img, None, 12, 12, 7, 21)
    return jpeg(den, 40)


def remove_watermark_requantise(img, step=24):
    """Coarse re-quantisation of pixel values to destroy small signal changes."""
    return (np.rint(img.astype(np.float64) / step) * step).clip(0, 255).astype(np.uint8)


def transplant(target, authentic_marked, authentic_original):
    """Watermark copy attack: add the (marked - original) difference of an authentic image onto another image."""
    diff = authentic_marked.astype(np.int16) - authentic_original.astype(np.int16)
    h, w = target.shape[:2]
    diff = cv2.resize(diff.astype(np.float32), (w, h)).astype(np.int16)
    return np.clip(target.astype(np.int16) + diff, 0, 255).astype(np.uint8)
