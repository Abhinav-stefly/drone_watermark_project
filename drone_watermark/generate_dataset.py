"""
Makes a synthetic 'drone / aerial' image dataset (fields, forest, water, roads, buildings,
with grain and blur like a real camera). Use it when you do not have a real dataset yet.

    python generate_dataset.py --out data/synthetic_aerial --count 20
"""
import argparse
import os

import cv2
import numpy as np


def smooth_noise(rng, h, w, scale):
    n = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)


def make_image(seed, h=480, w=640):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)

    # ground: patchwork of fields (BGR colours)
    palette = [(60, 140, 70), (70, 170, 110), (90, 120, 150), (50, 110, 60), (110, 160, 170), (80, 100, 90)]
    cell = rng.integers(60, 120)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            col = np.array(palette[rng.integers(len(palette))], np.float32) * rng.uniform(0.8, 1.15)
            img[y:y + cell, x:x + cell] = col
            if rng.random() < 0.6:  # crop rows
                ang = rng.uniform(0, np.pi)
                yy, xx = np.mgrid[y:min(y + cell, h), x:min(x + cell, w)]
                stripes = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * rng.uniform(0.4, 1.2))
                img[y:y + cell, x:x + cell] += stripes[..., None] * rng.uniform(4, 12)

    # forest blobs and a lake
    for _ in range(rng.integers(2, 5)):
        m = smooth_noise(rng, h, w, rng.integers(30, 70)) > 0.62
        tex = smooth_noise(rng, h, w, 4)[..., None] * 40
        img[m] = (np.array([30, 80, 40]) + tex)[m]
    lake = smooth_noise(rng, h, w, 120) > 0.72
    img[lake] = np.array([150, 110, 60], np.float32) + smooth_noise(rng, h, w, 10)[..., None][lake] * 25

    # roads
    for _ in range(rng.integers(2, 4)):
        p1 = (int(rng.integers(0, w)), 0)
        p2 = (int(rng.integers(0, w)), h)
        cv2.line(img, p1, p2, (120, 125, 130), int(rng.integers(6, 12)), cv2.LINE_AA)
        cv2.line(img, p1, p2, (200, 205, 205), 1, cv2.LINE_AA)

    # buildings with shadows
    for _ in range(rng.integers(15, 40)):
        bw, bh = rng.integers(10, 35, 2)
        x, y = rng.integers(0, w - 40), rng.integers(0, h - 40)
        cv2.rectangle(img, (x + 5, y + 5), (x + bw + 5, y + bh + 5), (30, 30, 30), -1)
        roof = tuple(int(v) for v in rng.integers(90, 230, 3))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), roof, -1)

    # camera effects: soft focus, vignette, sensor grain
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    yy, xx = np.mgrid[0:h, 0:w]
    vig = 1 - 0.25 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 4
    img *= vig[..., None]
    img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_aerial")
    ap.add_argument("--count", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.count):
        cv2.imwrite(os.path.join(a.out, f"aerial_{i:03d}.png"), make_image(1000 + i))
    print(f"Saved {a.count} images to {a.out}")
