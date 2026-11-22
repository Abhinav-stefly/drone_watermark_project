"""
Performance evaluation: embedding/extraction time, PSNR, SSIM and robustness
against compression, noise, rotation, cropping, scaling and blur.

    python evaluate.py --data data/synthetic_aerial --out results
"""
import argparse
import glob
import os
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dwm import DroneImageAuthenticator
from dwm import attacks as A
from dwm.metrics import ber, psnr, ssim
from dwm.payload import PAYLOAD_BITS

ATTACKS = {
    "JPEG Q90": lambda x: A.jpeg(x, 90), "JPEG Q70": lambda x: A.jpeg(x, 70),
    "JPEG Q50": lambda x: A.jpeg(x, 50), "JPEG Q30": lambda x: A.jpeg(x, 30),
    "Noise s=2": lambda x: A.gaussian_noise(x, 2), "Noise s=5": lambda x: A.gaussian_noise(x, 5),
    "Noise s=10": lambda x: A.gaussian_noise(x, 10),
    "Rotate 1deg": lambda x: A.rotate(x, 1), "Rotate 5deg": lambda x: A.rotate(x, 5),
    "Rotate 15deg": lambda x: A.rotate(x, 15),
    "Crop 10%": lambda x: A.crop_out(x, 0.10), "Crop 25%": lambda x: A.crop_out(x, 0.25),
    "Crop 50%": lambda x: A.crop_out(x, 0.50),
    "Scale 0.75": lambda x: A.scale(x, 0.75), "Scale 0.5": lambda x: A.scale(x, 0.5),
    "Blur 3x3": lambda x: A.blur(x, 3), "Blur 5x5": lambda x: A.blur(x, 5),
}


def main(data, out, key=b"demo-secret-key", delta=32.0):
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data, "*.png")) + glob.glob(os.path.join(data, "*.jpg")) +
                   glob.glob(os.path.join(data, "*.jpeg")))
    if not files:
        raise SystemExit(f"No images found in {data}")
    auth = DroneImageAuthenticator(key, delta=delta)

    quality = []   # per image: psnr, ssim, t_embed, t_extract
    robust = {k: {"ok": [], "bit_ber": [], "status": []} for k in ATTACKS}
    first_example = None
    for i, f in enumerate(files):
        img = cv2.imread(f)
        t0 = time.perf_counter()
        marked, payload = auth.sign(img, drone_id=7)
        t_embed = time.perf_counter() - t0
        true_bits = payload.to_bits(auth.mac_key)

        t0 = time.perf_counter()
        res = auth.verify(marked, check_replay=False)
        t_ext = time.perf_counter() - t0
        quality.append((psnr(img, marked), ssim(img, marked), t_embed, t_ext, res.tag_valid))
        if first_example is None:
            first_example = (img, marked)

        for name, fn in ATTACKS.items():
            att = fn(marked)
            bits = auth.wm.extract(att, PAYLOAD_BITS)[0]
            r = auth.verify(att, check_replay=False)
            robust[name]["ok"].append(r.tag_valid)
            robust[name]["bit_ber"].append(ber(bits, true_bits))
            robust[name]["status"].append(r.status)

    q = np.array(quality, dtype=float)
    lines = ["# Performance results", "",
             f"Images: {len(files)}  |  delta (embedding strength): {delta}", "",
             "## Quality and speed (no attack)", "",
             "| Metric | Mean | Min | Max |", "|---|---|---|---|"]
    for j, name in enumerate(["PSNR (dB)", "SSIM", "Embedding time (s)", "Extraction + verification time (s)"]):
        lines.append(f"| {name} | {q[:, j].mean():.4f} | {q[:, j].min():.4f} | {q[:, j].max():.4f} |")
    lines.append(f"| Verification success (no attack) | {q[:, 4].mean() * 100:.0f}% | | |")
    lines += ["", "## Robustness", "",
              "| Attack | Authentication success | Payload BER (mean) | Typical status |", "|---|---|---|---|"]
    rates = []
    for name, d in robust.items():
        rate = 100 * np.mean(d["ok"])
        rates.append(rate)
        common = max(set(d["status"]), key=d["status"].count)
        lines.append(f"| {name} | {rate:.0f}% | {np.mean(d['bit_ber']):.3f} | {common} |")
    open(os.path.join(out, "performance_results.md"), "w").write("\n".join(lines) + "\n")

    # ---- CSV
    with open(os.path.join(out, "robustness.csv"), "w") as fh:
        fh.write("attack,success_percent,mean_payload_ber\n")
        for (name, d), r in zip(robust.items(), rates):
            fh.write(f"{name},{r:.1f},{np.mean(d['bit_ber']):.4f}\n")

    # ---- charts
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(list(robust), rates, color=["#2a9d8f" if r >= 95 else "#e9c46a" if r >= 50 else "#e76f51" for r in rates])
    ax.set_ylabel("Authentication success (%)"); ax.set_ylim(0, 105)
    ax.set_title("Robustness of the watermark against attacks"); plt.xticks(rotation=60, ha="right")
    plt.tight_layout(); plt.savefig(os.path.join(out, "robustness.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(q[:, 0], "o-"); ax[0].set_title("PSNR per image (dB)"); ax[0].set_xlabel("image")
    ax[1].plot(q[:, 1], "o-", color="#e76f51"); ax[1].set_title("SSIM per image"); ax[1].set_xlabel("image")
    plt.tight_layout(); plt.savefig(os.path.join(out, "quality.png"), dpi=140); plt.close()

    img, marked = first_example
    diff = np.abs(img.astype(int) - marked.astype(int)).sum(axis=2)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].imshow(img[..., ::-1]); ax[0].set_title("Original")
    ax[1].imshow(marked[..., ::-1]); ax[1].set_title("Watermarked")
    im = ax[2].imshow(diff * 10, cmap="magma", vmax=255); ax[2].set_title("Difference x10")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(out, "example.png"), dpi=140); plt.close()
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_aerial")
    ap.add_argument("--out", default="results")
    ap.add_argument("--delta", type=float, default=32.0)
    a = ap.parse_args()
    main(a.data, a.out, delta=a.delta)
