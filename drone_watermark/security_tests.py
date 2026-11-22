"""
Security verification: forgery, watermark removal, unauthorised modification, replay,
and watermark-copy (transplant) attacks.

    python security_tests.py --data data/synthetic_aerial --out results
"""
import argparse
import glob
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dwm import DroneImageAuthenticator
from dwm import attacks as A
from dwm.authenticator import AUTHENTIC, AUTHENTIC_MODIFIED, NOT_AUTHENTIC, REPLAY, STALE
from dwm.metrics import psnr

KEY = b"demo-secret-key"


def run(data, out):
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data, "*.png")) + glob.glob(os.path.join(data, "*.jpg")))
    n = min(len(files), 10)
    imgs = [cv2.imread(f) for f in files[:n + 1]]
    rows = []   # (attack, detail, expected, got_counter, pass)

    def record(name, expected, outcomes, extra=""):
        ok = sum(o in expected for o in outcomes)
        rows.append((name, " / ".join(expected), f"{ok}/{len(outcomes)}", extra))
        return ok == len(outcomes)

    # --- 1. Forgery: image with no watermark / watermark made with a wrong key
    outs1, outs2 = [], []
    attacker = DroneImageAuthenticator(b"attacker-guess")
    for i in range(n):
        genuine = DroneImageAuthenticator(KEY)
        outs1.append(genuine.verify(imgs[i], check_replay=False).status)
        fake, _ = attacker.sign(imgs[i], drone_id=7)          # attacker signs with own key
        outs2.append(genuine.verify(fake, check_replay=False).status)
    record("Forgery: image without watermark", [NOT_AUTHENTIC], outs1)
    record("Forgery: watermark made with wrong key", [NOT_AUTHENTIC], outs2)

    # --- 2. Watermark removal attempts
    removal = {"Removal: denoise + JPEG Q40": A.remove_watermark_denoise,
               "Removal: coarse re-quantisation (step 24)": A.remove_watermark_requantise,
               "Removal: strong blur 9x9 + noise": lambda x: A.gaussian_noise(A.blur(x, 9), 8)}
    for name, fn in removal.items():
        outs, q = [], []
        for i in range(n):
            auth = DroneImageAuthenticator(KEY)
            marked, _ = auth.sign(imgs[i], 7)
            att = fn(marked)
            outs.append(auth.verify(att, check_replay=False).status)
            q.append(psnr(marked, att))
        # A removal attack "succeeds" for the attacker if status becomes NOT_AUTHENTIC while the image is still usable
        survived = sum(o != NOT_AUTHENTIC for o in outs)
        rows.append((name, "if watermark is wiped, image is rejected (NOT_AUTHENTIC)",
                     f"watermark survived {survived}/{n}", f"image PSNR after attack {np.mean(q):.1f} dB"))

    # --- 3. Unauthorised modification (splice, copy-move, small object)
    mods = {
        "Modification: splice 120x160 region from other image": lambda x, o: A.splice(x, o, (150, 270, 200, 360)),
        "Modification: copy-move 100x100 patch": lambda x, o: A.copy_move(x, (40, 40), (300, 400), (100, 100)),
        "Modification: small insertion 40x40 px": lambda x, o: A.splice(x, o, (100, 140, 200, 240)),
    }
    example = None
    for name, fn in mods.items():
        outs = []
        for i in range(n):
            auth = DroneImageAuthenticator(KEY)
            marked, _ = auth.sign(imgs[i], 7)
            att = fn(marked, imgs[n])
            r = auth.verify(att, check_replay=False)
            outs.append(r.status)
            if example is None and name.startswith("Modification: splice"):
                example = (att, r.tamper_map)
        record(name, [AUTHENTIC_MODIFIED, NOT_AUTHENTIC], outs)

    # --- 4. Replay attacks
    outs_first, outs_second, outs_stale = [], [], []
    for i in range(n):
        auth = DroneImageAuthenticator(KEY, max_age_seconds=600)
        marked, _ = auth.sign(imgs[i], 7, timestamp=1_800_000_000)
        outs_first.append(auth.verify(marked, now=1_800_000_005).status)     # genuine first delivery
        outs_second.append(auth.verify(marked, now=1_800_000_020).status)    # attacker re-sends same file
        old, _ = auth.sign(imgs[i], 7, timestamp=1_800_000_000 - 7200)       # captured 2 h ago, sent now
        outs_stale.append(auth.verify(old, now=1_800_000_000).status)
    record("Replay: first (genuine) delivery is accepted", [AUTHENTIC], outs_first)
    record("Replay: same image sent again", [REPLAY], outs_second)
    record("Replay: old image with expired timestamp", [STALE], outs_stale)

    # --- 5. Watermark copy (transplant) attack
    outs = []
    for i in range(n):
        auth = DroneImageAuthenticator(KEY)
        marked, _ = auth.sign(imgs[i], 7)
        fake = A.transplant(imgs[n], marked, imgs[i])
        outs.append(auth.verify(fake, check_replay=False).status)
    record("Copy attack: transplant watermark onto another image", [NOT_AUTHENTIC], outs)

    # ---- write report
    md = ["# Security verification results", "", f"Each test run on {n} images.", "",
          "| Attack | Expected / interpretation | Result | Note |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    open(os.path.join(out, "security_results.md"), "w").write("\n".join(md) + "\n")

    if example:
        att, tmap = example
        h, w = att.shape[:2]
        heat = cv2.resize(tmap.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(att[..., ::-1]); ax[0].set_title("Image received (region spliced in)")
        ax[1].imshow(att[..., ::-1]); ax[1].imshow(heat, cmap="Reds", alpha=0.55, vmin=0, vmax=1)
        ax[1].set_title("Tamper map (red = damaged watermark)")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(out, "tamper_map.png"), dpi=140); plt.close()
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_aerial")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run(a.data, a.out)
