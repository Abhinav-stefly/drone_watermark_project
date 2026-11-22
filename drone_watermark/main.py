"""
Command line tool.

  python main.py keygen   --out secret.key
  python main.py embed    --in photo.jpg --out photo_wm.png --key secret.key --drone-id 7
  python main.py verify   --in photo_wm.png --key secret.key [--heatmap map.png]
  python main.py embed-folder --in data/synthetic_aerial --out data/watermarked --key secret.key --drone-id 7

Save watermarked images as PNG (lossless). If you must use JPEG, use quality >= 70.
"""
import argparse
import glob
import os
import secrets

import cv2
import numpy as np

from dwm import DroneImageAuthenticator
from dwm.authenticator import ReplayStore


def load_key(path):
    return open(path, "rb").read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("keygen"); k.add_argument("--out", required=True)
    e = sub.add_parser("embed"); e.add_argument("--in", dest="inp", required=True); e.add_argument("--out", required=True)
    e.add_argument("--key", required=True); e.add_argument("--drone-id", type=int, default=1)
    f = sub.add_parser("embed-folder"); f.add_argument("--in", dest="inp", required=True); f.add_argument("--out", required=True)
    f.add_argument("--key", required=True); f.add_argument("--drone-id", type=int, default=1)
    v = sub.add_parser("verify"); v.add_argument("--in", dest="inp", required=True); v.add_argument("--key", required=True)
    v.add_argument("--replay-db", default=None, help="JSON file remembering accepted images (replay protection)")
    v.add_argument("--heatmap", default=None, help="save a tamper heat-map image here")
    a = ap.parse_args()

    if a.cmd == "keygen":
        open(a.out, "wb").write(secrets.token_bytes(32))
        print(f"Secret key written to {a.out}  (keep it private!)")
    elif a.cmd == "embed":
        auth = DroneImageAuthenticator(load_key(a.key))
        img = cv2.imread(a.inp)
        marked, p = auth.sign(img, a.drone_id)
        cv2.imwrite(a.out, marked)
        print(f"Saved {a.out}  (drone {p.drone_id}, nonce {p.nonce})")
    elif a.cmd == "embed-folder":
        auth = DroneImageAuthenticator(load_key(a.key))
        os.makedirs(a.out, exist_ok=True)
        for fp in sorted(glob.glob(os.path.join(a.inp, "*"))):
            img = cv2.imread(fp)
            if img is None:
                continue
            marked, _ = auth.sign(img, a.drone_id)
            cv2.imwrite(os.path.join(a.out, os.path.splitext(os.path.basename(fp))[0] + ".png"), marked)
        print(f"Done. Watermarked images are in {a.out}")
    elif a.cmd == "verify":
        auth = DroneImageAuthenticator(load_key(a.key), replay_store=ReplayStore(a.replay_db))
        img = cv2.imread(a.inp)
        r = auth.verify(img, check_replay=a.replay_db is not None)
        print(f"Status      : {r.status}")
        print(f"Tag valid   : {r.tag_valid}")
        if r.tag_valid:
            print(f"Drone ID    : {r.drone_id}\nTimestamp   : {r.timestamp}\nNonce       : {r.nonce}")
            print(f"Hash dist.  : {r.hash_distance}")
        print(f"Block errors: {r.block_error_rate:.3f}")
        if r.notes:
            print(f"Notes       : {r.notes}")
        if a.heatmap and r.tamper_map is not None:
            h, w = img.shape[:2]
            heat = cv2.resize((r.tamper_map * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = cv2.addWeighted(img, 0.6, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.4, 0)
            cv2.imwrite(a.heatmap, overlay)
            print(f"Heat-map saved to {a.heatmap}")


if __name__ == "__main__":
    main()
