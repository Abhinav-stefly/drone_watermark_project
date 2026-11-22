# Secure Drone Image Authentication using Digital Watermarking

Python project: hides a secret, signed, invisible watermark inside drone images and later
checks that an image is genuine, unmodified and not replayed.

## Quick start
```bash
pip install -r requirements.txt
python generate_dataset.py --count 20          # makes data/synthetic_aerial (or use a real dataset)
python evaluate.py --data data/synthetic_aerial --out results      # quality, speed, robustness
python security_tests.py --data data/synthetic_aerial --out results # forgery / removal / replay tests

python main.py keygen --out secret.key
python main.py embed  --in my_drone_photo.jpg --out my_photo_wm.png --key secret.key --drone-id 7
python main.py verify --in my_photo_wm.png --key secret.key --replay-db seen.json --heatmap map.png
```

## Using a real drone dataset
Put any `.jpg/.png` aerial images in a folder and pass it with `--data folder`.
Suggested public datasets: Semantic Drone Dataset (TU Graz), VisDrone, UAVid, UC Merced Land Use, AID.
Images should be at least about 256x256 pixels. Save watermarked images as PNG (or JPEG quality >= 30).

## Files
| File | Purpose |
|---|---|
| `dwm/payload.py` | Builds the 208-bit signed payload (drone id, time, nonce, image hash, HMAC tag) |
| `dwm/watermark.py` | DWT + DCT + keyed QIM embedding and blind extraction |
| `dwm/authenticator.py` | `sign()` and `verify()`; replay store; tamper map |
| `dwm/attacks.py` | All attacks used in the tests |
| `dwm/metrics.py` | PSNR, SSIM, BER |
| `evaluate.py` | Performance evaluation |
| `security_tests.py` | Security verification |
| `main.py` | Command line tool |
| `generate_dataset.py` | Synthetic aerial dataset generator |
