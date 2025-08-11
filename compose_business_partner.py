#!/usr/bin/env python3
import argparse
import io
import os
import sys
import math
import requests
from typing import Optional, Tuple, List
from PIL import Image, ImageEnhance
import numpy as np

# Lazy import cv2 and rembg to speed up help/arg parsing

def _lazy_imports():
    global cv2, remove
    import cv2  # type: ignore
    from rembg import remove  # type: ignore


STOCK_URLS: List[str] = [
    # Pexels direct image links (royalty-free). Multiple fallbacks included.
    "https://images.pexels.com/photos/1181686/pexels-photo-1181686.jpeg",
    "https://images.pexels.com/photos/4065896/pexels-photo-4065896.jpeg",
    "https://images.pexels.com/photos/3184405/pexels-photo-3184405.jpeg",
    "https://images.pexels.com/photos/5452201/pexels-photo-5452201.jpeg",
    "https://images.pexels.com/photos/3783725/pexels-photo-3783725.jpeg",
    "https://images.pexels.com/photos/2422293/pexels-photo-2422293.jpeg",
]


def download_first_available(urls: List[str], timeout: int = 20) -> Image.Image:
    last_error = None
    for url in urls:
        try:
            # Request a reasonably large image
            # Pexels supports query params like w=, h= to resize server-side
            if "images.pexels.com" in url and "?" not in url:
                url = url + "?auto=compress&cs=tinysrgb&w=1600"
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            return img
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(f"Failed to download any stock image. Last error: {last_error}")


def remove_background(pil_img: Image.Image) -> Image.Image:
    _lazy_imports()
    # Convert to bytes, then through rembg for robust results
    with io.BytesIO() as buf:
        pil_img.convert("RGBA").save(buf, format="PNG")
        data = buf.getvalue()
    cutout_bytes = remove(data)
    cutout_img = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
    return cutout_img


def detect_primary_face_bbox(pil_img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    _lazy_imports()
    # Use OpenCV Haar cascade
    cv2_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cv2_path)

    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Choose the largest face (by area)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


def compute_luminance(pil_img: Image.Image) -> float:
    rgb = pil_img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    # Perceived luminance approximation
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return float(luma.mean())


def match_brightness(src_img: Image.Image, target_img: Image.Image) -> Image.Image:
    # Adjust src_img brightness to roughly match target_img
    src_l = compute_luminance(src_img)
    tgt_l = compute_luminance(target_img)
    if src_l <= 0.0:
        return src_img
    factor = max(0.5, min(1.7, (tgt_l + 1e-4) / (src_l + 1e-4)))
    return ImageEnhance.Brightness(src_img).enhance(factor)


def place_right_shoulder_to_shoulder(
    base_img: Image.Image,
    person_rgba: Image.Image,
    base_face: Optional[Tuple[int, int, int, int]],
    person_face: Optional[Tuple[int, int, int, int]],
) -> Tuple[Image.Image, Tuple[int, int]]:
    base_w, base_h = base_img.size
    person = person_rgba

    # Determine target face height
    if base_face is not None and person_face is not None:
        _, _, base_fw, base_fh = base_face
        _, _, per_fw, per_fh = person_face
        if per_fh > 0:
            scale = (base_fh / per_fh) * 1.02  # small bias to look slightly closer
        else:
            scale = 1.0
    else:
        # Fallback: scale relative to image height
        scale = (0.78 * base_h) / max(1, person.height)

    new_w = max(1, int(person.width * scale))
    new_h = max(1, int(person.height * scale))
    person_resized = person.resize((new_w, new_h), Image.LANCZOS)

    # Compute y such that faces align vertically if available; else align mid-height
    if base_face is not None and person_face is not None:
        bx, by, bw, bh = base_face
        _, py, _, ph = person_face
        base_face_center_y = by + bh * 0.5
        per_face_center_y_scaled = py * (new_h / person.height) + (ph * (new_h / person.height)) * 0.5
        y = int(base_face_center_y - per_face_center_y_scaled)
    else:
        y = int((base_h - new_h) * 0.35)

    # Compute x to the right of the base subject face if available; else 60% width
    margin = int(0.03 * base_w)
    if base_face is not None:
        bx, by, bw, bh = base_face
        x = bx + bw + margin
    else:
        x = int(base_w * 0.58)

    # If overflow, shift left; if still overflow, downscale to fit
    if x + new_w > base_w - margin:
        overflow = x + new_w - (base_w - margin)
        x -= overflow
        if x < margin:
            # Need to downscale to fit
            fit_w = (base_w - margin) - margin
            scale2 = fit_w / new_w
            new_w2 = max(1, int(new_w * scale2))
            new_h2 = max(1, int(new_h * scale2))
            person_resized = person_resized.resize((new_w2, new_h2), Image.LANCZOS)
            new_w, new_h = new_w2, new_h2
            x = base_w - margin - new_w

    y = max(-int(0.05 * base_h), min(y, base_h - new_h + int(0.05 * base_h)))

    composed = base_img.convert("RGBA").copy()
    composed.alpha_composite(person_resized, (x, y))
    return composed.convert("RGB"), (x, y)


def main():
    parser = argparse.ArgumentParser(description="Compose a business woman to the right, shoulder-to-shoulder.")
    parser.add_argument("--input", default="/workspace/input.jpg", help="Path to the base image")
    parser.add_argument("--output", default="/workspace/output.jpg", help="Path to save the composed image")
    parser.add_argument("--debug", action="store_true", help="Save intermediate images for debugging")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input image not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    base = Image.open(args.input).convert("RGB")

    # Download and prepare stock image
    stock = download_first_available(STOCK_URLS)
    stock = match_brightness(stock, base)
    cutout = remove_background(stock)

    # Detect faces
    try:
        base_face = detect_primary_face_bbox(base)
    except Exception:
        base_face = None
    try:
        person_face = detect_primary_face_bbox(cutout)
    except Exception:
        person_face = None

    composed, pos = place_right_shoulder_to_shoulder(base, cutout, base_face, person_face)

    # Ensure JPEG quality
    composed.save(args.output, quality=95)

    if args.debug:
        dbg_dir = "/workspace/debug"
        os.makedirs(dbg_dir, exist_ok=True)
        base.save(os.path.join(dbg_dir, "base.jpg"))
        stock.save(os.path.join(dbg_dir, "stock_rgba.png"))
        cutout.save(os.path.join(dbg_dir, "cutout.png"))
        composed.save(os.path.join(dbg_dir, "composed.jpg"))
        # Visualize face boxes if any
        def draw_box(img: Image.Image, box: Optional[Tuple[int,int,int,int]], color=(255,0,0)) -> Image.Image:
            from PIL import ImageDraw
            vis = img.convert("RGB").copy()
            if box is not None:
                x, y, w, h = box
                d = ImageDraw.Draw(vis)
                d.rectangle([x, y, x+w, y+h], outline=color, width=3)
            return vis
        try:
            draw_box(base, base_face, (255,0,0)).save(os.path.join(dbg_dir, "base_face.jpg"))
        except Exception:
            pass
        try:
            draw_box(cutout, person_face, (0,255,0)).save(os.path.join(dbg_dir, "person_face.png"))
        except Exception:
            pass

    print(f"Done. Saved: {args.output}")


if __name__ == "__main__":
    main()