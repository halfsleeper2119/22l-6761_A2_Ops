"""Image loading shared by training AND serving.

Training (visionguard/data.py) and the model API both call `load_for_model`, so a file
becomes pixels in exactly one way. Pixel scaling (e.g. to [-1, 1]) is NOT done here: it is
the first layer inside each saved model, so the model file carries its own preprocessing.
"""
import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

# Allow large camera photos, but still refuse absurd "decompression bomb" files.
Image.MAX_IMAGE_PIXELS = 300_000_000


def to_rgb(im: Image.Image) -> Image.Image:
    """Return a 3-channel RGB copy of any PIL image.

    - Animated GIF/WebP: the first frame is used.
    - EXIF orientation is applied, so phone photos are upright.
    - Transparent pixels are composited onto white (a plain convert('RGB')
      would turn them black, which is an artefact the model could learn).
    """
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    im = ImageOps.exif_transpose(im)
    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    if has_alpha:
        im = im.convert("RGBA")
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.getchannel("A"))
        return background
    return im.convert("RGB")


def open_rgb(source: Union[str, Path, bytes]) -> Image.Image:
    """Open a path or raw bytes and return a fully decoded RGB image."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as im:
        im.load()
        return to_rgb(im)


def load_for_model(source: Union[str, Path, bytes], img_size: int) -> np.ndarray:
    """File/bytes -> float32 array of shape (img_size, img_size, 3) with values 0..255.

    The image is resized straight to a square ("squashed"), so the aspect ratio is NOT kept.
    We avoid letterbox padding on purpose: in this dataset AI images are mostly square and
    real photos are not, so padding bars would appear almost only on real photos and give
    the model a shortcut that has nothing to do with image content.
    """
    im = open_rgb(source).resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32)
