"""Image loading shared by training AND serving.

Both sides must turn a file into pixels in exactly the same way, so this is the only
place that does it. Task 2 adds resizing here; the Flask/model API imports the same code.
"""
import io
from pathlib import Path
from typing import Union

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
