"""Task 4.2 - upload validation shared by the web app and the model API.

Three independent checks, because each one alone can be fooled:
  1. extension      - cheap first filter (a .exe or .pdf is rejected immediately)
  2. size limit     - enforced by Flask's MAX_CONTENT_LENGTH before the body is read
  3. real decoding  - the bytes must actually decode as an image in an allowed format,
                      so a renamed text file or a truncated JPEG is caught
"""
import io
import warnings

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

import visionguard.preprocess  # noqa: F401  (sets Image.MAX_IMAGE_PIXELS)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "MPO"}  # MPO = multi-picture JPEG from phones


class UploadError(ValueError):
    def __init__(self, title: str, message: str, status: int = 400):
        super().__init__(message)
        self.title, self.message, self.status = title, message, status


def clean_filename(name: str) -> str:
    """'../../etc/passwd' -> 'etc_passwd'; never trust a client-supplied file name."""
    return (secure_filename(name or "") or "upload")[:200]


def check_extension(filename: str):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError("Unsupported file type",
                          f"'.{ext or '?'}' files are not accepted. Upload one of: "
                          + ", ".join(sorted(ALLOWED_EXTENSIONS)) + ".", 415)


def check_decodable(data: bytes):
    if not data:
        raise UploadError("Empty file", "The uploaded file is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(io.BytesIO(data)) as im:
                fmt = im.format
                im.verify()                       # structure check
            with Image.open(io.BytesIO(data)) as im:
                im.load()                         # full decode: catches truncated files
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise UploadError("Image could not be read",
                          "The file has an image extension but its contents are not a valid, "
                          "complete image. It may be corrupted or renamed from another file type.", 422)
    if fmt not in ALLOWED_FORMATS:
        raise UploadError("Unsupported image format", f"{fmt} images are not supported.", 415)
