from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.core.config import Settings
from app.core.errors import AppError, corrupted_document

# Pillow's built-in decompression-bomb check is replaced by the explicit max_image_pixels check below.
Image.MAX_IMAGE_PIXELS = None

SIXTEEN_BIT_MODES = ("I;16", "I;16L", "I;16B")


def open_image(path: Path, settings: Settings) -> Image.Image:
    """Opens an image after checking it parses and is within the pixel limit. Caller closes it."""
    try:
        with Image.open(path) as probe:
            probe.verify()
        image = Image.open(path)
    except Exception as error:
        raise corrupted_document("The image file is corrupted or cannot be read.") from error
    _check_pixels(image, settings)
    return image


def frame_count(image: Image.Image) -> int:
    return getattr(image, "n_frames", 1)


def load_frame(image: Image.Image, index: int, settings: Settings) -> Image.Image:
    """Returns frame `index` as an upright RGB image no larger than max_image_side."""
    try:
        image.seek(index)
        _check_pixels(image, settings)
        frame = ImageOps.exif_transpose(image)
        if frame.mode in SIXTEEN_BIT_MODES:
            # Pillow clips 16-bit values to 255 when converting; scale them to 8 bits instead.
            frame = Image.fromarray((np.asarray(frame) // 256).astype(np.uint8))
        if frame.mode in ("RGBA", "LA") or (frame.mode == "P" and "transparency" in frame.info):
            rgba = frame.convert("RGBA")
            frame = Image.new("RGB", rgba.size, "white")
            frame.paste(rgba, mask=rgba.getchannel("A"))
        else:
            frame = frame.convert("RGB")
    except AppError:
        raise
    except Exception as error:
        raise corrupted_document("The image file is corrupted or cannot be read.") from error

    longest_side = max(frame.size)
    if longest_side > settings.max_image_side:
        ratio = settings.max_image_side / longest_side
        frame = frame.resize((round(frame.width * ratio), round(frame.height * ratio)), Image.LANCZOS)
    return frame


def _check_pixels(image: Image.Image, settings: Settings) -> None:
    if image.width * image.height > settings.max_image_pixels:
        raise AppError(
            "IMAGE_TOO_LARGE",
            f"Image has more than {settings.max_image_pixels:,} pixels, which is the configured limit.",
            413,
        )
