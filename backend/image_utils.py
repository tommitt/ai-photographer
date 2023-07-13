from io import BytesIO, IOBase
from pathlib import Path
from typing import Union

import PIL.Image
import PIL.ImageDraw

from constants import IMG_OUTPUT_EXT


def format_image(
    image: Union[str, Path, IOBase, PIL.Image.Image], target_size: int
) -> PIL.Image:
    if isinstance(image, (str, Path, IOBase)):
        image = PIL.Image.open(image)

    # crop a square in the middle of the image
    width, height = image.size
    if width != height:
        square_size = min(width, height)

        left = (width - square_size) / 2
        top = (height - square_size) / 2
        right = (width + square_size) / 2
        bottom = (height + square_size) / 2

        image = image.crop((left, top, right, bottom))

    # resize the image
    if image.size[0] != target_size:
        image.thumbnail((target_size, target_size), PIL.Image.Resampling.LANCZOS)

    return image


def add_point_to_image(
    image: PIL.Image.Image,
    point: list[int],
    radius: int = 5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> PIL.Image.Image:
    image_mod = image.copy()
    draw = PIL.ImageDraw.Draw(image_mod)
    x, y = point
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)
    return image_mod


def pil_to_bytes(image: PIL.Image.Image):
    buff = BytesIO()
    image.save(buff, format=IMG_OUTPUT_EXT)
    return buff.getvalue()
