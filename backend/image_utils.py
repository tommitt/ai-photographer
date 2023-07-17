from io import BytesIO, IOBase
from pathlib import Path
from typing import Union

import numpy as np
import PIL.Image
import PIL.ImageDraw


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


def pil_to_bytes(image: PIL.Image.Image, ext: str) -> bytes:
    buff = BytesIO()
    image.save(buff, format=ext)
    return buff.getvalue()


def masked_image_w_white_bg(
    image: PIL.Image.Image, mask: PIL.Image.Image
) -> PIL.Image.Image:
    image_array = np.array(image)
    mask_array = np.array(mask)[:, :, np.newaxis]
    return PIL.Image.fromarray(np.where((mask_array == 0), image_array, 255))


def bg_image_w_black_mask(
    image: PIL.Image.Image, mask: PIL.Image.Image
) -> PIL.Image.Image:
    image_array = np.array(image)
    mask_array = np.array(mask)[:, :, np.newaxis]
    return PIL.Image.fromarray(np.where((mask_array == 0), 0, image_array))
