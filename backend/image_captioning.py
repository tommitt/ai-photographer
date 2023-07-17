import PIL.Image
from transformers import ImageToTextPipeline, pipeline

from constants import CAPTION_MAX_TOKENS, CAPTION_MODEL


def init_image_captioning() -> ImageToTextPipeline:
    pipe = pipeline(
        "image-to-text",
        model=CAPTION_MODEL,
        max_new_tokens=CAPTION_MAX_TOKENS,
    )
    return pipe


def generate_caption(pipe: ImageToTextPipeline, image: PIL.Image.Image) -> str:
    out = pipe(image)
    return out[0]["generated_text"].strip()
