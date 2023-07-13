import PIL.Image
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)

from constants import (
    SD_CONTROLNET_MODEL,
    SD_INPAINTING_MODEL,
    SD_NUM_INFERENCE_STEPS,
    SD_USE_CUDA,
)


def init_sd() -> StableDiffusionControlNetInpaintPipeline:
    controlnet = ControlNetModel.from_pretrained(
        SD_CONTROLNET_MODEL,
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        SD_INPAINTING_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    if SD_USE_CUDA:
        pipe = pipe.to("cuda")

    return pipe


def generate_inpainting(
    pipe: StableDiffusionControlNetInpaintPipeline,
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
    segm: PIL.Image.Image,
    pos_prompt: str,
    neg_prompt: str,
) -> PIL.Image.Image:
    output = pipe(
        pos_prompt,
        image,
        mask,
        segm,
        negative_prompt=neg_prompt,
        num_inference_steps=SD_NUM_INFERENCE_STEPS,
    )

    if SD_USE_CUDA:
        torch.cuda.empty_cache()

    return output.images[0]
