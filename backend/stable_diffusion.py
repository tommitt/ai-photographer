import PIL.Image
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)

from constants import (
    SD_CONDITIONING_SCALE,
    SD_CONTROLNET_MODEL,
    SD_INPAINTING_MODEL,
    SD_NUM_INFERENCE_STEPS,
    SD_USE_CUDA,
)


def init_sd() -> StableDiffusionControlNetInpaintPipeline:
    # TensorFloat32 mode for faster but slightly less accurate computations
    torch.backends.cuda.matmul.allow_tf32 = True

    controlnet = ControlNetModel.from_pretrained(
        SD_CONTROLNET_MODEL,
        torch_dtype=torch.float32,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        SD_INPAINTING_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float32,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if SD_USE_CUDA:
        pipe = pipe.to("cuda")
    else:
        # for minimal memory consumption
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing(1)

    pipe.enable_xformers_memory_efficient_attention()

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
        image=image,
        mask_image=mask,
        control_image=segm,
        prompt=pos_prompt,
        negative_prompt=None if neg_prompt == "" else neg_prompt,
        num_inference_steps=SD_NUM_INFERENCE_STEPS,
        controlnet_conditioning_scale=SD_CONDITIONING_SCALE,
    )

    torch.cuda.empty_cache()

    return output.images[0]
