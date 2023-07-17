from colorsys import hsv_to_rgb
from typing import Tuple

import numpy as np
import PIL.Image
import torch.cuda
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from segment_anything.modeling import Sam

from backend.image_utils import bg_image_w_black_mask
from constants import SAM_CHECKPOINT, SAM_REGISTRY, SAM_USE_CUDA


def init_sam() -> Sam:
    sam = sam_model_registry[SAM_REGISTRY](checkpoint=SAM_CHECKPOINT)
    if torch.cuda.is_available() and SAM_USE_CUDA:
        sam.to(device="cuda")
    return sam


def generate_mask(
    sam: Sam,
    image: PIL.Image.Image,
    points: list[list[int]],
    inpaint_background: bool = True,
) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
    image_array = np.array(image)

    # image mask
    sam_predictor = SamPredictor(sam)

    sam_predictor.set_image(image_array)
    mask_array, _, _ = sam_predictor.predict(
        point_coords=np.array(points),
        point_labels=np.ones(len(points), dtype=int),
        multimask_output=False,
    )

    if inpaint_background:
        mask_array = np.logical_not(mask_array)

    mask_image = PIL.Image.fromarray(mask_array[0, :, :])
    if SAM_USE_CUDA:
        torch.cuda.empty_cache()

    # image segmentation
    sam_auto_generator = SamAutomaticMaskGenerator(sam)

    bg_image = bg_image_w_black_mask(image, mask_image)
    bool_masks_out = sam_auto_generator.generate(np.array(bg_image))

    bool_masks = [s["segmentation"] for s in bool_masks_out]
    segm_array = np.zeros(
        (bool_masks[0].shape[0], bool_masks[0].shape[1], 3), dtype=np.uint8
    )
    # assign a unique color to each mask
    for class_id, bool_mask in enumerate(bool_masks):
        hue = float(class_id) / len(bool_masks)
        rgb = tuple(int(i * 255) for i in hsv_to_rgb(hue, 1, 1))
        rgb_mask = np.zeros((bool_mask.shape[0], bool_mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[:, :, 0] = bool_mask * rgb[0]
        rgb_mask[:, :, 1] = bool_mask * rgb[1]
        rgb_mask[:, :, 2] = bool_mask * rgb[2]
        segm_array += rgb_mask

    segm_image = PIL.Image.fromarray(segm_array)
    if SAM_USE_CUDA:
        torch.cuda.empty_cache()

    return mask_image, segm_image
