# Image settings
IMG_SIZE: int = 512
IMG_POINT_RADIUS: int = IMG_SIZE // 50
IMG_OUTPUT_NAME: str = "my_fantastic_product"
IMG_OUTPUT_EXT: str = "jpeg"  # jpeg, png

# Segment Anything
SAM_REGISTRY: str = "default"
SAM_CHECKPOINT: str = "sam_model/sam_vit_h_4b8939.pth"
SAM_USE_CUDA: bool = False

# Stable Diffusion
SD_INPAINTING_MODEL: str = "runwayml/stable-diffusion-inpainting"
SD_CONTROLNET_MODEL: str = "lllyasviel/sd-controlnet-seg"
SD_NUM_INFERENCE_STEPS: int = 20
SD_USE_CUDA: bool = False

# Developer
DEV_MODE: bool = True
DEV_IMAGE: str = "output/image.jpg"
DEV_MASK: str = "output/mask.jpg"
DEV_SEGM: str = "output/segm.jpg"
DEV_OUT: str = "output/out.jpg"
