# General
IMG_SIZE: int = 512
POINT_RADIUS: int = IMG_SIZE // 50

# Sam
SAM_REGISTRY: str = "default"
SAM_CHECKPOINT: str = "sam_model/sam_vit_h_4b8939.pth"
SAM_USE_CUDA: bool = False

# Developer
DEV_MODE: bool = True
DEV_IMAGE: str = "output/image.jpg"
DEV_MASK: str = "output/mask.jpg"
DEV_SEGM: str = "output/segm.jpg"
