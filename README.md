# AI Product Photographer

![Presentation](assets/presentation.png)

## How to set it up?
* Clone this repo
* Install the dependencies of the `requirements.txt` file
* Download the [Segment Anything Model](https://github.com/facebookresearch/segment-anything)

## How to use it?
* Run with `streamlit run main.py` (it takes some time at the first run to download the Stable Diffusion images from Hugging Face)
* Upload your product picture

![Shot1](assets/shot1.png)

* Click on the product for helping the segmentation and generate the mask

![Shot2](assets/shot2.png)

* Once the mask has been generated, write a prompt for the AI to generate the inpainting

![Shot3](assets/shot3.png)

* Obtain the final result (and iterate with the prompt if needed)

![Shot4](assets/shot4.png)
