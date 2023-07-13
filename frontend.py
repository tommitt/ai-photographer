import PIL.Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from backend.image_utils import add_point_to_image, format_image, pil_to_bytes
from backend.sam import generate_mask, init_sam
from backend.stable_diffusion import generate_inpainting, init_sd
from constants import (
    DEV_IMAGE,
    DEV_MASK,
    DEV_MODE,
    DEV_OUT,
    DEV_SEGM,
    IMG_OUTPUT_EXT,
    IMG_OUTPUT_NAME,
    IMG_POINT_RADIUS,
    IMG_SIZE,
)


def app():
    st.set_page_config(page_icon="📷", layout="centered")
    st.title("AI Product Photographer 📷")

    if "init_app" not in st.session_state:
        st.session_state["init_app"] = True
        st.session_state["points"] = []
        st.session_state["point_last_clicked"] = []

        if DEV_MODE:
            st.session_state["image_original"] = format_image(DEV_IMAGE, IMG_SIZE)
            st.session_state["image_displayed"] = st.session_state["image_original"]
            st.session_state["mask"] = PIL.Image.open(DEV_MASK)
            st.session_state["segm"] = PIL.Image.open(DEV_SEGM)
            st.session_state["output"] = PIL.Image.open(DEV_OUT)

    if "image_original" not in st.session_state:
        uploaded_picture = st.file_uploader(
            "Choose a picture", type=["png", "jpg", "jpeg"]
        )
        st.caption("Note, the picture is automatically cropped into a square")
        if uploaded_picture:
            st.session_state["image_original"] = format_image(
                uploaded_picture, IMG_SIZE
            )
            st.session_state["image_displayed"] = st.session_state["image_original"]
            st.experimental_rerun()
    else:
        st.write("Click on the product to show me where it is...")
        clicked_point = streamlit_image_coordinates(st.session_state["image_displayed"])
        if clicked_point is not None:
            xy = list(clicked_point.values())
            if (
                xy not in st.session_state["points"]
                and xy != st.session_state["point_last_clicked"]
            ):
                st.session_state["points"].append(xy)
                st.session_state["image_displayed"] = add_point_to_image(
                    st.session_state["image_displayed"], xy, radius=IMG_POINT_RADIUS
                )
                st.experimental_rerun()

        col1, col2 = st.columns(2)
        if col1.button("Delete points", use_container_width=True):
            st.session_state["point_last_clicked"] = st.session_state["points"][-1]
            st.session_state["points"] = []
            st.session_state["image_displayed"] = st.session_state["image_original"]
            st.experimental_rerun()

        if col2.button(
            "Generate mask",
            disabled=len(st.session_state["points"]) == 0,
            use_container_width=True,
        ):
            with st.spinner("Running Segment-Anything-Model (SAM for friends)..."):
                if "sam" not in st.session_state:
                    st.session_state["sam"] = init_sam()

                st.session_state["mask"], st.session_state["segm"] = generate_mask(
                    st.session_state["sam"],
                    st.session_state["image_original"],
                    st.session_state["points"],
                )

        if "mask" in st.session_state and "segm" in st.session_state:
            col1.image(st.session_state["mask"])
            col2.image(st.session_state["segm"])

            pos_prompt = st.text_input("Positive prompt")
            neg_prompt = st.text_input("Negative prompt")

            if st.button("Generate inpainting", use_container_width=True):
                with st.spinner("Running Stable Diffusion inpainting..."):
                    if "pipe" not in st.session_state:
                        st.session_state["pipe"] = init_sd()

                    st.session_state["output"] = generate_inpainting(
                        st.session_state["pipe"],
                        st.session_state["image_original"],
                        st.session_state["mask"],
                        st.session_state["segm"],
                        pos_prompt,
                        neg_prompt,
                    )

            if "output" in st.session_state:
                st.image(st.session_state["output"])
                st.download_button(
                    "Download image",
                    data=pil_to_bytes(st.session_state["output"]),
                    file_name=f"{IMG_OUTPUT_NAME}.{IMG_OUTPUT_EXT}",
                    mime=f"image/{IMG_OUTPUT_EXT}",
                    use_container_width=True,
                )
