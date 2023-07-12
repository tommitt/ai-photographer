import streamlit as st
from constants import IMG_SIZE
from streamlit_image_coordinates import streamlit_image_coordinates
from utils import add_point_to_image, format_image


def app():
    st.set_page_config(page_icon="📷", layout="centered")
    st.title("AI Product Photographer 📷")

    if "init_app" not in st.session_state:
        st.session_state["init_app"] = True
        st.session_state["points"] = []
        st.session_state["point_last_clicked"] = []

    if "image_original" not in st.session_state:
        uploaded_picture = st.file_uploader(
            "Choose a picture", type=["png", "jpg", "jpeg"]
        )
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
                    st.session_state["image_displayed"], xy, radius=(IMG_SIZE // 50)
                )
                st.experimental_rerun()

        if st.button("Delete points"):
            st.session_state["point_last_clicked"] = st.session_state["points"][-1]
            st.session_state["points"] = []
            st.session_state["image_displayed"] = st.session_state["image_original"]
            st.experimental_rerun()

        if st.button("Generate mask"):
            with st.spinner("Running Segment Anything Model..."):
                st.write("here we go let's generate a mask :)")

        st.button("Test button")
