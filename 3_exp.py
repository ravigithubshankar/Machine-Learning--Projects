import streamlit as st
import cv2
import numpy as np

def inpaint_image(image, mask, method='telea'):
    """
    Inpaints the given image using the provided mask.
    
    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Mask indicating areas to be inpainted.
        method (str): Inpainting method ('telea' or 'ns').
    
    Returns:
        numpy.ndarray: Inpainted image.
    """
    if method == 'telea':
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    elif method == 'ns':
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

def main():
    st.title("Image Inpainting")
    st.write("Upload an image and select the region to inpaint.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    try:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Original Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return

    # Load sample mask image
    try:
        mask = cv2.imread("mask.png", 0)
        st.image(mask, caption="Mask", use_column_width=True)
    except Exception as e:
        st.warning(f"Warning: Error loading sample mask image: {e}")
        mask = None

    if mask is not None:
        method = st.radio("Select inpainting method:", ('Telea', 'Navier-Stokes'))
        if st.button("Inpaint"):
            try:
                inpainted_image = inpaint_image(image, mask, method.lower())
                st.image(inpainted_image, caption="Inpainted Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error inpainting image: {e}")

if __name__ == "__main__":
    main()
