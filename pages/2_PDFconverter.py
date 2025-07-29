import streamlit as st
import fitz  # PyMuPDF
import io
from PIL import Image

# --- Core Conversion Function (Modified) ---
def convert_pdf_to_pil_images(pdf_file, start_page, end_page, dpi=300):
    """Converts PDF pages to a list of PIL Image objects."""
    pil_images = []
    try:
        # Open the PDF from the uploaded file's bytes
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

        # Validate page range
        total_pages = len(pdf_document)
        if start_page > end_page or start_page < 1 or end_page > total_pages:
            st.error(f"Error: Invalid page range. The PDF has {total_pages} pages.")
            return []

        # Convert selected pages to images
        for page_num in range(start_page - 1, end_page):
            page = pdf_document.load_page(page_num)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            pil_images.append(pil_image)

        pdf_document.close()
        return pil_images
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        return []

def rotate_image(image, angle):
    """Rotate a PIL image by the specified angle."""
    return image.rotate(angle, expand=True)

# --- Streamlit App UI ---
st.title("Step 1: 📂 High-Quality PDF to Image Converter")
st.markdown("Upload a PDF to convert its pages into images for the next steps in the workflow.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()
        # Use a 'with' block for safety
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            total_pages = len(doc)

        st.info(f"PDF uploaded successfully! It has **{total_pages}** pages.")

        st.header("Select Page Range and Quality")
        col1, col2 = st.columns(2)
        start_page_input = col1.number_input("Start Page", 1, total_pages, 1)
        end_page_input = col2.number_input("End Page", 1, total_pages, total_pages)
        dpi_input = st.slider(
            "Image Quality (DPI)", 100, 600, 300, 50, help="300 DPI is recommended for OCR."
        )

        if st.button("Convert to Images", type="primary"):
            with st.spinner("Converting pages to high-quality images..."):
                # Call the conversion and store the result in session state
                pil_images = convert_pdf_to_pil_images(
                    io.BytesIO(pdf_bytes), start_page_input, end_page_input, dpi_input
                )
                
                if pil_images:
                    # CRITICAL: Save images to session state for other pages to use
                    st.session_state.converted_pil_images = pil_images
                    # Initialize rotation angles for each image
                    st.session_state.image_rotations = [0] * len(pil_images)
                    
                    ### FIX: Clear ALL old data from subsequent steps to prevent using stale data
                    keys_to_clear = [
                        'extracted_df', 'corrected_df', 'validation_results', 
                        'confidence', 'reasoning', 'original_df', 'corrections',
                        'base64_image_to_validate', 'corrections_applied'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.success("✅ Conversion complete!")
                else:
                    st.session_state.converted_pil_images = []

    except Exception as e:
        st.error(f"Failed to read the PDF file. It might be corrupted. Error: {e}")

# Display images from session state with rotation controls
if st.session_state.get('converted_pil_images'):
    st.header("Converted Images")
    st.info("✅ Images are ready! You can rotate individual images if needed, then proceed to the **🔎 Table Extractor** page from the sidebar.")
    
    # Initialize rotation angles if not present
    if 'image_rotations' not in st.session_state:
        st.session_state.image_rotations = [0] * len(st.session_state.converted_pil_images)
    
    # Display images with rotation controls
    for i, img in enumerate(st.session_state.converted_pil_images):
        st.subheader(f"Page {i + 1}")
        
        # Rotation controls
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            if st.button(f"↺ 90° CCW", key=f"ccw_{i}", help="Rotate 90° counter-clockwise"):
                st.session_state.image_rotations[i] = (st.session_state.image_rotations[i] - 90) % 360
                st.rerun()
        
        with col2:
            if st.button(f"↻ 90° CW", key=f"cw_{i}", help="Rotate 90° clockwise"):
                st.session_state.image_rotations[i] = (st.session_state.image_rotations[i] + 90) % 360
                st.rerun()
        
        with col3:
            if st.button(f"↻ 180°", key=f"flip_{i}", help="Rotate 180°"):
                st.session_state.image_rotations[i] = (st.session_state.image_rotations[i] + 180) % 360
                st.rerun()
        
        with col4:
            current_rotation = st.session_state.image_rotations[i]
            if current_rotation != 0:
                st.info(f"Current rotation: {current_rotation}°")
        
        # Apply rotation and display image
        rotated_img = rotate_image(img, st.session_state.image_rotations[i])
        st.image(rotated_img, caption=f"Page {i + 1}", use_container_width=True)
        
        # Update the image in session state with the rotation applied
        st.session_state.converted_pil_images[i] = rotated_img
        
        st.divider()