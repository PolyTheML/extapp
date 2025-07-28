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
                    
                    # CRITICAL: Clear any old data from subsequent steps
                    st.session_state.extracted_df = None
                    st.session_state.corrected_df = None
                    st.session_state.validation_results = None
                    st.session_state.confidence = None
                    st.session_state.reasoning = None
                    st.session_state.original_df = None

                    st.success("✅ Conversion complete!")
                else:
                    st.session_state.converted_pil_images = []

    except Exception as e:
        st.error(f"Failed to read the PDF file. It might be corrupted. Error: {e}")

# Display images from session state
if st.session_state.get('converted_pil_images'):
    st.header("Converted Images")
    st.info("✅ Images are ready! Please proceed to the **🔎 Table Extractor** page from the sidebar.")
    # Display a preview of the generated images
    for i, img in enumerate(st.session_state.converted_pil_images):
        st.image(img, caption=f"Page {i + 1}", use_column_width=True)