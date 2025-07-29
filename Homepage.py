import streamlit as st
import os 

st.set_page_config(
    page_title="AI Document Processor",
    page_icon="🤖",
    layout="wide"
)

if 'STREAMLIT_SERVER_FILE_WATCHER_TYPE' not in os.environ:
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

st.title("Welcome to the AI Document Processor! 📄➡️📊")

st.markdown("""
This application is a multi-step workflow designed to extract, validate, and correct tabular data from PDF documents using powerful AI models.

### How to Use This App:
1.  **📂 PDF Converter:** Start by navigating to the PDF Converter page from the sidebar. Upload your PDF file to convert the relevant pages into high-quality images.
2.  **🔎 Table Extractor:** Once you have images, go to the Table Extractor page. Select an image, and the AI will automatically extract the data table from it.
3.  **🤖 File Validator:** Finally, use the File Validator page. The AI will compare the extracted data against the original image, identify any discrepancies, and suggest smart corrections.

**Select a step from the sidebar to begin.**
""")

# Initialize session state variables if they don't exist
# This ensures that the app doesn't crash on the first run
if 'converted_pil_images' not in st.session_state:
    st.session_state.converted_pil_images = []
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'corrected_df' not in st.session_state:
    st.session_state.corrected_df = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None