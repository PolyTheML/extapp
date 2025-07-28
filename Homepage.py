# Homepage.py
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="AI Document Workflow",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# This is crucial for passing data between pages
if 'converted_pil_images' not in st.session_state:
    st.session_state.converted_pil_images = []
if 'selected_image_index' not in st.session_state:
    st.session_state.selected_image_index = 0
if 'extracted_df' not in st.session_state:
    st.session_state.extracted_df = None
if 'corrected_df' not in st.session_state:
    st.session_state.corrected_df = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None

# --- Main Page UI ---
st.title("🤖 AI-Powered Document Workflow")
st.markdown("Follow the steps in the sidebar to extract, analyze, and validate data from your documents.")
st.sidebar.success("Select a workflow step.")

st.header("How It Works")
st.markdown("""
1.  **📂 PDF Converter**: Upload a PDF document. This tool will convert the selected pages into high-quality images.
2.  **🔎 Table Extractor**: Choose one of the converted images. The AI will analyze it to extract any tabular data into a structured format.
3.  **🤖 Validator**: The AI agent will compare the original image with the extracted data to find discrepancies and suggest smart corrections.
""")

st.info("To begin, please navigate to the **PDF Converter** page using the sidebar on the left.")