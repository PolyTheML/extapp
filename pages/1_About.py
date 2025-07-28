import streamlit as st

# Set the page title and icon
st.set_page_config(
    page_title="About This App",
    page_icon="ℹ️",
)

# Main title of the page
st.title("ℹ️ About This Application")

# --- UPDATED SECTION ---
# Added a prominent info box to highlight the client
st.info("This application was proudly developed for the **Rating Agency of Cambodia (RAC)** to enhance their document processing and data validation capabilities.")

# Updated the main description
st.markdown("""
This multi-page application was built for **RAC** to create a seamless workflow for internal document processing using AI.
""")

st.markdown("""
### Workflow Steps:
1.  **PDF Converter**: Converts pages from an uploaded PDF into high-quality images.
2.  **Table Extractor**: Uses advanced AI models (Google Gemini or Anthropic Claude) to extract tabular data from the converted images.
3.  **Validator**: An AI agent compares the extracted data against the original image to identify discrepancies and suggest corrections.

This project demonstrates the power of combining several modular apps into a single, cohesive user experience using Streamlit's multi-page functionality.
""")

st.header("Technologies Used")
st.markdown("""
- **Backend & UI**: [Streamlit](https://streamlit.io/)
- **PDF Processing**: [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
- **AI Models**: Google Gemini & Anthropic Claude APIs
- **Core Language**: Python
""")