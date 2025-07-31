import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import os
import pandas as pd
import base64
import json
from io import BytesIO
from dotenv import load_dotenv
import anthropic
import time

# --- Helper Functions ---
def prepare_image_from_pil(pil_image):
    """Resizes, encodes, and prepares a PIL image for API requests."""
    try:
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        max_dim = 2048; h, w, _ = img_cv.shape
        if h > max_dim or w > max_dim:
            if h > w: new_h, new_w = max_dim, int(w * (max_dim / h))
            else: new_h, new_w = int(h * (max_dim / w)), max_dim
            img_cv = cv2.resize(img_cv, (new_w, new_h))
        _, buffer = cv2.imencode('.png', img_cv)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        st.error(f"Error preparing image: {e}"); return None

def create_layout_aware_prompt():
    """Creates a standardized prompt for layout-aware table extraction."""
    return """
    You are an expert data extractor. Your primary task is to analyze the provided image, identify the main data table, and extract its contents with high precision, paying special attention to its orientation.

    **Step 1: Detect Table Layout**
    First, determine the table's layout. Is it:
    a) **Standard Layout:** Headers are in the top row, and data records are in subsequent rows.
    b) **Transposed Layout:** Headers are in the first column, and data records are in subsequent columns.

    **Step 2: Extract Data Based on Layout**
    - **If Standard Layout:** Proceed with normal extraction. The first row is your header list, and each row that follows is a data list.
    - **If Transposed Layout:** You must **un-pivot** the data. The first column contains the headers, and each following column is a record. You must reconstruct this into a standard format.
      - **Example of a Transposed Table:**
        [
          ["Name", "John Doe"],
          ["Age", "30"],
          ["City", "New York"]
        ]
      - **Your mandatory output for the above example must be:**
        [
          ["Name", "Age", "City"],
          ["John Doe", "30", "New York"]
        ]

    **Step 3: Final JSON Output**
    Regardless of the original layout, you must return a single, valid JSON object with the following three keys: "table_data", "confidence_score", and "reasoning".

    - **`table_data`**: MUST be a list of lists in the **standard format** (the first inner list is the header row).
    - **`confidence_score`**: A numerical score from 0.0 to 1.0 on your confidence in the accuracy and correct orientation of the extraction.
    - **`reasoning`**: **Crucially, state which layout you detected** (e.g., "Detected a transposed layout and un-pivoted the data.") and explain any challenges.

    Your top priority is returning the data in the correct, standard format.
    """

# --- AI Extraction Functions ---
def extract_table_with_gemini(base64_image_data, api_key, model_name):
    """Extracts table data using Google Gemini API."""
    if not api_key:
        return None, 0.0, "API key not configured."
    
    prompt = create_layout_aware_prompt()
    
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": base64_image_data}}] }], "generationConfig": {"responseMimeType": "application/json"}}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        json_response_text = result['candidates'][0]['content']['parts'][0]['text']
        parsed_json = json.loads(json_response_text)
        return parsed_json.get("table_data", []), parsed_json.get("confidence_score", 0.0), parsed_json.get("reasoning", "No reasoning provided.")
    except Exception as e:
        return None, 0.0, f"Extraction failed: {str(e)}"

def extract_table_with_claude(base64_image_data, api_key, model_name):
    """Extracts table data using Anthropic Claude API."""
    if not api_key:
        return None, 0.0, "API key not configured."
    
    prompt = create_layout_aware_prompt()
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(model=model_name, max_tokens=4096, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}}, {"type": "text", "text": prompt}]}])
        json_response_text = message.content[0].text
        # Clean potential markdown code fences
        if json_response_text.strip().startswith("```json"):
            json_response_text = json_response_text.strip()[7:-3].strip()
        parsed_json = json.loads(json_response_text)
        return parsed_json.get("table_data", []), parsed_json.get("confidence_score", 0.0), parsed_json.get("reasoning", "No reasoning provided.")
    except Exception as e:
        return None, 0.0, f"API error: {str(e)}"

def to_excel(df):
    """Converts a DataFrame to an in-memory Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 2: 🔎 AI-Powered Table Extractor")

if 'converted_pil_images' not in st.session_state or not st.session_state.converted_pil_images:
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Choose an image from the converted PDF and an AI model to extract the table.")
    
    # AI Configuration
    st.header("⚙️ AI Configuration")
    model_options = [
        "claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro"
    ]
    selected_model = st.selectbox("Choose AI Model:", options=model_options)
    
    st.divider()
    
    # Image Processing
    st.header("🖼️ Image Processing")
    
    image_options = {f"Image {i+1}": i for i in range(len(st.session_state.converted_pil_images))}
    if image_options:
        selected_image_key = st.selectbox("Choose an image to process:", options=list(image_options.keys()))
        selected_image_index = image_options[selected_image_key]
        st.session_state.selected_image_index = selected_image_index
        selected_pil_image = st.session_state.converted_pil_images[selected_image_index]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(selected_pil_image, caption="Selected Image for Extraction", use_container_width=True)
        
        with col2:
            if st.button("Extract Table from Image", type="primary"):
                # Determine provider and API key from selected model
                if "gemini" in selected_model:
                    ai_provider = "Google Gemini"
                    api_key_to_use = gemini_api_key
                    extraction_function = extract_table_with_gemini
                elif "claude" in selected_model:
                    ai_provider = "Anthropic Claude"
                    api_key_to_use = anthropic_api_key
                    extraction_function = extract_table_with_claude
                else:
                    st.error("Invalid model selected. Cannot determine AI provider.")
                    st.stop()
                
                if not api_key_to_use:
                    st.warning(f"Please add your {ai_provider} API Key to the .env file.")
                else:
                    with st.spinner(f"🤖 {ai_provider} is analyzing the image with **{selected_model}**. Please wait..."):
                        start_time = time.time()
                        
                        base64_image = prepare_image_from_pil(selected_pil_image)
                        if base64_image:
                            table_data, confidence, reasoning = extraction_function(base64_image, api_key_to_use, selected_model)
                        else:
                            table_data, confidence, reasoning = None, 0.0, "Failed to prepare image"
                        
                        processing_time = time.time() - start_time
                        
                        # Store results in session state
                        st.session_state.ai_provider = ai_provider
                        st.session_state.confidence = confidence
                        st.session_state.reasoning = reasoning
                        st.session_state.processing_time = processing_time
                        st.session_state.selected_model = selected_model
                        
                        if table_data and len(table_data) > 1:
                            try:
                                header, data = table_data[0], table_data[1:]
                                st.session_state.extracted_df = pd.DataFrame(data, columns=header)
                                st.session_state.original_df = st.session_state.extracted_df.copy()
                                st.success(f"✅ Table extracted successfully in {st.session_state.processing_time:.1f} seconds!")
                                st.info("✅ Data is ready! Please proceed to the **🤖 Validator** page.")
                            except Exception as e:
                                st.error(f"Data Mismatch Error: {e}. Trying to load without a header.")
                                st.session_state.extracted_df = pd.DataFrame(table_data)
                                st.session_state.original_df = st.session_state.extracted_df.copy()
                        elif table_data:
                            st.warning("Only one row of data was extracted.")
                            st.session_state.extracted_df = pd.DataFrame(table_data)
                            st.session_state.original_df = st.session_state.extracted_df.copy()
                        else:
                            st.error("The AI could not find a table in the image.")
                            st.session_state.extracted_df = None
                            st.session_state.original_df = None

    # --- Display Single Image Results ---
    if st.session_state.get('confidence') is not None:
        st.subheader("📊 Extraction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}")
        with col2:
            st.metric(label="Processing Time", value=f"{st.session_state.processing_time:.1f}s")
        with col3:
            st.metric(label="AI Provider", value=st.session_state.get('ai_provider', 'N/A'))
        
        with st.expander("🧠 AI's Reasoning"):
            st.info(st.session_state.get('reasoning', 'No reasoning provided.'))
            st.markdown(f"""
            **Model Info:**
            - **Model:** {st.session_state.get('selected_model', 'N/A')}
            - **Provider:** {st.session_state.get('ai_provider', 'N/A')}
            """)

    # Data Display and Download for single image
    if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("Extracted Data Preview")
        st.dataframe(st.session_state.extracted_df)
        
        st.divider()
        st.subheader("📥 Download Extracted Data")
        
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            file_name = st.text_input("File Name (without extension):", value="extracted_table", key="download_filename")
        with dl_col2:
            file_format = st.selectbox("Choose Format:", ["Excel (.xlsx)", "CSV (.csv)"], key="download_format")
        
        if file_name:
            if file_format == "Excel (.xlsx)":
                file_data = to_excel(st.session_state.extracted_df)
                file_extension = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                file_data = st.session_state.extracted_df.to_csv(index=False).encode('utf-8')
                file_extension = "csv"
                mime_type = "text/csv"
            
            st.download_button(
                label=f"📥 Download as {file_format}",
                data=file_data,
                file_name=f"{file_name}.{file_extension}",
                mime=mime_type,
                type="primary"
            )
        else:
            st.warning("Please enter a file name to enable download.")