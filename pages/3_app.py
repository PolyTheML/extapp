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

# --- Helper Functions (No changes needed in functions themselves) ---
def prepare_image_from_pil(pil_image):
    """Converts and resizes a PIL image to a base64 encoded string."""
    try:
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        max_dim = 2048
        h, w, _ = img_cv.shape
        if h > max_dim or w > max_dim:
            if h > w:
                new_h, new_w = max_dim, int(w * (max_dim / h))
            else:
                new_h, new_w = int(h * (max_dim / w)), max_dim
            img_cv = cv2.resize(img_cv, (new_w, new_h))
        _, buffer = cv2.imencode('.png', img_cv)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"Error preparing image: {e}")
        return None

def extract_table_with_ai(base64_image_data, api_key, model_name):
    if not api_key:
        st.error("Error: Gemini API key not found.")
        return None, 0.0, "API key not configured."
    prompt = "Analyze the provided image to identify the primary data table. Your task is to extract its content with high precision. Instructions: 1. **JSON Output:** Return a single JSON object with three keys: \"table_data\", \"confidence_score\", and \"reasoning\". 2. **table_data:** The value must be a list of lists, where each inner list represents a table row. The first inner list must be the header. 3. **confidence_score:** Provide a numerical score from 0.0 to 1.0, where 1.0 is absolute confidence in the extraction accuracy. 4. **reasoning:** Briefly explain your confidence score. Mention any blurry text, complex merged cells, or unusual formatting that might affect accuracy. 5. **Accuracy Rules:** Handle merged cells by repeating values, represent empty cells with an empty string(\"\"), and combine multi-line text. Begin the extraction now."
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": base64_image_data}}] }],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        json_response_text = result['candidates'][0]['content']['parts'][0]['text']
        parsed_json = json.loads(json_response_text)
        return parsed_json.get("table_data", []), parsed_json.get("confidence_score", 0.0), parsed_json.get("reasoning", "No reasoning provided.")
    except Exception as e:
        st.error(f"An error occurred during Gemini extraction: {e}")
        return None, 0.0, "Extraction failed due to a general error."

def extract_table_with_claude(base64_image_data, api_key, model_name):
    if not api_key:
        st.error("Error: Anthropic API key not found.")
        return None, 0.0, "API key not configured."
    prompt = "Analyze the provided image to identify the primary data table. Your task is to extract its content with high precision. Instructions: 1. **JSON Output:** Return a single JSON object with three keys: \"table_data\", \"confidence_score\", and \"reasoning\". 2. **table_data:** The value must be a list of lists, where each inner list represents a table row. The first inner list must be the header. 3. **confidence_score:** Provide a numerical score from 0.0 to 1.0, where 1.0 is absolute confidence in the extraction accuracy. 4. **reasoning:** Briefly explain your confidence score. Mention any blurry text, complex merged cells, or unusual formatting that might affect accuracy. 5. **Accuracy Rules:** Handle merged cells by repeating values, represent empty cells with an empty string(\"\"), and combine multi-line text. Begin the extraction now."
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(model=model_name, max_tokens=4096, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}}, {"type": "text", "text": prompt}]}])
        json_response_text = message.content[0].text
        parsed_json = json.loads(json_response_text)
        return parsed_json.get("table_data", []), parsed_json.get("confidence_score", 0.0), parsed_json.get("reasoning", "No reasoning provided.")
    except Exception as e:
        st.error(f"An error occurred with the Claude API: {e}")
        return None, 0.0, "Extraction failed due to an API error."

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 2: 🔎 AI-Powered Table Extractor")

if not st.session_state.get('converted_pil_images'):
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Select an image below to extract a table. The AI will analyze it and turn it into data.")
    
    image_options = {f"Image {i+1}": i for i in range(len(st.session_state.converted_pil_images))}
    if image_options:
        selected_image_key = st.selectbox("Choose an image to process:", options=list(image_options.keys()))
        selected_image_index = image_options[selected_image_key]
        st.session_state.selected_image_index = selected_image_index
        selected_pil_image = st.session_state.converted_pil_images[selected_image_index]
    else:
        st.error("No images found from the conversion step.")
        selected_pil_image = None

    if selected_pil_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(selected_pil_image, caption="Selected Image for Extraction", use_container_width=True)
        with col2:
            st.header("⚙️ AI Options")
            ai_provider = st.selectbox("Choose AI Provider:", ("Google Gemini", "Anthropic Claude"))
            
            ### FIX: Corrected and standardized model names
            if ai_provider == "Google Gemini":
                model_options = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
            else: # Anthropic Claude
                model_options = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
            
            selected_model = st.selectbox("Choose AI Model:", options=model_options)

            if st.button("Extract Table from Image", type="primary"):
                api_key_to_use, extraction_function = (gemini_api_key, extract_table_with_ai) if ai_provider == "Google Gemini" else (anthropic_api_key, extract_table_with_claude)
                
                if not api_key_to_use:
                    st.warning(f"Please add your {ai_provider} API Key to the .env file.")
                else:
                    with st.spinner(f"The AI ({ai_provider}) is analyzing the image with **{selected_model}**. Please wait..."):
                        base64_image = prepare_image_from_pil(selected_pil_image)
                        if base64_image:
                            table_data, confidence, reasoning = extraction_function(base64_image, api_key_to_use, selected_model)
                            st.session_state.confidence = confidence
                            st.session_state.reasoning = reasoning
                            
                            if table_data and len(table_data) > 1:
                                try:
                                    header, data = table_data[0], table_data[1:]
                                    st.session_state.extracted_df = pd.DataFrame(data, columns=header)
                                    st.session_state.original_df = st.session_state.extracted_df.copy()
                                    st.success("Table extracted successfully!")
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
                                st.error("The AI could not find a table or the API call failed.")
                                st.session_state.extracted_df = None
                                st.session_state.original_df = None

    if st.session_state.get('confidence') is not None:
        st.subheader("📊 Extraction Accuracy")
        st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}", delta_color="off")
        with st.expander("See AI's Reasoning"):
            st.info(st.session_state.get('reasoning', 'No reasoning provided.'))

    if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("Extracted Data Preview")
        st.dataframe(st.session_state.extracted_df)
        
        st.divider()
        st.subheader("📥 Download Extracted Data")
        
        col1, col2 = st.columns(2)
        with col1:
            file_name = st.text_input("File Name (without extension):", value="extracted_table", key="download_filename")
        with col2:
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