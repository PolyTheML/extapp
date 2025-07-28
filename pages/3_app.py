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
from sqlalchemy import create_engine
import sqlite3
import anthropic

# --- Keep ALL your original functions from app.py ---
# (prepare_image_from_upload, extract_table_with_ai, extract_table_with_claude,
# to_excel, transform_data, and all load_to_* functions)
# For brevity, they are pasted below.

def prepare_image_from_pil(pil_image):
    """Converts and resizes a PIL image to a base64 encoded string."""
    try:
        # Convert PIL Image to an OpenCV image
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
    """Sends the image to the Gemini API and asks it to extract the table and a confidence score."""
    if not api_key:
        st.error("Error: Gemini API key not found.")
        return None, 0.0, "API key not configured."

    prompt = """
    Analyze the provided image to identify the primary data table. Your task is to extract its content with high precision.
    Instructions:
    1.  **JSON Output:** Return a single JSON object with three keys: "table_data", "confidence_score", and "reasoning".
    2.  **table_data:** The value must be a list of lists, where each inner list represents a table row. The first inner list must be the header.
    3.  **confidence_score:** Provide a numerical score from 0.0 to 1.0, where 1.0 is absolute confidence in the extraction accuracy.
    4.  **reasoning:** Briefly explain your confidence score. Mention any blurry text, complex merged cells, or unusual formatting that might affect accuracy.
    5.  **Accuracy Rules:** Handle merged cells by repeating values, represent empty cells with an empty string(""), and combine multi-line text.
    Begin the extraction now.
    """
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
        table_data = parsed_json.get("table_data", [])
        confidence = parsed_json.get("confidence_score", 0.0)
        reasoning = parsed_json.get("reasoning", "No reasoning provided.")
        return table_data, confidence, reasoning
    except requests.exceptions.HTTPError as err:
        st.error(f"An HTTP error occurred with the Gemini API: {err}")
        st.code(err.response.text, language='json')
        return None, 0.0, "Extraction failed due to an HTTP error."
    except Exception as e:
        st.error(f"An error occurred during Gemini extraction: {e}")
        return None, 0.0, "Extraction failed due to a general error."

def extract_table_with_claude(base64_image_data, api_key, model_name):
    """Sends the image to the Claude API and asks it to extract the table data."""
    if not api_key:
        st.error("Error: Anthropic API key not found.")
        return None, 0.0, "API key not configured."

    prompt = """
    Analyze the provided image to identify the primary data table. Your task is to extract its content with high precision.
    Instructions:
    1.  **JSON Output:** Return a single JSON object with three keys: "table_data", "confidence_score", and "reasoning".
    2.  **table_data:** The value must be a list of lists, where each inner list represents a table row. The first inner list must be the header.
    3.  **confidence_score:** Provide a numerical score from 0.0 to 1.0, where 1.0 is absolute confidence in the extraction accuracy.
    4.  **reasoning:** Briefly explain your confidence score. Mention any blurry text, complex merged cells, or unusual formatting that might affect accuracy.
    5.  **Accuracy Rules:** Handle merged cells by repeating values, represent empty cells with an empty string(""), and combine multi-line text.
    Begin the extraction now.
    """
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model_name,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}},
                    {"type": "text", "text": prompt}
                ]}
            ],
        )
        json_response_text = message.content[0].text
        parsed_json = json.loads(json_response_text)
        table_data = parsed_json.get("table_data", [])
        confidence = parsed_json.get("confidence_score", 0.0)
        reasoning = parsed_json.get("reasoning", "No reasoning provided.")
        return table_data, confidence, reasoning
    except Exception as e:
        st.error(f"An error occurred with the Claude API: {e}")
        return None, 0.0, "Extraction failed due to an API error."

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def transform_data(df: pd.DataFrame, is_for_viz=False) -> pd.DataFrame:
    """Cleans, standardizes, and validates the extracted DataFrame."""
    if not is_for_viz:
        st.info("Transforming data for ETL...")
    
    df.columns = [str(col).strip().lower().replace(' ', '_').replace('%', 'pct') for col in df.columns]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            cleaned_col = df[col].str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.strip()
            df[col] = pd.to_numeric(cleaned_col, errors='ignore')
            
        if not pd.api.types.is_numeric_dtype(df[col]):
             try:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
             except (ValueError, TypeError):
                pass
                
    df.replace('', np.nan, inplace=True)
    if not is_for_viz:
        st.success("Data transformation complete!")
    return df

def load_to_bigquery(df: pd.DataFrame, project_id: str, table_id: str):
    try:
        df.to_gbq(destination_table=table_id, project_id=project_id, if_exists='append', progress_bar=True)
        st.success(f"Successfully loaded {len(df)} rows to BigQuery table: {table_id}")
    except Exception as e:
        st.error(f"Failed to load data to BigQuery: {e}")
        st.warning("Ensure your GCP credentials are set up correctly.")

def load_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str):
    """Loads the DataFrame into a specified SQLite database table using the native sqlite3 library."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, con=conn, if_exists='append', index=False)
        conn.commit()
        st.success(f"Successfully loaded {len(df)} rows to table '{table_name}' in {db_path}")
    except Exception as e:
        st.error(f"Failed to load data to SQLite: {e}")
    finally:
        if conn:
            conn.close()

def load_to_postgres(df: pd.DataFrame, user, password, host, port, dbname, table_name):
    """Loads the DataFrame into a specified PostgreSQL table."""
    try:
        db_url = f'postgresql://{user}:{password}@{host}:{port}/{dbname}'
        engine = create_engine(db_url)
        with engine.connect() as connection:
            df.to_sql(table_name, con=connection, if_exists='append', index=False)
        st.success(f"Successfully loaded {len(df)} rows to table '{table_name}' in database '{dbname}'")
    except Exception as e:
        st.error(f"Failed to load data to PostgreSQL: {e}")
        st.warning("Ensure the database and table exist and credentials are correct.")
        
# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 2: 🔎 AI-Powered Table Extractor")

# MODIFICATION: Check for images from the previous step
if not st.session_state.get('converted_pil_images'):
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Select an image below to extract a table. The AI will analyze it and turn it into data.")
    
    # MODIFICATION: Let user select which image to process
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
            st.image(selected_pil_image, caption="Selected Image for Extraction", use_column_width=True)
        with col2:
            st.header("⚙️ AI Options")
            ai_provider = st.selectbox("Choose AI Provider:", ("Google Gemini", "Anthropic Claude"))
            if ai_provider == "Google Gemini":
                model_options = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite","gemini-1.5-flash"]
            else:
                model_options = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
            selected_model = st.selectbox("Choose AI Model:", options=model_options)

            if st.button("Extract Table from Image", type="primary"):
                api_key_to_use, extraction_function = (gemini_api_key, extract_table_with_ai) if ai_provider == "Google Gemini" else (anthropic_api_key, extract_table_with_claude)
                if not api_key_to_use:
                    st.warning(f"Please add your {ai_provider} API Key to the .env file.")
                else:
                    with st.spinner(f"The AI ({ai_provider}) is analyzing the image with **{selected_model}**. Please wait..."):
                        # MODIFICATION: Use the selected PIL image
                        base64_image = prepare_image_from_pil(selected_pil_image)
                        if base64_image:
                            table_data, confidence, reasoning = extraction_function(base64_image, api_key_to_use, selected_model)
                            st.session_state.confidence = confidence
                            st.session_state.reasoning = reasoning
                            if table_data and len(table_data) > 1:
                                try:
                                    header, data = table_data[0], table_data[1:]
                                    st.session_state.extracted_df = pd.DataFrame(data, columns=header)
                                    # Also save to original_df for the validator page
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

    # This part of your UI remains the same, as it depends on session state
    if st.session_state.confidence is not None:
        st.subheader("📊 Extraction Accuracy")
        st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}", delta_color="off")
        with st.expander("See AI's Reasoning"):
            st.info(st.session_state.reasoning)

    if st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("Extracted Data Preview")
        st.dataframe(st.session_state.extracted_df)
