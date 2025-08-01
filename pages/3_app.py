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
import openai
import time

# --- Helper Functions ---
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object-type columns with mixed types to string,
    and ensures consistent serialization with PyArrow.
    """
    for col in df.select_dtypes(include='object').columns:
        try:
            # If all values are numeric strings, convert to float
            if pd.to_numeric(df[col], errors='coerce').notna().sum() >= len(df[col]) * 0.9:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(str)
        except Exception as e:
            df[col] = df[col].astype(str)
    return df

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
    """Creates a standardized prompt for layout-aware table extraction with merged cell support."""
    return """
    You are an expert data analyst specializing in complex financial and technical table extraction. Your primary task is to analyze the provided image, identify the main data table, and extract its contents with high precision.

    **CRITICAL REQUIREMENTS:**
    1. **EXTRACT EVERY SINGLE ROW** - Missing rows = Complete failure
    2. **NUMERICAL ACCURACY IS ABSOLUTE** - Every digit must be perfect
    3. **NO ROW LEFT BEHIND** - Scan systematically from top to bottom

    **Step 1: Table Analysis**
    - Identify the complete table boundary (top, bottom, left, right edges)
    - Count total rows visually BEFORE extraction
    - Determine table layout: Standard (headers top), Transposed (headers left), or Complex structure
    - Note any merged cells, multi-level headers, or rotated text

    **Step 2: Systematic Extraction**
    - Extract headers first, creating meaningful combined names for grouped headers
    - Extract data rows sequentially from top to bottom
    - Include ALL rows: data, subtotals, totals, summary rows
    - Maintain exact numerical formatting (commas, decimals, parentheses)
    - For unclear numbers, use [UNCLEAR] rather than guessing

    **Step 3: Quality Assurance**
    - Verify row count matches visual inspection
    - Ensure all columns have consistent data types
    - Double-check numerical accuracy
    - Maintain proper column alignment

    **Output Format:**
    Return a single, valid JSON object with these keys:
    - "table_data": List of lists (first = headers, rest = data rows)
    - "confidence_score": 0.0 to 1.0 score
    - "reasoning": Description of extraction approach
    - "structure_notes": Notes on table complexity
    - "numerical_accuracy_notes": Confirmation of numerical precision

    **Example Output:**
    ```json
    {
      "table_data": [
        ["Company", "Revenue", "Profit", "Loss"],
        ["ABC Corp", "1,250,000", "125,000", "(15,000)"],
        ["XYZ Inc", "2,100,500", "310,200", "(25,500)"]
      ],
      "confidence_score": 0.95,
      "reasoning": "Standard table layout with clear headers and numerical data",
      "structure_notes": "Simple 4-column table with financial data",
      "numerical_accuracy_notes": "All numbers extracted with 100% accuracy"
    }
    ```

    Remember: Numerical accuracy is critical for financial data. Take time to verify each number.
    """

# --- AI Extraction Functions ---
def extract_table_with_claude(base64_image_data, api_key, model_name):
    """Extracts table data using Anthropic Claude API with improved error handling."""
    
    if not api_key:
        return None, 0.0, "❌ Claude API key not configured."

    if not base64_image_data:
        return None, 0.0, "❌ Base64 image data is missing or corrupt."

    # Updated model names - these are the correct Claude model identifiers
    allowed_models = [
        "claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet
        "claude-3-5-sonnet-20240620",  # Previous Claude 3.5 Sonnet
        "claude-3-opus-20240229",      # Claude 3 Opus
        "claude-3-sonnet-20240229",    # Claude 3 Sonnet
        "claude-3-haiku-20240307"      # Claude 3 Haiku
    ]
    
    if model_name not in allowed_models:
        return None, 0.0, f"❌ Invalid model name. Available models: {', '.join(allowed_models)}"

    prompt = create_layout_aware_prompt()

    try:
        # Initialize Claude client
        client = anthropic.Anthropic(api_key=api_key)
        
        st.info(f"🤖 Sending request to Claude ({model_name})...")

        # Create the message
        response = client.messages.create(
            model=model_name,
            max_tokens=8192,
            temperature=0.1,  # Low temperature for consistent extraction
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        # Extract the response text
        raw_text = response.content[0].text.strip()
        
        # Debug output (optional - can be removed in production)
        with st.expander("🔍 Claude Raw Response (Debug)"):
            st.code(raw_text[:1500], language="json")

        # Clean JSON response (remove markdown code blocks if present)
        if raw_text.startswith("```json"):
            raw_text = raw_text.strip()[7:-3].strip()
        elif raw_text.startswith("```"):
            # Handle generic code blocks
            lines = raw_text.strip().split('\n')
            if len(lines) > 2:
                raw_text = '\n'.join(lines[1:-1]).strip()

        try:
            # Parse the JSON response
            parsed_json = json.loads(raw_text)
            
            # Extract required fields with defaults
            table_data = parsed_json.get("table_data", [])
            confidence_score = parsed_json.get("confidence_score", 0.0)
            reasoning = parsed_json.get("reasoning", "No reasoning provided.")
            
            # Additional validation
            if not table_data:
                return None, 0.0, "❌ No table data found in response"
            
            if len(table_data) < 2:
                return table_data, confidence_score, f"⚠️ Only {len(table_data)} row(s) extracted. {reasoning}"
            
            return table_data, confidence_score, reasoning
            
        except json.JSONDecodeError as e:
            return None, 0.0, f"⚠️ Claude returned invalid JSON: {str(e)[:200]}..."

    except anthropic.APIError as e:
        return None, 0.0, f"❌ Claude API error: {str(e)}"
    except Exception as e:
        return None, 0.0, f"❌ Unexpected error: {str(e)}"

def extract_table_with_gemini(base64_image_data, api_key, model_name):
    """Extracts table data using Google Gemini API."""
    if not api_key:
        return None, 0.0, "API key not configured."

    prompt = create_layout_aware_prompt()

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/png", "data": base64_image_data}}
            ]
        }],
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
        return None, 0.0, f"Extraction failed: {str(e)}"

def extract_table_with_openai(base64_image_data, api_key, model_name):
    """Extracts table data using OpenAI GPT-4 API."""
    if not api_key:
        return None, 0.0, "API key not configured."

    prompt = create_layout_aware_prompt()

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image_data}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        json_response_text = response.choices[0].message.content
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
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("Step 2: 🔎 AI-Powered Table Extractor")

if 'converted_pil_images' not in st.session_state or not st.session_state.converted_pil_images:
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Choose an image from the converted PDF and an AI model to extract the table.")

    # AI Configuration
    st.header("⚙️ AI Configuration")
    
    # Updated model options with correct Claude model names
    model_options = [
        # OpenAI Models
        "gpt-4o", "gpt-4-turbo",
        # Anthropic Claude Models (CORRECTED)
        "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", 
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
        # Google Models
        "gemini-1.5-pro", "gemini-1.5-flash"
    ]
    
    selected_model = st.selectbox("Choose AI Model:", options=model_options)
    
    # Show API key status
    col1, col2, col3 = st.columns(3)
    with col1:
        openai_status = "✅ Configured" if openai_api_key else "❌ Missing"
        st.info(f"**OpenAI:** {openai_status}")
    with col2:
        claude_status = "✅ Configured" if anthropic_api_key else "❌ Missing"
        st.info(f"**Claude:** {claude_status}")
    with col3:
        gemini_status = "✅ Configured" if gemini_api_key else "❌ Missing"
        st.info(f"**Gemini:** {gemini_status}")

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
                # Determine AI provider and function
                if "gemini" in selected_model:
                    ai_provider = "Google Gemini"
                    api_key_to_use = gemini_api_key
                    extraction_function = extract_table_with_gemini
                elif "claude" in selected_model:
                    ai_provider = "Anthropic Claude"
                    api_key_to_use = anthropic_api_key
                    extraction_function = extract_table_with_claude
                elif "gpt" in selected_model:
                    ai_provider = "OpenAI"
                    api_key_to_use = openai_api_key
                    extraction_function = extract_table_with_openai
                else:
                    st.error("Invalid model selected. Cannot determine AI provider.")
                    st.stop()

                if not api_key_to_use:
                    st.error(f"❌ {ai_provider} API Key is not configured. Please add it to your .env file.")
                else:
                    with st.spinner(f"🤖 {ai_provider} is analyzing the image with **{selected_model}**. Please wait..."):
                        start_time = time.time()

                        # Prepare image and extract
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
                                st.success(f"✅ Table extracted successfully in {processing_time:.1f} seconds!")
                                st.info("✅ Data is ready! Please proceed to the **🤖 Validator** page.")
                            except Exception as e:
                                st.warning(f"Data structure issue: {e}. Loading without strict header matching.")
                                st.session_state.extracted_df = pd.DataFrame(table_data)
                                st.session_state.original_df = st.session_state.extracted_df.copy()
                        elif table_data:
                            st.warning("Only one row of data was extracted.")
                            st.session_state.extracted_df = pd.DataFrame(table_data)
                            st.session_state.original_df = st.session_state.extracted_df.copy()
                        else:
                            st.error("❌ The AI could not find a table in the image.")
                            st.error(reasoning)  # Show the error reasoning
                            st.session_state.extracted_df = None
                            st.session_state.original_df = None

    # Display Results
    if st.session_state.get('confidence') is not None:
        st.subheader("📊 Extraction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            confidence_color = "normal" if st.session_state.confidence >= 0.8 else "inverse"
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

    # Data Display and Download
    if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("📋 Extracted Data Preview")
        clean_df = normalize_dataframe(st.session_state.extracted_df)
        st.dataframe(clean_df, use_container_width=True)

        # Show data summary
        st.markdown(f"**Data Summary:** {len(clean_df)} rows × {len(clean_df.columns)} columns")

        st.divider()
        st.subheader("📥 Download Extracted Data")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            file_name = st.text_input("File Name (without extension):", value="extracted_table", key="download_filename")
        with dl_col2:
            file_format = st.selectbox("Choose Format:", ["Excel (.xlsx)", "CSV (.csv)"], key="download_format")

        if file_name:
            if file_format == "Excel (.xlsx)":
                file_data = to_excel(clean_df)
                file_extension = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                file_data = clean_df.to_csv(index=False).encode('utf-8')
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