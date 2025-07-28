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
import re
from datetime import datetime

# --- Keep ALL your original functions from FileVerify.py ---
# (prepare_image_from_upload, validate_with_*, get_smart_corrections, 
# apply_corrections_to_dataframe, etc.)
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

def validate_with_ai_comparison(base64_image_data, extracted_df, api_key, model_name, provider):
    """Have AI directly compare the image with the CSV data to identify discrepancies."""
    
    csv_data = extracted_df.to_dict('records')
    headers = list(extracted_df.columns)
    
    prompt = f"""
    You are a data validation expert. I have an image of a table and extracted data from that table. 
    Your task is to compare the image with the extracted data and identify any discrepancies.

    EXTRACTED DATA (CSV):
    Headers: {headers}
    Data: {json.dumps(csv_data, indent=2)}

    IMPORTANT INSTRUCTIONS:
    - Look very carefully at the image and compare each cell with the CSV data
    - Be extremely precise - don't assume the CSV is wrong just because values look different
    - Consider that both the image reading AND the CSV extraction could have errors
    - Pay special attention to: digit recognition (0 vs O, 1 vs l, 5 vs S), decimal points, commas, spacing
    - Only report discrepancies when you are highly confident there's actually a difference

    For each discrepancy you find, analyze:
    1. What do you actually see in the image (be very specific)
    2. What's in the CSV data
    3. Which one is more likely to be correct based on context, formatting, and clarity
    4. Your confidence level in the discrepancy

    Return your analysis as a JSON object with this structure:
    {{
        "validation_results": [
            {{
                "row_index": 0,
                "column_name": "column_name",
                "issue_type": "number_mismatch|text_mismatch|missing_data|extra_data|unclear_image",
                "image_value": "what you actually see in the image (be specific)",
                "csv_value": "what's in the CSV",
                "likely_correct_value": "which value is more likely correct",
                "confidence": 0.9,
                "csv_likely_correct": true,
                "description": "Brief description of the issue and why you think one is more correct",
                "reasoning": "Detailed explanation of your analysis"
            }}
        ],
        "overall_accuracy": 0.95,
        "total_issues": 3,
        "summary": "Brief summary of findings"
    }}

    CRITICAL: 
    - Set "csv_likely_correct" to true if the CSV value appears more accurate than what you see in the image
    - Set "likely_correct_value" to whichever value you believe is actually correct
    - Only report issues where you're confident there's a real discrepancy (confidence > 0.7)
    - Consider image quality, OCR challenges, and context when making decisions
    """
    
    if provider == "Google Gemini":
        return validate_with_gemini(base64_image_data, api_key, model_name, prompt)
    else:
        return validate_with_claude(base64_image_data, api_key, model_name, prompt)

def get_smart_corrections(base64_image_data, issues, extracted_df, api_key, model_name, provider):
    """Get AI-powered smart corrections, considering that CSV might be correct."""
    
    if not issues:
        return {}
    
    csv_correct_issues = [issue for issue in issues if issue.get('csv_likely_correct', False)]
    image_correct_issues = [issue for issue in issues if not issue.get('csv_likely_correct', False)]
    
    if not image_correct_issues:
        st.info("All identified issues suggest the CSV data is already correct. No corrections needed!")
        return {}
    
    csv_data = extracted_df.to_dict('records')
    headers = list(extracted_df.columns)
    
    issues_summary = []
    for issue in image_correct_issues:
        issues_summary.append({
            "row_index": issue.get('row_index'),
            "column_name": issue.get('column_name'),
            "current_csv_value": issue.get('csv_value'),
            "what_ai_sees_in_image": issue.get('image_value'),
            "likely_correct_value": issue.get('likely_correct_value'),
            "issue_type": issue.get('issue_type'),
            "original_reasoning": issue.get('reasoning', '')
        })
    
    prompt = f"""
    You are a data correction specialist. Based on previous analysis, the following issues have been identified where the IMAGE appears to contain the correct values and the CSV needs correction.
    
    CONTEXT:
    - Original CSV Data: {json.dumps(csv_data, indent=2)}
    - Headers: {headers}
    
    ISSUES TO CORRECT (where IMAGE is likely correct):
    {json.dumps(issues_summary, indent=2)}
    
    For each issue, look at the image again very carefully and provide the corrected CSV value.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "corrections": [
            {{
                "row_index": 0,
                "column_name": "column_name",
                "corrected_value": "the exact value as seen in the image",
                "confidence": 0.95,
                "reasoning": "why this correction should be made based on image analysis"
            }}
        ]
    }}
    
    Be very precise with the corrected values. Match exactly what you see in the image.
    """
    
    if provider == "Google Gemini":
        result = validate_with_gemini(base64_image_data, api_key, model_name, prompt)
    else:
        result = validate_with_claude(base64_image_data, api_key, model_name, prompt)
    
    if result and 'corrections' in result:
        corrections_dict = {}
        for correction in result['corrections']:
            key = (correction.get('row_index'), correction.get('column_name'))
            corrections_dict[key] = {
                'corrected_value': correction.get('corrected_value'),
                'confidence': correction.get('confidence', 0.0),
                'reasoning': correction.get('reasoning', 'No reasoning provided')
            }
        return corrections_dict
    
    return {}

def apply_corrections_to_dataframe(df, corrections_dict):
    """Apply corrections to the dataframe and return the corrected version."""
    corrected_df = df.copy()
    correction_log = []
    
    for (row_idx, col_name), correction_info in corrections_dict.items():
        if row_idx < len(corrected_df) and col_name in corrected_df.columns:
            old_value = corrected_df.iloc[row_idx][col_name]
            new_value = correction_info['corrected_value']
            
            if pd.api.types.is_numeric_dtype(corrected_df[col_name]):
                try: new_value = pd.to_numeric(new_value)
                except: pass
            
            corrected_df.iloc[row_idx, corrected_df.columns.get_loc(col_name)] = new_value
            
            correction_log.append({
                'row': row_idx + 1, 'column': col_name, 'old_value': old_value,
                'new_value': new_value, 'confidence': correction_info['confidence'],
                'reasoning': correction_info['reasoning']
            })
    return corrected_df, correction_log

def validate_with_gemini(base64_image_data, api_key, model_name, prompt):
    """Validate using Gemini API."""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": base64_image_data}}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        json_response_text = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(json_response_text)
    except Exception as e:
        st.error(f"Gemini validation error: {e}")
        return None

def validate_with_claude(base64_image_data, api_key, model_name, prompt):
    """Validate using Claude API."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model_name, max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}},
                {"type": "text", "text": prompt}
            ]}]
        )
        json_response_text = message.content[0].text
        if "```json" in json_response_text:
            json_response_text = json_response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_response_text:
            json_response_text = json_response_text.split("```")[1].strip()
        return json.loads(json_response_text)
    except Exception as e:
        st.error(f"Claude validation error: {e}")
        return None

# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 3: 🤖 Table Extraction Validator")

# MODIFICATION: Check if data from previous steps is available
if st.session_state.get('original_df') is None or not st.session_state.get('converted_pil_images'):
    st.warning("⚠️ Please complete the previous steps first: Convert a PDF and then Extract a Table.")
else:
    st.markdown("The AI agent will now compare the original image with the extracted data to find errors and suggest corrections.")

    # Get the data from session state
    original_df = st.session_state.original_df
    image_to_validate = st.session_state.converted_pil_images[st.session_state.selected_image_index]
    
    # MODIFICATION: Display the inputs for validation instead of uploaders
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Image to Validate")
        st.image(image_to_validate, use_column_width=True)
    with col2:
        st.subheader("📊 Extracted Data")
        st.dataframe(original_df.head())
        st.info(f"Validating **{original_df.shape[0]} rows** and **{original_df.shape[1]} columns**.")

    # Sidebar configuration remains the same
    with st.sidebar:
        st.header("⚙️ Configuration")
        ai_provider = st.selectbox("Choose AI Provider:", ("Google Gemini", "Anthropic Claude"))
        if ai_provider == "Google Gemini":
            model_options = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
        else:
            model_options = ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        selected_model = st.selectbox("Choose AI Model:", options=model_options)

    # Main processing
    if st.button("🔍 Validate Extraction", type="primary"):
        api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
        if not api_key:
            st.error(f"Please add your {ai_provider} API key to the .env file.")
        else:
            base64_image = prepare_image_from_pil(image_to_validate)
            if not base64_image:
                st.error("Could not process the image for validation.")
            else:
                with st.spinner(f"AI is comparing the image with your data using {ai_provider}..."):
                    validation_results = validate_with_ai_comparison(base64_image, original_df, api_key, selected_model, ai_provider)
                    st.session_state.validation_results = validation_results
    
    # It will work automatically because it depends on st.session_state.validation_results
    if st.session_state.validation_results:
        # --- All your logic for displaying results, corrections, and downloads ---
        # --- will now work using the data from session state. ---
        st.subheader("🔍 AI Validation Results")
        # Example of how the rest of your code will just work:
        summary = st.session_state.validation_results.get('summary', 'No summary provided')
        st.info(f"**AI Summary:** {summary}")
        # (And so on for the rest of your detailed display logic)