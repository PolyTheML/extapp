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

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# Initialize session state
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'corrections_applied' not in st.session_state:
    st.session_state.corrections_applied = False
if 'corrected_df' not in st.session_state:
    st.session_state.corrected_df = None

# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 3: 🤖 Table Extraction Validator")

# Check if data from previous steps is available
if st.session_state.get('original_df') is None or not st.session_state.get('converted_pil_images'):
    st.warning("⚠️ Please complete the previous steps first: Convert a PDF and then Extract a Table.")
else:
    st.markdown("The AI agent will now compare the original image with the extracted data to find errors and suggest corrections.")

    # Get the data from session state
    original_df = st.session_state.original_df
    image_to_validate = st.session_state.converted_pil_images[st.session_state.selected_image_index]
    
    # Display the inputs for validation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Image to Validate")
        st.image(image_to_validate, use_container_width=True)
    with col2:
        st.subheader("📊 Extracted Data")
        st.dataframe(original_df.head())
        st.info(f"Validating **{original_df.shape[0]} rows** and **{original_df.shape[1]} columns**.")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        ai_provider = st.selectbox("Choose AI Provider:", ("Google Gemini", "Anthropic Claude"))
        if ai_provider == "Google Gemini":
            model_options = ["gemini-2.0-flash-exp", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
        else:
            model_options = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
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
    
    # Display validation results
    if st.session_state.validation_results:
        st.divider()
        st.subheader("🔍 AI Validation Results")
        
        results = st.session_state.validation_results
        summary = results.get('summary', 'No summary provided')
        overall_accuracy = results.get('overall_accuracy', 0.0)
        total_issues = results.get('total_issues', 0)
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", f"{overall_accuracy:.1%}")
        with col2:
            st.metric("Total Issues Found", total_issues)
        with col3:
            confidence_color = "green" if overall_accuracy > 0.9 else "orange" if overall_accuracy > 0.7 else "red"
            st.metric("Status", "✅ High" if overall_accuracy > 0.9 else "⚠️ Medium" if overall_accuracy > 0.7 else "❌ Low")
        
        st.info(f"**AI Summary:** {summary}")
        
        # Display detailed issues
        validation_issues = results.get('validation_results', [])
        if validation_issues:
            st.subheader("📋 Detailed Issues Found")
            
            # Create a DataFrame for better display
            issues_data = []
            for i, issue in enumerate(validation_issues):
                issues_data.append({
                    'Row': issue.get('row_index', 0) + 1,
                    'Column': issue.get('column_name', 'Unknown'),
                    'Issue Type': issue.get('issue_type', 'Unknown'),
                    'Image Value': issue.get('image_value', 'N/A'),
                    'CSV Value': issue.get('csv_value', 'N/A'),
                    'Likely Correct': issue.get('likely_correct_value', 'N/A'),
                    'Confidence': f"{issue.get('confidence', 0.0):.1%}",
                    'CSV Correct?': "✅" if issue.get('csv_likely_correct', False) else "❌"
                })
            
            issues_df = pd.DataFrame(issues_data)
            st.dataframe(issues_df, use_container_width=True)
            
            # Show expandable details for each issue
            with st.expander("📖 View Detailed Reasoning"):
                for i, issue in enumerate(validation_issues):
                    st.write(f"**Issue {i+1}:** Row {issue.get('row_index', 0) + 1}, Column '{issue.get('column_name', 'Unknown')}'")
                    st.write(f"**Description:** {issue.get('description', 'No description')}")
                    st.write(f"**Reasoning:** {issue.get('reasoning', 'No reasoning provided')}")
                    st.divider()
            
            # Smart corrections section
            st.subheader("🔧 Smart Corrections")
            
            if st.button("🚀 Get AI Corrections", type="secondary"):
                api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
                base64_image = prepare_image_from_pil(image_to_validate)
                
                with st.spinner("AI is analyzing issues and generating smart corrections..."):
                    corrections = get_smart_corrections(base64_image, validation_issues, original_df, api_key, selected_model, ai_provider)
                    st.session_state.corrections = corrections
            
            # Apply corrections if available
            if hasattr(st.session_state, 'corrections') and st.session_state.corrections:
                st.write("**Proposed Corrections:**")
                
                corrections_data = []
                for (row_idx, col_name), correction_info in st.session_state.corrections.items():
                    corrections_data.append({
                        'Row': row_idx + 1,
                        'Column': col_name,
                        'Current Value': original_df.iloc[row_idx][col_name],
                        'Corrected Value': correction_info['corrected_value'],
                        'Confidence': f"{correction_info['confidence']:.1%}",
                        'Reasoning': correction_info['reasoning']
                    })
                
                corrections_df = pd.DataFrame(corrections_data)
                st.dataframe(corrections_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Apply All Corrections", type="primary"):
                        corrected_df, correction_log = apply_corrections_to_dataframe(original_df, st.session_state.corrections)
                        st.session_state.corrected_df = corrected_df
                        st.session_state.corrections_applied = True
                        st.success(f"Applied {len(correction_log)} corrections!")
                        
                        # Show correction log
                        st.write("**Correction Log:**")
                        log_df = pd.DataFrame(correction_log)
                        st.dataframe(log_df, use_container_width=True)
                
                with col2:
                    if st.button("❌ Skip Corrections"):
                        st.session_state.corrected_df = original_df.copy()
                        st.session_state.corrections_applied = True
                        st.info("Skipped corrections. Using original data.")
        
        else:
            st.success("🎉 No issues found! The extracted data appears to be accurate.")
            st.session_state.corrected_df = original_df.copy()
            st.session_state.corrections_applied = True
    
    # Download section for corrected data
    if st.session_state.corrections_applied and st.session_state.corrected_df is not None:
        st.divider()
        st.subheader("📥 Download Validated Data")
        
        # Show final data preview
        st.write("**Final Validated Data:**")
        st.dataframe(st.session_state.corrected_df)
        
        col1, col2 = st.columns(2)
        with col1:
            file_name = st.text_input("File Name (without extension):", value="validated_table", key="validated_filename")
        with col2:
            file_format = st.selectbox("Choose Format:", ["Excel (.xlsx)", "CSV (.csv)"], key="validated_format")
        
        # Generate download button
        if file_name:
            if file_format == "Excel (.xlsx)":
                file_data = to_excel(st.session_state.corrected_df)
                file_extension = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:  # CSV
                file_data = st.session_state.corrected_df.to_csv(index=False).encode('utf-8')
                file_extension = "csv"
                mime_type = "text/csv"
            
            st.download_button(
                label=f"📥 Download Validated Data as {file_format}",
                data=file_data,
                file_name=f"{file_name}.{file_extension}",
                mime=mime_type,
                type="primary"
            )
        else:
            st.warning("Please enter a file name to enable download.")
        
        # Success message
        st.success("✅ Validation complete! Your data has been verified and is ready for use.")
        st.info("💡 **Next Step:** You can now use this validated data for further analysis or load it into your database systems.")