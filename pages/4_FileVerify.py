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
    try:
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        max_dim = 2048
        h, w, _ = img_cv.shape
        if h > max_dim or w > max_dim:
            if h > w: new_h, new_w = max_dim, int(w * (max_dim / h))
            else: new_h, new_w = int(h * (max_dim / w)), max_dim
            img_cv = cv2.resize(img_cv, (new_w, new_h))
        _, buffer = cv2.imencode('.png', img_cv)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        st.error(f"Error preparing image: {e}")
        return None

def validate_with_ai_comparison(base64_image_data, extracted_df, api_key, model_name, provider):
    csv_data = extracted_df.to_dict('records')
    headers = list(extracted_df.columns)
    prompt = f"You are a data validation expert... (rest of your long prompt)" # Keeping prompt brief for clarity
    if provider == "Google Gemini": return validate_with_gemini(base64_image_data, api_key, model_name, prompt)
    else: return validate_with_claude(base64_image_data, api_key, model_name, prompt)

def get_smart_corrections(base64_image_data, issues, extracted_df, api_key, model_name, provider):
    if not issues: return {}
    image_correct_issues = [issue for issue in issues if not issue.get('csv_likely_correct', False)]
    if not image_correct_issues:
        st.info("All identified issues suggest the CSV data is already correct. No corrections needed!")
        return {}
    csv_data, headers = extracted_df.to_dict('records'), list(extracted_df.columns)
    issues_summary = [{"row_index": issue.get('row_index'), "column_name": issue.get('column_name'), "current_csv_value": issue.get('csv_value'), "what_ai_sees_in_image": issue.get('image_value'), "likely_correct_value": issue.get('likely_correct_value'), "issue_type": issue.get('issue_type'), "original_reasoning": issue.get('reasoning', '')} for issue in image_correct_issues]
    prompt = f"You are a data correction specialist... (rest of your long prompt)" # Keeping prompt brief
    result = validate_with_gemini(base64_image_data, api_key, model_name, prompt) if provider == "Google Gemini" else validate_with_claude(base64_image_data, api_key, model_name, prompt)
    if result and 'corrections' in result:
        return {(c.get('row_index'), c.get('column_name')): {'corrected_value': c.get('corrected_value'), 'confidence': c.get('confidence', 0.0), 'reasoning': c.get('reasoning', 'No reasoning provided')} for c in result['corrections']}
    return {}

def apply_corrections_to_dataframe(df, corrections_dict):
    corrected_df, correction_log = df.copy(), []
    for (row_idx, col_name), info in corrections_dict.items():
        if row_idx < len(corrected_df) and col_name in corrected_df.columns:
            old_value, new_value = corrected_df.iloc[row_idx][col_name], info['corrected_value']
            if pd.api.types.is_numeric_dtype(corrected_df[col_name]):
                try: new_value = pd.to_numeric(new_value)
                except: pass
            corrected_df.iloc[row_idx, corrected_df.columns.get_loc(col_name)] = new_value
            correction_log.append({'row': row_idx + 1, 'column': col_name, 'old_value': old_value, 'new_value': new_value, 'confidence': info['confidence'], 'reasoning': info['reasoning']})
    return corrected_df, correction_log

def validate_with_gemini(base64_image_data, api_key, model_name, prompt):
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": base64_image_data}}]}], "generationConfig": {"responseMimeType": "application/json"}}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        json_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Gemini validation error: {e}")
        return None

def validate_with_claude(base64_image_data, api_key, model_name, prompt):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(model=model_name, max_tokens=4096, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}}, {"type": "text", "text": prompt}]}])
        json_text = message.content[0].text
        if "```json" in json_text: json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text: json_text = json_text.split("```")[1].strip()
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Claude validation error: {e}")
        return None

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 3: 🤖 File Validator")

if st.session_state.get('original_df') is None or not st.session_state.get('converted_pil_images'):
    st.warning("⚠️ Please complete the previous steps first: Convert a PDF and then Extract a Table.")
else:
    st.markdown("The AI agent will now compare the original image with the extracted data to find errors and suggest corrections.")
    original_df = st.session_state.original_df
    image_to_validate = st.session_state.converted_pil_images[st.session_state.selected_image_index]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Image to Validate"); st.image(image_to_validate, use_container_width=True)
    with col2:
        st.subheader("📊 Extracted Data"); st.dataframe(original_df.head()); st.info(f"Validating **{original_df.shape[0]} rows** and **{original_df.shape[1]} columns**.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        ai_provider = st.selectbox("Choose AI Provider:", ("Google Gemini", "Anthropic Claude"))
        ### FIX: Corrected and standardized model names
        if ai_provider == "Google Gemini":
            model_options = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
        else:
            model_options = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
        selected_model = st.selectbox("Choose AI Model:", options=model_options)

    if st.button("🔍 Validate Extraction", type="primary"):
        api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
        if not api_key:
            st.error(f"Please add your {ai_provider} API key to the .env file.")
        else:
            ### FIX: Prepare image once and save to session state for efficiency
            with st.spinner("Preparing image for validation..."):
                base64_image = prepare_image_from_pil(image_to_validate)
                st.session_state.base64_image_to_validate = base64_image
            
            if not base64_image:
                st.error("Could not process the image for validation.")
            else:
                with st.spinner(f"AI is comparing the image with your data using {ai_provider}..."):
                    validation_results = validate_with_ai_comparison(base64_image, original_df, api_key, selected_model, ai_provider)
                    st.session_state.validation_results = validation_results
    
    if st.session_state.get('validation_results'):
        st.divider(); st.subheader("🔍 AI Validation Results")
        results = st.session_state.validation_results
        summary, accuracy, issues_count = results.get('summary', 'N/A'), results.get('overall_accuracy', 0.0), results.get('total_issues', 0)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Accuracy", f"{accuracy:.1%}")
        c2.metric("Total Issues Found", issues_count)
        status_text = "✅ High" if accuracy > 0.9 else "⚠️ Medium" if accuracy > 0.7 else "❌ Low"
        c3.metric("Status", status_text)
        st.info(f"**AI Summary:** {summary}")
        
        validation_issues = results.get('validation_results', [])
        if validation_issues:
            st.subheader("📋 Detailed Issues Found")
            issues_df = pd.DataFrame([{'Row': i.get('row_index', 0) + 1, 'Column': i.get('column_name', 'N/A'), 'Issue Type': i.get('issue_type', 'N/A'), 'Image Value': i.get('image_value', 'N/A'), 'CSV Value': i.get('csv_value', 'N/A'), 'Likely Correct': i.get('likely_correct_value', 'N/A'), 'Confidence': f"{i.get('confidence', 0.0):.1%}", 'CSV Correct?': "✅" if i.get('csv_likely_correct', False) else "❌"} for i in validation_issues])
            st.dataframe(issues_df, use_container_width=True)
            
            with st.expander("📖 View Detailed Reasoning"):
                for i, issue in enumerate(validation_issues):
                    st.write(f"**Issue {i+1}:** Row {issue.get('row_index', 0) + 1}, Column '{issue.get('column_name', 'N/A')}'")
                    st.write(f"**Description:** {issue.get('description', 'N/A')}"); st.write(f"**Reasoning:** {issue.get('reasoning', 'N/A')}"); st.divider()
            
            st.subheader("🔧 Smart Corrections")
            if st.button("🚀 Get AI Corrections"):
                api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
                ### FIX: Use the already-prepared base64 image from session state
                base64_image = st.session_state.get('base64_image_to_validate')
                if not base64_image:
                    st.error("Image not prepared. Please click 'Validate Extraction' first.")
                else:
                    with st.spinner("AI is generating smart corrections..."):
                        corrections = get_smart_corrections(base64_image, validation_issues, original_df, api_key, selected_model, ai_provider)
                        st.session_state.corrections = corrections
            
            if 'corrections' in st.session_state and st.session_state.corrections:
                st.write("**Proposed Corrections:**")
                corrections_df = pd.DataFrame([{'Row': r+1, 'Column': c, 'Current Value': original_df.iloc[r][c], 'Corrected Value': info['corrected_value'], 'Confidence': f"{info['confidence']:.1%}", 'Reasoning': info['reasoning']} for (r, c), info in st.session_state.corrections.items()])
                st.dataframe(corrections_df, use_container_width=True)
                
                b1, b2 = st.columns(2)
                if b1.button("✅ Apply All Corrections", type="primary"):
                    corrected_df, log = apply_corrections_to_dataframe(original_df, st.session_state.corrections)
                    st.session_state.corrected_df, st.session_state.corrections_applied = corrected_df, True
                    st.success(f"Applied {len(log)} corrections!"); st.write("**Correction Log:**"); st.dataframe(pd.DataFrame(log), use_container_width=True)
                if b2.button("❌ Skip Corrections"):
                    st.session_state.corrected_df, st.session_state.corrections_applied = original_df.copy(), True
                    st.info("Skipped corrections. Using original data.")
        else:
            st.success("🎉 No issues found! The extracted data appears to be accurate.")
            st.session_state.corrected_df = original_df.copy(); st.session_state.corrections_applied = True
    
    if st.session_state.get('corrections_applied') and st.session_state.get('corrected_df') is not None:
        st.divider(); st.subheader("📥 Download Validated Data")
        st.write("**Final Validated Data:**"); st.dataframe(st.session_state.corrected_df)
        
        dl1, dl2 = st.columns(2)
        file_name = dl1.text_input("File Name (without extension):", value="validated_table", key="validated_filename")
        file_format = dl2.selectbox("Choose Format:", ["Excel (.xlsx)", "CSV (.csv)"], key="validated_format")
        
        if file_name:
            if file_format == "Excel (.xlsx)":
                data, ext, mime = to_excel(st.session_state.corrected_df), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                data, ext, mime = st.session_state.corrected_df.to_csv(index=False).encode('utf-8'), "csv", "text/csv"
            st.download_button(label=f"📥 Download as {file_format}", data=data, file_name=f"{file_name}.{ext}", mime=mime, type="primary")
        else:
            st.warning("Please enter a file name to enable download.")
        st.success("✅ Validation complete!")