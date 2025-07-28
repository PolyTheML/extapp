import streamlit as st
import pandas as pd
from PIL import Image
import os
import base64
import json
from io import BytesIO
from dotenv import load_dotenv

# --- Mock Helper Functions (for improved code structure) ---
# In your actual app, these would contain your full AI logic for Gemini/Claude
# The core UI and workflow improvements below do not depend on their internal code.

def prepare_image_from_pil(pil_image):
    """Converts a PIL image to a base64 string."""
    try:
        output_buffer = BytesIO()
        pil_image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        return base64.b64encode(byte_data).decode('utf-8')
    except Exception as e:
        st.error(f"Error preparing image: {e}")
        return None

def get_ai_validation(api_key, model_name, provider, base64_image, original_df):
    """Mocks a call to an AI to get validation results."""
    st.info(f"Pretending to call {provider} with model {model_name}...")
    # This would be your actual API call. For this example, we return a mock response.
    return {
        "summary": "Found potential discrepancies in date formats and numerical values.",
        "overall_accuracy": 0.85, "total_issues": 2,
        "validation_results": [
            {"row_index": 0, "column_name": "date", "image_value": "2025-07-28", "csv_value": "28/07/2025", "likely_correct_value": "2025-07-28", "confidence": 0.9, "csv_likely_correct": False, "description": "Date format mismatch.", "reasoning": "The image shows YYYY-MM-DD format."},
            {"row_index": 1, "column_name": "amount", "image_value": "1,500.00", "csv_value": "1500.00", "likely_correct_value": "1,500.00", "confidence": 0.95, "csv_likely_correct": False, "description": "Missing thousands separator.", "reasoning": "The image clearly shows a comma as a thousands separator."}
        ]
    }

def get_ai_corrections(api_key, model_name, provider, base64_image, issues):
    """Mocks a call to an AI to get smart corrections."""
    # This would be your actual API call. We return a mock response based on issues.
    corrections_dict = {}
    for issue in issues:
        if not issue.get('csv_likely_correct'):
            key = (issue.get('row_index'), issue.get('column_name'))
            corrections_dict[key] = {
                'corrected_value': issue.get('likely_correct_value'),
                'confidence': issue.get('confidence'),
                'reasoning': issue.get('reasoning')
            }
    return corrections_dict

def apply_corrections_to_dataframe(df, corrections_dict):
    """Applies a dictionary of corrections to a DataFrame."""
    corrected_df = df.copy()
    for (row_idx, col_name), info in corrections_dict.items():
        if row_idx < len(corrected_df) and col_name in corrected_df.columns:
            corrected_df.loc[row_idx, col_name] = info['corrected_value']
    return corrected_df

# --- Main App UI ---
st.set_page_config(layout="wide")
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 3: 🤖 Interactive File Validator")

# Check for data from previous steps
if 'original_df' not in st.session_state or st.session_state.original_df is None:
    st.warning("⚠️ Please complete the previous steps first: Convert a PDF and then Extract a Table.")
    st.stop()

## 1. Configuration & Initial Display
st.markdown("Compare the original image with the extracted data, then review and apply AI-powered corrections interactively.")

original_df = st.session_state.original_df
image_to_validate = st.session_state.converted_pil_images[st.session_state.selected_image_index]

col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Image to Validate")
    st.image(image_to_validate, use_container_width=True)
with col2:
    st.subheader("📊 Extracted Data")
    st.dataframe(original_df.head(), use_container_width=True)
    st.info(f"Validating **{original_df.shape[0]} rows** and **{original_df.shape[1]} columns**.")

with st.sidebar:
    st.header("⚙️ Configuration")
    ai_provider = st.selectbox("Choose AI Provider:", ("Google Gemini", "Anthropic Claude"), key="validator_provider")
    model_options = ["gemini-1.5-pro-latest"] if ai_provider == "Google Gemini" else ["claude-3-5-sonnet-20240620"]
    selected_model = st.selectbox("Choose AI Model:", options=model_options, key="validator_model")

## 2. Validation Engine
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔍 Validate Data", type="primary", use_container_width=True):
        api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
        if not api_key:
            st.error(f"Please add your {ai_provider} API key to the .env file.")
        else:
            with st.spinner("Preparing image and running AI validation..."):
                base64_image = prepare_image_from_pil(image_to_validate)
                st.session_state.base64_image_to_validate = base64_image
                if base64_image:
                    results = get_ai_validation(api_key, selected_model, ai_provider, base64_image, original_df)
                    st.session_state.validation_results = results
                    # Clear previous corrections when re-validating
                    if 'corrections_df' in st.session_state:
                        del st.session_state.corrections_df
with col2:
    if st.button("🔄 Reset Validation", use_container_width=True):
        keys_to_clear = ['validation_results', 'corrections_df', 'corrected_df']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Validation state has been reset.")
        st.rerun()

## 3. Interactive Correction Workflow
if 'validation_results' in st.session_state:
    results = st.session_state.validation_results
    st.subheader("🔍 AI Validation Results")

    # Display summary metrics
    c1, c2, c3 = st.columns(3)
    accuracy = results.get('overall_accuracy', 0.0)
    c1.metric("Overall Accuracy", f"{accuracy:.1%}")
    c2.metric("Total Issues Found", results.get('total_issues', 0))
    status = "✅ High" if accuracy > 0.9 else "⚠️ Medium" if accuracy > 0.7 else "❌ Low"
    c3.metric("Status", status)
    
    validation_issues = results.get('validation_results', [])
    if not validation_issues:
        st.success("🎉 No issues found! The extracted data appears to be accurate.")
        st.session_state.corrected_df = original_df.copy()
    else:
        st.info(f"**AI Summary:** {results.get('summary', 'No summary provided.')}")
        
        # This is the new, interactive part
        st.subheader("🔧 Review & Apply Corrections")
        
        # Generate the corrections dataframe if it doesn't exist
        if 'corrections_df' not in st.session_state:
            with st.spinner("AI is generating smart corrections..."):
                api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
                base64_image = st.session_state.base64_image_to_validate
                corrections_dict = get_ai_corrections(api_key, selected_model, ai_provider, base64_image, validation_issues)
                
                if corrections_dict:
                    corrections_data = []
                    for (row, col), info in corrections_dict.items():
                        corrections_data.append({
                            "Apply": True,
                            "Row": row,
                            "Column": col,
                            "Original Value": original_df.loc[row, col],
                            "Suggested Correction": info['corrected_value'],
                            "Reasoning": info['reasoning']
                        })
                    st.session_state.corrections_df = pd.DataFrame(corrections_data)
                else:
                    st.session_state.corrections_df = pd.DataFrame() # Empty df

        if not st.session_state.corrections_df.empty:
            st.info("Review the AI's suggestions below. Edit any correction directly in the table or uncheck 'Apply' to ignore it.")
            
            # Use the data editor for an interactive experience
            edited_df = st.data_editor(
                st.session_state.corrections_df,
                column_config={
                    "Apply": st.column_config.CheckboxColumn(default=True),
                    "Reasoning": st.column_config.TextColumn(width="large")
                },
                use_container_width=True,
                hide_index=True,
                key="correction_editor"
            )
            
            if st.button("✅ Apply Selected Corrections", type="primary"):
                corrections_to_apply = {}
                for _, row in edited_df.iterrows():
                    if row["Apply"]:
                        corrections_to_apply[(row["Row"], row["Column"])] = {'corrected_value': row["Suggested Correction"]}
                
                st.session_state.corrected_df = apply_corrections_to_dataframe(original_df, corrections_to_apply)
                st.success(f"Applied {len(corrections_to_apply)} corrections successfully!")
                st.rerun()


## 4. Final Download
if 'corrected_df' in st.session_state and st.session_state.corrected_df is not None:
    st.subheader("📥 Download Validated Data")
    st.markdown("Your data has been verified and corrected. You can now download the final version.")
    
    st.dataframe(st.session_state.corrected_df, use_container_width=True)
    
    final_csv = st.session_state.corrected_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download as CSV",
        data=final_csv,
        file_name="validated_table.csv",
        mime="text/csv",
        type="primary"
    )