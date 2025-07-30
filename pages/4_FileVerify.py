import streamlit as st
import pandas as pd
from PIL import Image
import os
import base64
import json
from io import BytesIO
from dotenv import load_dotenv
import requests
import tempfile
import time
import platform
import subprocess

# --- Windows Helper Functions ---
def is_windows():
    """Check if running on Windows"""
    return platform.system().lower() == 'windows'

def check_ollama_connection():
    """Check Ollama connection and get available models"""
    try:
        session = requests.Session()
        session.trust_env = False  # Avoid Windows proxy issues
        
        response = session.get("http://localhost:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            vision_models = [m['name'] for m in models 
                           if 'vision' in m['name'].lower() or 'llava' in m['name'].lower()]
            return True, vision_models, "Connected"
        else:
            return False, [], f"Server error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, [], "Server not running"
    except Exception as e:
        return False, [], f"Connection check failed: {e}"

# --- Helper Functions ---
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

def get_ai_validation_ollama(model_name, base64_image, original_df):
    """Get validation results using Ollama"""
    try:
        validation_prompt = f"""
        You are an expert data validator. Compare the extracted CSV data with the original image and identify any discrepancies.
        **CSV Data to Validate:**
        {original_df.to_string()}
        **Your Task:**
        1. Carefully examine the image and compare it with the CSV data above
        2. Look for differences in: numbers, text, dates, formatting, missing data, extra data
        3. For each discrepancy found, determine what the correct value should be based on the image
        **Required JSON Output:**
        {{
            "summary": "Brief description of findings",
            "overall_accuracy": 0.85,
            "total_issues": 2,
            "validation_results": [
                {{
                    "row_index": 0,
                    "column_name": "date",
                    "image_value": "2025-07-28",
                    "csv_value": "28/07/2025", 
                    "likely_correct_value": "2025-07-28",
                    "confidence": 0.9,
                    "csv_likely_correct": false,
                    "description": "Date format mismatch",
                    "reasoning": "Image shows YYYY-MM-DD format"
                }}
            ]
        }}
        """
        
        # Create temp file with proper Windows handling
        temp_dir = os.environ.get('TEMP', tempfile.gettempdir()) if is_windows() else tempfile.gettempdir()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir) as temp_file:
            # Convert base64 back to image and save
            image_data = base64.b64decode(base64_image)
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        # Read and encode for Ollama
        with open(temp_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Windows-specific request handling
        session = requests.Session()
        session.trust_env = False
        
        # Prepare the request payload
        payload = {
            "model": model_name,
            "prompt": validation_prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        # Make the request
        response = session.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx]
                    validation_data = json.loads(json_text)
                    return validation_data
                else:
                    # Fallback if no JSON found
                    return {
                        "summary": "Validation completed but no structured data returned",
                        "overall_accuracy": 0.5,
                        "total_issues": 0,
                        "validation_results": [],
                        "raw_response": response_text
                    }
            except json.JSONDecodeError:
                return {
                    "summary": "Error parsing validation response",
                    "overall_accuracy": 0.0,
                    "total_issues": 0,
                    "validation_results": [],
                    "raw_response": response_text
                }
        else:
            return {
                "summary": f"API Error: {response.status_code}",
                "overall_accuracy": 0.0,
                "total_issues": 0,
                "validation_results": [],
                "error": response.text
            }
            
    except Exception as e:
        return {
            "summary": f"Validation failed: {str(e)}",
            "overall_accuracy": 0.0,
            "total_issues": 0,
            "validation_results": [],
            "error": str(e)
        }

def apply_corrections(df, validation_results):
    """Apply corrections to the dataframe based on validation results"""
    corrected_df = df.copy()
    corrections_made = []
    
    for issue in validation_results.get('validation_results', []):
        if not issue.get('csv_likely_correct', True):
            row_idx = issue.get('row_index', 0)
            col_name = issue.get('column_name', '')
            new_value = issue.get('likely_correct_value', '')
            
            if col_name in corrected_df.columns and row_idx < len(corrected_df):
                old_value = corrected_df.iloc[row_idx][col_name]
                corrected_df.iloc[row_idx, corrected_df.columns.get_loc(col_name)] = new_value
                corrections_made.append({
                    'row': row_idx,
                    'column': col_name,
                    'old_value': old_value,
                    'new_value': new_value,
                    'confidence': issue.get('confidence', 0.5)
                })
    
    return corrected_df, corrections_made

def display_validation_results(validation_data):
    """Display validation results in Streamlit"""
    st.subheader("🔍 Validation Results")
    
    # Overall summary
    col1, col2, col3 = st.columns(3)
    with col1:
        accuracy = validation_data.get('overall_accuracy', 0.0)
        st.metric("Overall Accuracy", f"{accuracy:.1%}")
    
    with col2:
        total_issues = validation_data.get('total_issues', 0)
        st.metric("Issues Found", total_issues)
    
    with col3:
        summary = validation_data.get('summary', 'No summary available')
        st.info(f"Summary: {summary}")
    
    # Detailed issues
    validation_results = validation_data.get('validation_results', [])
    if validation_results:
        st.subheader("📋 Detailed Issues")
        
        for i, issue in enumerate(validation_results):
            with st.expander(f"Issue {i+1}: {issue.get('description', 'Unknown issue')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Details:**")
                    st.write(f"Row: {issue.get('row_index', 'N/A')}")
                    st.write(f"Column: {issue.get('column_name', 'N/A')}")
                    st.write(f"Confidence: {issue.get('confidence', 0.0):.1%}")
                
                with col2:
                    st.write("**Values:**")
                    st.write(f"Image shows: `{issue.get('image_value', 'N/A')}`")
                    st.write(f"CSV has: `{issue.get('csv_value', 'N/A')}`")
                    st.write(f"Suggested: `{issue.get('likely_correct_value', 'N/A')}`")
                
                if issue.get('reasoning'):
                    st.write(f"**Reasoning:** {issue.get('reasoning')}")
    
    # Show raw response if available (for debugging)
    if 'raw_response' in validation_data:
        with st.expander("🔧 Raw AI Response (Debug)"):
            st.text(validation_data['raw_response'])

# --- Main Streamlit App ---
def main():
    st.set_page_config(
        page_title="CSV Image Validator",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 CSV Image Validator")
    st.markdown("Upload an image and CSV to validate data accuracy using AI vision models")
    
    # Check Ollama connection
    st.sidebar.header("🔌 Connection Status")
    connected, vision_models, status = check_ollama_connection()
    
    if connected:
        st.sidebar.success(f"✅ Ollama: {status}")
        if vision_models:
            st.sidebar.success(f"Found {len(vision_models)} vision models")
            selected_model = st.sidebar.selectbox("Select Vision Model:", vision_models)
        else:
            st.sidebar.warning("No vision models found")
            st.sidebar.info("Please install a vision model like llava")
            selected_model = None
    else:
        st.sidebar.error(f"❌ Ollama: {status}")
        st.sidebar.info("Please start Ollama server")
        selected_model = None
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📷 Upload Image")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload the original image containing the data"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.header("📊 Upload CSV")
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload the CSV data to validate against the image"
        )
        
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.write("**CSV Preview:**")
            st.dataframe(df, use_container_width=True)
    
    # Validation section
    if uploaded_image and uploaded_csv and selected_model and connected:
        st.header("🔍 AI Validation")
        
        if st.button("🚀 Start Validation", type="primary"):
            with st.spinner("Validating data with AI vision model..."):
                # Prepare image
                base64_image = prepare_image_from_pil(image)
                
                if base64_image:
                    # Get validation results
                    validation_data = get_ai_validation_ollama(selected_model, base64_image, df)
                    
                    # Display results
                    display_validation_results(validation_data)
                    
                    # Option to apply corrections
                    if validation_data.get('validation_results'):
                        st.subheader("🔧 Apply Corrections")
                        
                        if st.button("Apply Suggested Corrections"):
                            corrected_df, corrections_made = apply_corrections(df, validation_data)
                            
                            if corrections_made:
                                st.success(f"Applied {len(corrections_made)} corrections!")
                                
                                # Show corrections made
                                st.write("**Corrections Applied:**")
                                for correction in corrections_made:
                                    st.write(f"Row {correction['row']}, Column '{correction['column']}': "
                                           f"`{correction['old_value']}` → `{correction['new_value']}` "
                                           f"(Confidence: {correction['confidence']:.1%})")
                                
                                # Show corrected data
                                st.write("**Corrected CSV:**")
                                st.dataframe(corrected_df, use_container_width=True)
                                
                                # Download corrected CSV
                                csv_buffer = BytesIO()
                                corrected_df.to_csv(csv_buffer, index=False)
                                csv_buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 Download Corrected CSV",
                                    data=csv_buffer.getvalue(),
                                    file_name="corrected_data.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("No corrections were applied.")
                else:
                    st.error("Failed to process the image")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Start Ollama**: Make sure Ollama is running with a vision model (like llava)
        2. **Upload Image**: Upload the original image containing your data
        3. **Upload CSV**: Upload the CSV file you want to validate
        4. **Select Model**: Choose a vision model from the sidebar
        5. **Run Validation**: Click "Start Validation" to compare image vs CSV
        6. **Review Results**: Check the validation results and apply corrections if needed
        
        **Supported Models**: llava, llava:13b, bakllava, or other vision-capable models
        """)

if __name__ == "__main__":
    main()