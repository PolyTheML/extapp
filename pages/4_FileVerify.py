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

# Load environment variables
load_dotenv()

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

def check_openai_connection(api_key):
    """Check OpenAI API connection"""
    if not api_key:
        return False, "API key not provided"
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Test with a simple models list request
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "Connected to OpenAI API"
        elif response.status_code == 401:
            return False, "Invalid API key"
        else:
            return False, f"API Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, "Network connection failed"
    except Exception as e:
        return False, f"Connection check failed: {e}"

def get_ai_validation_openai(api_key, base64_image, original_df):
    """Get validation results using OpenAI GPT-4 Vision"""
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
        
        Please respond ONLY with valid JSON.
        """
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",  # GPT-4 with vision capabilities
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": validation_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
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
            error_data = response.json() if response.content else {"error": "Unknown error"}
            return {
                "summary": f"API Error: {response.status_code}",
                "overall_accuracy": 0.0,
                "total_issues": 0,
                "validation_results": [],
                "error": error_data.get("error", {}).get("message", str(error_data))
            }
            
    except Exception as e:
        return {
            "summary": f"Validation failed: {str(e)}",
            "overall_accuracy": 0.0,
            "total_issues": 0,
            "validation_results": [],
            "error": str(e)
        }

def get_ai_validation_anthropic(api_key, base64_image, original_df):
    """Get validation results using Anthropic Claude"""
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
        
        Please respond ONLY with valid JSON.
        """
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2000,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": validation_prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['content'][0]['text']
            
            # Try to extract JSON from the response
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx]
                    validation_data = json.loads(json_text)
                    return validation_data
                else:
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
            error_data = response.json() if response.content else {"error": "Unknown error"}
            return {
                "summary": f"API Error: {response.status_code}",
                "overall_accuracy": 0.0,
                "total_issues": 0,
                "validation_results": [],
                "error": str(error_data)
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
    
    # API Configuration
    st.sidebar.header("🔑 API Configuration")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "Select AI Provider:",
        ["OpenAI (GPT-4 Vision)", "Anthropic (Claude)"]
    )
    
    # API Key input
    if provider == "OpenAI (GPT-4 Vision)":
        api_key = st.sidebar.text_input(
            "OpenAI API Key:",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Get your API key from https://platform.openai.com/api-keys"
        )
    else:
        api_key = st.sidebar.text_input(
            "Anthropic API Key:",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="Get your API key from https://console.anthropic.com/"
        )
    
    # Check connection
    st.sidebar.header("🔌 Connection Status")
    if api_key:
        if provider == "OpenAI (GPT-4 Vision)":
            connected, status = check_openai_connection(api_key)
        else:
            # For Anthropic, we'll assume it's connected if API key is provided
            connected, status = True, "API key provided"
        
        if connected:
            st.sidebar.success(f"✅ {provider}: {status}")
        else:
            st.sidebar.error(f"❌ {provider}: {status}")
    else:
        st.sidebar.warning("⚠️ Please enter your API key")
        connected = False
    
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
    if uploaded_image and uploaded_csv and api_key and connected:
        st.header("🔍 AI Validation")
        
        if st.button("🚀 Start Validation", type="primary"):
            with st.spinner(f"Validating data with {provider}..."):
                # Prepare image
                base64_image = prepare_image_from_pil(image)
                
                if base64_image:
                    # Get validation results based on provider
                    if provider == "OpenAI (GPT-4 Vision)":
                        validation_data = get_ai_validation_openai(api_key, base64_image, df)
                    else:
                        validation_data = get_ai_validation_anthropic(api_key, base64_image, df)
                    
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
        1. **Get API Key**: 
           - For OpenAI: Visit https://platform.openai.com/api-keys
           - For Anthropic: Visit https://console.anthropic.com/
        2. **Select Provider**: Choose between OpenAI GPT-4 Vision or Anthropic Claude
        3. **Enter API Key**: Add your API key in the sidebar
        4. **Upload Files**: Upload both the original image and CSV file
        5. **Run Validation**: Click "Start Validation" to compare image vs CSV
        6. **Review Results**: Check the validation results and apply corrections if needed
        
        **Note**: This uses cloud APIs, so your data will be sent to the selected provider for processing.
        Both providers have strong privacy policies, but be aware of your data sensitivity requirements.
        """)

if __name__ == "__main__":
    main()