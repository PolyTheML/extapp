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
import numpy as np

# --- Securely Load API Keys ---
# This loads variables from your .env file into the environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


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

def create_sample_image():
    """Create a sample data table image for demonstration"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Draw table
    data = [
        ["Name", "Age", "City"],
        ["John Smith", "25", "New York"],
        ["Jane Doe", "30", "Los Angeles"],
        ["Bob Johnson", "35", "Chicago"]
    ]
    
    y_offset = 20
    for row in data:
        x_offset = 20
        for cell in row:
            draw.text((x_offset, y_offset), cell, fill='black', font=font)
            x_offset += 100
        y_offset += 30
    
    # Draw table borders
    draw.rectangle([15, 15, 385, 135], outline='black', width=2)
    for i in range(1, 4):
        draw.line([15, 15 + i * 30, 385, 15 + i * 30], fill='black', width=1)
    for i in range(1, 3):
        draw.line([15 + i * 100, 15, 15 + i * 100, 135], fill='black', width=1)
    
    return img

def create_sample_csv():
    """Create sample CSV data"""
    return pd.DataFrame({
        'Name': ['John Smith', 'Jane Doe', 'Bob Johnson'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
    })

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
            "model": "claude-3-5-sonnet-20240620",
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
    st.markdown("Validate data accuracy by comparing images with CSV data using AI vision models")
    
    # --- MODIFIED: API Configuration Sidebar ---
    st.sidebar.header("🔑 API Configuration")
    
    provider = st.sidebar.selectbox(
        "Select AI Provider:",
        ["OpenAI (GPT-4 Vision)", "Anthropic (Claude)"]
    )
    
    api_key = None
    connected = False

    if provider == "OpenAI (GPT-4 Vision)":
        api_key = OPENAI_API_KEY
        if api_key:
            st.sidebar.success("✅ OpenAI API Key loaded.")
            connected = True
        else:
            st.sidebar.error("❌ OpenAI API Key not found.")
            st.sidebar.info("Please add your OPENAI_API_KEY to the .env file.")
    else: # Anthropic
        api_key = ANTHROPIC_API_KEY
        if api_key:
            st.sidebar.success("✅ Anthropic API Key loaded.")
            connected = True
        else:
            st.sidebar.error("❌ Anthropic API Key not found.")
            st.sidebar.info("Please add your ANTHROPIC_API_KEY to the .env file.")
            
    # --- END OF MODIFICATION ---

    # Input method selection
    st.header("📊 Choose Input Method")
    input_method = st.radio(
        "How would you like to provide your data?",
        ["📁 Upload Files", "✍️ Input Data Directly", "🎯 Use Sample Data"],
        horizontal=True
    )
    
    image = None
    df = None
    
    if input_method == "📁 Upload Files":
        # File uploads
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload the original image containing the data"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("📊 Upload CSV")
            uploaded_csv = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload the CSV data to validate against the image"
            )
            
            if uploaded_csv:
                df = pd.read_csv(uploaded_csv)
                st.write("**CSV Preview:**")
                st.dataframe(df, use_container_width=True)
    
    elif input_method == "✍️ Input Data Directly":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Image Input")
            
            # Option to upload image or take photo
            image_input_type = st.radio(
                "Image source:",
                ["Upload Image", "Camera Capture"],
                key="image_input_type"
            )
            
            if image_input_type == "Upload Image":
                uploaded_image = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg'],
                    key="direct_image_upload"
                )
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                # Camera capture
                camera_image = st.camera_input("Take a photo of your data")
                if camera_image:
                    image = Image.open(camera_image)
                    st.image(image, caption="Captured Image", use_column_width=True)
        
        with col2:
            st.subheader("📊 CSV Data Input")
            
            # CSV input options
            csv_input_type = st.radio(
                "CSV input method:",
                ["Manual Entry", "Paste CSV Text"],
                key="csv_input_type"
            )
            
            if csv_input_type == "Manual Entry":
                st.write("**Manual Data Entry:**")
                
                # Dynamic table creation
                if 'manual_data' not in st.session_state:
                    st.session_state.manual_data = [["Column1", "Column2", "Column3"], ["", "", ""]]
                
                # Controls for table size
                col_a, col_b = st.columns(2)
                with col_a:
                    num_cols = st.number_input("Number of columns:", min_value=1, max_value=10, value=3)
                with col_b:
                    num_rows = st.number_input("Number of data rows:", min_value=1, max_value=20, value=1)
                
                # Adjust session state based on inputs
                if len(st.session_state.manual_data[0]) != num_cols:
                    # Adjust columns
                    if num_cols > len(st.session_state.manual_data[0]):
                        for i in range(len(st.session_state.manual_data)):
                            st.session_state.manual_data[i].extend([""] * (num_cols - len(st.session_state.manual_data[i])))
                    else:
                        for i in range(len(st.session_state.manual_data)):
                            st.session_state.manual_data[i] = st.session_state.manual_data[i][:num_cols]
                
                if len(st.session_state.manual_data) - 1 != num_rows:
                    # Adjust rows (keeping header)
                    if num_rows > len(st.session_state.manual_data) - 1:
                        for _ in range(num_rows - (len(st.session_state.manual_data) - 1)):
                            st.session_state.manual_data.append([""] * num_cols)
                    else:
                        st.session_state.manual_data = st.session_state.manual_data[:num_rows + 1]
                
                # Create input fields
                for i, row in enumerate(st.session_state.manual_data):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        with cols[j]:
                            if i == 0:
                                st.session_state.manual_data[i][j] = st.text_input(
                                    f"Header {j+1}:", 
                                    value=st.session_state.manual_data[i][j],
                                    key=f"header_{j}"
                                )
                            else:
                                st.session_state.manual_data[i][j] = st.text_input(
                                    f"Row {i}, Col {j+1}:", 
                                    value=st.session_state.manual_data[i][j],
                                    key=f"data_{i}_{j}"
                                )
                
                # Convert to DataFrame
                if any(any(cell.strip() for cell in row) for row in st.session_state.manual_data):
                    headers = st.session_state.manual_data[0]
                    data_rows = st.session_state.manual_data[1:]
                    df = pd.DataFrame(data_rows, columns=headers)
                    
                    # Clean empty rows
                    df = df[df.apply(lambda x: x.astype(str).str.strip().ne('').any(), axis=1)]
                    
                    if not df.empty:
                        st.write("**Data Preview:**")
                        st.dataframe(df, use_container_width=True)
            
            else:  # Paste CSV Text
                csv_text = st.text_area(
                    "Paste your CSV data here:",
                    height=200,
                    placeholder="Name,Age,City\nJohn,25,NYC\nJane,30,LA"
                )
                
                if csv_text.strip():
                    try:
                        from io import StringIO
                        df = pd.read_csv(StringIO(csv_text))
                        st.write("**Data Preview:**")
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error parsing CSV: {e}")
    
    else:  # Use Sample Data
        st.subheader("🎯 Sample Data")
        st.info("Using sample data for demonstration. Click 'Generate Sample Data' below.")
        
        if st.button("Generate Sample Data"):
            image = create_sample_image()
            df = create_sample_csv()
            
            # We need to set the keys to something for the sample to work if they are not in .env
            if not api_key:
                st.warning("No API key found in .env. Sample data validation will not work.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Sample Image:**")
                st.image(image, caption="Sample Data Image", use_column_width=True)
            
            with col2:
                st.write("**Sample CSV:**")
                st.dataframe(df, use_container_width=True)
    
    # Validation section
    if image is not None and df is not None and api_key and connected:
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
    
    elif image is None and df is None:
        st.info("👆 Please provide both an image and CSV data using one of the methods above.")
    elif image is None:
        st.info("📷 Please provide an image containing the data to validate.")
    elif df is None:
        st.info("📊 Please provide CSV data to validate against the image.")
    elif not api_key:
        st.info("🔑 Please create a .env file with your API key and restart the app.")
    elif not connected:
        st.error("❌ API Key not found or invalid. Please check your .env file.")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ## Three Ways to Use This Tool:
        
        ### 1. 📁 Upload Files
        - Upload an image file (PNG, JPG, JPEG) containing your data
        - Upload a CSV file with the data to validate
        
        ### 2. ✍️ Input Data Directly  
        - **Image**: Upload an image or use your camera to capture data
        - **CSV**: Either manually enter data in the table or paste CSV text
        
        ### 3. 🎯 Use Sample Data
        - Try the tool with built-in sample data
        - Perfect for testing and understanding how it works
        
        ## API Setup (Secure Method):
        1. Create a file named `.env` in the same directory as this script.
        2. Add your API keys to it like this:
           ```
           OPENAI_API_KEY="your-openai-key"
           ANTHROPIC_API_KEY="your-anthropic-key"
           ```
        3. The app will automatically load these keys. **Never share your `.env` file.**
        
        ## Process:
        1. Set up your `.env` file.
        2. Choose your input method.
        3. Provide image and CSV data.
        4. Click "Start Validation".
        5. Review results and apply corrections if needed.
        """)

if __name__ == "__main__":
    main()