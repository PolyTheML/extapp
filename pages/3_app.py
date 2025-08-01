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
import re

# --- Helper Functions ---
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object-type columns with mixed types to string,
    handles duplicate column names, and ensures consistent serialization with PyArrow.
    """
    if df is None or df.empty:
        return df
    
    # Handle duplicate column names by adding suffixes
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    
    df.columns = new_cols
    
    # Clean up column names - remove leading/trailing whitespace
    df.columns = df.columns.str.strip()
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all')  # Remove rows where all values are NaN
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    
    # Convert object columns with proper handling
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

def create_comprehensive_extraction_prompt():
    """Creates an enhanced prompt specifically designed for complete table extraction."""
    return """
    You are an expert data extraction specialist. Your task is to extract EVERY SINGLE ROW from the table in this image with 100% completeness.

    🚨 CRITICAL SUCCESS METRICS:
    - EXTRACT ALL ROWS (missing even 1 row = failure)
    - MAINTAIN EXACT NUMERICAL ACCURACY
    - PRESERVE ALL FORMATTING (commas, parentheses, percentages)

    STEP-BY-STEP EXTRACTION PROTOCOL:

    **PHASE 1: COMPLETE TABLE MAPPING**
    1. Scan the ENTIRE image from top to bottom
    2. Identify ALL horizontal lines/rows containing data
    3. Count total rows BEFORE starting extraction (this is your target)
    4. Note any multi-page continuations or page breaks
    5. Look for headers, data rows, subtotals, totals, and footer rows

    **PHASE 2: SYSTEMATIC ROW EXTRACTION**
    Start from the very top and work down systematically:
    - Row 1: Extract header row completely
    - Row 2, 3, 4... Continue extracting EVERY subsequent row
    - Don't skip rows with different formatting (bold, italic, indented)
    - Include partial rows if they contain any data
    - Extract continuation rows that may wrap to next line
    - Include summary/total rows at the bottom

    **PHASE 3: HANDLE COMPLEX STRUCTURES**
    - If table spans multiple columns, extract left-to-right, top-to-bottom
    - For merged cells, repeat the value across all positions
    - For multi-line entries within a cell, combine into single entry
    - Preserve hierarchical structure (numbered items, sub-items)

    **PHASE 4: NUMERICAL PRECISION**
    - Extract numbers EXACTLY as shown: "1,361,196" not "1361196"
    - Preserve negative indicators: "(2,207)" not "-2207"
    - Keep percentage symbols: "2.5%" not "2.5"
    - Maintain decimal precision: "0.1%" not "0.1"

    **PHASE 5: QUALITY VERIFICATION**
    Before finalizing:
    - Count extracted rows vs. visual count
    - Verify no rows were accidentally merged
    - Check that all columns have consistent width
    - Ensure no data was truncated

    **SPECIAL INSTRUCTIONS FOR BANKING/FINANCIAL TABLES:**
    - Bank names may be long - extract completely
    - Look for numbered sequences (1, 2, 3...) to verify row count
    - Financial figures are critical - double-check every digit
    - Include all percentage columns
    - Don't skip rows that appear to be subtotals or summaries

    **OUTPUT FORMAT - CRITICAL:**
    Return ONLY a valid JSON object with these exact keys:
    ```json
    {
        "table_data": [
            ["Header1", "Header2", "Header3", ...],
            ["Row1Col1", "Row1Col2", "Row1Col3", ...],
            ["Row2Col1", "Row2Col2", "Row2Col3", ...],
            ...
        ],
        "total_rows_extracted": 60,
        "confidence_score": 0.95,
        "extraction_notes": "Extracted complete banking table with 60 institutions including all financial metrics"
    }
    ```

    **CRITICAL REQUIREMENTS:**
    1. Return ONLY the JSON object - no markdown, no explanations, no code blocks
    2. Extract EVERY visible row in the table
    3. Maintain exact numerical formatting
    4. If you see ~60 rows, extract all ~60 rows
    5. Double-check row count before returning

    Your success is measured by: Complete row extraction + Perfect numerical accuracy.
    """

def extract_table_with_claude_enhanced(base64_image_data, api_key, model_name):
    """Enhanced Claude extraction with better error handling and response processing."""
    
    if not api_key:
        return None, 0.0, "❌ Claude API key not configured."

    if not base64_image_data:
        return None, 0.0, "❌ Base64 image data is missing or corrupt."

    # Use the most capable Claude model for complex tables
    if model_name not in ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]:
        st.warning(f"⚠️ For complex tables with many rows, consider using claude-3-5-sonnet-20241022")

    prompt = create_comprehensive_extraction_prompt()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        st.info(f"🤖 Claude ({model_name}) is analyzing the complete table...")

        # Enhanced message creation with maximum token limit
        # Claude models support different max tokens:
        # - Claude 3.5 Sonnet: up to 8192 output tokens
        # - Claude 3 Opus: up to 4096 output tokens  
        # - Claude 3 Haiku: up to 4096 output tokens
        max_tokens_config = {
            "claude-3-5-sonnet-20241022": 8192,
            "claude-3-5-sonnet-20240620": 8192,
            "claude-3-opus-20240229": 4096,
            "claude-3-sonnet-20240229": 4096,
            "claude-3-haiku-20240307": 4096
        }
        
        max_tokens = max_tokens_config.get(model_name, 8192)
        
        # Use custom tokens if set by user
        if hasattr(st.session_state, 'custom_max_tokens'):
            max_tokens = st.session_state.custom_max_tokens
            
        st.info(f"🔧 Using {max_tokens} max tokens for {model_name}")
        
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,  # Use maximum available tokens for the model
            temperature=0.0,  # Zero temperature for consistency
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

        # Extract and clean the response
        raw_text = response.content[0].text.strip()
        
        # Debug info
        st.write(f"📊 Response length: {len(raw_text)} characters")
        
        # Enhanced JSON cleaning
        cleaned_text = clean_json_response(raw_text)
        
        # Show cleaned response for debugging
        with st.expander("🔍 Claude Response Analysis"):
            st.write(f"**Raw length:** {len(raw_text)} chars")
            st.write(f"**Cleaned length:** {len(cleaned_text)} chars")  # FIXED: Added missing quote
            st.code(cleaned_text[:2000], language="json")
            
            # Try to identify why JSON might be failing
            if not cleaned_text.strip():
                st.error("❌ Response is empty after cleaning")
            elif not cleaned_text.startswith('{'):
                st.error(f"❌ Response doesn't start with '{{': '{cleaned_text[:50]}...'")
            elif not cleaned_text.endswith('}'):
                st.error(f"❌ Response doesn't end with '}}': '...{cleaned_text[-50:]}'")

        try:
            # Parse the JSON response
            parsed_json = json.loads(cleaned_text)
            
            # Extract data with validation
            table_data = parsed_json.get("table_data", [])
            confidence_score = parsed_json.get("confidence_score", 0.0)
            total_rows = parsed_json.get("total_rows_extracted", len(table_data))
            extraction_notes = parsed_json.get("extraction_notes", "No notes provided")
            
            # Enhanced validation
            if not table_data:
                return None, 0.0, "❌ No table data found in response"
            
            if len(table_data) < 2:
                return table_data, confidence_score, f"⚠️ Only {len(table_data)} row(s) extracted. Expected ~60 rows."
            
            # Success message with row count
            success_msg = f"✅ Extracted {len(table_data)} rows (including header). {extraction_notes}"
            
            return table_data, confidence_score, success_msg
            
        except json.JSONDecodeError as e:
            # Enhanced JSON error handling
            error_position = getattr(e, 'pos', 0)
            context = cleaned_text[max(0, error_position-50):error_position+50]
            
            return None, 0.0, f"❌ JSON parsing failed at position {error_position}. Context: '...{context}...'. Error: {str(e)}"

    except anthropic.APIError as e:
        return None, 0.0, f"❌ Claude API error: {str(e)}"
    except Exception as e:
        return None, 0.0, f"❌ Unexpected error: {str(e)}"

def clean_json_response(raw_text):
    """Enhanced JSON cleaning function."""
    text = raw_text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    # Remove any leading/trailing non-JSON text
    # Find the first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx+1]
    
    # Handle common JSON formatting issues
    text = text.replace('\n', ' ')  # Remove newlines within JSON
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text.strip()

def to_excel(df):
    """Converts a DataFrame to an in-memory Excel file."""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def to_csv(df):
    """Convert DataFrame to CSV string for download."""
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        st.error(f"Error creating CSV file: {str(e)}")
        return None

def extract_table_with_gemini(base64_image_data, api_key, model_name):
    """Extracts table data using Google Gemini API."""
    if not api_key:
        return None, 0.0, "API key not configured."

    prompt = create_comprehensive_extraction_prompt()

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

    prompt = create_comprehensive_extraction_prompt()

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

def extract_table_with_claude_fallback(base64_image_data, api_key, model_name):
    """Fallback method for when primary extraction fails - uses chunked approach."""
    
    prompt_chunked = """
    This appears to be a large table that may have been cut off in previous extraction attempts.
    
    Please extract this table with special attention to:
    1. SCANNING THE ENTIRE IMAGE from top to bottom
    2. Not stopping at the first 10-15 rows
    3. Looking for continuation of numbered sequences
    4. Including ALL rows visible in the image
    
    Focus on extracting the complete table structure. If this is a banking report, there should be many more institutions listed.
    
    Return ONLY valid JSON:
    {
        "table_data": [["headers"], ["row1"], ["row2"], ...],
        "total_rows_found": number,
        "confidence_score": 0.95
    }
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=model_name,
            max_tokens=8192,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}},
                    {"type": "text", "text": prompt_chunked}
                ]
            }]
        )
        
        raw_text = response.content[0].text.strip()
        cleaned_text = clean_json_response(raw_text)
        parsed_json = json.loads(cleaned_text)
        
        return (
            parsed_json.get("table_data", []),
            parsed_json.get("confidence_score", 0.0),
            f"Fallback extraction: {parsed_json.get('total_rows_found', 0)} rows found"
        )
        
    except Exception as e:
        return None, 0.0, f"Fallback method also failed: {str(e)}"

# --- Main App UI with Enhanced Claude Integration ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("Step 2: 🔎 AI-Powered Table Extractor (Enhanced for Large Tables)")

if 'converted_pil_images' not in st.session_state or not st.session_state.converted_pil_images:
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Choose an image from the converted PDF and an AI model to extract the table.")

    # AI Configuration
    st.header("⚙️ AI Configuration")
    
    # Prioritize Claude models for complex tables
    model_options = [
        # Recommended Claude models (best for large tables)
        "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
        # Other models
        "claude-opus-4-20250514	", "claude-sonnet-4-20250514",
        "gpt-4o", "gpt-4-turbo",
        "gemini-2.5-pro", "gemini-2.5-flash"
    ]
    
    selected_model = st.selectbox("Choose AI Model:", options=model_options)
    
    # Special recommendation for large tables
    if "claude-3-5-sonnet" in selected_model:
        st.success("✅ Excellent choice for large tables with many rows!")
    elif "claude" in selected_model:
        st.info("👍 Good choice for complex table extraction.")

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
    
    # Enhanced settings for large tables
    st.subheader("⚙️ Extraction Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_fallback = st.checkbox("Enable fallback extraction if first attempt fails", value=True)
    with col2:
        show_debug = st.checkbox("Show detailed debug information", value=True)
    with col3:
        # Allow users to customize max tokens
        if "claude" in selected_model:
            max_tokens_options = {
                "Standard (4096)": 4096,
                "High (8192)": 8192,
                "Maximum (8192)": 8192  # Claude's current limit
            }
            selected_tokens = st.selectbox("Max Tokens:", options=list(max_tokens_options.keys()), index=2)
            custom_max_tokens = max_tokens_options[selected_tokens]
            st.session_state.custom_max_tokens = custom_max_tokens

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
            if st.button("🚀 Extract Complete Table", type="primary"):
                # Determine AI provider and function
                if "claude" in selected_model:
                    ai_provider = "Anthropic Claude"
                    api_key_to_use = anthropic_api_key
                    extraction_function = extract_table_with_claude_enhanced
                elif "gemini" in selected_model:
                    ai_provider = "Google Gemini"
                    api_key_to_use = gemini_api_key
                    extraction_function = extract_table_with_gemini
                elif "gpt" in selected_model:
                    ai_provider = "OpenAI"
                    api_key_to_use = openai_api_key
                    extraction_function = extract_table_with_openai
                else:
                    st.error("Invalid model selected.")
                    st.stop()

                if not api_key_to_use:
                    st.error(f"❌ {ai_provider} API Key is not configured.")
                else:
                    with st.spinner(f"🤖 {ai_provider} is extracting the complete table with **{selected_model}**..."):
                        start_time = time.time()

                        # Prepare image and extract
                        base64_image = prepare_image_from_pil(selected_pil_image)
                        if base64_image:
                            table_data, confidence, reasoning = extraction_function(base64_image, api_key_to_use, selected_model)
                            
                            # Try fallback if first attempt failed and it's enabled
                            if (not table_data or len(table_data) < 15) and use_fallback and "claude" in selected_model:
                                st.warning("🔄 First attempt extracted fewer rows than expected. Trying fallback method...")
                                table_data, confidence, reasoning = extract_table_with_claude_fallback(base64_image, api_key_to_use, selected_model)
                                reasoning = f"Fallback method used. {reasoning}"
                                
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
                                
                                # Enhanced success message
                                row_count = len(data)
                                col_count = len(header)
                                st.success(f"✅ Complete table extracted! {row_count} rows × {col_count} columns in {processing_time:.1f}s")
                                
                                if row_count >= 50:
                                    st.balloons()  # Celebrate successful large table extraction!
                                    
                            except Exception as e:
                                st.warning(f"Structure adjustment needed: {e}")
                                st.session_state.extracted_df = pd.DataFrame(table_data)
                                st.session_state.original_df = st.session_state.extracted_df.copy()
                        else:
                            st.error("❌ Extraction failed or returned insufficient data.")
                            if reasoning:
                                st.error(reasoning)

    # Display Results
    if st.session_state.get('confidence') is not None:
        st.subheader("📊 Extraction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}")
        with col2:
            st.metric(label="Processing Time", value=f"{st.session_state.processing_time:.1f}s")
        with col3:
            st.metric(label="AI Provider", value=st.session_state.get('ai_provider', 'N/A'))

        with st.expander("🧠 AI's Analysis"):
            st.info(st.session_state.get('reasoning', 'No reasoning provided.'))

    # Data Display and Download - ENHANCED WITH BETTER ERROR HANDLING
    if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("📋 Extracted Data Preview")
        
        try:
            # Use copy to avoid modifying original and apply normalization
            clean_df = normalize_dataframe(st.session_state.extracted_df.copy())
            st.dataframe(clean_df, use_container_width=True)

            # Enhanced data summary
            row_count = len(clean_df)
            col_count = len(clean_df.columns)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", row_count)
            with col2:
                st.metric("Total Columns", col_count)
            with col3:
                st.metric("Data Points", row_count * col_count)
            
            if row_count >= 50:
                st.success("🎉 Large table successfully extracted!")
            elif row_count < 20:
                st.warning("⚠️ Fewer rows than expected. Consider using fallback extraction or different model.")

            st.divider()
            st.subheader("💾 Download Processed Data")

            col1, col2 = st.columns(2)
            with col1:
                file_name = st.text_input("File Name:", value=f"banking_table_{row_count}rows", key="download_filename")
            with col2:
                file_format = st.selectbox("Format:", ["Excel (.xlsx)", "CSV (.csv)"], key="download_format")

            if file_name:
                try:
                    if file_format == "Excel (.xlsx)":
                        file_data = to_excel(clean_df)
                        file_extension = "xlsx"
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    else:
                        file_data = to_csv(clean_df)
                        file_extension = "csv"
                        mime_type = "text/csv"

                    if file_data is not None:
                        st.download_button(
                            label=f"📥 Download {row_count} rows as {file_format}",
                            data=file_data,
                            file_name=f"{file_name}.{file_extension}",
                            mime=mime_type,
                            type="primary"
                        )
                        st.success("✅ Data extraction completed successfully!")
                    else:
                        st.error("❌ Failed to prepare download file.")
                        
                except Exception as e:
                    st.error(f"❌ Error preparing download: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ Error processing extracted data: {str(e)}")
            st.info("Try extracting the data again or use a different AI model.")