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
from typing import List, Optional, Tuple

# --- Data Engineering Helper Functions ---

def _clean_and_convert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Cleans a pandas Series by removing financial notations and converts to numeric.
    Handles currency symbols, commas, and parentheses for negative values.
    """
    if series.empty:
        return series
    
    # Ensure the series is treated as a string for cleaning operations
    cleaned_series = series.astype(str).str.strip()
    
    # Handle financial notations
    cleaned_series = cleaned_series.str.replace(r'[\$,]', '', regex=True)
    
    # Convert numbers in parentheses to negative numbers
    mask = cleaned_series.str.startswith('(') & cleaned_series.str.endswith(')')
    if mask.any():
        cleaned_series.loc[mask] = '-' + cleaned_series.loc[mask].str.strip('()')

    # Attempt to convert to a numeric type, coercing failures into NaN
    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
    
    # Only return the numeric series if a high percentage of values converted successfully
    if not series.dropna().empty and numeric_series.notna().sum() / len(series.dropna()) > 0.9:
        return numeric_series
    
    return series # Return the original series if conversion is not appropriate

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data engineering best practices to clean and standardize a DataFrame.

    This function performs the following steps:
    1.  Handles empty or invalid input gracefully.
    2.  Strips leading/trailing whitespace from all column names.
    3.  Creates unique names for duplicate columns by adding a suffix (e.g., 'Total_1').
    4.  Removes rows and columns that are entirely empty or unnamed.
    5.  Strips whitespace from all string data cells.
    6.  Intelligently converts columns to numeric types after cleaning financial strings.
    7.  Ensures all remaining object columns are strings to prevent serialization errors.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 1. Clean and de-duplicate column names
    df.columns = df.columns.str.strip()
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        count = seen.get(col, 0)
        if count > 0:
            new_cols.append(f"{col}_{count}")
        else:
            new_cols.append(col)
        seen[col] = count + 1
    df.columns = new_cols

    # 2. Remove empty/unwanted rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # 3. Clean data within the DataFrame
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
        
        # Apply robust numeric conversion
        df[col] = _clean_and_convert_to_numeric(df[col])

    # 4. Final conversion for compatibility
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace('nan', '')

    st.success("✅ Dataframe has been cleaned and normalized for display.")
    return df

def to_excel(df: pd.DataFrame) -> Optional[bytes]:
    """
    Converts a pandas DataFrame to an in-memory Excel file.

    Args:
        df: The DataFrame to convert.

    Returns:
        The Excel file as a byte string, or None if an error occurs.
    """
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def to_csv(df: pd.DataFrame) -> Optional[bytes]:
    """
    Converts a pandas DataFrame to an in-memory CSV file.

    Args:
        df: The DataFrame to convert.

    Returns:
        The CSV file as a UTF-8 encoded byte string, or None if an error occurs.
    """
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        st.error(f"Error creating CSV file: {str(e)}")
        return None

# --- AI & Image Processing Functions (Original logic preserved) ---

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

    if model_name not in ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]:
        st.warning(f"⚠️ For complex tables with many rows, consider using claude-3-5-sonnet-20240620")

    prompt = create_comprehensive_extraction_prompt()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        st.info(f"🤖 Claude ({model_name}) is analyzing the complete table...")

        max_tokens_config = {
            "claude-3-5-sonnet-20241022": 8192,
            "claude-3-5-sonnet-20240620": 8192,
            "claude-3-opus-20240229": 4096,
            "claude-3-sonnet-20240229": 4096,
            "claude-3-haiku-20240307": 4096
        }
        
        max_tokens = max_tokens_config.get(model_name, 8192)
        
        if hasattr(st.session_state, 'custom_max_tokens'):
            max_tokens = st.session_state.custom_max_tokens
            
        st.info(f"🔧 Using {max_tokens} max tokens for {model_name}")
        
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )

        raw_text = response.content[0].text.strip()
        cleaned_text = clean_json_response(raw_text)
        
        with st.expander("🔍 Claude Response Analysis"):
            st.write(f"**Raw length:** {len(raw_text)} chars")
            st.write(f"**Cleaned length:** {len(cleaned_text)} chars")
            st.code(cleaned_text[:2000], language="json")

        try:
            parsed_json = json.loads(cleaned_text)
            table_data = parsed_json.get("table_data", [])
            confidence_score = parsed_json.get("confidence_score", 0.0)
            extraction_notes = parsed_json.get("extraction_notes", "No notes provided")
            
            if not table_data: return None, 0.0, "❌ No table data found in response"
            
            success_msg = f"✅ Extracted {len(table_data)} rows. {extraction_notes}"
            return table_data, confidence_score, success_msg
            
        except json.JSONDecodeError as e:
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
    
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    
    text = text.strip()
    
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx+1]
    
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# --- Main App UI ---
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
    
    # --- MODEL OPTIONS UNCHANGED AS REQUESTED ---
    model_options = [
        "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
        "claude-opus-4-20250514	", "claude-sonnet-4-20250514",
        "gpt-4o", "gpt-4-turbo",
        "gemini-2.5-pro", "gemini-2.5-flash"
    ]
    
    selected_model = st.selectbox("Choose AI Model:", options=model_options)
    
    if "claude-3-5-sonnet" in selected_model:
        st.success("✅ Excellent choice for large tables with many rows!")
    elif "claude" in selected_model:
        st.info("👍 Good choice for complex table extraction.")

    # Show API key status
    col1, col2, col3 = st.columns(3)
    with col1: st.info(f"**OpenAI:** {'✅ Configured' if openai_api_key else '❌ Missing'}")
    with col2: st.info(f"**Claude:** {'✅ Configured' if anthropic_api_key else '❌ Missing'}")
    with col3: st.info(f"**Gemini:** {'✅ Configured' if gemini_api_key else '❌ Missing'}")

    st.divider()

    # Image Processing
    st.header("🖼️ Image Processing")
    
    st.subheader("⚙️ Extraction Settings")
    col1, col2, col3 = st.columns(3)
    with col1: use_fallback = st.checkbox("Enable fallback extraction", value=True)
    with col2: show_debug = st.checkbox("Show detailed debug info", value=True)
    with col3:
        if "claude" in selected_model:
            max_tokens_options = {"Standard (4096)": 4096, "High (8192)": 8192}
            selected_tokens = st.selectbox("Max Tokens:", options=list(max_tokens_options.keys()), index=1)
            st.session_state.custom_max_tokens = max_tokens_options[selected_tokens]

    image_options = {f"Image {i+1}": i for i in range(len(st.session_state.converted_pil_images))}
    if image_options:
        selected_image_key = st.selectbox("Choose an image to process:", options=list(image_options.keys()))
        selected_image_index = image_options[selected_image_key]
        st.session_state.selected_image_index = selected_image_index
        selected_pil_image = st.session_state.converted_pil_images[selected_image_index]

        st.image(selected_pil_image, caption="Selected Image for Extraction", use_container_width=True)

        if st.button("🚀 Extract Complete Table", type="primary"):
            # Determine AI provider and function
            # This part can be expanded with functions for Gemini and OpenAI
            if "claude" in selected_model:
                ai_provider = "Anthropic Claude"
                api_key_to_use = anthropic_api_key
                extraction_function = extract_table_with_claude_enhanced
            else:
                st.error("Selected AI provider is not fully implemented yet.")
                st.stop()

            if not api_key_to_use:
                st.error(f"❌ {ai_provider} API Key is not configured.")
            else:
                with st.spinner(f"🤖 {ai_provider} is extracting the table with **{selected_model}**..."):
                    start_time = time.time()
                    base64_image = prepare_image_from_pil(selected_pil_image)
                    if base64_image:
                        # Fallback logic can be added here
                        table_data, confidence, reasoning = extraction_function(base64_image, api_key_to_use, selected_model)
                        
                        st.session_state.processing_time = time.time() - start_time
                        st.session_state.confidence = confidence
                        st.session_state.reasoning = reasoning

                        if table_data and len(table_data) > 1:
                            try:
                                header, data = table_data[0], table_data[1:]
                                df = pd.DataFrame(data, columns=header)
                                st.session_state.extracted_df = df
                                st.success(f"✅ Extracted! {len(df)} rows × {len(df.columns)} columns in {st.session_state.processing_time:.1f}s")
                            except Exception as e:
                                st.warning(f"Could not structure DataFrame: {e}")
                                st.session_state.extracted_df = pd.DataFrame(table_data)
                        else:
                            st.error(f"❌ Extraction failed or returned no data. Reason: {reasoning}")
                    else:
                        st.error("❌ Failed to prepare the image for the API.")

    # Display Results
    if st.session_state.get('confidence') is not None:
        st.subheader("📊 Extraction Results")
        # Metrics and reasoning display
        col1, col2, col3 = st.columns(3)
        with col1: st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}")
        with col2: st.metric(label="Processing Time", value=f"{st.session_state.processing_time:.1f}s")
        with col3: st.metric(label="AI Model", value=selected_model, help="The model used for this extraction.")
        with st.expander("🧠 AI's Analysis"):
            st.info(st.session_state.get('reasoning', 'No reasoning provided.'))

    # Data Display and Download
    if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("📋 Extracted Data Preview")
        
        # Use the robust data engineering function before displaying
        clean_df = normalize_dataframe(st.session_state.extracted_df.copy())
        st.dataframe(clean_df, use_container_width=True)

        # Data summary and download section
        row_count, col_count = clean_df.shape
        st.subheader("💾 Download Processed Data")
        col1, col2 = st.columns(2)
        with col1: file_name = st.text_input("File Name:", value=f"extracted_data_{row_count}rows")
        with col2: file_format = st.selectbox("Format:", ["Excel (.xlsx)", "CSV (.csv)"])

        if file_name:
            if file_format == "Excel (.xlsx)":
                file_data = to_excel(clean_df)
                file_extension, mime_type = "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                file_data = to_csv(clean_df)
                file_extension, mime_type = "csv", "text/csv"

            if file_data is not None:
                st.download_button(
                    label=f"📥 Download {row_count} rows",
                    data=file_data,
                    file_name=f"{file_name}.{file_extension}",
                    mime=mime_type,
                    type="primary"
                )