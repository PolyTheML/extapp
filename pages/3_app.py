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
import google.generativeai as genai
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

# --- AI & Image Processing Functions ---

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

def create_optimized_financial_extraction_prompt():
    """Creates the ultimate prompt for financial table extraction based on your requirements."""
    return """
You are a specialized financial document parser. Extract ALL tables from this page with complete accuracy.

## Core Requirements
- **EXTRACT EVERYTHING**: Every table, every row, every column, every cell
- **PRESERVE EXACTLY**: All numbers, formatting, punctuation, and text as shown
- **NO INTERPRETATION**: Extract exactly what you see, don't convert or standardize

## Critical Rules

### 1. Complete Extraction
- Scan entire page systematically (left-to-right, top-to-bottom)
- Extract tables regardless of orientation (portrait/landscape)
- Include ALL rows: headers, data, subtotals, totals, footnotes within table structure
- Count and verify: output row count MUST match source

### 2. Value Preservation
- **Numbers**: Keep exact formatting: "1,361,196", "(2,207)", "2.5%", "-"
- **Text**: Preserve case, spacing, special characters
- **Empty cells**: Use empty string "" (not null, "0", or "-" unless actually shown)
- **Merged cells**: Repeat the value for all positions it spans

### 3. Header Processing
- **Detection**: First row with text/labels = headers
- **Cleaning**: Trim whitespace, preserve original language and case
- **Duplicates**: Append "_2", "_3" etc: "Amount", "Amount_2", "Amount_3"
- **Missing**: Generate "Column_1", "Column_2" if no clear headers

### 4. Multi-language Support
- **Mixed content**: Extract exactly as shown (don't translate)
- **Bank names**: Keep full original text including English/Khmer
- **Special characters**: Preserve all Unicode characters
- **Numbers in text**: Keep as part of the string

### 5. Complex Structure Handling
- **Multi-level headers**: Combine with underscore: "2024_Deposits", "2023_Loans"
- **Rotated tables**: Read following the text orientation
- **Split tables**: If table continues across sections, treat as separate tables
- **Nested data**: Extract hierarchical info maintaining structure

## Output Format - CRITICAL
Return ONLY a valid JSON object with this exact structure:
```json
{
    "table_data": [
        ["Header1", "Header2", "Header3"],
        ["Row1Col1", "Row1Col2", "Row1Col3"],
        ["Row2Col1", "Row2Col2", "Row2Col3"]
    ],
    "total_rows_extracted": 52,
    "confidence_score": 0.95,
    "extraction_notes": "Extracted complete banking table with all financial metrics"
}
```

## Validation Checklist
Before returning results, verify:
- [ ] Every visible table identified
- [ ] Row counts match exactly
- [ ] All numbers preserved with original formatting
- [ ] No data invented or modified
- [ ] Headers are appropriate
- [ ] All text readable and preserved

## Example Transformation
**Source row**: `Advanced Bank of Asia Limited | 43,051,336 | 34,281,391 | 79.6%`
**JSON output**: 
```json
["Advanced Bank of Asia Limited", "43,051,336", "34,281,391", "79.6%"]
```

**CRITICAL INSTRUCTIONS**:
1. Return ONLY the JSON object - no markdown, no explanations, no code blocks
2. Extract EVERY visible row in the table
3. Maintain exact numerical formatting
4. Financial accuracy is critical - double-check all numerical values
5. If you see 50+ rows, extract all 50+ rows

Begin extraction now.
"""

def extract_table_with_claude_enhanced(base64_image_data, api_key, model_name):
    """Enhanced Claude extraction with optimized prompt for financial documents."""
    
    if not api_key:
        return None, 0.0, "❌ Claude API key not configured."

    if not base64_image_data:
        return None, 0.0, "❌ Base64 image data is missing or corrupt."

    # Use optimized prompt
    prompt = create_optimized_financial_extraction_prompt()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        st.info(f"🤖 Claude ({model_name}) is analyzing the financial table...")

        max_tokens_config = {
            "claude-3-5-sonnet-20241022": 8192,
            "claude-3-5-sonnet-20240620": 8192,
            "claude-3-opus-20240229": 4096,
            "claude-3-sonnet-20240229": 4096,
            "claude-3-haiku-20240307": 4096,
            "claude-opus-4-20250514": 8192,
            "claude-sonnet-4-20250514": 8192
        }
        
        max_tokens = max_tokens_config.get(model_name, 8192)
        
        if hasattr(st.session_state, 'custom_max_tokens'):
            max_tokens = st.session_state.custom_max_tokens
            
        st.info(f"🔧 Using {max_tokens} max tokens for {model_name}")
        
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.0,  # Keep deterministic for financial data
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
            if len(cleaned_text) > 2000:
                st.write("... (truncated for display)")

        try:
            parsed_json = json.loads(cleaned_text)
            table_data = parsed_json.get("table_data", [])
            confidence_score = parsed_json.get("confidence_score", 0.0)
            extraction_notes = parsed_json.get("extraction_notes", "No notes provided")
            total_rows = parsed_json.get("total_rows_extracted", len(table_data))
            
            if not table_data: 
                return None, 0.0, "❌ No table data found in response"
            
            # Enhanced success message with more details
            success_msg = f"✅ Extracted {len(table_data)} rows ({total_rows} reported). {extraction_notes}"
            
            # Validate data structure
            if len(table_data) > 0 and isinstance(table_data[0], list):
                return table_data, confidence_score, success_msg
            else:
                return None, 0.0, "❌ Invalid table data structure returned"
            
        except json.JSONDecodeError as e:
            error_position = getattr(e, 'pos', 0)
            context = cleaned_text[max(0, error_position-50):error_position+50]
            return None, 0.0, f"❌ JSON parsing failed at position {error_position}. Context: '...{context}...'. Error: {str(e)}"

    except anthropic.APIError as e:
        return None, 0.0, f"❌ Claude API error: {str(e)}"
    except Exception as e:
        return None, 0.0, f"❌ Unexpected error: {str(e)}"

def extract_table_with_gemini(base64_image_data, api_key, model_name):
    """Gemini extraction with optimized prompt for financial documents."""
    
    if not api_key:
        return None, 0.0, "❌ Gemini API key not configured."

    if not base64_image_data:
        return None, 0.0, "❌ Base64 image data is missing or corrupt."

    try:
        genai.configure(api_key=api_key)
        
        # Map model names to Gemini model identifiers
        model_mapping = {
            "gemini-2.5-flash-lite": "gemini-2.0-flash-exp",
            "gemini-2.0-flash-exp": "gemini-2.0-flash-exp", 
            "gemini-2.0-flash-lite": "gemini-2.0-flash-exp",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash": "gemini-2.5-flash"
        }
        
        actual_model = model_mapping.get(model_name, "gemini-2.0-flash-exp")
        
        st.info(f"🤖 Gemini ({actual_model}) is analyzing the financial table...")
        
        model = genai.GenerativeModel(actual_model)
        
        # Convert base64 to PIL Image for Gemini
        import io
        from PIL import Image as PILImage
        image_data = base64.b64decode(base64_image_data)
        pil_image = PILImage.open(io.BytesIO(image_data))
        
        prompt = create_optimized_financial_extraction_prompt()

        generation_config = genai.GenerationConfig(
            temperature=0.0,  # Keep deterministic for financial data
            max_output_tokens=8192,  # Gemini handles dynamic token allocation
            top_p=1.0,
            top_k=32
        )
        
        # Generate content with image
        response = model.generate_content([
            prompt,
            pil_image
        ])
        
        raw_text = response.text.strip()
        cleaned_text = clean_json_response(raw_text)
        
        with st.expander("🔍 Gemini Response Analysis"):
            st.write(f"**Raw length:** {len(raw_text)} chars")
            st.write(f"**Cleaned length:** {len(cleaned_text)} chars")
            st.code(cleaned_text[:2000], language="json")
            if len(cleaned_text) > 2000:
                st.write("... (truncated for display)")

        try:
            parsed_json = json.loads(cleaned_text)
            table_data = parsed_json.get("table_data", [])
            confidence_score = parsed_json.get("confidence_score", 0.85)  # Default confidence for Gemini
            extraction_notes = parsed_json.get("extraction_notes", "Extracted with Gemini AI")
            total_rows = parsed_json.get("total_rows_extracted", len(table_data))
            
            if not table_data: 
                return None, 0.0, "❌ No table data found in response"
            
            # Enhanced success message with more details
            success_msg = f"✅ Extracted {len(table_data)} rows ({total_rows} reported). {extraction_notes}"
            
            # Validate data structure
            if len(table_data) > 0 and isinstance(table_data[0], list):
                return table_data, confidence_score, success_msg
            else:
                return None, 0.0, "❌ Invalid table data structure returned"
            
        except json.JSONDecodeError as e:
            error_position = getattr(e, 'pos', 0)
            context = cleaned_text[max(0, error_position-50):error_position+50]
            return None, 0.0, f"❌ JSON parsing failed at position {error_position}. Context: '...{context}...'. Error: {str(e)}"

    except Exception as e:
        return None, 0.0, f"❌ Gemini API error: {str(e)}"

def clean_json_response(raw_text: str) -> str:
    """Enhanced JSON cleaning function."""
    text = raw_text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    
    text = text.strip()
    
    # Find JSON boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx+1]
    
    text = re.sub(r',\s*]', ']', text)
    
    # Clean up whitespace but preserve structure
    text = re.sub(r'\n\s*', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)    # Collapse multiple spaces
    
    return text.strip()

# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 2: 🔎 AI-Powered Financial Table Extractor")
st.markdown("*Optimized for complex banking and financial documents with multilingual support*")

if 'converted_pil_images' not in st.session_state or not st.session_state.converted_pil_images:
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Choose an image from the converted PDF and an AI model to extract the table.")

    # AI Configuration
    st.header("⚙️ AI Configuration")
    
    # Updated model options with both Claude and Gemini
    model_options = [
        "claude-3-5-sonnet-20241022", 
        "claude-3-5-sonnet-20240620", 
        "claude-3-opus-20240229",
        "claude-opus-4-20250514", 
        "claude-sonnet-4-20250514",
        "gemini-2.5-pro", 
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite", 
        "gemini-2.0-flash-exp", 
        "gemini-2.0-flash-lite"
    ]
    
    selected_model = st.selectbox("Choose AI Model:", options=model_options)
    
    # Model-specific recommendations
    if "claude-3-5-sonnet" in selected_model or "claude-sonnet-4" in selected_model:
        st.success("✅ Excellent choice for complex financial tables with many rows!")
    elif "claude-opus" in selected_model:
        st.success("✅ Premium choice for highest accuracy on complex tables!")
    elif "claude" in selected_model:
        st.info("👍 Good choice for complex table extraction.")
    elif "gemini" in selected_model:
        st.info("🚀 Fast and capable for financial table extraction!")
    else:
        st.info("ℹ️ Selected model ready for table extraction.")

    # Show API key status
    col1, col2 = st.columns(2)
    with col1: st.info(f"**Claude:** {'✅ Configured' if anthropic_api_key else '❌ Missing'}")
    with col2: st.info(f"**Gemini:** {'✅ Configured' if gemini_api_key else '❌ Missing'}")

    st.divider()

    # Image Processing
    st.header("🖼️ Image Processing")
    
    st.subheader("⚙️ Extraction Settings")
    col1, col2, col3 = st.columns(3)
    with col1: 
        use_fallback = st.checkbox("Enable fallback extraction", value=True, 
                                  help="Retry with different settings if first attempt fails")
    with col2: 
        show_debug = st.checkbox("Show detailed debug info", value=True,
                               help="Display API response details for troubleshooting")
    with col3:
        if "claude" in selected_model:
            max_tokens_options = {
                "Standard (4096)": 4096, 
                "High (8192)": 8192, 
                "Maximum (16384)": 16384
            }
            selected_tokens = st.selectbox("Max Tokens:", options=list(max_tokens_options.keys()), 
                                         index=1, help="Higher tokens allow for larger tables")
            st.session_state.custom_max_tokens = max_tokens_options[selected_tokens]
        elif "gemini" in selected_model:
            st.info("🔧 Gemini uses dynamic token allocation")

    image_options = {f"Page {i+1}": i for i in range(len(st.session_state.converted_pil_images))}
    if image_options:
        selected_image_key = st.selectbox("Choose a page to process:", options=list(image_options.keys()))
        selected_image_index = image_options[selected_image_key]
        st.session_state.selected_image_index = selected_image_index
        selected_pil_image = st.session_state.converted_pil_images[selected_image_index]

        st.image(selected_pil_image, caption=f"Selected {selected_image_key} for Table Extraction", use_container_width=True)

        # Enhanced extraction button with better styling
        extract_button = st.button("🚀 Extract Financial Table", type="primary", 
                                 help="Process the selected page to extract all table data")
        
        if extract_button:
            # Determine AI provider and function
            if "claude" in selected_model:
                ai_provider = "Anthropic Claude"
                api_key_to_use = anthropic_api_key
                extraction_function = extract_table_with_claude_enhanced
            elif "gemini" in selected_model:
                ai_provider = "Google Gemini"
                api_key_to_use = gemini_api_key
                extraction_function = extract_table_with_gemini
            else:
                st.error("❌ Unknown model selected. Please choose a Claude or Gemini model.")
                st.stop()

            if not api_key_to_use:
                st.error(f"❌ {ai_provider} API Key is not configured. Please check your .env file.")
            else:
                with st.spinner(f"🤖 {ai_provider} is processing with **{selected_model}**..."):
                    start_time = time.time()
                    base64_image = prepare_image_from_pil(selected_pil_image)
                    
                    if base64_image:
                        table_data, confidence, reasoning = extraction_function(base64_image, api_key_to_use, selected_model)
                        
                        processing_time = time.time() - start_time
                        st.session_state.processing_time = processing_time
                        st.session_state.confidence = confidence
                        st.session_state.reasoning = reasoning

                        if table_data and len(table_data) > 1:
                            try:
                                header, data = table_data[0], table_data[1:]
                                df = pd.DataFrame(data, columns=header)
                                st.session_state.extracted_df = df
                                
                                # Enhanced success message
                                st.success(f"✅ Successfully extracted {len(df)} rows × {len(df.columns)} columns in {processing_time:.1f}s")
                                st.balloons()  # Celebration for successful extraction!
                                
                            except Exception as e:
                                st.warning(f"⚠️ Data structuring issue: {e}. Attempting fallback...")
                                # Fallback: treat all data as rows without headers
                                st.session_state.extracted_df = pd.DataFrame(table_data)
                        else:
                            st.error(f"❌ Extraction failed or returned insufficient data.")
                            st.error(f"**Reason:** {reasoning}")
                            
                            # Provide troubleshooting suggestions
                            with st.expander("🔧 Troubleshooting Suggestions"):
                                st.markdown("""
                                **Try these solutions:**
                                1. **Switch to Claude 3.5 Sonnet** - Best for complex tables
                                2. **Try Gemini models** - Fast alternative for table extraction
                                3. **Increase Max Tokens** (Claude only) - Use 8192 or 16384 for large tables
                                4. **Check image quality** - Ensure text is clearly readable
                                5. **Try a different page** - Some pages may have better table structure
                                6. **Enable debug info** - Check the API response for clues
                                """)
                    else:
                        st.error("❌ Failed to prepare the image for processing. Please try a different image.")

    # Display Results (Enhanced)
    if st.session_state.get('confidence') is not None:
        st.divider()
        st.subheader("📊 Extraction Results")
        
        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            confidence_color = "normal" if st.session_state.confidence > 0.8 else "inverse"
            st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}")
        with col2: 
            st.metric(label="Processing Time", value=f"{st.session_state.processing_time:.1f}s")
        with col3: 
            st.metric(label="AI Model", value=selected_model.split('-')[0].title())
        with col4:
            if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None:
                row_count = len(st.session_state.extracted_df)
                st.metric(label="Rows Extracted", value=row_count)
        
        # AI Analysis section
        with st.expander("🧠 AI's Extraction Analysis"):
            st.info(st.session_state.get('reasoning', 'No reasoning provided.'))
            
            # Additional analysis if available
            if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None:
                df = st.session_state.extracted_df
                st.write(f"**Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(f"**Column Names:** {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}")

    # Data Display and Download (Enhanced)
    if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("📋 Extracted Financial Data")
        
        # Data quality indicator
        raw_df = st.session_state.extracted_df.copy()
        row_count, col_count = raw_df.shape
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Data Summary:** {row_count:,} rows × {col_count} columns extracted")
        with col2:
            if st.button("🧹 Clean & Normalize Data", help="Apply data engineering best practices"):
                st.session_state.show_clean_data = True
        
        # Show raw vs cleaned data
        if st.session_state.get('show_clean_data', False):
            clean_df = normalize_dataframe(raw_df)
            st.subheader("🔄 Cleaned & Normalized Data")
            st.dataframe(clean_df, use_container_width=True, height=400)
            display_df = clean_df
        else:
            st.subheader("📊 Raw Extracted Data")
            st.dataframe(raw_df, use_container_width=True, height=400)
            display_df = raw_df

        # Enhanced download section
        st.divider()
        st.subheader("💾 Download Options")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1: 
            file_name = st.text_input("File Name:", value=f"financial_data_{row_count}rows_{int(time.time())}")
        with col2: 
            file_format = st.selectbox("Format:", ["Excel (.xlsx)", "CSV (.csv)"])
        with col3:
            data_version = st.selectbox("Data Version:", ["Raw Data", "Cleaned Data"])

        # Select appropriate dataframe based on user choice
        final_df = display_df if data_version == "Cleaned Data" and st.session_state.get('show_clean_data', False) else raw_df

        if file_name:
            if file_format == "Excel (.xlsx)":
                file_data = to_excel(final_df)
                file_extension, mime_type = "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                icon = "📊"
            else:
                file_data = to_csv(final_df)
                file_extension, mime_type = "csv", "text/csv"
                icon = "📄"

            if file_data is not None:
                file_size_mb = len(file_data) / (1024 * 1024)
                st.download_button(
                    label=f"{icon} Download {row_count:,} rows ({file_size_mb:.1f} MB)",
                    data=file_data,
                    file_name=f"{file_name}.{file_extension}",
                    mime=mime_type,
                    type="primary",
                    help=f"Download {data_version.lower()} as {file_format}"
                )
                
                # Success message with file info
                st.success(f"✅ Ready to download {row_count:,} rows of financial data as {file_format}")
            else:
                st.error("❌ Failed to prepare download file. Please try again.")

# Footer with tips
st.divider()
with st.expander("💡 Pro Tips for Better Extraction"):
    st.markdown("""
    **For Best Results:**
    - **Claude Models**: Best for complex financial tables with 50+ rows
      - Use **Claude 3.5 Sonnet** or **Claude Opus 4** for highest accuracy
      - Set **Max Tokens to 8192+** for tables with many rows
    - **Gemini Models**: Fast and efficient for most financial documents
      - Good performance with automatic token management
      - Excellent for multilingual content (English/Khmer)
    - Ensure your PDF pages have **clear, readable text**
    - **Clean & normalize data** before download for analysis
    - Try different pages if one doesn't extract well
    
    **Troubleshooting:**
    - **Claude**: Enable debug info, increase max tokens for large tables
    - **Gemini**: Try different Gemini model variants if extraction fails
    - Check confidence scores - below 80% may need retry with different model
    - Mixed languages (English/Khmer) are fully supported by both providers
    - Compare results between Claude and Gemini for best accuracy
    
    **Model Comparison:**
    - **Claude**: Superior for complex tables, financial accuracy, large datasets
    - **Gemini**: Faster processing, good multilingual support, cost-effective
    """)