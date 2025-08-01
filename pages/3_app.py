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
import openai # Added for OpenAI
import time

# --- Helper Functions ---
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object-type columns with mixed types to string,
    and ensures consistent serialization with PyArrow.
    """
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

def create_layout_aware_prompt():
    """Creates a standardized prompt for layout-aware table extraction with merged cell support."""
    return """
    You are an expert data analyst specializing in complex financial and technical table extraction. Your primary task is to analyze the provided image, identify the main data table, and extract its contents with high precision, paying special attention to its orientation, cell structure, data density, and **COMPLETE ROW EXTRACTION**.

    **🚨 MISSION CRITICAL REQUIREMENTS 🚨**
    1. **EXTRACT EVERY SINGLE ROW** - Missing rows = Complete failure
    2. **NUMERICAL ACCURACY IS ABSOLUTE** - Every digit must be perfect
    3. **NO ROW LEFT BEHIND** - Scan systematically from top to bottom

    **Step 1: COMPLETE TABLE BOUNDARY DETECTION**
    
    **🔍 MANDATORY ROW SCANNING PROTOCOL:**
    1. **Identify the ENTIRE table boundary** - Find the absolute top, bottom, left, and right edges
    2. **Count total rows visually BEFORE extraction** - This is your target row count
    3. **Scan systematically from top to bottom** - Don't skip ANY horizontal line that contains data
    4. **Look for continuation indicators** - Tables may extend beyond obvious boundaries
    5. **Check for subtotals, totals, and summary rows** - These are data rows too!
    
    Determine the table's layout and structure:
    a) **Standard Layout:** Headers are in the top row, and data records are in subsequent rows.
    b) **Transposed Layout:** Headers are in the first column, and data records are in subsequent columns.  
    c) **Complex Structure:** The table contains merged cells, nested headers, or hierarchical organization.
    d) **Dense Financial Layout:** Multi-level headers, grouped columns, rotated text, and dense numerical data typical of financial reports.
    
    **⚠️ ROW EXTRACTION CHECKPOINTS:**
    - Did I scan the ENTIRE vertical space of the table?
    - Are there any faint lines or subtle row separators I missed?
    - Do I have continuation rows that might be formatted differently?
    - Are there summary/total rows at the bottom that I need to include?

    **Step 2: Handle Complex Cell Structures and Dense Data**
    Pay special attention to these common patterns in financial/technical tables:
    
    - **Rotated/Vertical Text Headers:** Text that appears rotated 90° should be read and included as column headers.
    
    - **Multi-Level Column Groups:** When columns are grouped under parent headers (like "1 USD / KHR" spanning multiple sub-columns), create descriptive combined headers.
      - Example: "1 USD / KHR" + "Profit Loss" = "1 USD KHR Profit Loss"
    
    - **Merged Header Cells:** When a header spans multiple columns, create meaningful combined names rather than repetition.
      - Better: ["Company", "USD Profit", "USD Loss", "KHR Profit", "KHR Loss"] 
      - Avoid: ["Company", "Currency", "Currency", "Currency", "Currency"]
    
    - **Merged Data Cells:** When a data cell spans multiple columns/rows, repeat the value in each position it occupies.
    
    - **Dense Numerical Data:** Handle parentheses indicating negative numbers, commas in large numbers, and decimal precision carefully.
    
    - **Row Headers with Categories:** When left-most columns contain categorical data (like company names, industry types), preserve these exactly.
    
    - **Empty/Dash Cells:** Distinguish between truly empty cells, cells with dashes (-), and cells that are part of merged ranges.

    **Step 3: SYSTEMATIC ROW-BY-ROW EXTRACTION**
    
    **🎯 COMPLETE ROW EXTRACTION PROTOCOL:**
    
    **Phase 1: ROW INVENTORY**
    - Count all visible rows in the table (including headers, data, subtotals, totals)
    - Note any rows that might be partially visible or cut off
    - Identify rows with different formatting (bold, italic, indented)
    - Mark rows that span multiple lines or have wrapped text
    
    **Phase 2: SEQUENTIAL EXTRACTION**
    - **Extract Row 1:** Headers (create meaningful combined names for grouped headers)
    - **Extract Row 2:** First data row (verify column count matches headers)
    - **Extract Row 3:** Second data row (verify alignment)
    - **Continue systematically:** Extract EVERY subsequent row without exception
    - **Include ALL special rows:** Subtotals, category breaks, summary rows, footnote references
    
    **Phase 3: EXTRACTION BY LAYOUT TYPE**
    - **If Standard Layout:** Row-by-row extraction, maintaining perfect column alignment
    - **If Transposed Layout:** Un-pivot while ensuring no columns (original rows) are missed
    - **If Complex/Dense Structure:**
      1. Map all column boundaries across the ENTIRE table height
      2. Extract each row systematically, ensuring no row is skipped
      3. Handle wrapped text and multi-line entries as single rows
      4. Preserve special formatting indicators
    
    **🚨 ROW COMPLETENESS VERIFICATION:**
    - **Before finishing:** Count extracted rows vs. visually counted rows
    - **If counts don't match:** Re-scan for missed rows
    - **Check bottom of table:** Often contains critical summary data
    - **Verify no rows were accidentally merged or skipped**

    **Step 4: Special Handling for Financial Tables - NUMERICAL ACCURACY IS CRITICAL**
    
    **⚠️ ABSOLUTE PRIORITY: NUMERICAL ACCURACY ⚠️**
    - **EVERY NUMBER MUST BE EXTRACTED EXACTLY AS SHOWN** - No approximations, no rounding, no guessing
    - **Double-check every digit** - A single incorrect digit can invalidate financial data
    - **If uncertain about a number, mark it clearly rather than guessing**
    - **Verify decimal places, commas, and parentheses precisely**
    
    Specific formatting rules:
    - **Currency Symbols:** Preserve currency information in headers or data as appropriate
    - **Negative Numbers:** Extract parenthetical negatives EXACTLY: "(1,250)" should remain "(1,250)" unless context clearly requires conversion
    - **Large Numbers:** Preserve comma separators EXACTLY as shown: "1,250,000" not "1250000"
    - **Decimal Precision:** Maintain exact decimal places: "1.50" not "1.5", "0.001" not ".001"
    - **Percentage Values:** Keep percentage symbols and exact decimal precision where they appear
    - **Date Formats:** Maintain original date formatting exactly
    - **Company/Entity Names:** Extract full company names even if they span multiple lines in the image
    
    **NUMERICAL VERIFICATION CHECKLIST:**
    ✓ Every digit matches the source exactly
    ✓ All commas, decimals, and parentheses are preserved
    ✓ No numbers are accidentally transposed or approximated
    ✓ Negative indicators (parentheses, minus signs) are captured correctly

    **Step 5: COMPREHENSIVE DATA QUALITY ASSURANCE**
    
    **🎯 ZERO TOLERANCE FOR MISSING ROWS OR INCORRECT NUMBERS:**
    
    **ROW COMPLETENESS REQUIREMENTS:**
    - **EVERY visible row must be extracted** - No exceptions, no shortcuts
    - **Total extracted rows must match visual count** - Recount if necessary
    - **Include header rows, data rows, subtotal rows, total rows, and any footnote rows**
    - **Don't skip rows with different formatting** (bold, italic, indented, highlighted)
    - **Extract continued/wrapped rows as single entries**
    - **Capture partial rows** if they contain any data
    
    **NUMERICAL PRECISION REQUIREMENTS:**
    - Every row must have the same number of columns as the header row
    - **ALL NUMBERS MUST BE PIXEL-PERFECT ACCURATE** - Treat each number as if millions of dollars depend on it
    - **NEVER estimate or approximate numerical values** - If you cannot read a number clearly, mark it as "[UNCLEAR]" rather than guess
    - **Maintain exact formatting:** Preserve commas, decimals, parentheses, and spacing exactly as shown
    - **Cross-verify large numbers:** For numbers with 4+ digits, double-check each digit sequence
    - Maintain consistent data types within columns (all numbers, all text, etc.)
    - No empty cells unless the original data is genuinely empty
    - When multiple header levels exist, create clear, descriptive combined column names
    - Ensure row labels/identifiers are properly captured from leftmost columns
    
    **MANDATORY VERIFICATION CHECKLIST:**
    ✓ Row count matches visual inspection
    ✓ Every number is digit-perfect accurate
    ✓ All formatting (commas, decimals, parentheses) preserved
    ✓ No rows accidentally merged or skipped
    ✓ Headers properly reflect hierarchical structure
    ✓ Column alignment maintained across all rows

    **Step 6: Final JSON Output - WITH NUMERICAL ACCURACY GUARANTEE**
    Return a single, valid JSON object with these five keys: "table_data", "confidence_score", "reasoning", "structure_notes", and "numerical_accuracy_notes".

    - **`table_data`**: MUST be a list of lists in standard format. First inner list = headers, subsequent lists = data rows. Every row must have identical column count. **ALL NUMERICAL VALUES MUST BE EXACTLY AS SHOWN IN SOURCE.**
    - **`confidence_score`**: Score from 0.0 to 1.0 reflecting extraction accuracy and completeness. **Reduce score significantly if any numbers were unclear or estimated.**
    - **`reasoning`**: Describe the layout detected and extraction approach used.
    - **`structure_notes`**: Document merged cells, grouped headers, rotated text, and complex formatting encountered.
    - **`numerical_accuracy_notes`**: **MANDATORY FIELD** - Explicitly confirm that all numbers were extracted with 100% accuracy, or note any numbers that were unclear/estimated with [UNCLEAR] markers.

    **Example for Dense Financial Table with Perfect Numerical Accuracy:**
    Original complex structure with grouped headers:
    ```
    |           | 1 USD / KHR        | Exchange Rate |
    | Company   | Profit | Loss      | Current       |
    | ABC Bank  | 1,250  | (500)     | 4,100.25      |
    | XYZ Corp  | 15,875 | (2,100.5) | 4,099.80      |
    ```
    
    Your output should be:
    ```json
    {
      "table_data": [
        ["Company", "1 USD KHR Profit", "1 USD KHR Loss", "Exchange Rate Current"],
        ["ABC Bank", "1,250", "(500)", "4,100.25"],
        ["XYZ Corp", "15,875", "(2,100.5)", "4,099.80"]
      ],
      "structure_notes": "Detected grouped headers with '1 USD / KHR' spanning two sub-columns",
      "numerical_accuracy_notes": "All numerical values extracted with 100% accuracy - verified each digit, decimal place, and formatting symbol"
    }
    ```

    **FINAL CRITICAL REMINDER:**
    🚨 **FINANCIAL DATA ACCURACY IS NON-NEGOTIABLE** 🚨
    - A single wrong digit can cause massive financial miscalculations
    - Take extra time to verify each number rather than rushing
    - When in doubt about a digit, use [UNCLEAR] rather than guessing
    - Your numerical accuracy directly impacts financial decisions and compliance

    **Critical Instructions for Dense Tables - NUMBERS ABOVE ALL:**
    1. **🎯 NUMERICAL ACCURACY IS THE TOP PRIORITY** - Everything else is secondary
    2. Read ALL text carefully, including rotated or small text, but TRIPLE-CHECK all numbers
    3. Identify column boundaries precisely - dense tables often have narrow spacing
    4. Create meaningful, unique column headers that capture the hierarchical structure
    5. **Verify every single digit in every number** - Use systematic left-to-right verification
    6. Extract every visible data point - dense tables contain valuable information in every cell
    7. **If any number is unclear, mark as [UNCLEAR] rather than estimate**

    **ZERO-ERROR NUMERICAL EXTRACTION PROTOCOL:**
    - Scan each number multiple times before recording
    - Pay attention to number formatting patterns (commas every 3 digits, decimal precision)
    - Verify negative indicators are correctly captured
    - Cross-reference similar numbers to check for consistency in formatting
    - Remember: Perfect accuracy on 90% of numbers is better than 95% accuracy on 100% of numbers

    Your top priority is complete, accurate extraction with **PERFECT NUMERICAL PRECISION** and meaningful column headers that reflect the table's hierarchical structure.
    """

# --- AI Extraction Functions ---
def extract_table_with_gemini(base64_image_data, api_key, model_name):
    """Extracts table data using Google Gemini API."""
    if not api_key:
        return None, 0.0, "API key not configured."

    prompt = create_layout_aware_prompt()

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": base64_image_data}}] }], "generationConfig": {"responseMimeType": "application/json"}}
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

def extract_table_with_claude(base64_image_data, api_key, model_name):
    """Extracts table data using Anthropic Claude API."""
    if not api_key:
        return None, 0.0, "API key not configured."

    prompt = create_layout_aware_prompt()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(model=model_name, max_tokens=4096, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}}, {"type": "text", "text": prompt}]}])
        json_response_text = message.content[0].text
        # Clean potential markdown code fences
        if json_response_text.strip().startswith("```json"):
            json_response_text = json_response_text.strip()[7:-3].strip()
        parsed_json = json.loads(json_response_text)
        return parsed_json.get("table_data", []), parsed_json.get("confidence_score", 0.0), parsed_json.get("reasoning", "No reasoning provided.")
    except Exception as e:
        return None, 0.0, f"API error: {str(e)}"

# --- NEW: Function to handle OpenAI extraction ---
def extract_table_with_openai(base64_image_data, api_key, model_name):
    """Extracts table data using OpenAI GPT-4 API."""
    if not api_key:
        return None, 0.0, "API key not configured."

    prompt = create_layout_aware_prompt()

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


def to_excel(df):
    """Converts a DataFrame to an in-memory Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY") # Added for OpenAI

st.title("Step 2: 🔎 AI-Powered Table Extractor")

if 'converted_pil_images' not in st.session_state or not st.session_state.converted_pil_images:
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Choose an image from the converted PDF and an AI model to extract the table.")

    # AI Configuration
    st.header("⚙️ AI Configuration")
    # --- MODIFIED: Added OpenAI models to the list ---
    model_options = [
        "gpt-4o", "gpt-4-turbo", # OpenAI Models
        "claude-3-5-sonnet-20240620" # Anthropic Models
        "gemini-2.5-pro", "gemini-2.5-flash" # Google Models
    ]
    selected_model = st.selectbox("Choose AI Model:", options=model_options)

    st.divider()

    # Image Processing
    st.header("🖼️ Image Processing")

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
            if st.button("Extract Table from Image", type="primary"):
                # --- MODIFIED: Added logic for OpenAI provider ---
                if "gemini" in selected_model:
                    ai_provider = "Google Gemini"
                    api_key_to_use = gemini_api_key
                    extraction_function = extract_table_with_gemini
                elif "claude" in selected_model:
                    ai_provider = "Anthropic Claude"
                    api_key_to_use = anthropic_api_key
                    extraction_function = extract_table_with_claude
                elif "gpt" in selected_model: # Logic for OpenAI
                    ai_provider = "OpenAI"
                    api_key_to_use = openai_api_key
                    extraction_function = extract_table_with_openai
                else:
                    st.error("Invalid model selected. Cannot determine AI provider.")
                    st.stop()

                if not api_key_to_use:
                    st.warning(f"Please add your {ai_provider} API Key to the .env file.")
                else:
                    with st.spinner(f"🤖 {ai_provider} is analyzing the image with **{selected_model}**. Please wait..."):
                        start_time = time.time()

                        base64_image = prepare_image_from_pil(selected_pil_image)
                        if base64_image:
                            table_data, confidence, reasoning = extraction_function(base64_image, api_key_to_use, selected_model)
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
                                st.success(f"✅ Table extracted successfully in {st.session_state.processing_time:.1f} seconds!")
                                st.info("✅ Data is ready! Please proceed to the **🤖 Validator** page.")
                            except Exception as e:
                                st.error(f"Data Mismatch Error: {e}. Trying to load without a header.")
                                st.session_state.extracted_df = pd.DataFrame(table_data)
                                st.session_state.original_df = st.session_state.extracted_df.copy()
                        elif table_data:
                            st.warning("Only one row of data was extracted.")
                            st.session_state.extracted_df = pd.DataFrame(table_data)
                            st.session_state.original_df = st.session_state.extracted_df.copy()
                        else:
                            st.error("The AI could not find a table in the image.")
                            st.session_state.extracted_df = None
                            st.session_state.original_df = None

    # --- Display Single Image Results ---
    if st.session_state.get('confidence') is not None:
        st.subheader("📊 Extraction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}")
        with col2:
            st.metric(label="Processing Time", value=f"{st.session_state.processing_time:.1f}s")
        with col3:
            st.metric(label="AI Provider", value=st.session_state.get('ai_provider', 'N/A'))

        with st.expander("🧠 AI's Reasoning"):
            st.info(st.session_state.get('reasoning', 'No reasoning provided.'))
            st.markdown(f"""
            **Model Info:**
            - **Model:** {st.session_state.get('selected_model', 'N/A')}
            - **Provider:** {st.session_state.get('ai_provider', 'N/A')}
            """)

    # Data Display and Download for single image
    if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("Extracted Data Preview")
        clean_df = normalize_dataframe(st.session_state.extracted_df)
        st.dataframe(clean_df)

        st.divider()
        st.subheader("📥 Download Extracted Data")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            file_name = st.text_input("File Name (without extension):", value="extracted_table", key="download_filename")
        with dl_col2:
            file_format = st.selectbox("Choose Format:", ["Excel (.xlsx)", "CSV (.csv)"], key="download_format")

        if file_name:
            if file_format == "Excel (.xlsx)":
                file_data = to_excel(st.session_state.extracted_df)
                file_extension = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                file_data = st.session_state.extracted_df.to_csv(index=False).encode('utf-8')
                file_extension = "csv"
                mime_type = "text/csv"

            st.download_button(
                label=f"📥 Download as {file_format}",
                data=file_data,
                file_name=f"{file_name}.{file_extension}",
                mime=mime_type,
                type="primary"
            )
        else:
            st.warning("Please enter a file name to enable download.")