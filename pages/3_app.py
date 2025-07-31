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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Helper Functions ---
def prepare_image_from_pil(pil_image):
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
    return """
    You are an expert data extractor. Your primary task is to analyze the provided image, identify the main data table, and extract its contents with high precision, paying special attention to its orientation.

    **Step 1: Detect Table Layout**
    First, determine the table's layout. Is it:
    a) **Standard Layout:** Headers are in the top row, and data records are in subsequent rows.
    b) **Transposed Layout:** Headers are in the first column, and data records are in subsequent columns.

    **Step 2: Extract Data Based on Layout**
    - **If Standard Layout:** Proceed with normal extraction. The first row is your header list, and each row that follows is a data list.
    - **If Transposed Layout:** You must **un-pivot** the data. The first column contains the headers, and each following column is a record. You must reconstruct this into a standard format.
      - **Example of a Transposed Table:**
        [
          ["Name", "John Doe"],
          ["Age", "30"],
          ["City", "New York"]
        ]
      - **Your mandatory output for the above example must be:**
        [
          ["Name", "Age", "City"],
          ["John Doe", "30", "New York"]
        ]

    **Step 3: Final JSON Output**
    Regardless of the original layout, you must return a single, valid JSON object with the following three keys: "table_data", "confidence_score", and "reasoning".

    - **`table_data`**: MUST be a list of lists in the **standard format** (the first inner list is the header row).
    - **`confidence_score`**: A numerical score from 0.0 to 1.0 on your confidence in the accuracy and correct orientation of the extraction.
    - **`reasoning`**: **Crucially, state which layout you detected** (e.g., "Detected a transposed layout and un-pivoted the data.") and explain any challenges.

    Your top priority is returning the data in the correct, standard format.
    """

# --- AI Extraction Functions ---
def extract_table_with_gemini(base64_image_data, api_key, model_name):
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
    if not api_key:
        return None, 0.0, "API key not configured."
    
    prompt = create_layout_aware_prompt()
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(model=model_name, max_tokens=4096, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}}, {"type": "text", "text": prompt}]}])
        json_response_text = message.content[0].text
        parsed_json = json.loads(json_response_text)
        return parsed_json.get("table_data", []), parsed_json.get("confidence_score", 0.0), parsed_json.get("reasoning", "No reasoning provided.")
    except Exception as e:
        return None, 0.0, f"API error: {str(e)}"

# --- Batch Processing Functions ---
def process_single_image(args):
    """Process a single image - used for batch processing"""
    image_index, pil_image, ai_provider, selected_model, api_key = args
    
    try:
        base64_image = prepare_image_from_pil(pil_image)
        if not base64_image:
            return {
                'index': image_index,
                'success': False,
                'error': 'Failed to prepare image',
                'table_data': None,
                'confidence': 0.0,
                'reasoning': 'Image preparation failed'
            }
        
        if ai_provider == "Google Gemini":
            table_data, confidence, reasoning = extract_table_with_gemini(base64_image, api_key, selected_model)
        elif ai_provider == "Anthropic Claude":
            table_data, confidence, reasoning = extract_table_with_claude(base64_image, api_key, selected_model)
        else:
            return {
                'index': image_index,
                'success': False,
                'error': 'Unsupported AI provider',
                'table_data': None,
                'confidence': 0.0,
                'reasoning': 'Invalid provider'
            }
        
        # Process the extracted data
        df = None
        if table_data and len(table_data) > 1:
            try:
                header, data = table_data[0], table_data[1:]
                df = pd.DataFrame(data, columns=header)
            except Exception as e:
                # Try without header if there's a mismatch
                df = pd.DataFrame(table_data)
        elif table_data:
            df = pd.DataFrame(table_data)
        
        return {
            'index': image_index,
            'success': df is not None and not df.empty,
            'error': None if df is not None else 'No table data extracted',
            'table_data': table_data,
            'dataframe': df,
            'confidence': confidence,
            'reasoning': reasoning
        }
        
    except Exception as e:
        return {
            'index': image_index,
            'success': False,
            'error': str(e),
            'table_data': None,
            'dataframe': None,
            'confidence': 0.0,
            'reasoning': f'Processing failed: {str(e)}'
        }

def batch_extract_tables(images, ai_provider, selected_model, api_key, max_workers=3):
    """Extract tables from multiple images in parallel"""
    # Prepare arguments for each image
    args_list = [
        (i, img, ai_provider, selected_model, api_key) 
        for i, img in enumerate(images)
    ]
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_image, args): args[0] 
            for args in args_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            result = future.result()
            results.append(result)
    
    # Sort results by original image index
    results.sort(key=lambda x: x['index'])
    return results

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def to_excel_multiple_sheets(dataframes_dict):
    """Create Excel file with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

def get_model_options(ai_provider):
    """Get model options based on AI provider"""
    if ai_provider == "Google Gemini":
        return ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro", "gemini-1.5-pro"]
    elif ai_provider == "Anthropic Claude":
        return ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
    else:
        return []

# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 2: 🔎 AI-Powered Table Extractor")

if not st.session_state.get('converted_pil_images'):
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Choose to extract tables from individual images or process all images at once.")
    
    # Processing mode selection
    processing_mode = st.radio(
        "**Processing Mode:**",
        ["Single Image", "Batch Process All Images"],
        horizontal=True
    )
    
    # AI Configuration (shared for both modes)
    st.header("⚙️ AI Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        ai_provider = st.selectbox("Choose AI Provider:", (
            "Google Gemini", 
            "Anthropic Claude"
        ))
    
    with col2:
        model_options = get_model_options(ai_provider)
        selected_model = st.selectbox("Choose AI Model:", options=model_options)
    
    st.divider()
    
    # Single Image Mode
    if processing_mode == "Single Image":
        st.header("🖼️ Single Image Processing")
        
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
                    # Check API key
                    if ai_provider == "Google Gemini":
                        api_key_to_use = gemini_api_key
                        extraction_function = extract_table_with_gemini
                    elif ai_provider == "Anthropic Claude":
                        api_key_to_use = anthropic_api_key
                        extraction_function = extract_table_with_claude
                    
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
                            
                            # Store results
                            st.session_state.confidence = confidence
                            st.session_state.reasoning = reasoning
                            
                            if table_data and len(table_data) > 1:
                                try:
                                    header, data = table_data[0], table_data[1:]
                                    st.session_state.extracted_df = pd.DataFrame(data, columns=header)
                                    st.session_state.original_df = st.session_state.extracted_df.copy()
                                    st.success(f"✅ Table extracted successfully in {processing_time:.1f} seconds!")
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
    
    # Batch Processing Mode
    else:
        st.header("🚀 Batch Processing All Images")
        
        total_images = len(st.session_state.converted_pil_images)
        st.info(f"Ready to process **{total_images}** images with **{ai_provider}** using **{selected_model}**")
        
        # Batch processing options
        col1, col2 = st.columns(2)
        with col1:
            batch_max_workers = st.slider(
                "Parallel Workers:", 
                min_value=1, 
                max_value=5,
                value=3,
                help="Number of parallel processes for faster processing."
            )
        
        with col2:
            estimated_cost = total_images * 0.05  # Rough estimate
            st.warning(f"💰 Estimated API cost: ~${estimated_cost:.2f}")
        
        # Preview images
        with st.expander("🖼️ Preview Images to Process"):
            cols = st.columns(min(4, total_images))
            for i, img in enumerate(st.session_state.converted_pil_images[:8]):  # Show first 8
                with cols[i % 4]:
                    st.image(img, caption=f"Image {i+1}", use_container_width=True)
            if total_images > 8:
                st.info(f"... and {total_images - 8} more images")
        
        if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
            # Check API key
            api_key_to_use = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
            if not api_key_to_use:
                st.error(f"Please add your {ai_provider} API Key to the .env file.")
                st.stop()
            
            # Start batch processing
            start_time = time.time()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            with st.spinner(f"Processing {total_images} images with {ai_provider}..."):
                # Run batch extraction
                results = batch_extract_tables(
                    st.session_state.converted_pil_images,
                    ai_provider,
                    selected_model,
                    api_key_to_use,
                    batch_max_workers
                )
                
                # Update progress
                progress_bar.progress(1.0)
                processing_time = time.time() - start_time
                
                # Analyze results
                successful_extractions = [r for r in results if r['success']]
                failed_extractions = [r for r in results if not r['success']]
                
                # Display summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", total_images)
                with col2:
                    st.metric("Successful", len(successful_extractions))
                with col3:
                    st.metric("Failed", len(failed_extractions))
                with col4:
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                
                if successful_extractions:
                    st.success(f"✅ Successfully extracted {len(successful_extractions)} tables!")
                    
                    # Store batch results in session state
                    st.session_state.batch_results = results
                    st.session_state.batch_dataframes = {
                        f"Image_{r['index']+1}": r['dataframe'] 
                        for r in successful_extractions if r['dataframe'] is not None
                    }
                    
                    # Show detailed results
                    with st.expander("📊 Detailed Results", expanded=True):
                        for result in results:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                st.image(
                                    st.session_state.converted_pil_images[result['index']], 
                                    caption=f"Image {result['index']+1}",
                                    width=150
                                )
                            
                            with col2:
                                if result['success']:
                                    st.success(f"✅ **Image {result['index']+1}** - Table extracted")
                                    if result['dataframe'] is not None:
                                        st.write(f"**Rows:** {len(result['dataframe'])}, **Columns:** {len(result['dataframe'].columns)}")
                                        st.write(f"**Confidence:** {result['confidence']:.1%}")
                                        
                                        # Show preview of extracted data with toggle button
                                        preview_key = f"show_preview_{result['index']}"
                                        if preview_key not in st.session_state:
                                            st.session_state[preview_key] = False
                                        
                                        if st.button(f"👁️ {'Hide' if st.session_state[preview_key] else 'Show'} Preview", 
                                                   key=f"preview_btn_{result['index']}", 
                                                   help="Click to show/hide data preview"):
                                            st.session_state[preview_key] = not st.session_state[preview_key]
                                        
                                        if st.session_state[preview_key]:
                                            st.dataframe(result['dataframe'].head(), use_container_width=True)
                                else:
                                    st.error(f"❌ **Image {result['index']+1}** - Failed")
                                    st.write(f"**Error:** {result['error']}")
                            
                            with col3:
                                if result['success'] and result['dataframe'] is not None:
                                    # Individual download buttons
                                    csv_data = result['dataframe'].to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="📥 CSV",
                                        data=csv_data,
                                        file_name=f"image_{result['index']+1}_table.csv",
                                        mime="text/csv",
                                        key=f"csv_{result['index']}"
                                    )
                                    
                                    excel_data = to_excel(result['dataframe'])
                                    st.download_button(
                                        label="📥 Excel",
                                        data=excel_data,
                                        file_name=f"image_{result['index']+1}_table.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=f"excel_{result['index']}"
                                    )
                    
                    # Bulk download options
                    st.divider()
                    st.subheader("📦 Bulk Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📋 Individual CSV Files (ZIP)**")
                        if st.button("📥 Download All CSVs as ZIP", use_container_width=True):
                            import zipfile
                            zip_buffer = BytesIO()
                            
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for result in successful_extractions:
                                    if result['dataframe'] is not None:
                                        csv_data = result['dataframe'].to_csv(index=False)
                                        zip_file.writestr(
                                            f"image_{result['index']+1}_table.csv", 
                                            csv_data
                                        )
                            
                            st.download_button(
                                label="📥 Download ZIP File",
                                data=zip_buffer.getvalue(),
                                file_name="all_extracted_tables.zip",
                                mime="application/zip"
                            )
                    
                    with col2:
                        st.markdown("**📊 Combined Excel File (Multiple Sheets)**")
                        if st.button("📥 Create Combined Excel File", use_container_width=True):
                            if st.session_state.batch_dataframes:
                                combined_excel = to_excel_multiple_sheets(st.session_state.batch_dataframes)
                                st.download_button(
                                    label="📥 Download Combined Excel",
                                    data=combined_excel,
                                    file_name="all_tables_combined.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                
                if failed_extractions:
                    st.warning(f"⚠️ {len(failed_extractions)} images failed to process")
                    
                    with st.expander("❌ Failed Extractions"):
                        for result in failed_extractions:
                            st.error(f"**Image {result['index']+1}:** {result['error']}")
                
                # Option to retry failed extractions
                if failed_extractions:
                    st.divider()
                    if st.button("🔄 Retry Failed Extractions", type="secondary"):
                        failed_images = [(r['index'], st.session_state.converted_pil_images[r['index']]) for r in failed_extractions]
                        
                        with st.spinner(f"Retrying {len(failed_images)} failed extractions..."):
                            retry_args = [
                                (idx, img, ai_provider, selected_model, api_key_to_use)
                                for idx, img in failed_images
                            ]
                            
                            retry_results = []
                            for args in retry_args:
                                result = process_single_image(args)
                                retry_results.append(result)
                            
                            # Update original results
                            for retry_result in retry_results:
                                for i, original_result in enumerate(st.session_state.batch_results):
                                    if original_result['index'] == retry_result['index']:
                                        st.session_state.batch_results[i] = retry_result
                                        break
                            
                            st.rerun()

    # Display single image results
    if processing_mode == "Single Image" and st.session_state.get('confidence') is not None:
        st.subheader("📊 Extraction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}")
        with col2:
            if 'processing_time' in locals():
                st.metric(label="Processing Time", value=f"{processing_time:.1f}s")
        with col3:
            st.metric(label="AI Provider", value=ai_provider)
        
        with st.expander("🧠 AI's Reasoning"):
            st.info(st.session_state.get('reasoning', 'No reasoning provided.'))
            
            st.markdown(f"""
            **Model Info:**
            - **Model:** {selected_model}
            - **Provider:** {ai_provider}
            - **Processing:** Cloud API
            """)

    # Data Display and Download for single image
    if processing_mode == "Single Image" and 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("Extracted Data Preview")
        st.dataframe(st.session_state.extracted_df)
        
        st.divider()
        st.subheader("📥 Download Extracted Data")
        
        col1, col2 = st.columns(2)
        with col1:
            file_name = st.text_input("File Name (without extension):", value="extracted_table", key="download_filename")
        with col2:
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