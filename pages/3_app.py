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
import tempfile
import time
import platform
import subprocess

# --- Windows Helper Functions ---
def is_windows():
    """Check if running on Windows"""
    return platform.system().lower() == 'windows'

def check_ollama_installation():
    """Check if Ollama is installed"""
    try:
        if is_windows():
            result = subprocess.run(['where', 'ollama'], 
                                  capture_output=True, text=True, shell=True)
            return result.returncode == 0
        else:
            result = subprocess.run(['which', 'ollama'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
    except:
        return False

def start_ollama_server():
    """Start Ollama server if not running"""
    try:
        if is_windows():
            # Check if ollama is already running
            tasklist = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq ollama.exe'], 
                                    capture_output=True, text=True, shell=True)
            
            if 'ollama.exe' not in tasklist.stdout:
                subprocess.Popen(['ollama', 'serve'], 
                               creationflags=subprocess.CREATE_NO_WINDOW)
                time.sleep(10)
                return True
            else:
                return True
        else:
            try:
                subprocess.run(['pgrep', 'ollama'], check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError:
                subprocess.Popen(['ollama', 'serve'])
                time.sleep(10)
                return True
    except Exception as e:
        st.error(f"Failed to start Ollama server: {e}")
        return False

def check_ollama_connection():
    """Check Ollama connection and get available models"""
    try:
        if not check_ollama_installation():
            return False, [], "Ollama not installed"
        
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
        if start_ollama_server():
            time.sleep(5)
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=10)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    vision_models = [m['name'] for m in models 
                                   if 'vision' in m['name'].lower() or 'llava' in m['name'].lower()]
                    return True, vision_models, "Started and connected"
                else:
                    return False, [], "Server started but not responding"
            except:
                return False, [], "Failed to start server"
        else:
            return False, [], "Server not running and failed to start"
    except Exception as e:
        return False, [], f"Connection check failed: {e}"

# --- Existing Helper Functions ---
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
def extract_table_with_ollama(pil_image, model_name="llama3.2-vision:11b"):
    """Extract table using Llama Vision via Ollama - Windows optimized"""
    try:
        # Create temp file with proper Windows handling
        temp_dir = os.environ.get('TEMP', tempfile.gettempdir()) if is_windows() else tempfile.gettempdir()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir) as temp_file:
            pil_image.save(temp_file.name, format='PNG', optimize=False, quality=100)
            temp_path = temp_file.name
        
        # Convert image to base64
        with open(temp_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = create_layout_aware_prompt()
        
        # Windows-specific request handling
        session = requests.Session()
        session.trust_env = False
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": 4096
            }
        }
        
        # Make request with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=600,  # 10 minutes timeout
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    break
                else:
                    if attempt == max_retries - 1:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                    time.sleep(5)
                    
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise Exception("Request timed out. The image might be too complex.")
                time.sleep(10)
            except requests.exceptions.ConnectionError:
                if attempt == max_retries - 1:
                    raise Exception("Cannot connect to Ollama. Make sure it's running.")
                time.sleep(5)
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except (PermissionError, FileNotFoundError):
            pass  # Windows sometimes locks files
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '{}')
            
            try:
                # Clean up response text if needed
                if not response_text.strip().startswith('{'):
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    if start != -1 and end != 0:
                        response_text = response_text[start:end]
                
                parsed_data = json.loads(response_text)
                
                return (
                    parsed_data.get("table_data", []), 
                    parsed_data.get("confidence_score", 0.0), 
                    parsed_data.get("reasoning", "No reasoning provided.")
                )
                
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse Ollama response as JSON: {e}")
                st.error(f"Raw response (first 500 chars): {response_text[:500]}")
                return None, 0.0, "JSON parsing failed"
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Ollama server.")
        st.error("Please ensure Ollama is running. Try:")
        st.code("ollama serve")
        return None, 0.0, "Connection failed - Ollama not running"
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The image might be too complex.")
        st.info("💡 Try using a smaller model like 'llava:13b'.")
        return None, 0.0, "Request timed out"
        
    except Exception as e:
        st.error(f"❌ Ollama extraction failed: {e}")
        return None, 0.0, f"Extraction failed: {str(e)}"

def extract_table_with_ai(base64_image_data, api_key, model_name):
    if not api_key:
        st.error("Error: Gemini API key not found.")
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
        st.error(f"An error occurred during Gemini extraction: {e}")
        return None, 0.0, "Extraction failed due to a general error."

def extract_table_with_claude(base64_image_data, api_key, model_name):
    if not api_key:
        st.error("Error: Anthropic API key not found.")
        return None, 0.0, "API key not configured."
    
    prompt = create_layout_aware_prompt()
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(model=model_name, max_tokens=4096, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}}, {"type": "text", "text": prompt}]}])
        json_response_text = message.content[0].text
        parsed_json = json.loads(json_response_text)
        return parsed_json.get("table_data", []), parsed_json.get("confidence_score", 0.0), parsed_json.get("reasoning", "No reasoning provided.")
    except Exception as e:
        st.error(f"An error occurred with the Claude API: {e}")
        return None, 0.0, "Extraction failed due to an API error."

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def get_model_options(ai_provider):
    """Get model options based on AI provider"""
    if ai_provider == "Google Gemini":
        return ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro", "gemini-1.5-pro"]
    elif ai_provider == "Anthropic Claude":
        return ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
    elif ai_provider == "Llama (Ollama)":
        is_connected, available_models, _ = check_ollama_connection()
        if is_connected and available_models:
            return available_models
        else:
            return ["llama3.2-vision:11b", "llama3.2-vision:90b", "llava:13b"]
    else:
        return []

def install_ollama_model(model_name):
    """Install an Ollama model with progress"""
    try:
        if is_windows():
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            progress_placeholder = st.empty()
            output_lines = []
            
            for line in process.stdout:
                output_lines.append(line.strip())
                if len(output_lines) > 10:
                    output_lines = output_lines[-10:]
                
                progress_placeholder.text("Downloading...\n" + "\n".join(output_lines))
            
            process.wait()
            return process.returncode == 0
        else:
            result = subprocess.run(['ollama', 'pull', model_name], 
                                  capture_output=True, text=True)
            return result.returncode == 0
            
    except Exception as e:
        st.error(f"Failed to install model: {e}")
        return False

# --- Main App UI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("Step 2: 🔎 AI-Powered Table Extractor")

if not st.session_state.get('converted_pil_images'):
    st.warning("⚠️ Please go back to the **📂 PDF Converter** page and convert a PDF first.")
else:
    st.markdown("Select an image below to extract a table. The AI will analyze it and turn it into data.")
    
    image_options = {f"Image {i+1}": i for i in range(len(st.session_state.converted_pil_images))}
    if image_options:
        selected_image_key = st.selectbox("Choose an image to process:", options=list(image_options.keys()))
        selected_image_index = image_options[selected_image_key]
        st.session_state.selected_image_index = selected_image_index
        selected_pil_image = st.session_state.converted_pil_images[selected_image_index]
    else:
        st.error("No images found from the conversion step.")
        selected_pil_image = None

    if selected_pil_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(selected_pil_image, caption="Selected Image for Extraction", use_container_width=True)
        with col2:
            st.header("⚙️ AI Options")
            ai_provider = st.selectbox("Choose AI Provider:", (
                "Google Gemini", 
                "Anthropic Claude", 
                "Llama (Ollama)"
            ))
            
            model_options = get_model_options(ai_provider)
            selected_model = st.selectbox("Choose AI Model:", options=model_options)
            
            # Ollama-specific status and controls
            if ai_provider == "Llama (Ollama)":
                is_connected, available_models, status_msg = check_ollama_connection()
                
                if is_connected:
                    st.success(f"✅ Ollama: {status_msg}")
                    if available_models:
                        st.info(f"📋 Available vision models: {len(available_models)}")
                    else:
                        st.warning("⚠️ No vision models found")
                        
                        # Quick install options
                        st.markdown("**Quick Install:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("📥 Install llama3.2-vision:11b", help="Recommended model"):
                                with st.spinner("Installing model... This may take several minutes."):
                                    if install_ollama_model("llama3.2-vision:11b"):
                                        st.success("✅ Model installed!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Installation failed")
                        
                        with col_b:
                            if st.button("📥 Install llava:13b", help="Smaller, faster model"):
                                with st.spinner("Installing model... This may take several minutes."):
                                    if install_ollama_model("llava:13b"):
                                        st.success("✅ Model installed!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Installation failed")
                else:
                    st.error(f"❌ Ollama: {status_msg}")
                    
                    with st.expander("🔧 Windows Setup Help"):
                        st.markdown("""
                        **If Ollama is not working:**
                        
                        1. **Install Ollama:**
                           - Download: https://ollama.ai/download/windows
                           - Or use: `winget install Ollama.Ollama`
                        
                        2. **Start Ollama:**
                           ```
                           ollama serve
                           ```
                        
                        3. **Check Windows Firewall** and allow Ollama
                        
                        4. **Restart PowerShell** after installation
                        """)

            if st.button("Extract Table from Image", type="primary"):
                # Determine extraction function and API key
                if ai_provider == "Google Gemini":
                    api_key_to_use = gemini_api_key
                    extraction_function = extract_table_with_ai
                elif ai_provider == "Anthropic Claude":
                    api_key_to_use = anthropic_api_key
                    extraction_function = extract_table_with_claude
                elif ai_provider == "Llama (Ollama)":
                    api_key_to_use = None
                    extraction_function = lambda img, key, model: extract_table_with_ollama(selected_pil_image, model)
                
                # Check requirements
                if ai_provider in ["Google Gemini", "Anthropic Claude"] and not api_key_to_use:
                    st.warning(f"Please add your {ai_provider} API Key to the .env file.")
                elif ai_provider == "Llama (Ollama)" and not check_ollama_connection()[0]:
                    st.error("Ollama is not running. Please start it with: `ollama serve`")
                else:
                    with st.spinner(f"🦙 {ai_provider} is analyzing the image with **{selected_model}**. Please wait..."):
                        start_time = time.time()
                        
                        if ai_provider == "Llama (Ollama)":
                            table_data, confidence, reasoning = extraction_function(None, None, selected_model)
                        else:
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

    # Results Display
    if st.session_state.get('confidence') is not None:
        st.subheader("📊 Extraction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}")
        with col2:
            if ai_provider == "Llama (Ollama)":
                st.metric(label="Processing Time", value=f"{processing_time:.1f}s")
        with col3:
            if ai_provider == "Llama (Ollama)":
                st.metric(label="Model", value=selected_model.split(':')[0])
        
        with st.expander("🧠 AI's Reasoning"):
            st.info(st.session_state.get('reasoning', 'No reasoning provided.'))
            
            if ai_provider == "Llama (Ollama)":
                st.markdown(f"""
                **Model Info:**
                - **Model:** {selected_model}
                - **Provider:** Ollama (Local)
                - **Privacy:** ✅ Fully local processing
                - **Cost:** ✅ Free
                """)

    # Data Display and Download
    if 'extracted_df' in st.session_state and st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
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