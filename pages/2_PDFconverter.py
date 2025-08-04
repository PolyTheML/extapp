import streamlit as st
import fitz  # PyMuPDF
import io
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
from skimage import restoration, filters, morphology, exposure
import matplotlib.pyplot as plt

# --- Enhanced Image Processing Functions ---
def enhance_image_quality(pil_image, enhancement_level="medium"):
    """
    Applies comprehensive image enhancement for better OCR/table extraction.
    
    Args:
        pil_image: PIL Image object
        enhancement_level: "light", "medium", "aggressive"
    
    Returns:
        Enhanced PIL Image
    """
    # Convert PIL to numpy array for OpenCV processing
    img_array = np.array(pil_image)
    
    # Convert to grayscale for processing (but keep original for color analysis)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        is_color = True
        original_color = img_array.copy()
    else:
        gray = img_array
        is_color = False
    
    # Apply enhancement based on level
    if enhancement_level == "light":
        enhanced = apply_light_enhancement(gray)
    elif enhancement_level == "medium":
        enhanced = apply_medium_enhancement(gray)
    else:  # aggressive
        enhanced = apply_aggressive_enhancement(gray)
    
    # Convert back to PIL
    if is_color and should_preserve_color(original_color, enhanced):
        # If original had useful color information, convert enhanced back to color
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_color)
    else:
        return Image.fromarray(enhanced)

def apply_light_enhancement(gray_image):
    """Light enhancement - minimal processing for already good quality images."""
    # Slight contrast enhancement
    enhanced = cv2.convertScaleAbs(gray_image, alpha=1.1, beta=10)
    
    # Gentle noise reduction
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    return enhanced

def apply_medium_enhancement(gray_image):
    """Medium enhancement - balanced approach for most images."""
    # Adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # Noise reduction while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Sharpen text
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1], 
                       [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Ensure proper contrast
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)
    
    return enhanced

def apply_aggressive_enhancement(gray_image):
    """Aggressive enhancement - for poor quality or scanned documents."""
    # Strong adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # Advanced denoising
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    # Strong sharpening
    kernel = np.array([[-1,-1,-1,-1,-1],
                       [-1, 2, 2, 2,-1],
                       [-1, 2, 8, 2,-1],
                       [-1, 2, 2, 2,-1],
                       [-1,-1,-1,-1,-1]]) / 8.0
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Final contrast adjustment
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=0)
    
    return enhanced

def should_preserve_color(original_color, enhanced_gray):
    """Determines if the original image had meaningful color information."""
    # Calculate color variance
    color_variance = np.var(original_color, axis=2).mean()
    return color_variance > 100  # Threshold for meaningful color

def analyze_image_quality(pil_image):
    """Analyzes image quality and suggests optimal processing."""
    img_array = np.array(pil_image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate quality metrics
    metrics = {
        'mean_brightness': np.mean(gray),
        'contrast_std': np.std(gray),
        'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'noise_level': estimate_noise_level(gray),
        'resolution': img_array.shape[:2]
    }
    
    # Suggest enhancement level
    if metrics['sharpness'] > 500 and metrics['noise_level'] < 10:
        suggested_enhancement = "light"
    elif metrics['sharpness'] > 100 and metrics['noise_level'] < 50:
        suggested_enhancement = "medium"
    else:
        suggested_enhancement = "aggressive"
    
    return metrics, suggested_enhancement

def estimate_noise_level(gray_image):
    """Estimates noise level in the image."""
    # Use Laplacian variance as noise estimate
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def estimate_memory_usage(width, height, color_channels, num_pages):
    """Estimate memory usage for high DPI conversions."""
    bytes_per_pixel = color_channels
    total_pixels = width * height * num_pages
    estimated_bytes = total_pixels * bytes_per_pixel
    estimated_mb = estimated_bytes / (1024 * 1024)
    return estimated_mb

def convert_pdf_to_pil_images(pdf_file, start_page, end_page, dpi=300, color_space="rgb"):
    """
    Enhanced PDF to PIL conversion with better quality control and high DPI support.
    
    Args:
        pdf_file: Uploaded PDF file
        start_page: Starting page number
        end_page: Ending page number  
        dpi: Resolution (up to 2400 supported)
        color_space: "rgb", "gray", or "auto"
    
    Returns:
        List of PIL Image objects
    """
    pil_images = []
    quality_reports = []
    
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        total_pages = len(pdf_document)
        
        if start_page > end_page or start_page < 1 or end_page > total_pages:
            st.error(f"Error: Invalid page range. The PDF has {total_pages} pages.")
            return [], []

        # Memory usage warning for very high DPI
        if dpi > 1200:
            sample_page = pdf_document.load_page(0)
            page_rect = sample_page.rect
            zoom = dpi / 72.0
            estimated_width = int(page_rect.width * zoom)
            estimated_height = int(page_rect.height * zoom)
            channels = 3 if color_space == "rgb" else 1
            estimated_mb = estimate_memory_usage(estimated_width, estimated_height, channels, end_page - start_page + 1)
            
            if estimated_mb > 500:  # More than 500MB
                st.warning(f"⚠️ High DPI Warning: Estimated memory usage is {estimated_mb:.0f}MB. This may cause performance issues or memory errors.")
                if estimated_mb > 2000:  # More than 2GB
                    st.error(f"🚨 DPI too high: Estimated memory usage is {estimated_mb:.0f}MB. Consider reducing DPI or processing fewer pages.")
                    return [], []

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, page_num in enumerate(range(start_page - 1, end_page)):
            progress = (i + 1) / (end_page - start_page + 1)
            progress_bar.progress(progress)
            status_text.text(f"Processing page {page_num + 1} at {dpi} DPI...")
            
            page = pdf_document.load_page(page_num)
            
            # Higher quality matrix calculation
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Get pixmap with better quality settings
            try:
                if color_space == "gray":
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                elif color_space == "rgb":
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                else:  # auto
                    # Analyze page content to determine best color space
                    test_pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                    if has_meaningful_color(test_pix):
                        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    else:
                        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Analyze quality
                metrics, suggested_enhancement = analyze_image_quality(pil_image)
                quality_reports.append({
                    'page': page_num + 1,
                    'metrics': metrics,
                    'suggested_enhancement': suggested_enhancement
                })
                
                pil_images.append(pil_image)
                
            except Exception as page_error:
                st.error(f"Error processing page {page_num + 1}: {page_error}")
                if dpi > 1800:
                    st.warning(f"Try reducing DPI if you encounter memory issues with very high resolutions.")
                continue

        pdf_document.close()
        progress_bar.empty()
        status_text.empty()
        
        return pil_images, quality_reports
        
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        return [], []

def has_meaningful_color(pixmap):
    """Check if pixmap has meaningful color information."""
    # Simple heuristic: if it's already grayscale or low color variance
    return pixmap.colorspace.n > 1

def rotate_image(image, angle):
    """Rotate a PIL image by the specified angle with better quality."""
    return image.rotate(angle, expand=True, fillcolor='white', resample=Image.BICUBIC)

# --- Streamlit App UI ---
st.title("Step 1: 📂 High-Quality PDF to Image Converter")
st.markdown("Upload a PDF to convert its pages into optimized images for table extraction.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            total_pages = len(doc)

        st.info(f"PDF uploaded successfully! It has **{total_pages}** pages.")

        st.header("⚙️ Conversion Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            start_page_input = st.number_input("Start Page", 1, total_pages, 1)
            end_page_input = st.number_input("End Page", 1, total_pages, min(5, total_pages))
        
        with col2:
            # Enhanced DPI selection with warnings
            dpi_options = [150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000, 2400]
            dpi_input = st.selectbox(
                "Image Quality (DPI)", 
                options=dpi_options,
                index=3,  # Default to 400 DPI
                help="Higher DPI = better quality but larger files and more memory usage. 400-600 recommended for most uses. 1200+ for extreme detail."
            )
            
            # DPI guidance
            if dpi_input <= 400:
                st.success("✅ Good balance of quality and performance")
            elif dpi_input <= 800:
                st.info("ℹ️ High quality, larger file sizes")
            elif dpi_input <= 1600:
                st.warning("⚠️ Very high quality, significant memory usage")
            else:
                st.error("🚨 Extreme quality, may cause memory issues")
            
            color_space = st.selectbox(
                "Color Space",
                ["auto", "rgb", "gray"],
                help="Auto detects optimal color space. RGB preserves colors, Gray reduces file size."
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            enable_enhancement = st.checkbox("Enable Image Enhancement", value=True, 
                                           help="Applies AI-powered image enhancement for better OCR")
            if enable_enhancement:
                enhancement_level = st.selectbox(
                    "Enhancement Level",
                    ["light", "medium", "aggressive"],
                    index=1,
                    help="Light: minimal processing, Medium: balanced, Aggressive: maximum enhancement"
                )
            
            # Memory usage estimation
            if dpi_input > 600:
                num_pages = end_page_input - start_page_input + 1
                estimated_mb = estimate_memory_usage(8000, 11000, 3 if color_space == "rgb" else 1, num_pages)  # Rough estimate
                st.info(f"📊 Estimated memory usage: ~{estimated_mb:.0f}MB for {num_pages} page(s)")

        if st.button("🚀 Convert to High-Quality Images", type="primary"):
            with st.spinner(f"Converting pages to {dpi_input} DPI images..."):
                pil_images, quality_reports = convert_pdf_to_pil_images(
                    io.BytesIO(pdf_bytes), start_page_input, end_page_input, dpi_input, color_space
                )
                
                if pil_images:
                    # Apply enhancement if enabled
                    if enable_enhancement:
                        st.info("🔧 Applying image enhancement...")
                        enhanced_images = []
                        enhancement_progress = st.progress(0)
                        for idx, img in enumerate(pil_images):
                            enhancement_progress.progress((idx + 1) / len(pil_images))
                            enhanced_img = enhance_image_quality(img, enhancement_level)
                            enhanced_images.append(enhanced_img)
                        enhancement_progress.empty()
                        pil_images = enhanced_images
                    
                    # Store in session state
                    st.session_state.converted_pil_images = pil_images
                    st.session_state.image_rotations = [0] * len(pil_images)
                    st.session_state.quality_reports = quality_reports
                    
                    # Clear old data
                    keys_to_clear = [
                        'extracted_df', 'corrected_df', 'validation_results', 
                        'confidence', 'reasoning', 'original_df', 'corrections',
                        'base64_image_to_validate', 'corrections_applied'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.success(f"✅ High-quality conversion complete at {dpi_input} DPI!")
                    
                    # Display quality summary
                    if quality_reports:
                        st.subheader("📊 Quality Analysis")
                        avg_sharpness = np.mean([r['metrics']['sharpness'] for r in quality_reports])
                        avg_noise = np.mean([r['metrics']['noise_level'] for r in quality_reports])
                        avg_resolution = np.mean([r['metrics']['resolution'][0] * r['metrics']['resolution'][1] for r in quality_reports])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average Sharpness", f"{avg_sharpness:.0f}")
                        with col2:
                            st.metric("Average Noise Level", f"{avg_noise:.1f}")
                        with col3:
                            st.metric("Avg Resolution", f"{avg_resolution/1000000:.1f}MP")
                        with col4:
                            enhancement_count = sum(1 for r in quality_reports if r['suggested_enhancement'] != 'light')
                            st.metric("Pages Needing Enhancement", f"{enhancement_count}/{len(quality_reports)}")

    except Exception as e:
        st.error(f"Failed to read the PDF file. Error: {e}")

# Display images with enhanced controls
if st.session_state.get('converted_pil_images'):
    st.header("🖼️ Converted Images")
    st.info("✅ High-quality images ready! You can rotate and fine-tune individual images, then proceed to table extraction.")
    
    # Initialize rotation angles if not present
    if 'image_rotations' not in st.session_state:
        st.session_state.image_rotations = [0] * len(st.session_state.converted_pil_images)
    
    # Display images with enhanced controls
    for i, img in enumerate(st.session_state.converted_pil_images):
        st.subheader(f"📄 Page {i + 1}")
        
        # Quality info if available
        if 'quality_reports' in st.session_state and i < len(st.session_state.quality_reports):
            report = st.session_state.quality_reports[i]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sharpness", f"{report['metrics']['sharpness']:.0f}")
            with col2:
                st.metric("Brightness", f"{report['metrics']['mean_brightness']:.0f}")
            with col3:
                resolution = report['metrics']['resolution']
                megapixels = (resolution[0] * resolution[1]) / 1000000
                st.metric("Resolution", f"{megapixels:.1f}MP")
            with col4:
                st.info(f"Suggested: {report['suggested_enhancement']} enhancement")
        
        # Rotation and enhancement controls
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
        
        with col1:
            if st.button(f"↺ 90° CCW", key=f"ccw_{i}"):
                st.session_state.image_rotations[i] = (st.session_state.image_rotations[i] - 90) % 360
                st.rerun()
        
        with col2:
            if st.button(f"↻ 90° CW", key=f"cw_{i}"):
                st.session_state.image_rotations[i] = (st.session_state.image_rotations[i] + 90) % 360
                st.rerun()
        
        with col3:
            if st.button(f"↻ 180°", key=f"flip_{i}"):
                st.session_state.image_rotations[i] = (st.session_state.image_rotations[i] + 180) % 360
                st.rerun()
        
        with col4:
            if st.button(f"🔧 Enhance", key=f"enhance_{i}"):
                enhanced_img = enhance_image_quality(img, "medium")
                st.session_state.converted_pil_images[i] = enhanced_img
                st.rerun()
        
        with col5:
            current_rotation = st.session_state.image_rotations[i]
            if current_rotation != 0:
                st.info(f"Rotation: {current_rotation}°")
        
        # Apply rotation and display
        rotated_img = rotate_image(img, st.session_state.image_rotations[i])
        st.image(rotated_img, caption=f"Page {i + 1} - Optimized for Table Extraction", use_container_width=True)
        
        # Update session state
        st.session_state.converted_pil_images[i] = rotated_img
        
        # Image stats
        with st.expander(f"📊 Image Details - Page {i + 1}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Resolution:** {rotated_img.size[0]} × {rotated_img.size[1]}")
            with col2:
                st.write(f"**Mode:** {rotated_img.mode}")
            with col3:
                # Estimate file size
                img_bytes = io.BytesIO()
                rotated_img.save(img_bytes, format='PNG')
                size_mb = len(img_bytes.getvalue()) / (1024 * 1024)
                st.write(f"**Size:** {size_mb:.1f} MB")
        
        st.divider()