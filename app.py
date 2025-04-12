import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
from typing import Dict, Tuple
import sys
import asyncio

# =============================================
# CRITICAL FIXES FOR TORCH.CLASSES AND ASYNCIO
# =============================================
sys.modules['torch._classes'] = None  # Disables torch.classes inspection
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =============================================
# CONSTANTS AND CONFIGURATION
# =============================================
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov"]
DEFAULT_CONFIDENCE = 0.5
MODEL_PATH = "car_defect_detection_model.pt"

# =============================================
# SESSION STATE MANAGEMENT
# =============================================
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'stop_processing' not in st.session_state:
    st.session_state.stop_processing = False

# =============================================
# MODEL LOADING WITH ERROR HANDLING
# =============================================
@st.cache_resource
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """Load and cache the YOLO model with error handling"""
    try:
        import torch
        torch._C._disable_torch_class_checks = True  # Additional protection
        
        model = YOLO(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# =============================================
# DEFECT DETECTION FUNCTIONS
# =============================================
def detect_defects(image: np.ndarray, 
                 model: YOLO, 
                 confidence_threshold: float = DEFAULT_CONFIDENCE) -> Tuple[np.ndarray, Dict[str, int]]:
    """Detect defects with comprehensive error handling"""
    try:
        # Run prediction
        results = model.predict(
            source=image,
            imgsz=640,
            conf=confidence_threshold,
            stream=False  # Disable streaming for more stable behavior
        )
        
        # Process results
        result = results[0]
        annotated_image = result.plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Count defects
        defect_counts = {}
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        
        return annotated_image, defect_counts
        
    except Exception as e:
        st.error(f"⚠️ Detection error: {str(e)}")
        return image, {}  # Return original image if detection fails

# =============================================
# VIDEO PROCESSING FUNCTIONS
# =============================================
def process_video(video_path: str, model: YOLO, confidence_threshold: float) -> None:
    """Process video with proper resource cleanup"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("🚨 Could not open video file")
            return
            
        stframe = st.empty()
        st.session_state.stop_processing = False
        
        while cap.isOpened() and not st.session_state.stop_processing:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_image, _ = detect_defects(frame_rgb, model, confidence_threshold)
            stframe.image(result_image, channels="RGB", use_container_width=True)
            
    except Exception as e:
        st.error(f"🎥 Video processing error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

def webcam_detection(model: YOLO, confidence_threshold: float) -> None:
    """Webcam processing with proper state management"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("📷 Could not access webcam")
            st.session_state.webcam_active = False
            return
            
        stframe = st.empty()
        
        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not read webcam frame")
                break
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_image, _ = detect_defects(frame_rgb, model, confidence_threshold)
            stframe.image(result_image, channels="RGB", use_container_width=True)
            
    except Exception as e:
        st.error(f"🎥 Webcam error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        st.session_state.webcam_active = False

# =============================================
# STREAMLIT UI COMPONENTS
# =============================================
def display_results(annotated_image: np.ndarray, 
                   defect_counts: Dict[str, int], 
                   original_image: np.ndarray = None) -> None:
    """Display results in a clean layout"""
    if original_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_image, caption="Detected Defects", use_container_width=True)
    else:
        st.image(annotated_image, caption="Detected Defects", use_container_width=True)
    
    # Show defect counts in expandable section
    with st.expander("🔍 Defect Analysis Details", expanded=True):
        if defect_counts:
            for defect, count in defect_counts.items():
                st.markdown(f"- **{defect.capitalize()}**: {count}")
            st.markdown(f"**🔢 Total defects detected**: {sum(defect_counts.values())}")
        else:
            st.warning("No defects detected or analysis failed")

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application entry point"""
    # Configure page
    st.set_page_config(
        page_title="Car Defect Detection",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load model first
    try:
        model = load_model()
    except Exception as e:
        st.error(f"🔥 Critical error: {str(e)}")
        return
    
    # UI Header
    st.title("🚗 Real-time Car Defect Detection System")
    st.markdown("""
    Upload images or videos to detect car defects using advanced computer vision.
    Adjust the confidence threshold in the sidebar to control detection sensitivity.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.1, 1.0, DEFAULT_CONFIDENCE, 0.01,
            help="Higher values reduce false positives but may miss some defects"
        )
        
        st.markdown("---")
        st.markdown("**ℹ️ About**")
        st.markdown("""
        This system uses YOLOv8 to detect various car defects.
        For best results, use clear, well-lit images/videos.
        """)
    
    # Main content area
    option = st.radio(
        "Select Input Type:",
        ("Image", "Video", "Webcam"),
        horizontal=True,
        index=0
    )
    
    # Image processing
    if option == "Image":
        uploaded_file = st.file_uploader(
            "Upload car image",
            type=SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=False,
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                if st.button("🔍 Detect Defects", type="primary"):
                    with st.spinner("🔄 Processing image..."):
                        result_image, defect_counts = detect_defects(image_np, model, confidence_threshold)
                        display_results(result_image, defect_counts, image_np)
            except Exception as e:
                st.error(f"📷 Image processing error: {str(e)}")
    
    # Video processing
    elif option == "Video":
        uploaded_file = st.file_uploader(
            "Upload car video",
            type=SUPPORTED_VIDEO_TYPES,
            accept_multiple_files=False,
            help="Supported formats: MP4, AVI, MOV"
        )
        
        if uploaded_file is not None:
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name
                
                # Add stop button
                if st.button("🛑 Stop Processing"):
                    st.session_state.stop_processing = True
                
                # Process video
                process_video(temp_path, model, confidence_threshold)
                
            except Exception as e:
                st.error(f"🎥 Video processing error: {str(e)}")
            finally:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    # Webcam processing
    elif option == "Webcam":
        webcam_toggle = st.checkbox(
            "🎥 Start Webcam",
            value=st.session_state.webcam_active,
            help="Requires camera access permission"
        )
        
        if webcam_toggle:
            st.session_state.webcam_active = True
            webcam_detection(model, confidence_threshold)
        else:
            st.session_state.webcam_active = False
            st.info("Webcam is currently off")

# =============================================
# APPLICATION ENTRY POINT
# =============================================
if __name__ == "__main__":
    main()