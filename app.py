import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
from typing import Dict, Tuple

# Constants
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov"]
DEFAULT_CONFIDENCE = 0.5

# Load the YOLO model with caching
@st.cache_resource
def load_model(model_path: str = "car_defect_detection_model.pt") -> YOLO:
    """Load and cache the YOLO model"""
    return YOLO(model_path)

# Function to process image and detect defects
def detect_defects(image: np.ndarray, 
                  model: YOLO, 
                  confidence_threshold: float = DEFAULT_CONFIDENCE) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Detect defects in an image using YOLO model
    Args:
        image: Input image as numpy array
        model: Loaded YOLO model
        confidence_threshold: Confidence threshold for detection
    Returns:
        Tuple of (annotated_image, defect_counts)
    """
    results = model.predict(source=image, imgsz=640, conf=confidence_threshold)
    result = results[0]
    
    # Convert the result plot to RGB
    annotated_image = result.plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Count defects
    defect_counts = {}
    for box in result.boxes:
        class_name = model.names[int(box.cls)]
        defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
    
    return annotated_image, defect_counts

def display_results(annotated_image: np.ndarray, 
                   defect_counts: Dict[str, int], 
                   original_image: np.ndarray = None) -> None:
    """Display detection results"""
    col1, col2 = st.columns(2)
    
    if original_image is not None:
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_image, caption="Detected Defects", use_container_width=True)
    else:
        st.image(annotated_image, caption="Detected Defects", use_container_width=True)
    
    # Show defect counts
    with st.expander("Defect Analysis Details", expanded=True):
        for defect, count in defect_counts.items():
            st.markdown(f"- **{defect}**: {count}")
        st.markdown(f"**Total defects detected**: {sum(defect_counts.values())}")

def process_video(video_path: str, model: YOLO, confidence_threshold: float) -> None:
    """Process video frame by frame"""
    video_capture = cv2.VideoCapture(video_path)
    stframe = st.empty()
    stop_button = st.button("Stop Processing")
    
    while video_capture.isOpened() and not stop_button:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_image, _ = detect_defects(frame_rgb, model, confidence_threshold)
        stframe.image(result_image, channels="RGB", use_container_width=True)
    
    video_capture.release()

def webcam_detection(model: YOLO, confidence_threshold: float) -> None:
    """Process webcam feed"""
    st.warning("Note: Webcam feature works only when running locally")
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Webcam")
    
    while video_capture.isOpened() and not stop_button:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_image, _ = detect_defects(frame_rgb, model, confidence_threshold)
        stframe.image(result_image, channels="RGB", use_container_width=True)
    
    video_capture.release()

def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="Car Defect Detection",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Real-time Car Defect Detection System")
    st.markdown("Upload an image or video to detect defects in a car")
    
    # Load model
    model = load_model()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.1, 1.0, DEFAULT_CONFIDENCE, 0.01,
            help="Adjust the sensitivity of defect detection"
        )
    
    # Main content
    option = st.radio(
        "Select Input Type:", 
        ("Image", "Video", "Webcam"),
        horizontal=True
    )
    
    if option == "Image":
        uploaded_file = st.file_uploader(
            "Upload a car image", 
            type=SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if st.button("Detect Defects", type="primary"):
                with st.spinner("Processing image..."):
                    result_image, defect_counts = detect_defects(image_np, model, confidence_threshold)
                    display_results(result_image, defect_counts, image_np)
    
    elif option == "Video":
        uploaded_file = st.file_uploader(
            "Upload a car video", 
            type=SUPPORTED_VIDEO_TYPES,
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tfile:
                tfile.write(uploaded_file.read())
                temp_path = tfile.name
            
            try:
                process_video(temp_path, model, confidence_threshold)
            finally:
                os.unlink(temp_path)
    
    elif option == "Webcam":
        if st.checkbox("Start Webcam", key="webcam_toggle"):
            webcam_detection(model, confidence_threshold)

if __name__ == "__main__":
    main()