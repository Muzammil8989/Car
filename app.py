import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import subprocess
import sys

# =============================================
# SYSTEM DEPENDENCIES FIX FOR STREAMLIT CLOUD
# =============================================
try:
    import cv2
except ImportError:
    # Install required system dependencies
    subprocess.run(['apt-get', 'update'], check=True)
    subprocess.run(['apt-get', 'install', '-y', 'libgl1'], check=True)
    import cv2

# Disable problematic torch inspections
sys.modules['torch._classes'] = None

# =============================================
# CONSTANTS AND CONFIGURATION
# =============================================
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov"]
DEFAULT_CONFIDENCE = 0.5
MODEL_PATH = "car_defect_detection_model.pt"

# =============================================
# MODEL LOADING WITH CACHING
# =============================================
@st.cache_resource
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """Load and cache the YOLO model with error handling"""
    try:
        model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# =============================================
# DEFECT DETECTION FUNCTION
# =============================================
def detect_defects(image: np.ndarray, model: YOLO, confidence_threshold: float) -> tuple:
    """Detect defects in an image and return annotated image and defect counts"""
    try:
        results = model.predict(
            source=image,
            imgsz=640,
            conf=confidence_threshold,
            stream=False  # Disable streaming for more stable behavior
        )
        
        result = results[0]
        annotated_image = result.plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        defect_counts = {}
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        
        return annotated_image, defect_counts
    except Exception as e:
        st.error(f"⚠️ Detection error: {str(e)}")
        return image, {}

# =============================================
# VIDEO PROCESSING FUNCTION
# =============================================
def process_video(video_path: str, model: YOLO, confidence_threshold: float) -> None:
    """Process video file frame by frame"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("🚨 Could not open video file")
            return
            
        stframe = st.empty()
        stop_button = st.button("🛑 Stop Video Processing")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_image, _ = detect_defects(frame_rgb, model, confidence_threshold)
            stframe.image(result_image, channels="RGB", use_container_width=True)
        
        cap.release()
    except Exception as e:
        st.error(f"🎥 Video processing error: {str(e)}")

# =============================================
# RESULT DISPLAY FUNCTION
# =============================================
def display_results(original_image: np.ndarray, 
                   result_image: np.ndarray, 
                   defect_counts: dict) -> None:
    """Display original and annotated images with defect analysis"""
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(result_image, caption="Detected Defects", use_container_width=True)
    
    with st.expander("🔍 Detailed Defect Analysis", expanded=True):
        if defect_counts:
            st.subheader("Defect Counts:")
            for defect, count in defect_counts.items():
                st.markdown(f"- **{defect.capitalize()}**: {count}")
            st.markdown(f"**🔢 Total defects detected**: {sum(defect_counts.values())}")
        else:
            st.warning("No defects detected or analysis failed")

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    # Configure page
    st.set_page_config(
        page_title="Car Defect Detection",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Car Defect Detection System")
    st.markdown("""
    Upload images or videos to detect car defects using computer vision.
    Adjust the confidence threshold in the sidebar to control detection sensitivity.
    """)
    
    # Load model first to catch errors early
    model = load_model()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.1, 1.0, DEFAULT_CONFIDENCE, 0.01,
            help="Higher values reduce false positives but may miss subtle defects"
        )
        
        st.markdown("---")
        st.markdown("**ℹ️ About**")
        st.markdown("This system uses YOLOv8 to detect various car defects.")
    
    # Main interface
    option = st.radio(
        "Select Input Type:",
        ("Image", "Video"),
        horizontal=True
    )
    
    # Image processing
    if option == "Image":
        uploaded_file = st.file_uploader(
            "Upload car image",
            type=SUPPORTED_IMAGE_TYPES,
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                if st.button("🔍 Detect Defects", type="primary"):
                    with st.spinner("🔄 Processing image..."):
                        result_image, defect_counts = detect_defects(
                            image_np, 
                            model, 
                            confidence_threshold
                        )
                        display_results(image_np, result_image, defect_counts)
            except Exception as e:
                st.error(f"📷 Image processing error: {str(e)}")
    
    # Video processing
    elif option == "Video":
        uploaded_file = st.file_uploader(
            "Upload car video",
            type=SUPPORTED_VIDEO_TYPES,
            help="Supported formats: MP4, AVI, MOV"
        )
        
        if uploaded_file is not None:
            try:
                # Save to temp file with correct extension
                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name
                
                # Process video
                process_video(temp_path, model, confidence_threshold)
                
            except Exception as e:
                st.error(f"🎥 Video processing error: {str(e)}")
            finally:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)

if __name__ == "__main__":
    main()



#     import streamlit as st
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import tempfile
# import os
# import subprocess
# import sys
# import cv2

# # =============================================
# # SYSTEM DEPENDENCIES FIX
# # =============================================
# try:
#     import cv2
# except ImportError:
#     # Install required system dependencies
#     subprocess.run(['apt-get', 'update'], check=True)
#     subprocess.run(['apt-get', 'install', '-y', 'libgl1'], check=True)
#     import cv2

# # Disable problematic torch inspections
# sys.modules['torch._classes'] = None

# # =============================================
# # CONSTANTS AND CONFIGURATION
# # =============================================
# SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
# SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov"]
# DEFAULT_CONFIDENCE = 0.5
# MODEL_PATH = "car_defect_detection_model.pt"

# # Initialize session state for webcam
# if 'webcam_active' not in st.session_state:
#     st.session_state.webcam_active = False

# # =============================================
# # MODEL LOADING WITH CACHING
# # =============================================
# @st.cache_resource
# def load_model(model_path: str = MODEL_PATH) -> YOLO:
#     """Load and cache the YOLO model with error handling"""
#     try:
#         model = YOLO(model_path)
#         st.success("✅ Model loaded successfully!")
#         return model
#     except Exception as e:
#         st.error(f"❌ Model loading failed: {str(e)}")
#         st.stop()

# # =============================================
# # DEFECT DETECTION FUNCTION
# # =============================================
# def detect_defects(image: np.ndarray, model: YOLO, confidence_threshold: float) -> tuple:
#     """Detect defects in an image and return annotated image and defect counts"""
#     try:
#         results = model.predict(
#             source=image,
#             imgsz=640,
#             conf=confidence_threshold,
#             stream=False
#         )
        
#         result = results[0]
#         annotated_image = result.plot()
#         annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
#         defect_counts = {}
#         for box in result.boxes:
#             class_name = model.names[int(box.cls)]
#             defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        
#         return annotated_image, defect_counts
#     except Exception as e:
#         st.error(f"⚠️ Detection error: {str(e)}")
#         return image, {}

# # =============================================
# # VIDEO PROCESSING FUNCTION
# # =============================================
# def process_video(video_path: str, model: YOLO, confidence_threshold: float) -> None:
#     """Process video file frame by frame"""
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             st.error("🚨 Could not open video file")
#             return
            
#         stframe = st.empty()
#         stop_button = st.button("🛑 Stop Video Processing")
        
#         while cap.isOpened() and not stop_button:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result_image, _ = detect_defects(frame_rgb, model, confidence_threshold)
#             stframe.image(result_image, channels="RGB", use_container_width=True)
        
#         cap.release()
#     except Exception as e:
#         st.error(f"🎥 Video processing error: {str(e)}")

# # =============================================
# # WEBCAM PROCESSING FUNCTION
# # =============================================
# def process_webcam(model: YOLO, confidence_threshold: float) -> None:
#     """Process webcam feed in real-time"""
#     try:
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("📷 Could not access webcam")
#             st.session_state.webcam_active = False
#             return
            
#         stframe = st.empty()
#         stop_button = st.button("🛑 Stop Webcam")
        
#         while st.session_state.webcam_active and not stop_button:
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning("⚠️ Could not read webcam frame")
#                 break
                
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result_image, _ = detect_defects(frame_rgb, model, confidence_threshold)
#             stframe.image(result_image, channels="RGB", use_container_width=True)
        
#         cap.release()
#         st.session_state.webcam_active = False
#     except Exception as e:
#         st.error(f"📷 Webcam error: {str(e)}")
#         st.session_state.webcam_active = False

# # =============================================
# # RESULT DISPLAY FUNCTION
# # =============================================
# def display_results(original_image: np.ndarray, 
#                    result_image: np.ndarray, 
#                    defect_counts: dict) -> None:
#     """Display original and annotated images with defect analysis"""
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(original_image, caption="Original Image", use_container_width=True)
#     with col2:
#         st.image(result_image, caption="Detected Defects", use_container_width=True)
    
#     with st.expander("🔍 Detailed Defect Analysis", expanded=True):
#         if defect_counts:
#             st.subheader("Defect Counts:")
#             for defect, count in defect_counts.items():
#                 st.markdown(f"- **{defect.capitalize()}**: {count}")
#             st.markdown(f"**🔢 Total defects detected**: {sum(defect_counts.values())}")
#         else:
#             st.warning("No defects detected or analysis failed")

# # =============================================
# # MAIN APPLICATION
# # =============================================
# def main():
#     # Configure page
#     st.set_page_config(
#         page_title="Car Defect Detection",
#         page_icon="🚗",
#         layout="wide"
#     )
    
#     st.title("🚗 Car Defect Detection System")
#     st.markdown("""
#     Upload images/videos or use webcam to detect car defects using computer vision.
#     Adjust the confidence threshold in the sidebar to control detection sensitivity.
#     """)
    
#     # Load model first to catch errors early
#     model = load_model()
    
#     # Sidebar controls
#     with st.sidebar:
#         st.header("⚙️ Settings")
#         confidence_threshold = st.slider(
#             "Confidence Threshold",
#             0.1, 1.0, DEFAULT_CONFIDENCE, 0.01,
#             help="Higher values reduce false positives but may miss subtle defects"
#         )
        
#         st.markdown("---")
#         st.markdown("**ℹ️ About**")
#         st.markdown("This system uses YOLOv8 to detect various car defects.")
    
#     # Main interface
#     option = st.radio(
#         "Select Input Type:",
#         ("Image", "Video", "Webcam"),
#         horizontal=True
#     )
    
#     # Image processing
#     if option == "Image":
#         uploaded_file = st.file_uploader(
#             "Upload car image",
#             type=SUPPORTED_IMAGE_TYPES,
#             help="Supported formats: JPG, JPEG, PNG"
#         )
        
#         if uploaded_file is not None:
#             try:
#                 image = Image.open(uploaded_file)
#                 image_np = np.array(image)
                
#                 if st.button("🔍 Detect Defects", type="primary"):
#                     with st.spinner("🔄 Processing image..."):
#                         result_image, defect_counts = detect_defects(
#                             image_np, 
#                             model, 
#                             confidence_threshold
#                         )
#                         display_results(image_np, result_image, defect_counts)
#             except Exception as e:
#                 st.error(f"📷 Image processing error: {str(e)}")
    
#     # Video processing
#     elif option == "Video":
#         uploaded_file = st.file_uploader(
#             "Upload car video",
#             type=SUPPORTED_VIDEO_TYPES,
#             help="Supported formats: MP4, AVI, MOV"
#         )
        
#         if uploaded_file is not None:
#             try:
#                 # Save to temp file with correct extension
#                 file_ext = os.path.splitext(uploaded_file.name)[1]
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
#                     tfile.write(uploaded_file.read())
#                     temp_path = tfile.name
                
#                 # Process video
#                 process_video(temp_path, model, confidence_threshold)
                
#             except Exception as e:
#                 st.error(f"🎥 Video processing error: {str(e)}")
#             finally:
#                 if 'temp_path' in locals() and os.path.exists(temp_path):
#                     os.unlink(temp_path)
    
#     # Webcam processing
#     elif option == "Webcam":
#         if st.checkbox("🎥 Start Webcam", value=st.session_state.webcam_active,
#                       help="Requires camera access permission"):
#             st.session_state.webcam_active = True
#             process_webcam(model, confidence_threshold)
#         else:
#             st.session_state.webcam_active = False
#             st.info("Webcam is currently off. Check the box above to start.")

# if __name__ == "__main__":
#     main()