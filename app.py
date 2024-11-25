import os
import cv2
import numpy as np
import streamlit as st
import torch
from tensorflow.keras.models import load_model
from pathlib import Path

# Load both models
script_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model = load_model(os.path.join(script_dir, 'models', 'model_1.h5'))
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Constants
IMG_LENGTH = 50
IMG_WIDTH = 50

def generate_css(primary_color="#4786a5", secondary_background_color="#f0f2f6"):
    css = f"""
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #ffffff;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
            justify-content: center;
        }}
        .input-side, .output-side {{
            width: 80%;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .input-side {{
            background-color: {secondary_background_color};
        }}
        .output-side {{
            background-color: #fff;
        }}
        .title {{
            font-size: 2rem;
            color: {primary_color};
            margin-bottom: 10px;
        }}
        .stage-result {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            background-color: #f8f9fa;
        }}
    </style>
    """
    return css

def process_cnn_classification(img):
    """First stage: CNN classification"""
    resized_img = cv2.resize(img, (IMG_LENGTH, IMG_WIDTH))
    input_data = np.array([resized_img], dtype=np.float32) / 255.0
    prediction = cnn_model.predict(input_data)
    has_polyp = prediction[0][0] > 0.5
    return has_polyp, prediction[0][0]

def process_yolo_detection(img, confidence_threshold):
    """Second stage: YOLO detection"""
    results = yolo_model(img)
    
    # Get the original image
    img_with_boxes = img.copy()
    
    # Get predictions
    pred = results.pred[0]
    pred = pred[pred[:, 4] >= confidence_threshold]
    
    # Draw boxes on the image
    detections = []
    for det in pred:
        bbox = det[:4].round().int().tolist()
        conf = float(det[4])
        cls = int(det[5])
        label = f'{results.names[cls]} {conf:.2f}'
        
        detections.append({
            'bbox': bbox,
            'confidence': conf,
            'class': results.names[cls]
        })
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, 
                     (bbox[0], bbox[1]), 
                     (bbox[2], bbox[3]), 
                     (0, 255, 0), 
                     2)
        
        # Add label
        cv2.putText(img_with_boxes, 
                    label, 
                    (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2)
    
    return img_with_boxes, detections

def main():
    st.set_page_config(page_title="Two-Stage Polyp Detection", layout="wide")
    
    css = generate_css()
    st.markdown(css, unsafe_allow_html=True)

    st.title('Two-Stage Polyp Detection System')
    st.write("""
    This system uses a two-stage approach for polyp detection:
    1. First, a CNN model determines if a polyp is present in the image
    2. If a polyp is detected, a YOLO model then localizes the polyp(s) in the image
    """)
    
    # Sidebar controls
    st.sidebar.title("Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "YOLO Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    # File uploader
    st.markdown('<div class="input-side">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Read and process image
        img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if st.button('Analyze Image'):
            st.markdown('<div class="output-side">', unsafe_allow_html=True)
            
            # Stage 1: CNN Classification
            has_polyp, polyp_probability = process_cnn_classification(img)
            
            st.markdown("### Stage 1: Initial Classification")
            st.markdown('<div class="stage-result">', unsafe_allow_html=True)
            st.write(f"Polyp Detected: {has_polyp}")
            st.write(f"Confidence: {polyp_probability:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

            # If polyp detected, proceed to Stage 2
            if has_polyp:
                st.markdown("### Stage 2: Polyp Localization")
                result_img, detections = process_yolo_detection(img_rgb, confidence_threshold)
                
                # Display results side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption='Original Image', use_container_width=True)
                with col2:
                    st.image(result_img, caption='Detection Result', use_container_width=True)
                
                # Display detection details
                if detections:
                    st.markdown('<div class="stage-result">', unsafe_allow_html=True)
                    st.write("Detected Polyps:")
                    for i, det in enumerate(detections, 1):
                        st.write(f"Polyp {i}:")
                        st.write(f"- Confidence: {det['confidence']:.2%}")
                        st.write(f"- Location: {det['bbox']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No polyps localized by YOLO above the confidence threshold.")
            else:
                st.info("No polyps detected in initial classification. Skipping localization stage.")
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()