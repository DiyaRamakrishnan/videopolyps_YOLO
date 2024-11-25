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
yolo_model.conf = 0.25  # Default confidence threshold
yolo_model.iou = 0.45   # Default IoU threshold

# Constants
IMG_LENGTH = 50
IMG_WIDTH = 50
YOLO_IMG_SIZE = 640  # Standard YOLO input size

def enhance_image(image):
    """Apply image enhancement techniques"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge((cl,a,b))
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def preprocess_for_yolo(image, target_size=YOLO_IMG_SIZE):
    """Preprocess image for YOLO model"""
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size/w, target_size/h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create square image with black padding
    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    # Calculate position to paste resized image
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return square_img, (scale, x_offset, y_offset)

def process_yolo_detection(img, confidence_threshold, iou_threshold=0.45):
    """Improved YOLO detection with preprocessing and post-processing"""
    # Preprocess image
    processed_img, transform_params = preprocess_for_yolo(img)
    scale, x_offset, y_offset = transform_params
    
    # Apply image enhancement
    enhanced_img = enhance_image(processed_img)
    
    # Run inference with adjusted settings
    yolo_model.conf = confidence_threshold
    yolo_model.iou = iou_threshold
    results = yolo_model(enhanced_img)
    
    # Get the original image for drawing
    img_with_boxes = img.copy()
    
    # Process predictions
    detections = []
    if len(results.pred[0]) > 0:
        # Apply NMS (Non-Maximum Suppression)
        pred = results.pred[0]
        
        # Convert boxes back to original image coordinates
        for det in pred:
            bbox = det[:4].clone()
            
            # Adjust coordinates back to original image space
            bbox[0] = (bbox[0] - x_offset) / scale  # x1
            bbox[1] = (bbox[1] - y_offset) / scale  # y1
            bbox[2] = (bbox[2] - x_offset) / scale  # x2
            bbox[3] = (bbox[3] - y_offset) / scale  # y2
            
            # Ensure coordinates are within image bounds
            bbox[0] = max(0, min(bbox[0], img.shape[1]))
            bbox[1] = max(0, min(bbox[1], img.shape[0]))
            bbox[2] = max(0, min(bbox[2], img.shape[1]))
            bbox[3] = max(0, min(bbox[3], img.shape[0]))
            
            conf = float(det[4])
            cls = int(det[5])
            
            bbox = bbox.round().int().tolist()
            detections.append({
                'bbox': bbox,
                'confidence': conf,
                'class': results.names[cls]
            })
            
            # Draw rectangle with semi-transparency
            overlay = img_with_boxes.copy()
            cv2.rectangle(overlay, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         (0, 255, 0), 
                         2)
            
            # Add semi-transparent overlay
            cv2.addWeighted(overlay, 0.7, img_with_boxes, 0.3, 0, img_with_boxes)
            
            # Add label with confidence
            label = f'Polyp {conf:.2f}'
            cv2.putText(img_with_boxes, 
                        label, 
                        (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2)
    
    return img_with_boxes, detections, enhanced_img

def main():
    st.set_page_config(page_title="Enhanced Two-Stage Polyp Detection", layout="wide")
    
    # [Previous CSS code remains the same]

    st.title('Enhanced Two-Stage Polyp Detection System')
    
    # Enhanced sidebar controls
    st.sidebar.title("Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "YOLO Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05
    )
    
    show_preprocessing = st.sidebar.checkbox("Show Preprocessing Steps", value=False)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read and process image
        img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if st.button('Analyze Image'):
            # Stage 1: CNN Classification
            has_polyp, polyp_probability = process_cnn_classification(img)
            
            st.markdown("### Stage 1: Initial Classification")
            st.write(f"Polyp Detected: {has_polyp}")
            st.write(f"Confidence: {polyp_probability:.2%}")

            # If polyp detected, proceed to Stage 2
            if has_polyp:
                st.markdown("### Stage 2: Enhanced Polyp Localization")
                result_img, detections, enhanced_img = process_yolo_detection(
                    img_rgb, 
                    confidence_threshold,
                    iou_threshold
                )
                
                # Display results
                if show_preprocessing:
                    cols = st.columns(3)
                    with cols[0]:
                        st.image(img_rgb, caption='Original Image')
                    with cols[1]:
                        st.image(enhanced_img, caption='Preprocessed Image')
                    with cols[2]:
                        st.image(result_img, caption='Detection Result')
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_rgb, caption='Original Image')
                    with col2:
                        st.image(result_img, caption='Detection Result')
                
                # Display detection details
                if detections:
                    st.markdown("#### Detection Details:")
                    for i, det in enumerate(detections, 1):
                        st.write(f"Polyp {i}:")
                        st.write(f"- Confidence: {det['confidence']:.2%}")
                        st.write(f"- Bounding Box: {det['bbox']}")
                else:
                    st.warning("No polyps localized above the confidence threshold.")
            else:
                st.info("No polyps detected in initial classification.")

if __name__ == "__main__":
    main()