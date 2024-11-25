import os
import cv2
import numpy as np
import streamlit as st
import torch
from tensorflow.keras.models import load_model
from pathlib import Path

# Load models
script_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model = load_model(os.path.join(script_dir, 'models', 'model_1.h5'))
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
# Add new classification model
polyp_classifier = load_model(os.path.join(script_dir, 'models', 'polyp_classifier.h5'))

# Constants
IMG_LENGTH = 50
IMG_WIDTH = 50
YOLO_IMG_SIZE = 640
CLASSIFIER_SIZE = 224  # Standard size for classification models

def classify_polyp(img, bbox):
    """Classify polyp type from detected region"""
    # Extract polyp region using bbox
    x1, y1, x2, y2 = bbox
    polyp_region = img[y1:y2, x1:x2]
    
    # Resize for classifier
    resized_region = cv2.resize(polyp_region, (CLASSIFIER_SIZE, CLASSIFIER_SIZE))
    
    # Normalize
    input_data = np.array([resized_region], dtype=np.float32) / 255.0
    
    # Get prediction
    prediction = polyp_classifier.predict(input_data)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    
    # Map index to class name
    class_names = ['Hyperplastic', 'Adenomatous']
    return class_names[class_idx], confidence

def process_yolo_detection(img, confidence_threshold, iou_threshold=0.45):
    """Enhanced YOLO detection with polyp classification"""
    processed_img, transform_params = preprocess_for_yolo(img)
    scale, x_offset, y_offset = transform_params
    
    enhanced_img = enhance_image(processed_img)
    
    yolo_model.conf = confidence_threshold
    yolo_model.iou = iou_threshold
    results = yolo_model(enhanced_img)
    
    img_with_boxes = img.copy()
    
    detections = []
    if len(results.pred[0]) > 0:
        pred = results.pred[0]
        
        for det in pred:
            bbox = det[:4].clone()
            
            # Adjust coordinates back to original image space
            bbox[0] = (bbox[0] - x_offset) / scale
            bbox[1] = (bbox[1] - y_offset) / scale
            bbox[2] = (bbox[2] - x_offset) / scale
            bbox[3] = (bbox[3] - y_offset) / scale
            
            bbox = torch.clamp(bbox, min=0)
            bbox[0] = min(bbox[0], img.shape[1])
            bbox[1] = min(bbox[1], img.shape[0])
            bbox[2] = min(bbox[2], img.shape[1])
            bbox[3] = min(bbox[3], img.shape[0])
            
            conf = float(det[4])
            bbox = bbox.round().int().tolist()
            
            # Classify polyp type
            polyp_type, type_conf = classify_polyp(img, bbox)
            
            detections.append({
                'bbox': bbox,
                'confidence': conf,
                'type': polyp_type,
                'type_confidence': float(type_conf)
            })
            
            # Draw rectangle with color based on polyp type
            color = (0, 255, 0) if polyp_type == 'Hyperplastic' else (255, 0, 0)
            overlay = img_with_boxes.copy()
            cv2.rectangle(overlay, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 
                         2)
            
            cv2.addWeighted(overlay, 0.7, img_with_boxes, 0.3, 0, img_with_boxes)
            
            # Add label with type and confidence
            label = f'{polyp_type} {conf:.2f}'
            cv2.putText(img_with_boxes, 
                       label, 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       color, 
                       2)
    
    return img_with_boxes, detections, enhanced_img

def main():
    st.set_page_config(page_title="Enhanced Polyp Detection and Classification", layout="wide")
    
    css = generate_css()
    st.markdown(css, unsafe_allow_html=True)

    st.title('Enhanced Polyp Detection and Classification System')
    
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

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if st.button('Analyze Image'):
            has_polyp, polyp_probability = process_cnn_classification(img)
            
            st.markdown("### Stage 1: Initial Detection")
            st.write(f"Polyp Detected: {has_polyp}")
            st.write(f"Confidence: {polyp_probability:.2%}")

            if has_polyp:
                st.markdown("### Stage 2: Polyp Localization and Classification")
                result_img, detections, enhanced_img = process_yolo_detection(
                    img_rgb, 
                    confidence_threshold,
                    iou_threshold
                )
                
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
                
                if detections:
                    st.markdown("#### Detection Details:")
                    for i, det in enumerate(detections, 1):
                        st.write(f"Polyp {i}:")
                        st.write(f"- Type: {det['type']}")
                        st.write(f"- Detection Confidence: {det['confidence']:.2%}")
                        st.write(f"- Classification Confidence: {det['type_confidence']:.2%}")
                        st.write(f"- Bounding Box: {det['bbox']}")
                else:
                    st.warning("No polyps localized above the confidence threshold.")
            else:
                st.info("No polyps detected in initial classification.")

if __name__ == "__main__":
    main()