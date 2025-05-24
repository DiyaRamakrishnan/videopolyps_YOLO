import os
import cv2
import numpy as np
import streamlit as st
import torch
from tensorflow.keras.models import load_model
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model = load_model(os.path.join(script_dir, 'models', 'model_1.h5'))
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
yolo_model.conf = 0.25  
yolo_model.iou = 0.45   

IMG_LENGTH = 50
IMG_WIDTH = 50
YOLO_IMG_SIZE = 640 

CLASS_COLORS = {
    'adenomatous': (0, 255, 0),  
    'hyperplastic': (255, 165, 0)  
}

def process_cnn_classification(img):
    """First stage: CNN classification"""
    resized_img = cv2.resize(img, (IMG_LENGTH, IMG_WIDTH))
    input_data = np.array([resized_img], dtype=np.float32) / 255.0
    prediction = cnn_model.predict(input_data)
    has_polyp = prediction[0][0] > 0.5
    return has_polyp, prediction[0][0]

def enhance_image(image):
    """Apply image enhancement techniques"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl,a,b))
    
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def preprocess_for_yolo(image, target_size=YOLO_IMG_SIZE):
    """Preprocess image for YOLO model"""
    h, w = image.shape[:2]
    scale = min(target_size/w, target_size/h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return square_img, (scale, x_offset, y_offset)

def process_yolo_detection(img, confidence_threshold, iou_threshold=0.45):
    """Improved YOLO detection with polyp classification"""
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
            
            bbox[0] = (bbox[0] - x_offset) / scale  
            bbox[1] = (bbox[1] - y_offset) / scale  
            bbox[2] = (bbox[2] - x_offset) / scale  
            bbox[3] = (bbox[3] - y_offset) / scale  
            
            bbox[0] = max(0, min(bbox[0], img.shape[1]))
            bbox[1] = max(0, min(bbox[1], img.shape[0]))
            bbox[2] = max(0, min(bbox[2], img.shape[1]))
            bbox[3] = max(0, min(bbox[3], img.shape[0]))
            
            conf = float(det[4])
            cls = int(det[5])
            class_name = results.names[cls] 
            
            bbox = bbox.round().int().tolist()
            detections.append({
                'bbox': bbox,
                'confidence': conf,
                'class': class_name
            })
            
            color = CLASS_COLORS[class_name]
            overlay = img_with_boxes.copy()
            cv2.rectangle(overlay, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 
                         2)
            
            cv2.addWeighted(overlay, 0.7, img_with_boxes, 0.3, 0, img_with_boxes)
            
            label = f'{class_name.capitalize()} {conf:.2f}'
            cv2.putText(img_with_boxes, 
                       label, 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       color, 
                       2)
    
    return img_with_boxes, detections, enhanced_img

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
        .polyp-adenomatous {{
            color: #008000;
        }}
        .polyp-hyperplastic {{
            color: #FFA500;
        }}
    </style>
    """
    return css

def main():
    st.set_page_config(page_title="PolypDetect", layout="wide")
    
    css = generate_css()
    st.markdown(css, unsafe_allow_html=True)

    st.title('PolypDetect')
    
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

    st.sidebar.markdown("### Classification Legend")
    st.sidebar.markdown('<p class="polyp-adenomatous">■ Adenomatous Polyp</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="polyp-hyperplastic">■ Hyperplastic Polyp</p>', unsafe_allow_html=True)

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
            
            st.markdown("### Stage 1: Initial Classification")
            st.write(f"Polyp Detected: {has_polyp}")
            st.write(f"Confidence: {polyp_probability:.2%}")

            if has_polyp:
                st.markdown("### Stage 2: YOLO Classification")
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
                        polyp_class = det['class']
                        st.markdown(
                            f"<div class='polyp-{polyp_class}'>Polyp {i}:</div>",
                            unsafe_allow_html=True
                        )
                        st.write(f"- Type: {polyp_class.capitalize()}")
                        st.write(f"- Confidence: {det['confidence']:.2%}")
                        st.write(f"- Bounding Box: {det['bbox']}")
                else:
                    st.warning("No polyps localized above the confidence threshold.")
            else:
                st.info("No polyps detected in initial classification.")

if __name__ == "__main__":
    main()
