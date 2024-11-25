import os
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import torch

# Load YOLO model
script_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

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
        .button {{
            background-color: {primary_color};
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
            transition: background-color 0.3s;
        }}
        .button:hover {{
            background-color: #4786a5;
        }}
        .prediction {{
            font-size: 1.5rem;
            margin-bottom: 10px;
        }}
        .probability {{
            font-size: 1.5rem;
            margin-bottom: 20px;
        }}
        .output-image {{
            max-width: 400px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
    </style>
    """
    return css

def process_yolo_image(img, confidence_threshold):
    # Run YOLO inference with confidence threshold
    results = yolo_model(img)
    
    # Get the original image
    img_with_boxes = img.copy()
    
    # Get predictions
    pred = results.pred[0]
    pred = pred[pred[:, 4] >= confidence_threshold]
    
    # Draw boxes on the image
    for det in pred:
        bbox = det[:4].round().int().tolist()
        conf = float(det[4])
        cls = int(det[5])
        label = f'{results.names[cls]} {conf:.2f}'
        
        # Draw rectangle
        color = (0, 255, 0)  # BGR Green color
        cv2.rectangle(img_with_boxes, 
                     (bbox[0], bbox[1]), 
                     (bbox[2], bbox[3]), 
                     color, 
                     2)
        
        # Add label
        cv2.putText(img_with_boxes, 
                    label, 
                    (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2)
    
    return img_with_boxes, results

def process_video_frame(frame, confidence_threshold):
    # Process a single video frame with YOLO
    return process_yolo_image(frame, confidence_threshold)

def main():
    st.set_page_config(page_title="Object Detection App", layout="wide")
    
    css = generate_css()
    st.markdown(css, unsafe_allow_html=True)

    page = st.sidebar.selectbox("Go to", ["Object Detection", "Info", "Comments"])

    if page == "Object Detection":
        st.title('Object Detection')
        
        # Sidebar controls
        st.sidebar.title("Detection Settings")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05
        )

        # File uploader
        st.markdown('<div class="input-side">', unsafe_allow_html=True)
        st.markdown('<h2 class="title">Upload Media</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image or video...", 
            type=["jpg", "jpeg", "png", "mp4", "mov"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            st.markdown('<div class="output-side">', unsafe_allow_html=True)
            st.markdown('<h2 class="title">Detection Result</h2>', unsafe_allow_html=True)

            if uploaded_file.type.startswith('image'):
                # Process image
                img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if st.button('Detect Objects'):
                    result_img, results = process_yolo_image(img, confidence_threshold)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption='Original Image', use_column_width=True)
                    with col2:
                        st.image(result_img, caption='Detection Result', use_column_width=True)
                    
                    # Display detection information
                    st.markdown("### Detection Details")
                    # Convert detection results to DataFrame
                    if len(results.pred[0]) > 0:
                        df_data = []
                        for det in results.pred[0]:
                            if det[4] >= confidence_threshold:
                                df_data.append({
                                    'name': results.names[int(det[5])],
                                    'confidence': float(det[4]),
                                    'xmin': int(det[0]),
                                    'ymin': int(det[1]),
                                    'xmax': int(det[2]),
                                    'ymax': int(det[3])
                                })
                        if df_data:
                            import pandas as pd
                            df = pd.DataFrame(df_data)
                            st.dataframe(df)
                        else:
                            st.write("No objects detected above the confidence threshold.")
                    else:
                        st.write("No objects detected.")

            elif uploaded_file.type.startswith('video'):
                # Save uploaded video temporarily
                temp_video_path = os.path.join(script_dir, 'temp_video.mp4')
                with open(temp_video_path, 'wb') as f:
                    f.write(uploaded_file.read())

                # Video frame selection
                video = cv2.VideoCapture(temp_video_path)
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_number = st.slider("Select Frame", 0, total_frames-1, 0)
                
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video.read()
                video.release()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result_img, results = process_video_frame(frame, confidence_threshold)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(frame, caption='Original Frame', use_column_width=True)
                    with col2:
                        st.image(result_img, caption='Detection Result', use_column_width=True)
                    
                    # Display detection information
                    st.markdown("### Detection Details")
                    # Convert detection results to DataFrame
                    if len(results.pred[0]) > 0:
                        df_data = []
                        for det in results.pred[0]:
                            if det[4] >= confidence_threshold:
                                df_data.append({
                                    'name': results.names[int(det[5])],
                                    'confidence': float(det[4]),
                                    'xmin': int(det[0]),
                                    'ymin': int(det[1]),
                                    'xmax': int(det[2]),
                                    'ymax': int(det[3])
                                })
                        if df_data:
                            import pandas as pd
                            df = pd.DataFrame(df_data)
                            st.dataframe(df)
                        else:
                            st.write("No objects detected above the confidence threshold.")
                    else:
                        st.write("No objects detected.")
                
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Info":
        st.title("Information")
        st.write("""
        This application uses a YOLO (You Only Look Once) model for object detection.
        The model can detect objects in both images and videos.
        
        ### Features:
        - Upload images or videos for detection
        - Adjust confidence threshold for detections
        - View detection results with bounding boxes
        - Get detailed information about detected objects
        
        ### How to use:
        1. Select 'Object Detection' from the sidebar
        2. Upload an image or video file
        3. Adjust the confidence threshold if needed
        4. Click 'Detect Objects' for images or select a frame for videos
        5. View the results and detection details
        """)

    elif page == "Comments":
        st.title('Comments')
        st.write("Leave your comments and feedback below:")

        user_name = st.text_input("Your Name", max_chars=50)
        comment = st.text_area("Your Comment", max_chars=200)
        
        if st.button("Submit"):
            if len(comment.strip()) > 0:
                comments_file = os.path.join(script_dir, "comments.txt")
                with open(comments_file, "a") as file:
                    file.write(f"{user_name}: {comment}\n")
                st.success("Comment submitted successfully!")
            else:
                st.warning("Please enter a comment before submitting.")
        
        # Display existing comments
        st.write("### Previous Comments:")
        comments_file = os.path.join(script_dir, "comments.txt")
        if os.path.exists(comments_file):
            with open(comments_file, "r") as file:
                comments = file.readlines()
                for comment in comments:
                    st.text(comment.strip())

if __name__ == "__main__":
    main()