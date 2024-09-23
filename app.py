import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from info_page import show_info_page 
import sys
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Diagnostic information
st.write("Python version:", sys.version)
st.write("OpenCV version:", cv2.__version__)
st.write("NumPy version:", np.__version__)
st.write("PyTorch version:", torch.__version__)

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
st.write("Script directory:", script_dir)

# Add YOLOv5 directory to Python path
yolov5_dir = os.path.join(script_dir, 'yolov5')
sys.path.append(yolov5_dir)
logger.info(f"Added YOLOv5 directory to Python path: {yolov5_dir}")

# Load the classification model
model_file_path = os.path.join(script_dir, 'models', 'model_1.h5')
st.write("Looking for classification model at:", model_file_path)

try:
    model = load_model(model_file_path)
    st.success("Classification model loaded successfully!")
except Exception as e:
    st.error(f"Error loading classification model: {str(e)}")
    st.info("Please check if the 'model_1.h5' file is in the correct location.")
    logger.exception("Error loading classification model:")

# Load YOLOv5 model
yolo_path = os.path.join(script_dir, 'models', 'best.pt')
st.write("Looking for YOLO model at:", yolo_path)

yolo_model = None
try:
    if not os.path.exists(yolo_path):
        st.error(f"YOLO model file not found at: {yolo_path}")
        st.info("Please make sure you have placed the 'best.pt' file in the 'models' directory.")
    else:
        # Try to load the YOLO model
        st.info("Attempting to load YOLO model...")
        logger.info(f"Attempting to load YOLO model from {yolo_path}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
        
        # Import YOLO functions
        from models.experimental import attempt_load
        from utils.general import non_max_suppression
        
        # Load YOLOv5 model
        yolo_model = attempt_load(yolo_path, map_location='cpu')
        st.success("YOLO model loaded successfully!")
        logger.info("YOLO model loaded successfully")
except Exception as e:
    st.error(f"Error loading YOLO model: {str(e)}")
    st.info("YOLO model could not be loaded. The app will continue with limited functionality.")
    logger.exception("Error loading YOLO model:")

# Define image dimensions
img_length = 50
img_width = 50

# Define CSS styles dynamically based on theme settings
def generate_css(primary_color, secondary_background_color):
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

def process_image(img):
    # Resize for the classification model
    resized_img = cv2.resize(img, (img_length, img_width))
    input_data = np.array([resized_img], dtype=np.float32) / 255.0
    prediction = model.predict(input_data)
    result = "True" if prediction[0][0] > 0.5 else "False"
    
    # If classified as having a polyp and YOLO model is loaded, perform YOLO detection
    if result == "True" and yolo_model is not None:
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for YOLO
        img_for_yolo = torch.from_numpy(rgb_img).to('cpu')
        img_for_yolo = img_for_yolo.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        
        # Perform YOLO detection
        with torch.no_grad():
            detections = yolo_model(img_for_yolo)[0]
        
        # Non-maximum suppression
        detections = non_max_suppression(detections, 0.25, 0.45)[0]
        
        # Draw bounding boxes
        img_with_boxes = rgb_img.copy()
        if detections is not None and len(detections):
            detections[:, :4] = detections[:, :4].round()
            for *xyxy, conf, cls in detections:
                cv2.rectangle(img_with_boxes, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        
        return result, prediction[0][0], img_with_boxes
    
    return result, prediction[0][0], img

def process_video(video_path, frame_number):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    video.release()
    return frame

def main():
    # Get theme settings from config.toml
    primary_color = st.config.get_option("theme.primaryColor")
    secondary_background_color = st.config.get_option("theme.secondaryBackgroundColor")

    # Render CSS styles
    css = generate_css(primary_color, secondary_background_color)
    st.markdown(css, unsafe_allow_html=True)

    # Main content
    page = st.sidebar.selectbox("Go to", ["PolypDetect", "Info Page", "Comments", "QR Code"])

    if page == "PolypDetect":
        st.title('PolypDetect')
        st.write("""
        This website utilizes a Machine Learning Model to detect polyps in the colon.
        Polyps are clumps of cells that form on the lining of the colon.
        Polyps have been linked to high severity in patients who have an Inflammatory Bowl Disease (IBS).
        This website can help doctors to ensure that they identify all polyps, as some can be discrete.
        Please remember that the model is not perfect, so use it as a second method.
        """)

        # Input side
        st.markdown('<div class="input-side">', unsafe_allow_html=True)
        st.markdown('<h2 class="title" style="color: #4786a5;">Upload Image or Video</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Output side
        st.markdown('<div class="output-side">', unsafe_allow_html=True)
        if uploaded_file is not None:
            st.markdown('<h2 class="title" style="color: #4786a5;">Detection Result</h2>', unsafe_allow_html=True)
            if uploaded_file.type.startswith('image'):
                img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if st.button('Detect Polyps'):
                    result, probability, processed_img = process_image(img)
                    st.markdown(f'<p class="prediction">Prediction: {result}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="probability">Model Output: {probability}</p>', unsafe_allow_html=True)
                    st.image(processed_img, caption='Processed Image', width=500, channels="RGB")
            elif uploaded_file.type.startswith('video'):
                video_path = os.path.join(script_dir, 'temp_video.mp4')
                with open(video_path, 'wb') as f:
                    f.write(uploaded_file.read())
                frame_number = st.number_input("Frame Number", value=0, step=1)
                selected_frame = process_video(video_path, frame_number)
                st.image(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB), caption='Selected Frame', channels='RGB', width=500)
                if st.button('Detect Polyps'):
                    result, probability, processed_frame = process_image(selected_frame)
                    st.markdown(f'<p class="prediction">Prediction: {result}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="probability">Model Output: {probability}</p>', unsafe_allow_html=True)
                    st.image(processed_frame, caption='Processed Frame', width=500, channels="RGB")
                
    elif page == "Info Page":
        show_info_page(primary_color, secondary_background_color)

    elif page == "QR Code":
        st.title("QR Code")
        qr_image_path = "polypdetect_qr_code.png"
        st.image(qr_image_path, caption="Please use the QR code to send this app to people you know!", width=500)

    elif page == "Comments":
        st.title('Comments')
        st.write("""
        Leave your comments and feedback below:
        """)

        # Add comment box
        user_name = st.text_input("Your Name", max_chars=50)
        comment = st.text_area("Your Comment", max_chars=200)
        if st.button("Submit"):
            if len(comment.strip()) > 0:
                # Add the comment to the list
                with open("comments.txt", "a") as file:
                    file.write(f"{user_name}: {comment}\n")
                st.success("Comment submitted successfully!")
                comment = ""
            else:
                st.warning("Please enter a comment before submitting.")
        
        # Display comments
        st.write("### Comments:")
        comments = []
        try:
            with open("comments.txt", "r") as file:
                comments = file.readlines()
        except FileNotFoundError:
            st.info("No comments yet. Be the first to comment!")
        if comments:
            for comment_text in comments:
                # Split comment into name and message parts
                parts = comment_text.split(":", 1)
                if len(parts) == 2:
                    name, comment_msg = parts
                    # Display the name above the comment
                    st.write(f"**{name.strip()}**")
                    st.write(f"{comment_msg.strip()}")
                else:
                    st.write(comment_text.strip())

        # Add button to delete all comments
        if st.button("Delete All Comments"):
            # Clear the comments file
            with open("comments.txt", "w") as file:
                file.truncate(0)
            st.success("All comments deleted successfully!")
            # Reset the page
            st.experimental_rerun()

if __name__ == "__main__":
    main()