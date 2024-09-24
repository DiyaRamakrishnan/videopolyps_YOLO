import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from info_page import show_info_page 

# Check for required packages
try:
    import torch
    import ultralytics
except ImportError:
    st.error("Required packages are missing. Please install them using the following commands:")
    st.code("pip install torch torchvision torchaudio")
    st.code("pip install ultralytics")
    st.stop()

# Load the CNN model
script_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model_file_path = os.path.join(script_dir, 'models', 'model_1.h5')
cnn_model = load_model(cnn_model_file_path)

# Load the YOLO model
yolo_model_file_path = os.path.join(script_dir, 'models', 'best.pt')
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_file_path)
except Exception as e:
    st.error(f"Error loading YOLO model: {str(e)}")
    st.error("Please make sure the 'best.pt' file is in the 'models' directory and all required packages are installed.")
    st.stop()

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

def process_image_cnn(img):
    img = cv2.resize(img, (img_length, img_width))
    input_data = np.array([img], dtype=np.float32) / 255.0
    prediction = cnn_model.predict(input_data)
    result = "True" if prediction[0][0] > 0.5 else "False"
    return result, prediction[0][0]

def process_image_yolo(img):
    results = yolo_model(img)
    return results

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
        This website utilizes Machine Learning Models to detect polyps in the colon.
        Polyps are clumps of cells that form on the lining of the colon.
        Polyps have been linked to high severity in patients who have an Inflammatory Bowel Disease (IBD).
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
                    # CNN prediction
                    cnn_result, cnn_probability = process_image_cnn(img)
                    st.markdown(f'<p class="prediction">CNN Prediction: {cnn_result}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="probability">CNN Model Output: {cnn_probability:.4f}</p>', unsafe_allow_html=True)
                    
                    # If CNN predicts a polyp, perform YOLO detection
                    if cnn_result == "True":
                        yolo_results = process_image_yolo(img)
                        
                        # Display YOLO results
                        st.image(yolo_results.render()[0], caption='YOLO Detection', use_column_width=True)
                        
                        # Display detection information
                        for detection in yolo_results.xyxy[0]:
                            confidence = detection[4].item()
                            class_id = int(detection[5].item())
                            class_name = yolo_model.names[class_id]
                            st.write(f"Detected: {class_name}, Confidence: {confidence:.4f}")
                    else:
                        st.write("No polyp detected by CNN model.")
                    
                    # Display the original image
                    st.image(img, caption='Original Image', use_column_width=True)
            
            elif uploaded_file.type.startswith('video'):
                video_path = os.path.join(script_dir, 'temp_video.mp4')
                with open(video_path, 'wb') as f:
                    f.write(uploaded_file.read())
                
                video = cv2.VideoCapture(video_path)
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                video.release()

                frame_number = st.slider("Select frame", 0, total_frames - 1, 0)
                selected_frame = process_video(video_path, frame_number)
                
                if st.button('Detect Polyps'):
                    # CNN prediction
                    cnn_result, cnn_probability = process_image_cnn(selected_frame)
                    st.markdown(f'<p class="prediction">CNN Prediction: {cnn_result}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="probability">CNN Model Output: {cnn_probability:.4f}</p>', unsafe_allow_html=True)
                    
                    # If CNN predicts a polyp, perform YOLO detection
                    if cnn_result == "True":
                        yolo_results = process_image_yolo(selected_frame)
                        
                        # Display YOLO results
                        st.image(yolo_results.render()[0], caption='YOLO Detection', use_column_width=True)
                        
                        # Display detection information
                        for detection in yolo_results.xyxy[0]:
                            confidence = detection[4].item()
                            class_id = int(detection[5].item())
                            class_name = yolo_model.names[class_id]
                            st.write(f"Detected: {class_name}, Confidence: {confidence:.4f}")
                    else:
                        st.write("No polyp detected by CNN model.")
                    
                    # Display the original frame
                    st.image(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB), caption='Selected Frame', use_column_width=True)

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
        with open("comments.txt", "r") as file:
            comments = file.readlines()
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