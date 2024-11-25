import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from info_page import show_info_page
from pathlib import Path
import torch

# Load both models
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, 'models', 'model_1.h5')
polyp_model = load_model(model_file_path)
yolo_model = torch.hub.load('.', 'custom', path='best.pt', source='local')

img_length = 50
img_width = 50

def generate_css(primary_color, secondary_background_color):
    # Previous CSS remains the same
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

def process_polyp_image(img):
    img = cv2.resize(img, (img_length, img_width))
    input_data = np.array([img], dtype=np.float32) / 255.0
    prediction = polyp_model.predict(input_data)
    result = "True" if prediction[0][0] > 0.5 else "False"
    return result, prediction[0][0]

def process_yolo_image(img):
    # Run YOLO inference
    results = yolo_model(img)
    
    # Plot results on image
    results.render()  # Updates results.imgs with boxes and labels
    
    # Get the rendered image with detections
    return results.imgs[0]  # Return the first image

def process_video(video_path, frame_number):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    video.release()
    return frame

def main():
    primary_color = st.config.get_option("theme.primaryColor")
    secondary_background_color = st.config.get_option("theme.secondaryBackgroundColor")

    css = generate_css(primary_color, secondary_background_color)
    st.markdown(css, unsafe_allow_html=True)

    page = st.sidebar.selectbox("Go to", ["PolypDetect", "YOLO Detection", "Info Page", "Comments", "QR Code"])

    if page == "PolypDetect":
        # Original PolypDetect page remains the same
        st.title('PolypDetect')
        st.write("""
        This website utilizes a Machine Learning Model to detect polyps in the colon.
        Polyps are clumps of cells that form on the lining of the colon.
        Polyps have been linked to high severity in patients who have an Inflammatory Bowl Disease (IBS).
        This website can help doctors to ensure that they identify all polyps, as some can be discrete.
        Please remember that the model is not perfect, so use it as a second method.
        """)

        st.markdown('<div class="input-side">', unsafe_allow_html=True)
        st.markdown('<h2 class="title" style="color: #4786a5;">Upload Image or Video</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov"], key="polyp_uploader")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="output-side">', unsafe_allow_html=True)
        if uploaded_file is not None:
            st.markdown('<h2 class="title" style="color: #4786a5;">Detection Result</h2>', unsafe_allow_html=True)
            if uploaded_file.type.startswith('image'):
                img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if st.button('Detect Polyps'):
                    result, probability = process_polyp_image(img)
                    st.markdown(f'<p class="prediction">Prediction: {result}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="probability">Model Output: {probability}</p>', unsafe_allow_html=True)
                    st.image(img, caption='Original Image', width=500, output_format='JPEG')

    elif page == "YOLO Detection":
        st.title('YOLO Object Detection')
        st.write("""
        This section uses a YOLO model for object detection. Upload an image to detect objects.
        """)

        st.markdown('<div class="input-side">', unsafe_allow_html=True)
        st.markdown('<h2 class="title" style="color: #4786a5;">Upload Image</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="yolo_uploader")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="output-side">', unsafe_allow_html=True)
        if uploaded_file is not None:
            st.markdown('<h2 class="title" style="color: #4786a5;">Detection Result</h2>', unsafe_allow_html=True)
            img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if st.button('Detect Objects'):
                result_img = process_yolo_image(img)
                st.image(result_img, caption='Detection Result', width=500)
        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Info Page":
        show_info_page(primary_color, secondary_background_color)

    elif page == "QR Code":
        st.title("QR Code")
        qr_image_path = "polypdetect_qr_code.png"
        st.image(qr_image_path, caption="Please use the QR code to send this app to people you know!", width=500)

    elif page == "Comments":
        # Comments page remains the same
        st.title('Comments')
        st.write("""
        Leave your comments and feedback below:
        """)

        user_name = st.text_input("Your Name", max_chars=50)
        comment = st.text_area("Your Comment", max_chars=200)
        if st.button("Submit"):
            if len(comment.strip()) > 0:
                with open("comments.txt", "a") as file:
                    file.write(f"{user_name}: {comment}\n")
                st.success("Comment submitted successfully!")
                comment = ""
            else:
                st.warning("Please enter a comment before submitting.")

if __name__ == "__main__":
    main()