import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from grad_cam import GradCAMModel, get_grad_cam
from info_page import show_info_page 

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, 'models', 'model_1.h5')
model = load_model(model_file_path)
grad_cam_model = GradCAMModel(model, layer_name="conv2d_173")

# Define image dimensions
img_length = 50
img_width = 50

def process_image(img):
    img = cv2.resize(img, (img_length, img_width))
    input_data = np.array([img], dtype=np.float32) / 255.0
    prediction = model.predict(input_data)
    result = "True" if prediction[0][0] > 0.5 else "False"
    return result, prediction[0][0]

def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames

def main():
    page = st.sidebar.selectbox("Go to", ["PolypDetect", "Info Page", "Comments", "QR Code"])

    if page == "PolypDetect":
        st.title('PolypDetect')
        st.write("""
        This website utilizes a Machine Learning Model to detect polyps in the colon.
        Polyps are clumps of cells that form on the lining of the colon.
        Polyps have been linked to high severity in patients who have an Inflammatory Bowel Disease (IBS).
        This website can help doctors to ensure that they identify all polyps, as some can be discrete.
        Please remember that the model is not perfect, so use it as a second method.
        """)

        uploaded_file = st.file_uploader("Upload Image or Video")

        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                result, probability = process_image(img)
                st.write(f"Prediction: {result}")
                st.write(f"Probability: {probability}")
                
                # Display Grad-CAM image
                grad_cam_img = get_grad_cam(grad_cam_model, 1, img, img_length, img_width)  # Assuming class index 1
                st.image(grad_cam_img, caption='Grad-CAM', channels='RGB', use_column_width=True)
                
            elif uploaded_file.type.startswith('video'):
                video_path = os.path.join(script_dir, 'temp_video.mp4')  # Temporarily save video as .mp4
                with open(video_path, 'wb') as f:
                    f.write(uploaded_file.read())
                frames = process_video(video_path)
                selected_frame_index = st.select_slider('Select a frame for processing', range(len(frames)))
                selected_frame = frames[selected_frame_index]
                result, probability = process_image(selected_frame)
                st.write(f"Prediction: {result}")
                st.write(f"Probability: {probability}")

                # Display a preview image of the selected frame
                st.image(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB), caption='Selected Frame', channels='RGB', use_column_width=True)

    elif page == "Info Page":
        show_info_page(primary_color, secondary_background_color)  # Call the show_info_page function with theme colors

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
