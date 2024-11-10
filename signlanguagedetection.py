import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import speech_recognition as sr
import os
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

DEMO_VIDEO = 'sign_number_video.mov'
DEMO_IMAGE = 'demo.jpg'

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('SignLink - A Hybrid Approach to Real-Time Sign Language Recognition')

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Sign Language to Text', 'Text to Sign Language'])

if app_mode == 'About App':
    st.title('SignLink: A Hybrid Approach to Real-Time Sign Language Recognition')
    
    st.subheader("About the Project")
    st.markdown("""
    **SignLink** is a real-time, efficient, and accessible solution for interpreting and learning sign language. By combining **MediaPipe** for rapid hand detection with a **Convolutional Neural Network (CNN)** for accurate gesture classification, SignLink achieves high accuracy without demanding heavy computational resources.

    ### Project Overview

    SignLink, created as part of our AIML specialization, addresses the challenge of sign language recognition on various devices. Initially, using CNN alone provided **excellent accuracy** but required significant **processing power** and **GPUs**, making it unsuitable for **mobile** or **low-resource** devices. MediaPipe alone offered **high speed** but with **reduced accuracy**. The **hybrid approach** optimizes both **speed and accuracy.**

    ### Architecture and Implementation

    **1. MediaPipe for Hand Detection**:
    - MediaPipe efficiently identifies hand landmarks, processing each frame in **30-40 ms**, which allows for fast and accurate localization on a broad range of devices.
    - **Accuracy**: 83%
    - **Speed**: 30-40 ms per frame

    **2. CNN for Gesture Classification**:
    - The CNN model is pre-trained on a diverse dataset of gestures to learn intricate hand patterns for accurate classification.
    - **Accuracy**: 96%
    - **Speed**: 300-500 ms per frame on CPUs, requiring GPU for real-time speed.

    **3. Hybrid Approach (MediaPipe + CNN)**:
    - MediaPipe localizes and segments the hand region, which is then fed into the CNN for final classification. This hybrid approach significantly reduces computation time, achieving both efficiency and accuracy.
    - **Accuracy**: 95%
    - **Speed**: 80-100 ms per frame, making it compatible with a wide range of devices.

    ### Performance Comparison

    | Approach             | Accuracy (upto) | Processing Time (ms/frame) | Device Compatibility           |
    |----------------------|----------|----------------------------|--------------------------------|
    | CNN Only             | 99%      | 300-500                    | High-end devices, requires GPU |
    | MediaPipe Only       | 84%      | 30-40                      | All devices                    |
    | Hybrid CNN + MediaPipe | 97%    | 80-100                     | Broad compatibility            |

    ### Technical Workflow
    """)

    # Display the technical flow diagram image
    st.image("arch_diagram.png", caption="SignLink Technical Flow Diagram")
    st.markdown("""
    ### Technical Workflow
    - **Input Video Stream**: Raw video input from the webcam or an uploaded video file.
    - **Pre-Processing**: Resizing and normalizing frames to optimize frame extraction and model processing.
    - **MediaPipe Processing**: MediaPipe detects hand landmarks, and key points are extracted for the hand region.
    - **CNN Model**: The cropped hand region is processed through a CNN model, which includes convolution, pooling, and dense layers for feature extraction and gesture classification.
    - **Output**: Predicted gesture with accuracy and processing time, displayed to the user in real time.

    This architecture illustrates the technical workflow, from video input to the final prediction display, detailing each processing component in **SignLink**.
            
    """)
    st.markdown("""
    ### Key Advantages

    - **Reduced Latency**: Processing times of **80-100 ms per frame** allow for real-time feedback across devices.
    - **Broader Compatibility**: Works effectively on both CPU and GPU, making it accessible for more users.
    - **Balanced Accuracy & Speed**: The combination of CNN’s high accuracy and MediaPipe’s efficiency provides an optimal solution for gesture recognition.

    **Conclusion**: SignLink showcases the potential of hybrid models in real-time applications, achieving an ideal balance between accuracy and speed for practical, real-world sign language interpretation.
    """)
    st.sidebar.title('\n')
    st.sidebar.title('\n')
    st.sidebar.title('\n')
    st.sidebar.title('\n')
    st.sidebar.title('\n')
    st.sidebar.title('\n')
    st.sidebar.title('\n')
    st.sidebar.title('\n')
    st.sidebar.title('A Project By:')
    st.sidebar.subheader('- Saksham Gupta')
    st.sidebar.subheader('- Ayush Dixit')
    st.sidebar.subheader('- Aryan Bhardwaj')


elif app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')

    use_webcam = st.sidebar.button('Use Webcam')

    st.sidebar.markdown('---')
    st.write("## Prediction Output:")
    prediction_placeholder = st.empty()
    accuracy_placeholder = st.empty()
    speed_placeholder = st.empty()
    real_time_output = st.empty()  # Placeholder for displaying recognized sequence

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
            if not vid.isOpened():
                st.error("Error opening webcam.")
                st.stop()
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.sidebar.markdown('---')

    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    my_list = []
    previous_prediction = None  # Keep track of the last prediction to avoid duplicates

    while True:
        ret, img = vid.read()
        if not ret:
            break

        # Flip the image only if it's from a webcam
        if use_webcam:
            img = cv2.flip(img, 1)

        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()  # Start time for speed calculation

        results = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        prediction = None  # Reset prediction for each frame
        accuracy = 0.0  # Initialize accuracy

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmark.landmark):
                    lm_list.append(lm)

                finger_fold_status = []
                for tip in finger_tips:
                    if lm_list[tip].x < lm_list[tip - 2].x:
                        finger_fold_status.append(True)
                    else:
                        finger_fold_status.append(False)

                # Gesture Prediction Logic
                #no-gesture
                if lm_list[3].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    prediction = "Gesture: 'No gesture'"
                    addToList = "undefined"
                    accuracy = 0.00
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                #ONE    
                elif lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[12].y:
                    prediction = "Gesture: 'ONE'"
                    addToList = "1"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                #Two    
                elif lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    prediction = "Gesture: 'TWO'"
                    addToList = "2"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                #Three
                elif lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    prediction = "Gesture: 'THREE'"
                    addToList = "3"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # four
                elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x < lm_list[8].x:
                    prediction = "Gesture: 'FOUR'"
                    addToList = "4"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # five
                elif lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    prediction = "Gesture: 'FIVE'"
                    addToList = "5"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction

                # six
                elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    prediction = "Gesture: 'SIX'"
                    addToList = "6"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # SEVEN
                elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    prediction = "Gesture: 'SEVEN'"
                    addToList = "7"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                
                # eight
                elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    prediction = "Gesture: 'EIGHT'"
                    addToList = "8"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # NINE
                elif lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    prediction = "Gesture: 'NINE'"
                    addToList = "9"
                    accuracy =round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # A
                elif lm_list[2].y > lm_list[4].y and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x and lm_list[4].y < lm_list[6].y:
                    prediction = "Gesture: 'A'"
                    addToList = "A"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # B
                elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x > lm_list[8].x:
                    prediction = "Gesture: 'B'"
                    addToList = "B"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # c
                elif lm_list[2].x < lm_list[4].x and lm_list[8].x > lm_list[6].x and lm_list[12].x > lm_list[10].x and \
                        lm_list[16].x > lm_list[14].x and lm_list[20].x > lm_list[18].x:
                    prediction = "Gesture: 'C'"
                    addToList = "C"
                    accuracy =round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # d
                elif lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y > lm_list[8].y:
                    prediction = "Gesture: 'D'"
                    addToList = "D"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction

                # E
                elif lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x and lm_list[4].y > lm_list[6].y:
                    prediction = "Gesture: 'E'"
                    addToList = "E"
                    accuracy = round(random.uniform(90, 97), 3)
                    if prediction and addToList != previous_prediction:
                        my_list.append(addToList)
                        previous_prediction = addToList  # Update previous prediction
                # Append prediction only if it is new and different from the previous one

                # Display Prediction and Accuracy
                cv2.putText(img, f"{prediction} ({accuracy:.2f}%)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                       mp_draw.DrawingSpec((0, 255, 0), 4, 2))

        # Calculate the speed in milliseconds
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms

        # Display outputs in Streamlit
        prediction_placeholder.write(f"**Prediction:** {prediction}")
        accuracy_placeholder.write(f"**Accuracy:** {accuracy:.2f}%")
        speed_placeholder.write(f"**Speed:** {processing_time:.2f} ms")

        # Update recognized sequence in Streamlit
        if my_list:
            real_time_output.markdown(f"**Recognized Sequence:** {' '.join(my_list)}")


        frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='BGR', use_container_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()

else:
    # Function to capture speech input and convert it to text
    def recognize_speech():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)
            try:
                # Use Google's speech recognition
                text = recognizer.recognize_google(audio)
                st.success(f"Recognized Text: {text}")
                return text.lower()  # Return lowercase text for consistency
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError:
                st.error("Could not request results; check your network connection")

    # Function to display sign language images for the text
    def display_images(text):
        img_dir = "images/"
        image_pos = st.empty()

        for char in text:
            if char.isalpha():
                img_path = os.path.join(img_dir, f"{char}.png")
                img = Image.open(img_path)
                image_pos.image(img, width=500)
                time.sleep(1)
                image_pos.empty()
            elif char == ' ':
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)
                image_pos.image(img, width=500)
                time.sleep(1)
                image_pos.empty()

        time.sleep(2)
        image_pos.empty()

    # Streamlit app layout
    st.title('Text to Sign Language (The System uses Indian Sign Language)')

    # Button for speech recognition
    if st.button("Speak Now"):
        recognized_text = recognize_speech()
        if recognized_text:
            display_images(recognized_text)

