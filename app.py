import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import requests
import threading
import simpleaudio as sa  # For buzzer sound

# Backend URL (Update with actual AWS endpoint)
API_URL = "http://65.2.10.134:8000"  # Replace with your backend URL

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP, LEFT_CHEEK, RIGHT_CHEEK = 1, 234, 454

# Thresholds
EYE_CLOSED_THRESHOLD = 0.25
DROWSINESS_TIME_THRESHOLD = 2
HEAD_TURN_THRESHOLD = 0.08
HEAD_TURN_TIME_THRESHOLD = 2

# Alert timing
last_alert_time = 0
ALERT_INTERVAL = 5  # Minimum time (seconds) between alerts

# Load a buzzer sound file (you can replace this with your own buzzer sound)
buzzer_wave = sa.WaveObject.from_wave_file("buzzer.wav")  # Ensure you have a buzzer.wav file

def play_buzzer():
    buzzer_wave.play()

def get_landmark_coords(face_landmarks, indices, w, h):
    return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

def eye_aspect_ratio(eye):
    v1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    v2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    h = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (v1 + v2) / (2.0 * h)

# Streamlit UI
st.title("🚗 Vehicle & Driver Monitoring System")
option = st.radio("Select an application:", ["Vehicle Guidance System", "Driver Monitoring System"])

if option == "Vehicle Guidance System":
    st.subheader("📤 Upload a video for processing")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        st.video(uploaded_file)
        st.write("Uploading and processing...")

        files = {"file": (uploaded_file.name, uploaded_file, "video/mp4")}
        response = requests.post(f"{API_URL}/upload-video/", files=files)

        if response.status_code == 200:
            data = response.json()
            output_video_url = f"{API_URL}/download/{data['output_video'].split('/')[-1]}"

            st.success("✅ Processing complete! Watch/download the result below.")
            st.video(output_video_url)
            st.download_button("📥 Download Processed Video", output_video_url)
        else:
            st.error("❌ Failed to process the video. Please try again.")

elif option == "Driver Monitoring System":
    st.subheader("🛑 Real-time Driver Monitoring System")
    st.write("Detects drowsiness & distractions using a webcam.")

    start_monitoring = st.button("Start Monitoring")

    if start_monitoring:
        cap = cv2.VideoCapture(0)
        eye_closed_start, head_turn_start = None, None
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()  # For on-screen alerts

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            h, w, _ = frame.shape
            alert_message = ""

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    left_eye = get_landmark_coords(landmarks, LEFT_EYE, w, h)
                    right_eye = get_landmark_coords(landmarks, RIGHT_EYE, w, h)

                    nose_x = landmarks.landmark[NOSE_TIP].x
                    left_cheek_x = landmarks.landmark[LEFT_CHEEK].x
                    right_cheek_x = landmarks.landmark[RIGHT_CHEEK].x

                    avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                    # Drowsiness Detection
                    if avg_ear < EYE_CLOSED_THRESHOLD:
                        if eye_closed_start is None:
                            eye_closed_start = time.time()
                        elif time.time() - eye_closed_start > DROWSINESS_TIME_THRESHOLD:
                            alert_message = "⚠️ Warning! Drowsiness detected. Slow down!"
                            play_buzzer()
                    else:
                        eye_closed_start = None

                    # Head Turn Detection
                    head_direction = nose_x - (left_cheek_x + right_cheek_x) / 2
                    if abs(head_direction) > HEAD_TURN_THRESHOLD:
                        if head_turn_start is None:
                            head_turn_start = time.time()
                        elif time.time() - head_turn_start > HEAD_TURN_TIME_THRESHOLD:
                            alert_message = "⚠️ Please look straight at the road!"
                            play_buzzer()
                    else:
                        head_turn_start = None

            # Update Streamlit UI
            alert_placeholder.markdown(f"<h3 style='color:red;'>{alert_message}</h3>", unsafe_allow_html=True)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()
