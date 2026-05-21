import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. Streamlit UI Setup / 设置 Streamlit 界面 ---
st.set_page_config(page_title="AI Motion Tracking", page_icon="🤖", layout="wide")

st.title("🤖 Real-Time AI Face Tracking")
st.markdown("Welcome to the workshop! Allow camera access to see the AI map your face in real-time.")

# Sidebar for controls / 侧边栏控件
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Adjust the AI parameters below:")
    dot_color_hex = st.color_picker("Dot Color (点颜色)", "#00FF00")
    dot_size = st.slider("Dot Size (点大小)", min_value=1, max_value=5, value=2)

# Convert Hex color to BGR for OpenCV / 将十六进制颜色转换为 OpenCV 的 BGR
dot_color = tuple(int(dot_color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

# --- 2. Initialize MediaPipe / 初始化 MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh

# --- 3. WebRTC Video Processor / WebRTC 视频处理器 ---
# This class runs on every single frame of your webcam video
class FaceMeshProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        # Convert the video frame to an OpenCV array
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Process image with MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        # Draw the Face Dots
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # Draw dots using the dynamic color and size from the sidebar
                    cv2.circle(img, (x, y), dot_size, dot_color, -1)
            
            cv2.putText(img, 'AI Tracking Active', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'Looking for face...', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Return the modified image back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. Start the WebRTC Stream / 启动 WebRTC 流 ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="face-tracking",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=FaceMeshProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.markdown("Built for the AI Motion Tracking Workshop")