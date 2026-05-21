import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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

# --- 2. Initialize Modern MediaPipe Tasks API / 初始化现代 MediaPipe Tasks API ---
# We use @st.cache_resource so the server only loads the heavy AI model once!
# 我们使用 @st.cache_resource 这样服务器只需加载一次繁重的 AI 模型！
@st.cache_resource
def load_face_detector():
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1 # Track one face / 跟踪一张脸
    )
    return vision.FaceLandmarker.create_from_options(options)

face_detector = load_face_detector()

# --- 3. WebRTC Video Processor / WebRTC 视频处理器 ---
class FaceMeshProcessor:
    def recv(self, frame):
        # Convert the video frame to an OpenCV array
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # The Tasks API requires an mp.Image format / Tasks API 需要 mp.Image 格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect the pose / 检测姿态
        detection_result = face_detector.detect(mp_image)

        # Draw the Face Dots
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                for landmark in face_landmarks:
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
# ICE servers help bypass firewalls for webcam access / ICE 服务器帮助绕过防火墙以访问网络摄像头
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
