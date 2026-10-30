import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import av
# Imported VideoHTMLAttributes to safely communicate parameters to your web browser
from streamlit_webrtc import webrtc_streamer, get_cloudflare_ice_servers, VideoHTMLAttributes

st.set_page_config(page_title="YOLOv8 마스크 탐지", layout="centered")
st.title("😷 마스크 착용 상태 탐지 - YOLOv8")

# 모델 불러오기
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# 웹캠 전용: 웹캠(RGB) -> YOLO -> 올바른 RGB 결과 반환
def process_webrtc_frame(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model(image_bgr)
    result_bgr = results[0].plot() 
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb

# 이미지/비디오 파일 전용: 파일(BGR) -> YOLO -> 올바른 RGB 결과 반환
def process_file_frame(image_bgr):
    results = model(image_bgr)
    result_bgr = results[0].plot()
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb

# FIXED: Removed external calls to session_state and .image() inside this worker loop. 
# This stops the background thread from crashing and freezing on frame 1.
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img_rgb = frame.to_ndarray(format="rgb24")
    annotated_rgb = process_webrtc_frame(img_rgb)
    return av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")


# 탐지 모드 선택
mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠", "동영상"])

# -------------------------------------------------------------
# 1. 이미지 탐지
# -------------------------------------------------------------
if mode == "이미지":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        
        preview_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, caption="원본 이미지", use_container_width=True)
        st.subheader("탐지 결과")

        result_rgb = process_file_frame(image_bgr)
        st.image(result_rgb, caption="탐지된 이미지", use_container_width=True)

# -------------------------------------------------------------
# 2. 웹캠 탐지
# -------------------------------------------------------------
elif mode == "웹캠":
    st.write("웹캠 실시간 탐지")
    
    try:
        ice_servers = get_cloudflare_ice_servers()
    except Exception:
        ice_servers = None
        
    webrtc_streamer(
        key="mask-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": ice_servers} if ice_servers else None,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        
        # FIXED 1: This completely deletes that broken camera icon toggle button next to 시작/종료
        media_toggle_controls=False, 
        
        # FIXED 2: Keeps the browser from attempting to show native play/pause overlay panels
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=False, 
            playsInline=True,
            style={"width": "100%"}
        ),
        translations={
            "start": "시작",
            "stop": "종료",
            "select_device": "카메라 선택",
            "choose_device": "카메라를 고르세요",
            "live_status": "🔴 실시간 화면",
        }
    )

# -------------------------------------------------------------
# 3. 동영상 탐지
# -------------------------------------------------------------
elif mode == "동영상":
    uploaded_video = st.file_uploader("동영상을 업로드하세요", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.subheader("탐지 결과 (실시간 재생)")

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            result_rgb = process_file_frame(frame_bgr)
            stframe.image(result_rgb, channels="RGB", use_container_width=True)

        cap.release()
