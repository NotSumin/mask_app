import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

st.set_page_config(page_title="YOLOv8 마스크 탐지", layout="centered")
st.title("😷 마스크 착용 상태 탐지 - YOLOv8")

# 모델 불러오기
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # 같은 디렉토리에 best.pt 필요
    return model

model = load_model()

# 탐지 함수 (RGB 입력 → BGR 결과 반환됨)
def detect_image(image_rgb):
    results = model(image_rgb)
    result_bgr = results[0].plot()
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb

# 탐지 모드 선택
mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠", "동영상"])

# 이미지 탐지
if mode == "이미지":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        st.image(image_rgb, caption="원본 이미지", use_container_width=True)
        st.subheader("탐지 결과")

        result_bgr = detect_image(image_rgb)  # detect_image가 BGR로 반환하는 경우
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        st.image(result_rgb, caption="탐지된 이미지", use_container_width=True)


# 웹캠 탐지
elif mode == "웹캠":
    run = st.checkbox("웹캠 실시간 탐지 시작")
    stframe = st.empty()

    if run:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("웹캠을 열 수 없습니다.")
            else:
                while run:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        st.warning("웹캠 프레임을 가져올 수 없습니다.")
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    result_bgr = detect_image(frame_rgb)  # detect_image가 BGR 반환한다고 가정
                    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

                    stframe.image(result_rgb, channels="RGB", use_container_width=True)
                cap.release()
        except Exception as e:
            st.error(f"웹캠 실행 중 오류 발생: {e}")


# 동영상 탐지
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
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            result_bgr = detect_image(frame_rgb)  # 탐지 결과는 BGR로 반환된다고 가정
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            stframe.image(result_rgb, channels="RGB", use_container_width=True)

        cap.release()
