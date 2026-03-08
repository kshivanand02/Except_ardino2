# import streamlit as st
# import cv2
# import dlib
# import numpy as np
# import time
# import os
# import threading
# import requests
# import base64
# from collections import deque
# from groq import Groq
# from queue import Queue

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
# st.title("🚗 Driver Drowsiness Detection System (Hybrid AI + LLM, Stable & Fast)")

# # ---------------- SESSION STATE ----------------
# for k, v in {
#     "run": False,
#     "llm_state": "ACTIVE",
#     "frame_count": 0,
#     "total_yawns": 0,
#     "last_llm_time": 0.0,
#     "llm_busy": False,
# }.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ---------------- LOAD MODELS ----------------
# @st.cache_resource
# def load_models():
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     return detector, predictor

# detector, predictor = load_models()

# # ---------------- GROQ CLIENT ----------------
# api_key = "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"
# groq_client = Groq(api_key=api_key) if api_key else None
# MODEL_NAME = "llama-3.1-8b-instant"

# llm_result_queue = Queue()

# # ================= NEW: ASYNC CNN MODEL CALL =================
# def send_frame_to_models(frame):
#     try:
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_base64 = base64.b64encode(buffer).decode('utf-8')

#         response = requests.post(
#             "http://localhost:5000/predict_all",
#             json={"frame": frame_base64},
#             timeout=0.3
#         )

#         if response.status_code == 200:
#             data = response.json()

#             eye = "CLOSED" if data["eye"]["eye_label"] == "sleep" else "OPEN"
#             head = "BENT" if data["head"]["head_label"] == "Head Bent" else "STRAIGHT"
#             mouth = "OPEN" if data["mouth"]["mouth_label"] == "Mouth Open" else "CLOSED"

#             print("\n====== CNN MODEL OUTPUT ======")
#             print("Eye   :", eye)
#             print("Head  :", head)
#             print("Mouth :", mouth)
#             print("================================")

#     except Exception as e:
#         print("Model API error:", e)

# # ---------------- UTILITIES ----------------
# def euclidean(p1, p2):
#     return np.linalg.norm(np.array(p1) - np.array(p2))

# def eye_ratio(a, b, c, d, e, f):
#     up = euclidean(b, d) + euclidean(c, e)
#     down = euclidean(a, f) + 1e-6
#     return up / (2.0 * down)

# def head_bent_angle(nose, chin):
#     dx = chin[0] - nose[0]
#     dy = chin[1] - nose[1]
#     return np.degrees(np.arctan2(dy, dx))

# def weighted_ratio(binary_hist, alpha=0.85):
#     if not binary_hist:
#         return 0.0
#     w = np.linspace(alpha, 1.0, len(binary_hist))
#     arr = np.array(binary_hist, dtype=float)
#     return float(np.sum(w * arr) / np.sum(w))

# # ---------------- ASYNC LLM WORKER ----------------
# def llm_worker(features: dict, out_q: Queue):
#     try:
#         prompt = f"""
# Classify driver drowsiness as one of: ACTIVE, LOW, MEDIUM, HIGH.
# Features:
# eye_closed_ratio={features['eye_closed_ratio']}
# blink_rate={features['blink_rate']}
# yawn_rate={features['yawn_rate']}
# head_bent_freq={features['head_bent_freq']}
# face_lost_ratio={features['face_lost_ratio']}
# Return only one word.
# """
#         resp = groq_client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             timeout=3.0,
#         )
#         out = resp.choices[0].message.content.strip().upper()
#         if out in ["ACTIVE", "LOW", "MEDIUM", "HIGH"]:
#             out_q.put(out)
#     except Exception as e:
#         print("Groq error:", e)
#     finally:
#         out_q.put(None)

# # ---------------- SIDEBAR ----------------
# st.sidebar.header("⚙️ Controls")
# start = st.sidebar.button("▶ Start Camera")
# stop = st.sidebar.button("⏹ Stop Camera")

# MOUTH_THRESH = st.sidebar.slider("Mouth Threshold", 12, 40, 25)
# HEAD_BENT_THRESH = st.sidebar.slider("Head Angle Threshold", 60, 120, 90)
# EAR_THRESH = 0.20
# LLM_INTERVAL = st.sidebar.slider("LLM Interval (sec)", 2, 6, 3)

# if start: st.session_state.run = True
# if stop: st.session_state.run = False

# # ---------------- UI ----------------
# left_col, right_col = st.columns([2, 1])
# with left_col:
#     frame_window = st.empty()
# with right_col:
#     status_box = st.empty()
#     llm_box = st.empty()
#     ear_box = st.empty()
#     mouth_box = st.empty()
#     head_box = st.empty()
#     yawn_box = st.empty()
#     blink_rate_box = st.empty()

# # ---------------- TEMPORAL MEMORY ----------------
# WINDOW = 20
# STATE_WINDOW = 12
# ear_hist = deque(maxlen=WINDOW)
# eye_closed_hist = deque(maxlen=WINDOW)
# head_hist = deque(maxlen=WINDOW)
# face_hist = deque(maxlen=WINDOW)
# blink_events = deque(maxlen=WINDOW)
# yawn_events = deque(maxlen=WINDOW)
# state_hist = deque(maxlen=STATE_WINDOW)
# fps_hist = deque(maxlen=10)

# # ---------------- CAMERA LOOP ----------------
# if st.session_state.run:
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#     prev_yawn_status = False
#     prev_eye_closed = False
#     last_t = time.time()
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#     while st.session_state.run:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (640, 480))

#         # 🔥 NEW: Async CNN call (NON BLOCKING)
#         threading.Thread(
#             target=send_frame_to_models,
#             args=(frame.copy(),),
#             daemon=True
#         ).start()

#         now = time.time()
#         dt = max(1e-3, now - last_t)
#         last_t = now

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = clahe.apply(gray)

#         faces = detector(gray)

#         yawn_status = False
#         head_status = "Normal"
#         face_detected = len(faces) > 0
#         head_angle = 0.0
#         eye_closed = False
#         ear_avg = 0.0

#         for face in faces:
#             landmarks = predictor(gray, face)

#             left = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
#             right = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

#             left_ear = eye_ratio(left[0], left[1], left[2], left[5], left[4], left[3])
#             right_ear = eye_ratio(right[0], right[1], right[2], right[5], right[4], right[3])
#             ear_avg = float(np.clip((left_ear + right_ear) / 2, 0.05, 0.5))
#             eye_closed = ear_avg < EAR_THRESH

#             upper = (landmarks.part(62).x, landmarks.part(62).y)
#             lower = (landmarks.part(66).x, landmarks.part(66).y)
#             yawn_status = euclidean(upper, lower) > MOUTH_THRESH

#             nose = (landmarks.part(30).x, landmarks.part(30).y)
#             chin = (landmarks.part(8).x, landmarks.part(8).y)
#             head_angle = head_bent_angle(nose, chin)
#             head_status = "Head Bent" if head_angle > HEAD_BENT_THRESH else "Normal"

#         frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

#         status_box.success("RUNNING")
#         llm_box.info(f"🤖 LLM Decision: {st.session_state.llm_state}")
#         ear_box.info(f"EAR Avg: {ear_avg:.2f}")
#         mouth_box.info(f"Mouth: {'OPEN' if yawn_status else 'CLOSED'}")
#         head_box.info(f"Head Angle: {head_angle:.1f}°")
#         yawn_box.info(f"Yawns: {st.session_state.total_yawns}")
#         blink_rate_box.info("Processing...")

#     cap.release()
# else:
#     st.info("Click **Start Camera** to begin detection")


import streamlit as st
import cv2
import dlib
import numpy as np
import time
import os
import threading
import requests
import base64
from collections import deque
from groq import Groq
from queue import Queue

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection System (Hybrid AI + LLM + CNN, Stable & Fast)")

# ---------------- SESSION STATE ----------------
for k, v in {
    "run": False,
    "llm_state": "ACTIVE",
    "frame_count": 0,
    "total_yawns": 0,
    "last_llm_time": 0.0,
    "llm_busy": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

detector, predictor = load_models()

# ---------------- GROQ CLIENT ----------------
api_key = "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"
groq_client = Groq(api_key=api_key) if api_key else None
MODEL_NAME = "llama-3.1-8b-instant"

# --------------- QUEUE FOR LLM RESULTS ---------------
llm_result_queue = Queue()

# ================= CNN MODEL API INTEGRATION =================
def send_frame_to_models(frame):
    try:
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send to Flask API (NON-BLOCKING)
        response = requests.post(
            "http://localhost:5000/predict_all",
            json={"frame": frame_base64},
            timeout=0.3
        )

        if response.status_code == 200:
            data = response.json()

            eye = "CLOSED" if data["eye"]["eye_label"] == "sleep" else "OPEN"
            head = "BENT" if data["head"]["head_label"] == "Head Bent" else "STRAIGHT"
            mouth = "OPEN" if data["mouth"]["mouth_label"] == "Mouth Open" else "CLOSED"

            print("\n====== CNN MODEL OUTPUT ======")
            print("Eye   :", eye)
            print("Head  :", head)
            print("Mouth :", mouth)
            print("================================")

    except Exception as e:
        print("Model API error:", e)

# ---------------- UTILITIES ----------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_ratio(a, b, c, d, e, f):
    up = euclidean(b, d) + euclidean(c, e)
    down = euclidean(a, f) + 1e-6
    return up / (2.0 * down)

def head_bent_angle(nose, chin):
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    return np.degrees(np.arctan2(dy, dx))

def weighted_ratio(binary_hist, alpha=0.85):
    if not binary_hist:
        return 0.0
    w = np.linspace(alpha, 1.0, len(binary_hist))
    arr = np.array(binary_hist, dtype=float)
    return float(np.sum(w * arr) / np.sum(w))

# ---------------- ASYNC LLM WORKER ----------------
def llm_worker(features: dict, out_q: Queue):
    try:
        prompt = f"""
Classify driver drowsiness as one of: ACTIVE, LOW, MEDIUM, HIGH.
Features:
eye_closed_ratio={features['eye_closed_ratio']}
blink_rate={features['blink_rate']}
yawn_rate={features['yawn_rate']}
head_bent_freq={features['head_bent_freq']}
face_lost_ratio={features['face_lost_ratio']}
Return only one word.
"""
        resp = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=3.0,
        )
        out = resp.choices[0].message.content.strip().upper()
        if out in ["ACTIVE", "LOW", "MEDIUM", "HIGH"]:
            out_q.put(out)
    except Exception as e:
        print("Groq error:", e)
    finally:
        out_q.put(None)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
start = st.sidebar.button("▶ Start Camera")
stop = st.sidebar.button("⏹ Stop Camera")

MOUTH_THRESH = st.sidebar.slider("Mouth Threshold", 12, 40, 25)
HEAD_BENT_THRESH = st.sidebar.slider("Head Angle Threshold", 60, 120, 90)
EAR_THRESH = 0.20
LLM_INTERVAL = st.sidebar.slider("LLM Interval (sec)", 2, 6, 3)

if start: 
    st.session_state.run = True
if stop: 
    st.session_state.run = False

# ---------------- UI ----------------
left_col, right_col = st.columns([2, 1])
with left_col:
    frame_window = st.empty()
with right_col:
    status_box = st.empty()
    llm_box = st.empty()
    ear_box = st.empty()
    mouth_box = st.empty()
    head_box = st.empty()
    yawn_box = st.empty()
    blink_rate_box = st.empty()
    cnn_status_box = st.empty()  # 🔥 NEW: CNN Status Display

# ---------------- TEMPORAL MEMORY ----------------
WINDOW = 20
STATE_WINDOW = 12
ear_hist = deque(maxlen=WINDOW)
eye_closed_hist = deque(maxlen=WINDOW)
head_hist = deque(maxlen=WINDOW)
face_hist = deque(maxlen=WINDOW)
blink_events = deque(maxlen=WINDOW)
yawn_events = deque(maxlen=WINDOW)
state_hist = deque(maxlen=STATE_WINDOW)
fps_hist = deque(maxlen=10)

# ---------------- CAMERA LOOP ----------------
if st.session_state.run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev_yawn_status = False
    prev_eye_closed = False
    last_t = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = max(1e-3, now - last_t)
        last_t = now

        frame = cv2.resize(frame, (640, 480))
        
        # 🔥 CNN MODEL CALL (NON-BLOCKING - runs in parallel)
        threading.Thread(
            target=send_frame_to_models,
            args=(frame.copy(),),
            daemon=True
        ).start()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        faces = detector(gray)

        yawn_status = False
        head_status = "Normal"
        face_detected = len(faces) > 0
        head_angle = 0.0
        eye_closed = False
        ear_avg = 0.0

        for face in faces:
            landmarks = predictor(gray, face)

            # Eye landmarks (dlib EAR)
            left = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            left_ear = eye_ratio(left[0], left[1], left[2], left[5], left[4], left[3])
            right_ear = eye_ratio(right[0], right[1], right[2], right[5], right[4], right[3])
            ear_avg = float(np.clip((left_ear + right_ear) / 2, 0.05, 0.5))
            eye_closed = ear_avg < EAR_THRESH

            # Mouth landmarks (yawn detection)
            upper = (landmarks.part(62).x, landmarks.part(62).y)
            lower = (landmarks.part(66).x, landmarks.part(66).y)
            yawn_status = euclidean(upper, lower) > MOUTH_THRESH

            # Head pose
            nose = (landmarks.part(30).x, landmarks.part(30).y)
            chin = (landmarks.part(8).x, landmarks.part(8).y)
            head_angle = head_bent_angle(nose, chin)
            head_status = "Head Bent" if head_angle > HEAD_BENT_THRESH else "Normal"

        # ALL ORIGINAL EVENT DETECTION LOGIC (UNCHANGED)
        blink_event = 1 if (prev_eye_closed and not eye_closed) else 0
        prev_eye_closed = eye_closed
        yawn_event = 1 if (prev_yawn_status and not yawn_status) else 0
        prev_yawn_status = yawn_status

        if yawn_event:
            st.session_state.total_yawns += 1

        ear_hist.append(ear_avg if face_detected else 0)
        eye_closed_hist.append(1 if eye_closed else 0)
        head_hist.append(1 if head_status == "Head Bent" else 0)
        face_hist.append(1 if face_detected else 0)
        blink_events.append(blink_event)
        yawn_events.append(yawn_event)

        fps_est = 1.0 / dt
        fps_hist.append(fps_est)
        fps_smooth = sum(fps_hist) / len(fps_hist)

        blink_rate_fps = (sum(blink_events) / max(1, len(blink_events))) * fps_smooth
        yawn_rate_win = sum(yawn_events) / max(1, len(yawn_events))
        head_bent_freq = sum(head_hist) / max(1, len(head_hist))

        closed_ratio_w = weighted_ratio(list(eye_closed_hist), 0.85)
        face_lost_ratio_w = weighted_ratio([1 - x for x in face_hist], 0.85)

        # ---- Rule-based (UNCHANGED) ----
        if face_lost_ratio_w > 0.6:
            state_raw = "HIGH"
        elif closed_ratio_w > 0.45:
            state_raw = "MEDIUM"
        elif closed_ratio_w > 0.15:
            state_raw = "LOW"
        else:
            state_raw = "ACTIVE"

        # ---- ACTIVE override (UNCHANGED) ----
        is_clearly_active = (
            face_detected and ear_avg > EAR_THRESH + 0.05 and
            not yawn_status and head_status == "Normal" and
            closed_ratio_w < 0.10 and yawn_rate_win < 0.02
        )

        # ---- Trigger async LLM (UNCHANGED) ----
        if groq_client and not st.session_state.llm_busy and (time.time() - st.session_state.last_llm_time > LLM_INTERVAL):
            st.session_state.last_llm_time = time.time()
            st.session_state.llm_busy = True
            feats = {
                "eye_closed_ratio": round(closed_ratio_w, 3),
                "blink_rate": round(blink_rate_fps, 3),
                "yawn_rate": round(yawn_rate_win, 3),
                "head_bent_freq": round(head_bent_freq, 3),
                "face_lost_ratio": round(face_lost_ratio_w, 3),
            }
            threading.Thread(target=llm_worker, args=(feats, llm_result_queue), daemon=True).start()

        # ---- Read LLM results (thread-safe) (UNCHANGED) ----
        while not llm_result_queue.empty():
            res = llm_result_queue.get()
            if res in ["ACTIVE", "LOW", "MEDIUM", "HIGH"]:
                st.session_state.llm_state = res
            st.session_state.llm_busy = False

        # ---- Fusion Logic (UNCHANGED) ----
        if is_clearly_active:
            state = "ACTIVE"
        else:
            llm_state = st.session_state.llm_state
            if "HIGH" in [state_raw, llm_state]:
                state = "HIGH"
            elif "MEDIUM" in [state_raw, llm_state]:
                state = "MEDIUM"
            elif "LOW" in [state_raw, llm_state]:
                state = "LOW"
            else:
                state = "ACTIVE"

        # ---- Temporal smoothing (UNCHANGED) ----
        state_hist.append(state)
        weights = np.linspace(0.5, 1.0, len(state_hist))
        score = {"ACTIVE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for s, w in zip(state_hist, weights):
            score[s] += w
        state = max(score, key=score.get)

        # ---------------- UI (ENHANCED) ----------------
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Status display (UNCHANGED)
        if state == "ACTIVE":
            status_box.success("ACTIVE")
        elif state == "LOW":
            status_box.warning("LOW DROWSINESS")
        elif state == "MEDIUM":
            status_box.markdown(
                "<div style='background-color:#ffc0cb; padding:10px; border-radius:8px; color:black; font-weight:bold;'>MEDIUM DROWSINESS</div>",
                unsafe_allow_html=True
            )
        else:
            status_box.error("HIGH DROWSINESS 🚨")

        # llm_box.info(f"🤖 LLM Decision: {st.session_state.llm_state}")
        ear_box.info(f"EAR Avg: {ear_avg:.2f}")
        mouth_box.info(f"Mouth: {'OPEN' if yawn_status else 'CLOSED'}")
        head_box.info(f"Head Angle: {head_angle:.1f}°")
        yawn_box.info(f"Yawns: {st.session_state.total_yawns}")
        blink_rate_box.info(f"Blink Rate: {blink_rate_fps:.2f} | FPS: {fps_smooth:.1f}")
        # cnn_status_box.info("🧠 CNN Models: ACTIVE")  # 🔥 CNN Status

    cap.release()
else:
    st.info("Click **Start Camera** to begin detection")
