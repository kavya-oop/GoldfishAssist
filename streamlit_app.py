### FocusFace with Debug Mode: Pose + Object Detection + GPT Coaching

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import numpy as np
import time
import openai
import os

# OpenAI API setup
from openai import OpenAI
client = OpenAI()  # Assumes OPENAI_API_KEY is set in environment

# Load YOLOv8 models
pose_model = YOLO("yolov8n-pose.pt")
obj_model = YOLO("yolov8n.pt")

# Streamlit UI setup
st.set_page_config(page_title="GoldfishAssist", layout="centered")
st.title("🎯 FocusFace - AI Distraction Monitor (Debug Mode)")
st.markdown("Stay focused with real-time AI pose + object tracking and GPT coaching.")

# Task selector
st.subheader("What are you working on?")
task = st.selectbox("Select a task", [
    "Math (looking down allowed)",
    "Coding", "Reading", "Break", "Other"])

# Timer logic
if "start_time" not in st.session_state:
    st.session_state["start_time"] = None
if "paused" not in st.session_state:
    st.session_state["paused"] = False
if "pause_time" not in st.session_state:
    st.session_state["pause_time"] = 0

col1, col2 = st.columns(2)
if col1.button("▶️ Start Focus Block"):
    st.session_state["start_time"] = time.time()
    st.session_state["paused"] = False
    st.session_state["pause_time"] = 0

if col2.button("⏸ Pause/Resume"):
    if st.session_state["paused"]:
        st.session_state["start_time"] += time.time() - st.session_state["pause_time"]
        st.session_state["paused"] = False
    else:
        st.session_state["paused"] = True
        st.session_state["pause_time"] = time.time()

if st.session_state["start_time"] and not st.session_state["paused"]:
    elapsed = int(time.time() - st.session_state["start_time"])
    st.metric("Focus Time", f"{elapsed//60}:{elapsed%60:02}")
elif st.session_state["paused"]:
    st.info("⏸ Focus Block Paused")

# Chat + Reasoning
if "chat_log" not in st.session_state:
    st.session_state["chat_log"] = []
    st.session_state["pause_until"] = 0
    st.session_state["last_distraction_reason"] = ""

user_input = st.chat_input("Say something to FocusFace...")

def process_chat(user_message, context=""):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI focus assistant helping the user stay on task."},
                {"role": "user", "content": context + "\n" + user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error from GPT] {e}"

if user_input:
    st.session_state["chat_log"].append(("user", user_input))
    reason = process_chat(user_input, "The user was just flagged as distracted. Is this a reasonable excuse to pause focus tracking?")
    st.session_state["chat_log"].append(("assistant", reason))
    st.chat_message("assistant").markdown(reason)
    if "yes" in reason.lower():
        st.session_state["pause_until"] = time.time() + 300
        st.success("Break accepted. Alerts paused for 5 minutes.")

# Distraction alert display
if "distraction_alert" not in st.session_state:
    st.session_state["distraction_alert"] = ""

st.subheader("🔍 Focus Feedback")
if st.session_state["distraction_alert"]:
    st.error(st.session_state["distraction_alert"])

# Debug info toggle
show_debug = st.checkbox("🛠 Show Debug Output")

# Distraction condition from pose

def is_pose_distracted(keypoints):
    try:
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        vertical_diff = abs(nose[1] - avg_shoulder_y)

        if task.startswith("Math"):
            return False if nose[1] > avg_shoulder_y else True
        if task == "Break":
            return False
        return vertical_diff > 100
    except:
        return False

# Distraction condition from objects

def is_object_distracting(classes):
    distractions = {"cell phone", "cup", "bottle", "banana", "apple", "sandwich", "orange"}
    return any(c in distractions for c in classes)

# Video processor
class PoseAndObjectDetector(VideoTransformerBase):
    def __init__(self):
        self.last_alert_time = 0
        self.alert_text = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if time.time() < st.session_state.get("pause_until", 0):
            return img

        now = time.time()
        alert = ""

        # Pose detection
        pose_results = pose_model(img)
        if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
            pose_distracted = is_pose_distracted(pose_results[0].keypoints.xy[0])
            if pose_distracted:
                alert = "⚠️ You seem distracted by posture."
                if show_debug:
                    st.write("[DEBUG] Pose distraction triggered")

        # Object detection
        obj_results = obj_model(img)
        classes = obj_results[0].names
        if hasattr(obj_results[0], 'boxes') and obj_results[0].boxes is not None:
            class_ids = obj_results[0].boxes.cls.cpu().numpy()
            labels = [classes[int(i)] for i in class_ids]
            if show_debug:
                st.write("[DEBUG] Detected objects:", labels)
            if is_object_distracting(labels):
                alert = "📱 Snacking or phone use detected. Stay focused."
                if show_debug:
                    st.write("[DEBUG] Object-based distraction detected")

        # Trigger alert
        if alert and now - self.last_alert_time > 15:
            st.session_state["distraction_alert"] = alert
            self.last_alert_time = now
        elif not alert:
            st.session_state["distraction_alert"] = ""

        return img

# Webcam stream
webrtc_streamer(
    key="focusface",
    video_processor_factory=PoseAndObjectDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Chat log
with st.expander("🗨️ Chat History"):
    for speaker, msg in st.session_state["chat_log"]:
        st.markdown(f"**{speaker}**: {msg}")
