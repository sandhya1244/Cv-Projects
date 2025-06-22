import streamlit as st
import cv2
import numpy as np
from PIL import Image

from hand_recognition import HandTracker
from expression_converter import extract_expression_from_drawing
from gemini_solver import solve_expression
from utils import create_blank_canvas, draw_circle

# Streamlit page setup
st.set_page_config(page_title="Gesture-Controlled Math Solver", layout="centered")
st.title("🖐️ Gesture-Controlled Math Solver 🤖")

st.markdown("""
Use your finger to draw a math expression on the canvas.
The app will recognize it and solve it using Gemini AI!
""")

# Initialize hand tracker
tracker = HandTracker()

# App states
canvas = create_blank_canvas()
drawing = False
points = []

# Webcam capture
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

st.sidebar.title("✏️ Drawing Controls")
draw_mode = st.sidebar.toggle("Enable Drawing Mode", value=True)
clear_canvas = st.sidebar.button("🧹 Clear Canvas")
solve_button = st.sidebar.button("🧠 Solve Expression")

# Main app loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.error("Failed to access the webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame, landmarks = tracker.get_hand_landmarks(frame, draw=True)

    if draw_mode and landmarks:
        index_finger = landmarks[8]  # Tip of index finger
        x, y = index_finger[1], index_finger[2]
        draw_circle(canvas, x, y)
        points.append((x, y))

    if clear_canvas:
        canvas = create_blank_canvas()
        points = []

    # Combine webcam and canvas
    combined = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
    frame_placeholder.image(combined, channels="BGR", use_column_width=True)

    if solve_button:
        with st.spinner("Recognizing expression..."):
            expr = extract_expression_from_drawing(canvas)
            st.success(f"🧾 Expression detected: `{expr}`")

            with st.spinner("Solving using Gemini AI..."):
                result = solve_expression(expr)
                st.subheader("📌 Gemini AI Solution:")
                st.write(result)
        break

cap.release()
cv2.destroyAllWindows()
