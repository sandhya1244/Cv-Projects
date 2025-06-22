import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import time
import requests  # For Gemini AI API (placeholder)

# Initialize Mediapipe and CVZone
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
drawing = False
last_point = None
expression = ""

# Placeholder for Gemini AI API call
def solve_equation(equation):
    """
    Placeholder for Gemini AI API to solve equations.
    Replace with actual API call.
    """
    try:
        # Simulated response (replace with actual API call)
        result = eval(equation, {"__builtins__": {}}, {"sin": np.sin, "cos": np.cos, "tan": np.tan})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Process hand gestures and draw
def process_gestures(frame, canvas):
    global drawing, last_point, expression
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame = cv2.flip(frame, 1)
    canvas = cv2.flip(canvas, 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)
            thumb_tip = hand_landmarks.landmark[4]
            distance = np.sqrt((index_finger.x - thumb_tip.x)**2 + (index_finger.y - thumb_tip.y)**2)

            if distance < 0.05:  # Pinch to draw
                if not drawing:
                    drawing = True
                    last_point = (cx, cy)
            else:
                drawing = False
                last_point = None

            if drawing and last_point:
                cv2.line(canvas, last_point, (cx, cy), (255, 255, 255), 5)
                last_point = (cx, cy)

    return frame, canvas

# Convert canvas to expression (placeholder)
def canvas_to_expression(canvas):
    """
    Convert drawn canvas to math expression (simplified).
    Replace with OCR model for production.
    """
    return expression if expression else "2+3"  # Placeholder

# Streamlit UI
def main():
    global expression, canvas
    st.title("Gesture-Controlled Math Solver")
    st.write("Draw math expressions with hand gestures!")

    if 'canvas' not in st.session_state:
        st.session_state.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    if 'expression' not in st.session_state:
        st.session_state.expression = ""
    if 'result' not in st.session_state:
        st.session_state.result = ""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
        return

    video_placeholder = st.empty()
    solve_button = st.button("Solve Equation")
    clear_button = st.button("Clear Canvas")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        frame, st.session_state.canvas = process_gestures(frame, st.session_state.canvas)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", caption="Gesture Input")

        if solve_button:
            st.session_state.expression = canvas_to_expression(st.session_state.canvas)
            st.session_state.result = solve_equation(st.session_state.expression)
            st.write(f"Expression: {st.session_state.expression}")
            st.write(f"Result: {st.session_state.result}")

        if clear_button:
            st.session_state.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            st.session_state.expression = ""
            st.session_state.result = ""

        st.image(st.session_state.canvas, channels="BGR", caption="Drawn Expression")
        time.sleep(0.03)

    cap.release()

if __name__ == "__main__":
    main()
    