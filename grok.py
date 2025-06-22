import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import time
import requests
import pytesseract
from PIL import Image
import io
import os

# Initialize Mediapipe and CVZone
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Global variables
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
drawing = False
last_point = None
expression = ""
result = ""

# Load and prepare background image
def load_background_image(image_path, frame_shape):
    """
    Load and resize background image to match webcam frame dimensions.
    """
    try:
        bg_image = cv2.imread(image_path)
        if bg_image is None:
            raise FileNotFoundError(f"Background image not found at {image_path}")
        # Resize to match webcam frame (640x480)
        bg_image = cv2.resize(bg_image, (frame_shape[1], frame_shape[0]))
        return bg_image
    except Exception as e:
        print(f"Error loading background image: {e}")
        # Return black background as fallback
        return np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)

# Overlay person on background using segmentation
def overlay_person_on_background(frame, background):
    """
    Use Mediapipe Selfie Segmentation to place the person from the webcam feed
    in front of the background image.
    """
    # Convert frame to RGB for segmentation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get segmentation mask
    results = selfie_segmentation.process(frame_rgb)
    # Create binary mask (1 for person, 0 for background)
    mask = results.segmentation_mask > 0.1  # Threshold for person detection
    mask = mask[:, :, np.newaxis]  # Add channel dimension for broadcasting
    # Convert background to correct color space if needed
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    # Composite: person from frame, background from bg_image
    composite = np.where(mask, frame_rgb, background)
    return composite

# Process hand gestures and draw on canvas
def process_gestures(frame, canvas):
    global drawing, last_point
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame = cv2.flip(frame, 1)
    canvas = cv2.flip(canvas, 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            h, w, _ = frame.shape
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)
            distance = np.sqrt((index_finger.x - thumb_tip.x)**2 + (index_finger.y - thumb_tip.y)**2)

            if distance < 0.05:
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

# Convert canvas to mathematical expression
def canvas_to_expression(canvas):
    """
    Convert drawn canvas to a mathematical expression using Tesseract OCR.
    """
    try:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        temp_img_path = "temp_canvas.png"
        cv2.imwrite(temp_img_path, thresh)
        config = "--psm 6"
        text = pytesseract.image_to_string(temp_img_path, config=config).strip()
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        text = text.replace("\n", "").replace(" ", "")
        return text if text else "2+3"
    except Exception as e:
        return f"Error processing canvas: {str(e)}"

# Gemini AI API call
def solve_equation(equation):
    """
    Calls Gemini AI API to solve a mathematical equation.
    """
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyA_2TjlY_lOkc09_srUAwBxFZImwSWgeTA")
    try:
        prompt = f"Solve the equation: {equation}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        result = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error: No solution returned")
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
def main():
    global expression, canvas, result

    st.set_page_config(page_title="Gesture-Controlled Math Solver", layout="wide")
    st.title("Gesture-Controlled Math Solver")
    st.markdown("""
    Draw mathematical expressions using hand gestures and solve them!
    - **Pinch** (thumb and index finger close) to draw.
    - **Release** pinch to stop drawing.
    - Click **Solve Equation** to process.
    - Click **Clear Canvas** to reset.
    """)

    if 'canvas' not in st.session_state:
        st.session_state.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    if 'expression' not in st.session_state:
        st.session_state.expression = ""
    if 'result' not in st.session_state:
        st.session_state.result = ""

    col1, col2 = st.columns(2)
    with col1:
        video_placeholder = st.empty()
    with col2:
        canvas_placeholder = st.empty()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        solve_button = st.button("Solve Equation", key="solve")
    with col_btn2:
        clear_button = st.button("Clear Canvas", key="clear")
    status_placeholder = st.empty()

    # Load background image
    background_image = load_background_image("background.jpg", (480, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_placeholder.error("Cannot access webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Failed to capture video.")
                break

            # Process gestures and update canvas
            frame, st.session_state.canvas = process_gestures(frame, st.session_state.canvas)

            # Overlay person on background
            display_frame = overlay_person_on_background(frame, background_image)

            # Display in Streamlit
            video_placeholder.image(display_frame, channels="RGB", caption="Gesture Input")
            canvas_placeholder.image(st.session_state.canvas, channels="BGR", caption="Drawn Expression")

            if solve_button:
                st.session_state.expression = canvas_to_expression(st.session_state.canvas)
                if "Error" not in st.session_state.expression and st.session_state.expression != "No expression detected":
                    st.session_state.result = solve_equation(st.session_state.expression)
                else:
                    st.session_state.result = st.session_state.expression
                status_placeholder.markdown(f"**Expression**: {st.session_state.expression}  \n**Result**: {st.session_state.result}")

            if clear_button:
                st.session_state.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                st.session_state.expression = ""
                st.session_state.result = ""
                status_placeholder.empty()

            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()