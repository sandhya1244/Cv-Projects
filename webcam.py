import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('AirPointer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

###################################STEP-2#################################################


import pyautogui

screen_width, screen_height = pyautogui.size()

if results.multi_hand_landmarks:
    hand_landmarks = results.multi_hand_landmarks[0]
    index_tip = hand_landmarks.landmark[8]  # Index finger tip
    x = int(index_tip.x * screen_width)     # Scale to screen width
    y = int(index_tip.y * screen_height)    # Scale to screen height
    pyautogui.moveTo(x, y)


###################################STEP-3#################################################



thumb_tip = hand_landmarks.landmark[4]
dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
if dist < 0.05:  # Threshold for pinch
    pyautogui.click()