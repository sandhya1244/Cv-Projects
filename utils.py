import cv2
import numpy as np

def capture_hand_image(window_name="Capture", width=640, height=480):
    """
    Opens the webcam and captures a single frame when 'c' is pressed.
    Returns:
        frame (np.array): Captured image frame from webcam.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    print("📷 Press 'c' to capture image or 'q' to quit.")
    frame = None

    while True:
        success, img = cap.read()
        if not success:
            print("❌ Failed to access camera")
            break

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('c'):
            frame = img.copy()
            print("✅ Image captured!")
            break
        elif key & 0xFF == ord('q'):
            print("⚠️ Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

def create_blank_canvas(width=640, height=480, color=(255, 255, 255)):
    """
    Creates a blank canvas for finger drawing or writing.
    Returns:
        canvas (np.array): White image to draw on.
    """
    return np.ones((height, width, 3), dtype=np.uint8) * 255

def draw_circle(canvas, x, y, radius=8, color=(0, 0, 0)):
    """
    Draws a small circle on the canvas (useful for finger drawing).
    """
    cv2.circle(canvas, (x, y), radius, color, -1)
