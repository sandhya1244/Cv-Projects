import cv2
import numpy as np
import pytesseract
import os

# Optional: Configure tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """Convert to grayscale and apply thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_expression_from_drawing(drawn_image):
    """
    Given an image of a hand-drawn equation, extract the math expression using OCR.
    Returns a string like "3+2".
    """
    processed_img = preprocess_image(drawn_image)

    # Use pytesseract to extract expression
    expression = pytesseract.image_to_string(
        processed_img,
        config='--psm 7 -c tessedit_char_whitelist=0123456789+-*/=()'
    )
    expression = expression.strip().replace(" ", "")
    
    return expression

def convert_gesture_to_expression(landmarks):
    """
    Placeholder for actual ML-based gesture recognition.
    This function would ideally use the landmark patterns to predict digits/operators.
    """
    # Example: Hardcoded for demo/testing
    return "3+2"

# Optional utility if you want to use drawing interface instead of gesture landmarks
def save_drawing_to_file(image, filename="drawing.png"):
    cv2.imwrite(filename, image)
    return os.path.abspath(filename)
