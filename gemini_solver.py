import google.generativeai as genai
import os

# You can set your Gemini API key here or load it from an environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA_2TjlY_lOkc09_srUAwBxFZImwSWgeTA")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def solve_expression(expression):
    """
    Sends a math expression to Gemini and returns the solution.
    Args:
        expression (str): Math expression like '3+2*4'
    Returns:
        str: Result/solution provided by Gemini
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"Solve this math expression step by step and give the final answer: {expression}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error solving expression: {str(e)}"
