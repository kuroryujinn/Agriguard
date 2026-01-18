import os
import google.generativeai as genai
import json
import time

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def classify_crop(image_path, prompt, model=None):
    if model is None:
        try:
            model = get_gemini_client()
        except ValueError as e:
            return {"status": "UNCERTAIN", "confidence": 0, "evidence": ["API Key Missing"], "error": str(e)}

    # Load image
    # Gemini accepts paths or PIL images.
    # We will upload or pass data? 'gemini-1.5-flash' supports image data inline for small images.
    
    try:
        from PIL import Image
        img = Image.open(image_path)
    except Exception as e:
        return {"status": "UNCERTAIN", "confidence": 0, "evidence": ["Image Load Error"], "error": str(e)}

    try:
        # Generate content
        response = model.generate_content([prompt, img])
        
        # Parse JSON
        text = response.text
        # Cleanup markdown json
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        data = json.loads(text)
        return data
        
    except Exception as e:
        return {"status": "UNCERTAIN", "confidence": 0, "evidence": ["LLM Error"], "error": str(e)}
