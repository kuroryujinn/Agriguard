import os
import json
import google.generativeai as genai

def classify_gemini(image_path):
    """
    Tier 2 classification using Gemini Vision.
    Returns: (stage, confidence, reason)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "UNKNOWN", 0.0, "Missing GEMINI_API_KEY"
    
    genai.configure(api_key=api_key)
    
    # Upload/prepare file
    # For speed, we can upload the small preview file directly or use PIL image
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash") # Use flash for speed/cost
        
        sample_file = genai.upload_file(path=image_path, display_name="Aerial Sample")
        
        prompt = """
        Analyze this aerial image of an agricultural field.
        Classify it into exactly one of these stages:
        - OP1_POST_PITTING (Mostly bare soil, visible circular pits/holes, very low vegetation)
        - OP2_POST_PLANTING (Small saplings barely visible, patchy weak green, pits less dominant)
        - OP3_POST_SOWING (Clear soil circles around saplings, brown ring + green center pattern, higher vegetation)
        - UNKNOWN (Unreadable or does not fit)
        
        Return stricly VALID JSON:
        {
          "stage": "ENUM_VALUE",
          "confidence": <integer 0-100>,
          "reason": "<short string>"
        }
        """
        
        response = model.generate_content([sample_file, prompt])
        
        # Parse JSON
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        return data.get("stage", "UNKNOWN"), float(data.get("confidence", 0))/100.0, data.get("reason", "Gemini")
        
    except Exception as e:
        return "UNKNOWN", 0.0, f"Gemini Error: {e}"
