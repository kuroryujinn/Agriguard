from src.stage.heuristic import classify_heuristic
from src.stage.gemini import classify_gemini

def classify_image(preview_path, mode='heuristic'):
    """
    Orchestrates classification based on mode.
    Modes: heuristic, gemini, hybrid
    """
    
    stage, conf, reason = "UNKNOWN", 0.0, "Init"
    method = "none"

    # Tier 1
    if mode in ['heuristic', 'hybrid']:
        stage, conf, reason = classify_heuristic(preview_path)
        method = "heuristic"
        
        # If hybrid and low confidence, promote to Gemini
        if mode == 'hybrid' and conf < 0.75: # Threshold for "borderline"
            g_stage, g_conf, g_reason = classify_gemini(preview_path)
            if g_conf > conf:
                stage = g_stage
                conf = g_conf
                reason = g_reason
                method = "gemini_fallback"
    
    elif mode == 'gemini':
        stage, conf, reason = classify_gemini(preview_path)
        method = "gemini"

    return stage, conf, reason, method
