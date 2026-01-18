import cv2
import numpy as np

def classify_heuristic(image_path):
    """
    Tier 1 classification based on simple visual features.
    Returns: (stage, confidence, reason)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "UNKNOWN", 0.0, "Could not read preview"

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Green Pixel Ratio (Vegetation)
        # Green range in HSV roughly (35, 40, 40) to (85, 255, 255)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
        green_ratio = np.sum(mask_green > 0) / (img.shape[0] * img.shape[1])

        # 2. Dark Spot Density (Pits)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold for dark spots
        _, mask_dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.sum(mask_dark > 0) / (img.shape[0] * img.shape[1])
        
        # 3. Simple Logic
        # OP1: Mostly bare soil (low green), visible pits (high dark spots? or just soil color)
        # OP2: Patchy weak green
        # OP3: Higher vegetation, rings
        
        reason = f"Green: {green_ratio:.2f}, Dark: {dark_ratio:.2f}"
        
        if green_ratio > 0.30:
            return "OP3_POST_SOWING", 0.8, reason + " (High vegetation)"
        elif green_ratio < 0.05:
            if dark_ratio > 0.05: # Pits might be dark shadows
                 return "OP1_POST_PITTING", 0.7, reason + " (Low veg, potential pits)"
            else:
                 return "OP1_POST_PITTING", 0.6, reason + " (Very low veg)"
        else:
            return "OP2_POST_PLANTING", 0.6, reason + " (Moderate vegetation)"

        # Fallback
        return "UNKNOWN", 0.1, reason

    except Exception as e:
        return "UNKNOWN", 0.0, f"Error: {e}"
