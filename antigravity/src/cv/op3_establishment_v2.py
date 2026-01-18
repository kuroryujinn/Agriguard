import cv2
import numpy as np
import tifffile
import os
from src.cv.op2_planting_v2 import check_planting_v2

# OP3 logic is similar to OP2 but applies to OP3 images and might use different thresholds/metrics.
# We can reuse logic but wrap it for clarity and config separation.

def check_establishment_v2(image_path, planted_pits, config=None, debug_overlay_path=None):
    if config is None:
        config = {
            "crop_size_px": 256,
            "establishment_threshold": 0.03, # Higher threshold for strong establishment
            "hsv_green_lower": [25, 40, 40],
            "hsv_green_upper": [95, 255, 255]
        }
        
    # Map config keys to reuse generic planting logic
    base_config = {
        "crop_size_px": config.get("crop_size_px", 256),
        "planted_threshold": config.get("establishment_threshold", 0.03),
        "hsv_green_lower": config.get("hsv_green_lower"),
        "hsv_green_upper": config.get("hsv_green_upper")
    }
    
    # We can reuse the planting checker because "Establishment" is essentially "Is there a bigger plant?"
    # The filtering of 'planted_pits' happens before calling this function (caller responsibility).
    
    results = check_planting_v2(image_path, planted_pits, config=base_config, debug_overlay_path=debug_overlay_path)
    
    # Rename keys for output clarity
    final_results = []
    for r in results:
        r['established'] = r.pop('planted')
        final_results.append(r)
        
    return final_results
