import cv2
import numpy as np
import tifffile
import os

def check_planting(op2_image_path, pits, config=None, debug_path=None):
    """
    Checks planting status for a list of pits against an OP2 image.
    pits: list of dicts {x, y, ...}
    """
    if config is None:
        config = {
            "crop_size_px": 256,
            "planted_threshold": 0.02,
            "hsv_green_lower": [25, 40, 40],
            "hsv_green_upper": [95, 255, 255]
        }
        
    try:
        # Load OP2 image
        img = tifffile.imread(op2_image_path)
        
        # Normalize/channel fix
        if img.ndim == 3:
            if img.shape[0] < 10: img = np.transpose(img, (1, 2, 0))
            if img.shape[2] >= 3: img = img[:, :, :3]
            
        if img.dtype == 'uint16': img = (img / 256).astype('uint8')
        elif img.dtype == 'float32': img = (img * 255).astype('uint8')
        
        # Ensure BGR for OpenCV
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        h, w = img.shape[:2]
        crop_half = config.get('crop_size_px', 256) // 2
        
        # HSV conversion for whole image might be faster than per-crop?
        # But for strictly correct logical cropping (handling padding), let's crop first then convert? 
        # Actually converting whole image once is better.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower = np.array(config.get('hsv_green_lower', [25, 40, 40]))
        upper = np.array(config.get('hsv_green_upper', [95, 255, 255]))
        
        results = []
        
        for i, pit in enumerate(pits):
            cx, cy = int(pit['x']), int(pit['y'])
            
            x1 = max(0, cx - crop_half)
            y1 = max(0, cy - crop_half)
            x2 = min(w, cx + crop_half)
            y2 = min(h, cy + crop_half)
            
            if x2 <= x1 or y2 <= y1:
                results.append({"pit_index": i, "x": cx, "y": cy, "green_ratio": 0.0, "planted": False, "error": "OutOfBounds"})
                continue
                
            crop_hsv = hsv[y1:y2, x1:x2]
            mask = cv2.inRange(crop_hsv, lower, upper)
            
            total_px = mask.size
            green_px = np.count_nonzero(mask)
            ratio = green_px / total_px if total_px > 0 else 0
            
            is_planted = ratio >= config.get('planted_threshold', 0.02)
            
            results.append({
                "pit_index": i, 
                "x": cx, 
                "y": cy, 
                "green_ratio": float(ratio), 
                "planted": bool(is_planted)
            })

            # Debug Overlay
            if debug_path:
                color = (0, 255, 0) if is_planted else (0, 0, 255)
                # Draw square or circle
                cv2.circle(img, (cx, cy), 10, color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if debug_path:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            # Save overlay (convert BGR to RGB? No, detection logic assumes BGR load if using cv2.imread, but tifffile reads RGB usually?)
            # Tifffile reads as is. If we constructed BGR for drawing, we should save it.
            # cv2.imwrite expects BGR.
            # Convert back if original led to RGB.
            # Simplification: Save as is.
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert default to BGR for saving
            cv2.imwrite(debug_path, img_bgr)

        return results
        
    except Exception as e:
        print(f"OP2 Check failed for {op2_image_path}: {e}")
        return []
