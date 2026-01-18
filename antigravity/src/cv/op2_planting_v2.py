import cv2
import numpy as np
import tifffile
import os

def check_planting_v2(image_path, pits, config=None, debug_overlay_path=None):
    if config is None:
        config = {
            "crop_size_px": 256,
            "planted_threshold": 0.02, # Lower threshold for noise
            "hsv_green_lower": [25, 40, 40],
            "hsv_green_upper": [95, 255, 255]
        }
    
    try:
        img = tifffile.imread(image_path)
        
        # Norm
        if img.ndim == 3:
            if img.shape[0] < 10: img = np.transpose(img, (1, 2, 0))
            if img.shape[2] >= 3: img = img[:, :, :3]
        if img.dtype == 'uint16': img = (img / 256).astype('uint8')
        elif img.dtype == 'float32': img = (img * 255).astype('uint8')
        
        # Ensure BGR for OpenCV
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Use RGB for processing if preferred, but OpenCV standard is BGR.
        # Let's assume input is RGB (tifffile default).
        # Convert RGB to HSV directly?
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Assuming RGB input from Tifffile
        
        h, w = img.shape[:2]
        crop_half = config.get('crop_size_px', 256) // 2
        
        lower = np.array(config.get('hsv_green_lower', [25, 40, 40]))
        upper = np.array(config.get('hsv_green_upper', [95, 255, 255]))
        
        results = []
        
        debug_img = img.copy() if debug_overlay_path else None
        
        for i, pit in enumerate(pits):
            cx, cy = int(pit['x']), int(pit['y'])
            
            x1 = max(0, cx - crop_half)
            y1 = max(0, cy - crop_half)
            x2 = min(w, cx + crop_half)
            y2 = min(h, cy + crop_half)
            
            if x2 <= x1 or y2 <= y1:
                 results.append({"pit_index": i, "x": cx, "y": cy, "green_ratio": 0.0, "planted": False})
                 continue
                 
            crop_hsv = hsv[y1:y2, x1:x2]
            mask = cv2.inRange(crop_hsv, lower, upper)
            
            total = mask.size
            green = np.count_nonzero(mask)
            ratio = green / total if total > 0 else 0
            
            is_planted = ratio >= config.get('planted_threshold', 0.02)
            
            results.append({
                "pit_index": i, "x": cx, "y": cy, 
                "green_ratio": float(ratio), 
                "planted": bool(is_planted)
            })
            
            if debug_img is not None:
                color = (0, 255, 0) if is_planted else (255, 0, 0) # Green (Planted) or Red (Empty)
                # Note: debug_img is RGB because we copied 'img' which is RGB
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                cv2.circle(debug_img, (cx, cy), 5, color, -1)

        if debug_overlay_path:
            os.makedirs(os.path.dirname(debug_overlay_path), exist_ok=True)
            # Save BGR
            save_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(debug_overlay_path, save_img)
            
        return results

    except Exception as e:
        print(f"Error check_planting_v2: {e}")
        return []
