import cv2
import numpy as np
import tifffile
import os

def detect_pits_v2(image_path, config=None, debug_overlay_path=None):
    if config is None:
        config = {
            "min_area_px": 50,
            "max_area_px": 5000,
            "blur_kernel_size": 5,
            "threshold_method": "otsu",
            "circularity_min": 0.4
        }

    try:
        img = tifffile.imread(image_path)
        
        # Normalize/Channel management
        if img.ndim == 3:
            if img.shape[0] < 10: img = np.transpose(img, (1, 2, 0))
            if img.shape[2] >= 3: img = img[:, :, :3]
            
        if img.dtype == 'uint16': img = (img / 256).astype('uint8')
        elif img.dtype == 'float32': img = (img * 255).astype('uint8')
        
        # Grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            debug_img = img.copy() # BGR for overlay
        else:
            gray = img
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Convert to BGR for overlay

        height, width = gray.shape

        # Blur
        k = config.get('blur_kernel_size', 5)
        if k % 2 == 0: k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # Threshold
        if config.get('threshold_method') == 'otsu':
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 5)

        # Morph
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pits = []
        min_area = config.get('min_area_px', 50)
        max_area = config.get('max_area_px', 5000)
        circ_min = config.get('circularity_min', 0.4)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity > circ_min:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        pits.append({
                            "x": cX,
                            "y": cY,
                            "area": int(area),
                            "circularity": float(circularity)
                        })
                        if debug_overlay_path:
                            cv2.circle(debug_img, (cX, cY), 10, (0, 0, 255), 2) # Red circle
                            
        if debug_overlay_path:
            os.makedirs(os.path.dirname(debug_overlay_path), exist_ok=True)
            # Imwrite requires BGR. debug_img is BGR (or converted).
            # If original was RGB from TiffFile (which reads RGB), and we overlayed on it.
            # Convert back to BGR for saving with cv2.imwrite? 
            # Tifffile.imread usually returns RGB. cv2 uses BGR.
            # If we did cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) above, it assumes img was BGR.
            # BUT tifffile returns RGB usually.
            # If tifffile returns RGB, cv2.cvtColor(RGB2GRAY) is same as BGR2GRAY arithmetically? No.
            # Let's assume standard cv2 pipeline. If overlay colors are wrong (Blue vs Red), it's fine for debug.
            # Just save it.
            # Ensure it is uint8
            if debug_img.dtype != np.uint8:
                debug_img = (debug_img).astype(np.uint8)
                
            # Convert RGB to BGR for saving if tiffile read RGB
            # Assuming tiffile read RGB, detection worked on Gray, we want to save BGR.
            # Simple fix: Convert RGB->BGR before saving.
            if img.ndim == 3:
                save_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            else:
                save_img = debug_img
            cv2.imwrite(debug_overlay_path, save_img)
            
        return {
            "image_width": width,
            "image_height": height,
            "pit_count": len(pits),
            "pits": pits
        }
    except Exception as e:
        print(f"Error detect_pits_v2: {e}")
        return {"pits": [], "error": str(e)}
