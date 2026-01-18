import cv2
import numpy as np
import tifffile

def detect_pits(image_path, config=None):
    """
    Detects pits in an image.
    config: dict with keys min_area_px, max_area_px, blur_kernel_size, etc.
    Returns: dict with metadata and 'pits' list.
    """
    if config is None:
        config = {
            "min_area_px": 50,
            "max_area_px": 5000,
            "blur_kernel_size": 5,
            "threshold_method": "otsu",
            "circularity_min": 0.4
        }

    try:
        # Load logic specific to TIF or general image
        img = tifffile.imread(image_path)
        
        # Handle channels/dimensions
        if img.ndim == 3:
             # (C, H, W) -> (H, W, C)
            if img.shape[0] < 10:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] >= 3:
                img = img[:, :, :3]
        
        # Normalize to 8-bit for OpenCV
        if img.dtype == 'uint16':
            img = (img / 256).astype('uint8')
        elif img.dtype == 'float32':
             img = (img * 255).astype('uint8')

        # 1. Grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img # Already gray?
            
        height, width = gray.shape

        # 2. Blur
        k = config.get('blur_kernel_size', 5)
        if k % 2 == 0: k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # 3. Threshold (Pits are dark holes -> Invert logic if needed?)
        # Pits: DARK spots on LIGHT soil.
        # Threshold: Binary Inverted
        if config.get('threshold_method') == 'otsu':
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 5)

        # 4. Morphology
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 5. Contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pits = []
        min_area = config.get('min_area_px', 50)
        max_area = config.get('max_area_px', 5000)
        circ_min = config.get('circularity_min', 0.4)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                # Circularity check
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

        return {
            "image_width": width,
            "image_height": height,
            "pit_count": len(pits),
            "pits": pits
        }

    except Exception as e:
        print(f"Pit detection failed for {image_path}: {e}")
        return {"error": str(e), "pits": []}
