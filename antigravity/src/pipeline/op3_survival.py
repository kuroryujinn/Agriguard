import os
import cv2
import numpy as np
import pandas as pd
import json
import tifffile
from src.llm.gemini_client import classify_crop, get_gemini_client
from src.llm.op3_survival_prompts import OP3_SURVIVAL_PROMPT

def analyze_survival(
    field,
    op2_planting_csv,
    op3_manifest_path,
    output_dir,
    config=None, # dict with thresholds
    driver_download_fn=None, # function to download file given item
    mode="hybrid", # cv, hybrid, gemini
    max_gemini_calls=200,
    debug_overlay=False
):
    if config is None:
        config = {
            "crop_size_px": 256,
            "offset_px": 40,
            "alive_threshold": 0.05,
            "dead_threshold": 0.01
        }
        
    crop_size = config.get("crop_size_px", 256)
    offset_px = config.get("offset_px", 40)
    alive_thresh = config.get("alive_threshold", 0.05)
    dead_thresh = config.get("dead_threshold", 0.01)

    # Load Data
    op2_df = pd.read_csv(op2_planting_csv)
    # Filter PLANTED
    planted_df = op2_df[op2_df['planted'] == True]
    
    with open(op3_manifest_path, 'r') as f:
        op3_items = json.load(f)
        
    # Helpers
    hsv_lower = np.array([25, 40, 40])
    hsv_upper = np.array([95, 255, 255])
    
    def get_green_ratio(img_hsv):
        mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
        return np.count_nonzero(mask) / mask.size

    # Prepare Outputs
    v3_dir = output_dir
    os.makedirs(v3_dir, exist_ok=True)
    overlay_dir = os.path.join(v3_dir, "overlays", field) if debug_overlay else None
    if overlay_dir: os.makedirs(overlay_dir, exist_ok=True)

    results = []
    gemini_calls = 0
    
    # Pre-init Gemini if needed
    gemini_model = None
    if mode in ["hybrid", "gemini"]:
        try:
            gemini_model = get_gemini_client()
        except:
            print("Warning: Gemini client init failed. Falling back to CV.")
            mode = "cv"

    # Map OP3 images by name (assuming alignment)
    # Filter OP3 items that are sampled
    op3_map = {x['name']: x for x in op3_items if x.get('sampled') == True and x['stage'] == 'op3_post_sowing'}
    
    print(f"Analyzing survival for {len(planted_df)} planted pits across {len(op3_map)} OP3 images...")
    
    # Iterate Images (to load once)
    unique_images = planted_df['op2_image_name'].unique()
    
    for img_name in unique_images:
        img_item = op3_map.get(img_name)
        if not img_item:
            # Maybe OP3 name differs? For hackathon assume same.
            continue
            
        # Download
        img_path = driver_download_fn(img_item)
        
        # Load Image
        try:
            img = tifffile.imread(img_path)
            if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Enforce BGR/RGB structure
            if img.ndim == 3 and img.shape[0] < 10: img = np.transpose(img, (1, 2, 0)) # Channel last
            if img.shape[2] >= 3: img = img[:, :, :3] # Remove alpha
            
            # Ensure uint8
            if img.dtype == 'uint16': img = (img / 256).astype('uint8')
            elif img.dtype == 'float32': img = (img * 255).astype('uint8')
            
            # For HSV
            if img.ndim == 3:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Assuming RGB from tifffile
            else:
                hsv_img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        h, w = img.shape[:2]
        img_pits = planted_df[planted_df['op2_image_name'] == img_name]
        
        debug_img = img.copy() if debug_overlay else None

        for _, pit in img_pits.iterrows():
            px, py = int(pit['x']), int(pit['y'])
            
            # LOCAL SEARCH
            offsets = [(0,0), (-offset_px, 0), (offset_px, 0), (0, -offset_px), (0, offset_px)]
            best_crop = None
            best_score = -1.0
            best_xy = (px, py)
            
            half = crop_size // 2
            
            for dx, dy in offsets:
                nx, ny = px + dx, py + dy
                x1, y1 = max(0, nx-half), max(0, ny-half)
                x2, y2 = min(w, nx+half), min(h, ny+half)
                
                if x2<=x1 or y2<=y1: continue
                
                crop_h = hsv_img[y1:y2, x1:x2]
                score = get_green_ratio(crop_h)
                
                if score > best_score:
                    best_score = score
                    best_xy = (nx, ny)
                    best_crop = img[y1:y2, x1:x2] # RGB Crop for Gemini
            
            # DECISION
            status = "UNCERTAIN"
            method = "cv"
            confidence = 0.0
            evidence = []
            
            if best_score >= alive_thresh:
                status = "ALIVE"
            elif best_score <= dead_thresh:
                status = "DEAD"
            else:
                status = "UNCERTAIN"
                
            # GEMINI override
            if (mode == "gemini") or (mode == "hybrid" and status == "UNCERTAIN" and gemini_calls < max_gemini_calls):
                if best_crop is not None and best_crop.size > 0:
                    # Save temp crop
                    temp_crop_path = os.path.join(v3_dir, "temp_crop.jpg")
                    # Save as BGR for CV2
                    cv2.imwrite(temp_crop_path, cv2.cvtColor(best_crop, cv2.COLOR_RGB2BGR))
                    
                    g_res = classify_crop(temp_crop_path, OP3_SURVIVAL_PROMPT, model=gemini_model)
                    
                    if g_res.get("status") in ["ALIVE", "DEAD"]:
                        status = g_res["status"]
                        confidence = g_res.get("confidence", 0)
                        evidence = g_res.get("evidence", [])
                        method = "gemini"
                        gemini_calls += 1
                    
                    # Cleanup
                    try: os.remove(temp_crop_path)
                    except: pass

            results.append({
                "field": field,
                "pit_index": pit['pit_index'],
                "x": px, "y": py,
                "planted": True,
                "status": status,
                "method_used": method,
                "green_ratio": float(best_score),
                "gemini_confidence": confidence,
                "evidence": ";".join(evidence) if isinstance(evidence, list) else str(evidence)
            })
            
            # Overlay
            if debug_overlay:
                color = (0, 255, 0) if status == "ALIVE" else (0, 0, 255) if status == "DEAD" else (0, 255, 255)
                bx, by = best_xy
                cv2.rectangle(debug_img, (bx-half, by-half), (bx+half, by+half), color, 2)
                
        if debug_overlay and debug_img is not None:
            ov_path = os.path.join(overlay_dir, f"{img_name}.png")
            cv2.imwrite(ov_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    # SAVE RESULTS
    # 1. Full CSV
    out_csv = os.path.join(v3_dir, f"{field}.op3_survival.csv")
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)
    
    # 2. Dead CSV
    dead_df = res_df[res_df['status'] == 'DEAD']
    dead_csv = os.path.join(v3_dir, f"{field}.dead_locations.csv")
    dead_df.to_csv(dead_csv, index=False)
    
    # 3. Summary JSON
    summary = {
        "total_analyzed": len(res_df),
        "alive": len(res_df[res_df['status'] == 'ALIVE']),
        "dead": len(dead_df),
        "uncertain": len(res_df[res_df['status'] == 'UNCERTAIN']),
        "survival_pct": (len(res_df[res_df['status'] == 'ALIVE']) / len(res_df) * 100) if len(res_df) > 0 else 0,
        "gemini_calls_made": gemini_calls
    }
    sum_path = os.path.join(v3_dir, f"{field}.survival_summary.json")
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
