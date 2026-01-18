import os
import sys
import argparse
import cv2
import numpy as np
import json
import csv
import time
import tifffile
from PIL import Image

# 2) FIX IMPORT/REPO ROOT ROBUSTNESS
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# Attempt imports from existing modules
try:
    from src.cv.pit_detector_v2 import detect_pits_v2
    # 3) CLEAN GEMINI IMPORT/CLIENT USAGE
    from src.llm.gemini_client import get_gemini_client
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please verify you are running from the repo root and 'src' directory exists.")
    sys.exit(1)

def validate_image_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
        raise ValueError(f"Unsupported format {ext} for file {path}")
    return path

def load_image(path):
    """Load image as BGR numpy array using cv2 or tifffile."""
    try:
        if path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(path)
            if img.dtype == 'uint16':
                img = (img / 256).astype('uint8')
            elif img.dtype == 'float32':
                img = (img * 255).astype('uint8')
            
            if img.ndim == 3:
                # Tifffile usually returns RGB, OpenCV needs BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("cv2.imread failed")
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

def get_crop(img, x, y, size):
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def compute_green_ratio(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return np.count_nonzero(mask) / mask.size

def call_gemini_analysis(crop_bgr, model):
    """
    Calls Gemini to classify the crop.
    Returns dict with status, confidence, evidence.
    """
    if model is None:
        return {"status": "UNCERTAIN", "confidence": 0, "evidence": "Model init failed"}
        
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    
    prompt = """
    Analyze this crop of a tree pit. 
    Determine if there is a LIVE sapling/tree, a DEAD/dried sapling, or if it is UNCLEAR.
    Ignore weeds (grass). Focus on the central pit area.
    Return JSON only:
    {
        "status": "ALIVE" | "DEAD" | "UNCERTAIN",
        "confidence": <float 0.0-1.0>,
        "evidence": "<brief explanation>"
    }
    """
    
    try:
        # 3) Robust Gemini client API usage
        if hasattr(model, 'generate_content'):
            response = model.generate_content([prompt, pil_img])
        elif hasattr(model, 'generate'):
            response = model.generate([prompt, pil_img])
        elif hasattr(model, 'predict'):
            response = model.predict([prompt, pil_img])
        else:
            return {"status": "UNCERTAIN", "confidence": 0, "evidence": "Unsupported Gemini client API"}

        text = response.text
        # 7) GEMINI JSON PARSING HARDENING
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            data = json.loads(json_str)
            if "evidence" in data and isinstance(data["evidence"], list):
                data["evidence"] = "; ".join(map(str, data["evidence"]))
            return data
        else:
            raise ValueError("No JSON object found in Gemini response text")
    except Exception as e:
        return {"status": "UNCERTAIN", "confidence": 0, "evidence": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Agriguard Product Mode Runner")
    parser.add_argument("--op1", required=True, help="Path to OP1 image (Pits)")
    parser.add_argument("--op2", required=True, help="Path to OP2 image (Planting)")
    parser.add_argument("--op3", required=True, help="Path to OP3 image (Survival)")
    parser.add_argument("--mode", default="hybrid", choices=["cv", "hybrid"], help="Analysis mode")
    parser.add_argument("--max-gemini-calls", type=int, default=50, help="Max LLM calls")
    parser.add_argument("--crop-size", type=int, default=256, help="Crop size px")
    parser.add_argument("--offset-px", type=int, default=40, help="Local search offset px")
    parser.add_argument("--alive-threshold", type=float, default=0.05, help="Green ratio threshold for ALIVE status")
    parser.add_argument("--dead-threshold", type=float, default=0.01, help="Green ratio threshold for DEAD status")
    parser.add_argument("--planted-threshold", type=float, default=0.02, help="Green ratio threshold for PLANTED status")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    op1_path = validate_image_path(args.op1)
    op2_path = validate_image_path(args.op2)
    op3_path = validate_image_path(args.op3)
    
    mode = args.mode
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = None
    
    if mode == "hybrid":
        if not gemini_key:
            print("WARNING: GEMINI_API_KEY not found. Downgrading to cv mode.")
            mode = "cv"
        else:
            try:
                gemini_model = get_gemini_client()
            except Exception as e:
                print(f"WARNING: Gemini initialization failed ({e}). Downgrading to cv mode.")
                mode = "cv"

    print(f"--- Running Agriguard User Pipeline ---")
    print(f"Mode: {mode}")
    print(f"Inputs:\n  OP1: {op1_path}\n  OP2: {op2_path}\n  OP3: {op3_path}")
    
    # 1) Setup absolute output directory deterministicly
    out_dir = os.path.join(REPO_ROOT, "user_outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading images...")
    # 1) Load OP1 image into User_OP1 just like OP2/OP3
    User_OP1 = load_image(op1_path)
    User_OP2 = load_image(op2_path)
    User_OP3 = load_image(op3_path)
    
    # 2) Robust pit detection calls
    print("Detecting pits in OP1...")
    pit_config = {
        "min_area_px": 50,
        "max_area_px": 5000,
        "blur_kernel_size": 5,
        "threshold_method": "otsu",
        "circularity_min": 0.3
    }
    
    try:
        # Try calling with the path first
        pit_results = detect_pits_v2(op1_path, config=pit_config)
    except (TypeError, Exception) as path_err:
        try:
            # Fallback to calling with the loaded image
            pit_results = detect_pits_v2(User_OP1, config=pit_config)
        except Exception as img_err:
            print(f"CRITICAL ERROR: Pit detection failed with both path and image buffer.")
            print(f"Path error: {path_err}")
            print(f"Image error: {img_err}")
            sys.exit(1)

    pits = pit_results.get("pits", [])
    print(f"Found {len(pits)} pits.")
    
    if not pits:
        # 2) Early exit robustness: Write empty outputs
        print("No pits detected in OP1. Writing empty reports.")
        
        # CSV Headers
        surv_headers = ["pit_index", "op1_x", "op1_y", "op2_best_x", "op2_best_y", "op2_best_green_ratio", "op3_best_x", "op3_best_y", "planted", "status", "method_used", "green_ratio", "confidence", "evidence"]
        dead_headers = ["pit_index", "x", "y"]
        
        with open(os.path.join(out_dir, "op3_survival.csv"), 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=surv_headers).writeheader()
        with open(os.path.join(out_dir, "dead_locations.csv"), 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=dead_headers).writeheader()
            
        summary = {
            "total_pits": 0, "total_planted": 0, "alive": 0, "dead": 0, "survival_pct": 0,
            "gemini_calls_used": 0, "runtime_seconds": round(time.time() - start_time, 2),
            "note": "No pits detected in OP1"
        }
        with open(os.path.join(out_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Outputs saved to: {os.path.abspath(out_dir)}")
        sys.exit(0)

    # 4) OP2 LOCAL SEARCH (DRIFT HANDLING)
    print("Verifying planting in OP2 with local search...")
    planted_count = 0
    scan_offsets = [
        (0, 0),
        (0, -args.offset_px), (0, args.offset_px), # N, S
        (-args.offset_px, 0), (args.offset_px, 0)  # W, E
    ]
    
    for pit in pits:
        cx, cy = pit['x'], pit['y']
        best_ratio = -1.0
        best_coords = (cx, cy)
        
        for ox, oy in scan_offsets:
            tx, ty = cx + ox, cy + oy
            crop = get_crop(User_OP2, tx, ty, args.crop_size)
            ratio = compute_green_ratio(crop)
            if ratio > best_ratio:
                best_ratio = ratio
                best_coords = (tx, ty)
        
        pit['planted'] = best_ratio >= args.planted_threshold
        pit['op2_best_green_ratio'] = float(best_ratio)
        pit['op2_best_x'] = int(best_coords[0])
        pit['op2_best_y'] = int(best_coords[1])
        if pit['planted']:
            planted_count += 1
            
    print(f"Planted pits: {planted_count}/{len(pits)}")

    print("Assessing survival in OP3 with local search...")
    survival_results = []
    dead_locations = []
    gemini_calls = 0
    alive_count = 0
    dead_count = 0
    
    for i, pit in enumerate(pits):
        cx, cy = pit['x'], pit['y']
        
        if not pit.get('planted', False):
            survival_results.append({
                "pit_index": i,
                "op1_x": cx, "op1_y": cy,
                "op2_best_x": pit.get('op2_best_x', cx),
                "op2_best_y": pit.get('op2_best_y', cy),
                "op2_best_green_ratio": pit.get('op2_best_green_ratio', 0.0),
                "op3_best_x": cx, "op3_best_y": cy,
                "planted": False,
                "status": "NOT_PLANTED",
                "method_used": "cv",
                "green_ratio": 0.0,
                "confidence": 1.0,
                "evidence": "Not planted in OP2"
            })
            continue
            
        best_ratio = -1.0
        best_coords = (cx, cy)
        best_crop = None
        
        for ox, oy in scan_offsets:
            tx, ty = cx + ox, cy + oy
            crop = get_crop(User_OP3, tx, ty, args.crop_size)
            if crop is None: continue
            ratio = compute_green_ratio(crop)
            if ratio > best_ratio:
                best_ratio = ratio
                best_coords = (tx, ty)
                best_crop = crop
        
        status = "UNCERTAIN"
        method = "cv"
        confidence = 0.0
        evidence = ""
        
        if best_ratio >= args.alive_threshold:
            status = "ALIVE"
            confidence = 1.0
            evidence = f"High green ratio {best_ratio:.3f}"
        elif best_ratio <= args.dead_threshold:
            status = "DEAD"
            confidence = 1.0
            evidence = f"Low green ratio {best_ratio:.3f}"
        else:
            if mode == "hybrid" and gemini_calls < args.max_gemini_calls:
                gem_res = call_gemini_analysis(best_crop, gemini_model)
                status = gem_res.get("status", "UNCERTAIN").upper()
                confidence = gem_res.get("confidence", 0)
                evidence = str(gem_res.get("evidence", ""))
                method = "hybrid"
                gemini_calls += 1
            else:
                status = "UNCERTAIN"
                evidence = f"Ambiguous ratio {best_ratio:.3f}"
        
        if status == "ALIVE": alive_count += 1
        elif status == "DEAD": dead_count += 1
        
        if status == "DEAD":
            dead_locations.append({"pit_index": i, "x": best_coords[0], "y": best_coords[1]})
            
        survival_results.append({
            "pit_index": i,
            "op1_x": cx, "op1_y": cy,
            "op2_best_x": pit['op2_best_x'],
            "op2_best_y": pit['op2_best_y'],
            "op2_best_green_ratio": pit['op2_best_green_ratio'],
            "op3_best_x": best_coords[0],
            "op3_best_y": best_coords[1],
            "planted": True,
            "status": status,
            "method_used": method,
            "green_ratio": float(best_ratio),
            "confidence": confidence,
            "evidence": evidence
        })
        
    # Final output writing
    csv_path = os.path.join(out_dir, "op3_survival.csv")
    with open(csv_path, 'w', newline='') as f:
        headers = [
            "pit_index", "op1_x", "op1_y", 
            "op2_best_x", "op2_best_y", "op2_best_green_ratio",
            "op3_best_x", "op3_best_y",
            "planted", "status", "method_used", "green_ratio", "confidence", "evidence"
        ]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(survival_results)
        
    dead_path = os.path.join(out_dir, "dead_locations.csv")
    with open(dead_path, 'w', newline='') as f:
        headers = ["pit_index", "x", "y"]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dead_locations)
        
    survival_pct = (alive_count / planted_count * 100) if planted_count > 0 else 0
    summary = {
        "total_pits": len(pits),
        "total_planted": planted_count,
        "alive": alive_count,
        "dead": dead_count,
        "survival_pct": round(survival_pct, 2),
        "gemini_calls_used": gemini_calls,
        "runtime_seconds": round(time.time() - start_time, 2)
    }
    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n--- Analysis Complete ---")
    print(f"Total Pits:    {summary['total_pits']}")
    print(f"Total Planted: {summary['total_planted']}")
    print(f"Alive:         {summary['alive']}")
    print(f"Dead:          {summary['dead']}")
    print(f"Survival %:    {summary['survival_pct']}%")
    print(f"Gemini Calls:  {summary['gemini_calls_used']}")
    print(f"Outputs saved to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
