import os
import sys
import json

def find_repo_root():
    """
    Robustly locate REPO_ROOT by checking for:
    1. 'config/datasets.json'
    2. 'src' AND 'data' directories (strong signal for repo root)
    3. fallback to script directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.dirname(script_dir)
    ]
    
    for c in candidates:
        # Check for config
        if os.path.exists(os.path.join(c, "config", "datasets.json")):
            return c
        # Check for structural markers (src + data)
        if os.path.exists(os.path.join(c, "src")) and os.path.exists(os.path.join(c, "data")):
            return c
            
    return script_dir # Default to script dir if nothing else found

REPO_ROOT = find_repo_root()
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "datasets.json")
CREDENTIALS_PATH = os.path.join(REPO_ROOT, "credentials.json")

TEMPLATE_DATASETS = {
  "benkanmura": {
    "op1_post_pitting_folder_id": "REPLACE_WITH_FOLDER_ID",
    "op2_post_plantating_folder_id": "REPLACE_WITH_FOLDER_ID",
    "op3_post_sowing_folder_id": "REPLACE_WITH_FOLDER_ID"
  },
  "debadihi": {
    "op1_post_pitting_folder_id": "REPLACE_WITH_FOLDER_ID",
    "op2_post_plantating_folder_id": "REPLACE_WITH_FOLDER_ID",
    "op3_post_sowing_folder_id": "REPLACE_WITH_FOLDER_ID"
  },
  "test": {
    "op1_post_pitting_folder_id": "TEST_ID",
    "op2_post_plantating_folder_id": "TEST_ID",
    "op3_post_sowing_folder_id": "TEST_ID"
  },
  "pit_detection": {
    "min_area_px": 50,
    "max_area_px": 5000,
    "blur_kernel_size": 5,
    "threshold_method": "otsu",
    "circularity_min": 0.4
  },
  "op2_confirmation": {
    "crop_size_px": 256,
    "planted_threshold": 0.02,
    "hsv_green_lower": [25, 40, 40],
    "hsv_green_upper": [95, 255, 255]
  }
}

def check_environment(auto_fix=True, verbose=True):
    """
    Checks environment health.
    Returns: (is_healthy, messages)
    """
    is_healthy = True
    messages = []
    
    if verbose: print(f"[SETUP] Checking environment at: {REPO_ROOT}")
    
    # 1. Config Check
    if not os.path.exists(CONFIG_PATH):
        is_healthy = False
        msg = "[WARN] config/datasets.json is MISSING."
        messages.append(msg)
        if verbose: print(msg)
        
        if auto_fix:
            try:
                os.makedirs(CONFIG_DIR, exist_ok=True)
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(TEMPLATE_DATASETS, f, indent=2)
                print(f"[FIX] Created template config at: {CONFIG_PATH}")
                print("[ACTION] Please edit this file and add your Drive keys (from problem statement).")
            except Exception as e:
                print(f"[ERROR] Failed to create config: {e}")
    else:
        if verbose: print("[OK] config/datasets.json found.")

    # 2. Credentials Check
    if not os.path.exists(CREDENTIALS_PATH):
        # We cannot auto-fix this securely
        is_healthy = False
        msg = "[WARN] credentials.json is MISSING (Required for Drive ingest)."
        messages.append(msg)
        if verbose:
            print(msg)
            print("-" * 60)
            print("HOW TO GET CREDENTIALS:")
            print("1. Go to Google Cloud Console (https://console.cloud.google.com)")
            print("2. Create a Project > Enable 'Google Drive API'")
            print("3. APIs & Services > Credentials > Create Credentials > OAuth Client ID")
            print("4. Select 'Desktop App'")
            print(f"5. Download JSON, rename to 'credentials.json', and place in: {REPO_ROOT}")
            print("-" * 60)

    else:
        if verbose: print("[OK] credentials.json found.")
        
    return is_healthy

if __name__ == "__main__":
    print(f"{'#'*60}")
    print("AGRIGUARD ENVIRONMENT SETUP")
    print(f"{'#'*60}")
    
    healthy = check_environment(verbose=True)
    
    print(f"\n{'#'*60}")
    if healthy:
        print("STATUS: ENVIRONMENT READY")
        print("You can now run 'python main.py' or 'python train.py'")
    else:
        print("STATUS: SETUP INCOMPLETE")
        print("Review the warnings above. You may need to edit config or add credentials.")
    print(f"{'#'*60}")
