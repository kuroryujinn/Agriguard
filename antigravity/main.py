import argparse
import sys
import subprocess
import os
import shutil

# Detect Repository Root (Robust)
def find_repo_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [script_dir, os.path.dirname(script_dir)]
    for c in candidates:
        if os.path.exists(os.path.join(c, "config", "datasets.json")): return c
        if os.path.exists(os.path.join(c, "src")) and os.path.exists(os.path.join(c, "data")): return c
    return script_dir

REPO_ROOT = find_repo_root()
sys.path.append(REPO_ROOT)

# Try Import Setup Helper
try:
    from setup_env import check_environment
except ImportError:
    def check_environment(auto_fix=True, verbose=True): return True

def run_command(cmd, desc, cwd=None):
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    try:
        # Run command from REPO_ROOT to ensure imports (src.main) work correctly
        subprocess.run(cmd, check=True, text=True, cwd=cwd or REPO_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step failed: {desc}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Exit Code: {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Agriguard Antigravity: End-to-End Orchestrator")
    parser.add_argument("--field", help="Field name (e.g., benkanmura). If omitted, defaults to first dataset in config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--mode", choices=["cv", "hybrid", "gemini"], default="hybrid", help="Survival analysis mode")
    parser.add_argument("--max-gemini-calls", type=int, default=200, help="Max Gemini API calls")
    parser.add_argument("--skip-download", action="store_true", help="Skip indexing and downloading if data exists")
    
    args = parser.parse_args()
    
    # --- 0. Self-Healing Check ---
    check_environment(auto_fix=True, verbose=True)
    
    # --- Path Resolution ---
    config_path = os.path.join(REPO_ROOT, "config", "datasets.json")
    credentials_path = os.path.join(REPO_ROOT, "credentials.json")
    
    # --- Config Check & Demo Mode Fallback ---
    if not os.path.exists(config_path):
        print(f"\n[INFO] config/datasets.json not found at {config_path}")
        print("[INFO] Running in LOCAL DEMO MODE using 'test' field.")
        
        args.field = "test"
        args.skip_download = True
        
        # Default to CV mode in demo unless user explicitly requested otherwise
        if "--mode" not in sys.argv:
            args.mode = "cv"
            print(f"[INFO] Defaulting to mode='{args.mode}' for demo.")
    
    # --- Field Generation / Default (Only if config exists and field not set) ---
    elif not args.field:
        try:
            import json
            with open(config_path, 'r') as f:
                data = json.load(f)
                # Filter out settings keys
                keys = [k for k in data.keys() if k not in ["pit_detection", "op2_confirmation"]]
                if keys:
                    args.field = keys[0]
                    print(f"\n[INFO] No field provided. Defaulting to: {args.field}")
                else:
                    args.field = "default"
                    print(f"\n[INFO] No datasets found in config. Using: {args.field}")
        except Exception:
             args.field = "default"
             print(f"\n[INFO] Could not read config. Defaulting to: {args.field}")

    # --- 1. Environment Validation ---
    print("\n[1/5] Validating Environment...")
    
    # Only check credentials if we are definitely downloading
    if not args.skip_download:
        if not os.path.exists(credentials_path):
            print(f"ERROR: credentials.json not found at {credentials_path}. Required for Google Drive access.")
            sys.exit(1)
            
    # Check Gemini key only if mode requires it
    if args.mode in ["hybrid", "gemini"]:
        if not os.getenv("GEMINI_API_KEY"):
            print("ERROR: GEMINI_API_KEY environment variable is missing (Required for hybrid/gemini mode).")
            print("Set it using: $env:GEMINI_API_KEY='your_key'")
            sys.exit(1)

    python_exe = sys.executable
    
    # --- 2. Data Ingestion ---
    if not args.skip_download:
        print("\n[2/5] Data Ingestion...")
        # Index
        run_command([python_exe, "-m", "src.main", "index", "--field", args.field], "Indexing Drive Folder")
        
        # Sample
        run_command([python_exe, "-m", "src.main", "sample", 
                     "--field", args.field, 
                     "--seed", str(args.seed), 
                     "--stage", "all"], "Sampling Dataset")
                     
        # Sync (Download) - OP1, OP2, OP3
        run_command([python_exe, "-m", "src.main", "sync-sample", "--field", args.field, "--stage", "op1"], "Syncing OP1 Images")
        run_command([python_exe, "-m", "src.main", "sync-sample", "--field", args.field, "--stage", "op2"], "Syncing OP2 Images")
        run_command([python_exe, "-m", "src.main", "sync-sample", "--field", args.field, "--stage", "op3"], "Syncing OP3 Images")
    else:
        print("\n[2/5] Skipping Data Ingestion (User Request or Demo Mode)...")

    # --- 3. OpenCV V2 Pipeline ---
    print("\n[3/5] Running OpenCV V2 Pipeline...")
    
    # Detect Pits (OP1)
    run_command([python_exe, "-m", "src.main", "v2-detect-pits", 
                 "--field", args.field], "V2: Detect Pits (OP1)")
                 
    # Confirm Planting (OP2)
    run_command([python_exe, "-m", "src.main", "v2-confirm-planting", 
                 "--field", args.field], "V2: Confirm Planting (OP2)")
                 
    # Confirm Establishment (OP2/OP3 checks - Optional but good for cache)
    # The requirement says: "python -m src.main v2-confirm-establishment"
    run_command([python_exe, "-m", "src.main", "v2-confirm-establishment", 
                 "--field", args.field], "V2: Confirm Establishment (OP3)")

    # --- 4. Survival Prediction ---
    print("\n[4/5] Running Survival Analysis...")
    
    cmd_survival = [
        python_exe, "-m", "src.main", "op3-survival",
        "--field", args.field,
        "--mode", args.mode
    ]
    if args.max_gemini_calls:
        cmd_survival.extend(["--max-gemini-calls", str(args.max_gemini_calls)])
        
    run_command(cmd_survival, f"Survival Analysis (Mode: {args.mode})")

    # --- 5. Completion ---
    print("\n[5/5] Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Results available in: data/manifests/v3/")
    print(f"  - {args.field}.op3_survival.csv")
    print(f"  - {args.field}.dead_locations.csv")
    print(f"  - {args.field}.survival_summary.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
