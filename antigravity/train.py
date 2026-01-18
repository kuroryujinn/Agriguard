import argparse
import sys
import subprocess
import os
import json
import glob
import pandas as pd
import numpy as np
import time

# --- 1. Robust Root Detection ---
# --- 1. Robust Root Detection ---
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
        # Update: Also trust structural markers if config is mistakenly missing (so we can fix it)
        if os.path.exists(os.path.join(c, "src")) and os.path.exists(os.path.join(c, "data")):
            return c
            
    return script_dir

REPO_ROOT = find_repo_root()
sys.path.append(REPO_ROOT) # Ensure root is in path for imports

# Try Import Setup Helper
try:
    from setup_env import check_environment
except ImportError:
    # Fallback if setup_env not found (e.g. separate deployment), define dummy
    def check_environment(auto_fix=True, verbose=True):
        return True
REPO_ROOT = find_repo_root()
# Note: subprocess calls will use cwd=REPO_ROOT
# So imports like 'from src.main' work if 'src' is in REPO_ROOT.

print(f"[INFO] REPO_ROOT = {REPO_ROOT}")

# Data PaEWths
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "datasets.json")
CREDENTIALS_PATH = os.path.join(REPO_ROOT, "credentials.json")
DATA_DIR = os.path.join(REPO_ROOT, "data")
MANIFEST_DIR = os.path.join(DATA_DIR, "manifests")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
V2_DIR = os.path.join(MANIFEST_DIR, "v2")
V3_DIR = os.path.join(MANIFEST_DIR, "v3")
MODELS_DIR = os.path.join(DATA_DIR, "models")
EXAMPLES_DIR = os.path.join(MODELS_DIR, "gemini_examples")

def run_step(cmd, desc):
    print(f"\n[TRAIN] STEP: {desc}")
    print(f"[TRAIN] CMD:  {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, text=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step failed: {desc}")
        # Build Summary even on fail? No, just exit non-zero for robust orchestrator.
        sys.exit(e.returncode)

def get_training_fields():
    """
    Discover fields from Config AND Local Artifacts.
    """
    fields = []
    # 1. Config
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)
                # Heuristic: Key is a field IF value is dict AND contains dataset-like keys
                candidate_keys = ["op1", "op2", "op3", "folder_id", "drive_id"]
                for k, v in data.items():
                    if isinstance(v, dict):
                        # check if any key in the valid dict contains one of our candidates
                        if any(cand in subk for subk in v.keys() for cand in candidate_keys):
                            fields.append(k)
        except Exception as e:
            print(f"[WARN] Failed to read config: {e}")
    
    # 2. Local Artifacts (Union)
    # Cache
    if os.path.exists(CACHE_DIR):
        try:
            candidates = [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
            for c in candidates:
                if c in ["previews", "tmp", "__pycache__"]: continue
                if c not in fields: fields.append(c)
        except: pass

    # Manifests V2
    if os.path.exists(V2_DIR):
        pit_files = glob.glob(os.path.join(V2_DIR, "*.pits.csv"))
        for p in pit_files:
            basename = os.path.basename(p)
            if basename.endswith(".pits.csv"):
                f = basename.replace(".pits.csv", "")
                if f not in fields: fields.append(f)

    # Manifests V3
    if os.path.exists(V3_DIR):
        surv_files = glob.glob(os.path.join(V3_DIR, "*.survival_summary.json"))
        for s in surv_files:
            basename = os.path.basename(s)
            f = basename.replace(".survival_summary.json", "")
            if f not in fields: fields.append(f)
                
    return sorted(list(set(fields)))

def needs_ingest(field, force=False):
    """
    Returns True if we suspect we need to download data (Ingest).
    Checks for completeness of Cache OR Manifests.
    """
    # Conservatively: If credentials missing, we MUST return False (skip ingest) to avoid crash.
    if not os.path.exists(CREDENTIALS_PATH):
        if force:
            print(f"[WARN] Cannot force ingest for '{field}' without credentials.")
        return False

    if force:
        return True

    # 1. Check Manifests Existence (OP1, OP2, OP3)
    missing_manifests = False
    for stage in ["op1", "op2", "op3"]:
        man_path = os.path.join(MANIFEST_DIR, f"{field}.{stage}.sample.json")
        if not os.path.exists(man_path):
            missing_manifests = True
            break
            
    # 2. Check Cache Content Breadth
    # If we have manifests, we can proceed. But "train" uses sync-sample to build cache.
    # If any manifests are missing, we definitely need ingest.
    if missing_manifests:
        return True
        
    # Check recursive existence of enough tifs in cache
    field_cache = os.path.join(CACHE_DIR, field)
    if not os.path.exists(field_cache):
        return True
        
    tifs = list(glob.glob(os.path.join(field_cache, "**", "*.tif"), recursive=True))
    tifs += list(glob.glob(os.path.join(field_cache, "**", "*.tiff"), recursive=True))
    
    # Heuristic: If we have very few images (e.g. <3), we probably need to sync.
    if len(tifs) < 3: 
        return True

    return False

def generate_cv_calibration(fields):
    print("\n[TRAIN] Generating CV Calibration Artifacts...")
    
    stats = {
        "green_ratio_alive": [],
        "green_ratio_dead": [],
        "pit_areas": [],
        "pit_circularities": []
    }
    
    counts = {k: 0 for k in stats.keys()}
    
    for field in fields:
        # Pits
        pit_csv = os.path.join(V2_DIR, f"{field}.pits.csv")
        if os.path.exists(pit_csv):
            try:
                df = pd.read_csv(pit_csv)
                # Robust Columns
                area_col = next((c for c in ["area", "pit_area"] if c in df.columns), None)
                circ_col = next((c for c in ["circularity", "circ"] if c in df.columns), None)
                
                if area_col: 
                    stats['pit_areas'].extend(df[area_col].tolist())
                    counts["pit_areas"] += len(df)
                if circ_col: 
                    stats['pit_circularities'].extend(df[circ_col].tolist())
                    counts["pit_circularities"] += len(df)
            except Exception as e:
                print(f"[WARN] Error reading {pit_csv}: {e}")
            
        # Survival
        surv_csv = os.path.join(V3_DIR, f"{field}.op3_survival.csv")
        if os.path.exists(surv_csv):
            try:
                df = pd.read_csv(surv_csv)
                # Robust Status Column (Case Insensitive)
                status_col = next((c for c in df.columns if c.lower() == "status"), None)
                gr_col = next((c for c in ["green_ratio", "green_ratio_best", "score_green", "green_score"] if c in df.columns), None)
                
                if status_col and gr_col:
                    # Normalize status to uppercase for filtering
                    df[status_col] = df[status_col].astype(str).str.upper()
                    
                    alive = df[df[status_col] == 'ALIVE'][gr_col]
                    dead = df[df[status_col] == 'DEAD'][gr_col]
                    
                    stats['green_ratio_alive'].extend(alive.tolist())
                    stats['green_ratio_dead'].extend(dead.tolist())
                    
                    counts["green_ratio_alive"] += len(alive)
                    counts["green_ratio_dead"] += len(dead)
                else:
                    if not gr_col: 
                         print(f"[WARN] {surv_csv} missing green_ratio column. Columns found: {list(df.columns)}")
                    if not status_col:
                         print(f"[WARN] {surv_csv} missing 'status' column. Columns found: {list(df.columns)}")
            except Exception as e:
                print(f"[WARN] Error reading {surv_csv}: {e}")

    # Compute Aggregates
    res_alive = float(np.median(stats['green_ratio_alive'])) if stats['green_ratio_alive'] else 0.05
    res_dead = float(np.median(stats['green_ratio_dead'])) if stats['green_ratio_dead'] else 0.01
    
    if not stats['green_ratio_alive']: print("[WARN] Calibration stats empty for green_ratio_alive.")
    if not stats['green_ratio_dead']: print("[WARN] Calibration stats empty for green_ratio_dead.")

    calibration = {
        "recommended_thresholds": {
            "alive": res_alive,
            "dead": res_dead
        },
        "pit_stats": {
            "median_area": float(np.median(stats['pit_areas'])) if stats['pit_areas'] else 0,
            "median_circularity": float(np.median(stats['pit_circularities'])) if stats['pit_circularities'] else 0
        },
        "metadata": {
            "fields_contributed": fields,
            "counts": counts
        },
        "timestamp": time.time()
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "cv_calibration.json"), 'w') as f:
        json.dump(calibration, f, indent=2)
    
    return calibration

def save_gemini_examples_append(fields, epoch):
    print("[TRAIN] Persisting Gemini Example Library (Append-Only)...")
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    new_files = 0
    timestamp = int(time.time())
    
    for field in fields:
        surv_csv = os.path.join(V3_DIR, f"{field}.op3_survival.csv")
        if os.path.exists(surv_csv):
            try:
                df = pd.read_csv(surv_csv)
                
                # Filter Criteria:
                # 1. method_used contains "gemini" (case insensitive)
                # 2. columns like gemini_confidence, evidence present and valid
                
                # Check available columns
                cols = df.columns
                is_gemini_row = pd.Series([False] * len(df))
                
                if 'method_used' in cols:
                    is_gemini_row |= df['method_used'].astype(str).str.lower().str.contains("gemini", na=False)
                    
                target_cols = ["gemini_confidence", "gemini_label", "evidence"]
                for c in target_cols:
                    if c in cols:
                        # logical OR: if column exists and is not null/empty
                        is_gemini_row |= df[c].notna() & (df[c] != "")
                
                gemini_cases = df[is_gemini_row]
                
                for _, row in gemini_cases.iterrows():
                    ex = row.to_dict()
                    ex['epoch_provenance'] = epoch
                    
                    pit_id = ex.get('pit_index', 'u')
                    # Unique ID: Field + Pit + Epoch + Timestamp + Hash(optional but ts is likely enough for sequential)
                    fname = f"{field}_{pit_id}_e{epoch}_{timestamp}.json"
                    
                    with open(os.path.join(EXAMPLES_DIR, fname), 'w') as f:
                        json.dump(ex, f, indent=2)
                    new_files += 1
            except: pass
    
    print(f"[TRAIN] Saved {new_files} NEW Gemini examples.")
    return new_files

def get_total_gemini_calls(fields):
    total = 0
    for field in fields:
        sum_path = os.path.join(V3_DIR, f"{field}.survival_summary.json")
        if os.path.exists(sum_path):
            try:
                with open(sum_path, 'r') as f:
                    d = json.load(f)
                    total += d.get("gemini_calls_made", 0)
            except: pass
    return total

def main():
    parser = argparse.ArgumentParser(description="Agriguard Antigravity: Batch Trainer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gemini-calls", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    # New Mode Flag
    parser.add_argument("--mode", choices=["cv", "hybrid", "auto"], default="auto")
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"AGRIGUARD TRAINING & CACHE BUILDER")
    print(f"{'#'*60}\n")
    
    # 0. Self-Healing Check
    is_healthy = check_environment(auto_fix=True, verbose=True)
    if not is_healthy:
        print("\n[INFO] Environment incomplete. attempting to run in OFFLINE/CACHE mode if possible.")
    
    # 1. Field Discovery
    fields = get_training_fields()
    if not fields:
        print("[ERROR] No training fields found.")
        print("ACTION: Add 'config/datasets.json' (root) or place cached manifests in 'data/manifests/v2/'.")
        sys.exit(1)
        
    print(f"[INFO] Training fields: {fields}")
    
    # 2. Mode Resolution
    final_mode = "cv"
    if args.mode == "auto":
        if os.getenv("GEMINI_API_KEY"):
            final_mode = "hybrid"
        else:
            final_mode = "cv"
            print("[INFO] GEMINI_API_KEY missing. Auto-downgrading to 'cv' mode.")
    else:
        final_mode = args.mode
        
    print(f"[INFO] Survival Analysis Mode: {final_mode}")

    start_time = time.time()
    python_exe = sys.executable

    # 3. Training Loop
    fields_ingested = []
    epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n>>> EPOCH {epoch}/{args.epochs}")
        epoch_ingest_count = 0
        
        for field in fields:
            print(f"\n--- Processing Field: {field} ---")
            
            # 1. Early Skip Check for Missing Credentials & Data
            has_creds = os.path.exists(CREDENTIALS_PATH)
            has_manifests = any(os.path.exists(os.path.join(MANIFEST_DIR, f"{field}.{s}.sample.json")) for s in ["op1", "op2", "op3"])
            has_cache = os.path.exists(os.path.join(CACHE_DIR, field))
            
            if not has_creds and not has_manifests and not has_cache:
                print(f"[SKIP] {field}: no local artifacts and Drive ingest disabled (missing credentials.json).")
                continue
            
            try:
                # A) Ingest
                # If credentials exist, we try ingest. If not, needs_ingest returns False anyway.
                do_ingest = needs_ingest(field, force=args.force)
                if do_ingest:
                     run_step([python_exe, "-m", "src.main", "index", "--field", field], "Index")
                     run_step([python_exe, "-m", "src.main", "sample", "--field", field, "--seed", str(args.seed)], "Sample")
                     run_step([python_exe, "-m", "src.main", "sync-sample", "--field", field, "--stage", "op1"], "Sync OP1")
                     run_step([python_exe, "-m", "src.main", "sync-sample", "--field", field, "--stage", "op2"], "Sync OP2")
                     run_step([python_exe, "-m", "src.main", "sync-sample", "--field", field, "--stage", "op3"], "Sync OP3")
                     fields_ingested.append(field)
                     epoch_ingest_count += 1
                else:
                     # Only print skip info if we actually have data (implying we skipped because we already have it, or we simply can't ingest)
                     if has_creds:
                         print("[INFO] Skipping Ingest (Cache ready).")
                     else:
                         print("[INFO] Skipping Ingest (Local Cache used; Drive disabled).")

                # B) V2 Pipeline
                # Prerequisite: OP1 Manifest
                op1_manifest = os.path.join(MANIFEST_DIR, f"{field}.op1.sample.json")
                if not os.path.exists(op1_manifest):
                    print(f"[SKIP] Detect Pits (Missing manifest: {op1_manifest})")
                else:
                    pits_out = os.path.join(V2_DIR, f"{field}.pits.csv")
                    if args.force or not os.path.exists(pits_out):
                        run_step([python_exe, "-m", "src.main", "v2-detect-pits", "--field", field], "Detect Pits")
                    else:
                        print(f"[SKIP] Pits detected for {field}")

                # Prerequisite: Pits CSV + OP2 Manifest
                # Actually cmd_v2_confirm_planting checks for op2.sample.json
                # We should check it here to be safe.
                op2_manifest = os.path.join(MANIFEST_DIR, f"{field}.op2.sample.json")
                pits_csv = os.path.join(V2_DIR, f"{field}.pits.csv")
                
                if not os.path.exists(op2_manifest) or not os.path.exists(pits_csv):
                     print(f"[SKIP] Confirm Planting (Missing OP2 manifest or Pits CSV)")
                else:
                    plant_out = os.path.join(V2_DIR, f"{field}.op2_planting.csv")
                    if args.force or not os.path.exists(plant_out):
                         run_step([python_exe, "-m", "src.main", "v2-confirm-planting", "--field", field], "Confirm Planting")
                    else:
                         print(f"[SKIP] Planting confirmed for {field}")
                     
                # C) Survival
                # Prerequisite: Planted CSV + OP3 Manifest
                op3_manifest = os.path.join(MANIFEST_DIR, f"{field}.op3.sample.json")
                plant_csv = os.path.join(V2_DIR, f"{field}.op2_planting.csv")
                
                if not os.path.exists(op3_manifest) or not os.path.exists(plant_csv):
                     print(f"[SKIP] Survival Analysis (Missing OP3 manifest or Planting CSV)")
                else:
                    surv_out = os.path.join(V3_DIR, f"{field}.op3_survival.csv")
                    should_run_survival = args.force or not os.path.exists(surv_out) or (epoch > 1 and final_mode != 'cv')
                    
                    if should_run_survival:
                        run_step([
                            python_exe, "-m", "src.main", "op3-survival", 
                            "--field", field, 
                            "--mode", final_mode,
                            "--max-gemini-calls", str(args.max_gemini_calls)
                        ], "Survival Analysis")
                    else:
                        print(f"[SKIP] Survival analyzed for {field}")
                        
            except Exception as e:
                print(f"[ERROR] Failed to process field {field}: {e}")
                continue # Continue to next field

        # Post-Epoch artifacts
        new_examples = save_gemini_examples_append(fields, epoch)
        
        # Early Stopping
        if args.epochs > 1:
            if epoch_ingest_count == 0 and new_examples == 0:
                print(f"\n[INFO] No new learning signals (ingest or gemini examples) in Epoch {epoch}. Stopping early.")
                break

    # 4. Final Artifact Generation
    print("\n[TRAIN] Building Global Artifacts...")
    generate_cv_calibration(fields)
    
    # Summary
    gemini_calls = get_total_gemini_calls(fields)
    summary = {
        "fields_processed": fields,
        "epochs_completed": epoch,
        "runtime_seconds": int(time.time() - start_time),
        "fields_ingested": sorted(list(set(fields_ingested))),
        "gemini_calls_total": gemini_calls,
        "calibration_path": os.path.join(MODELS_DIR, "cv_calibration.json"),
        "gemini_examples_dir": EXAMPLES_DIR,
        "stats_timestamp": time.time()
    }
    with open(os.path.join(MODELS_DIR, "train_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[SUCCESS] Training complete.")
    print(f"Artifacts stored in {MODELS_DIR}")

if __name__ == "__main__":
    main()
