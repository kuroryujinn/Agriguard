# ... Existing imports ...
import argparse
import json
import os
import pandas as pd
from src.drive.listing import list_files
from src.drive.download import download_file
from src.data.sampling import sample_dataset
from src.cv.pit_detector import detect_pits
from src.cv.op2_sapling_detector import check_planting

# V2 Imports
from src.cv.pit_detector_v2 import detect_pits_v2
from src.cv.op2_planting_v2 import check_planting_v2
from src.cv.op3_establishment_v2 import check_establishment_v2

# Configuration
CONFIG_PATH = os.path.join("config", "datasets.json")
MANIFEST_DIR = os.path.join("data", "manifests")
CACHE_DIR = os.path.join("data", "cache")

# V2 Output Paths
V2_MANIFEST_DIR = os.path.join("data", "manifests", "v2")
V2_PITS_DIR = os.path.join(V2_MANIFEST_DIR, "pits")
V2_OP2_DIR = os.path.join(V2_MANIFEST_DIR, "op2")
V2_OP3_DIR = os.path.join(V2_MANIFEST_DIR, "op3")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

# ... Keeping existing commands (cmd_index, cmd_sample, cmd_sync_sample, cmd_detect_pits, cmd_confirm_planting) ...
# ... for backward compatibility/reference, but implementing V2 commands below ...

# (Assuming existing commands are here, I will append new ones and updated main)

def cmd_v2_detect_pits(args):
    # 1. Load OP1 Samples
    sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
    if not os.path.exists(sample_path):
        # Try stage specific?
        sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.op1.sample.json")
        if not os.path.exists(sample_path):
             print("Error: OP1 sample manifest not found.")
             return

    with open(sample_path, 'r') as f:
        data = json.load(f)
    
    op1_items = [x for x in data if x.get('sampled')==True and x['stage']=='op1_post_pitting']
    print(f"V2: Detecting pits for {len(op1_items)} images...")
    
    config = load_config()
    pit_conf = config.get('pit_detection', {})
    if args.min_area: pit_conf['min_area_px'] = args.min_area
    if args.max_area: pit_conf['max_area_px'] = args.max_area
    
    csv_rows = []
    
    for i, item in enumerate(op1_items):
        if args.max_files and i >= args.max_files: break
        
        subfolder = f"{args.field}/{item['stage']}"
        local_path = download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)
        
        debug_path = None
        if args.debug_overlay:
            debug_path = os.path.join(V2_PITS_DIR, args.field, "overlays", f"{item['name']}.png")
            
        result = detect_pits_v2(local_path, config=pit_conf, debug_overlay_path=debug_path)
        
        # Save V2 JSON
        out_dir = os.path.join(V2_PITS_DIR, args.field)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{item['name']}.pits.json"), 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"[{i+1}] {item['name']}: {result['pit_count']} pits")
        
        for idx, pit in enumerate(result['pits']):
            csv_rows.append({
                "field": args.field,
                "image_name": item['name'],
                "pit_index": idx,
                "x": pit['x'], "y": pit['y'], "area": pit['area'], "circularity": pit['circularity']
            })
            
    csv_path = os.path.join(V2_MANIFEST_DIR, f"{args.field}.pits.csv")
    os.makedirs(V2_MANIFEST_DIR, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved V2 pits to {csv_path}")

def cmd_v2_confirm_planting(args):
    # 1. Load V2 Pits
    pits_csv = os.path.join(V2_MANIFEST_DIR, f"{args.field}.pits.csv")
    if not os.path.exists(pits_csv):
        print("Error: V2 Pits CSV not found. Run v2-detect-pits first.")
        return
    pits_df = pd.read_csv(pits_csv)
    
    # 2. Load OP2 Samples
    # Try stage specific first
    sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.op2.sample.json")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
        
    if not os.path.exists(sample_path):
        print("Error: OP2 sample manifest not found.")
        return
        
    with open(sample_path, 'r') as f:
        data = json.load(f)
    
    op2_items = [x for x in data if x.get('sampled')==True and x['stage']=='op2_post_plantating']
    print(f"V2: Confirming planting for {len(op2_items)} images...")
    
    config = load_config()
    op2_conf = config.get('op2_confirmation', {})
    if args.crop_size: op2_conf['crop_size_px'] = args.crop_size
    if args.op2_threshold: op2_conf['planted_threshold'] = args.op2_threshold
    
    csv_rows = []
    
    for i, item in enumerate(op2_items):
        if args.max_files and i >= args.max_files: break
        
        # Download
        subfolder = f"{args.field}/{item['stage']}"
        local_path = download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)
        
        # Pits check
        img_pits = pits_df[pits_df['image_name'] == item['name']]
        if img_pits.empty: continue
        
        debug_path = None
        if args.debug_overlay:
            debug_path = os.path.join(V2_OP2_DIR, args.field, "overlays", f"{item['name']}.png")
            
        results = check_planting_v2(local_path, img_pits.to_dict('records'), config=op2_conf, debug_overlay_path=debug_path)
        
        out_dir = os.path.join(V2_OP2_DIR, args.field)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{item['name']}.op2.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        planted_count = sum(1 for r in results if r['planted'])
        print(f"[{i+1}] {item['name']}: {planted_count}/{len(results)} planted")
        
        for r in results:
            csv_rows.append({
                "field": args.field, "op2_image_name": item['name'],
                "pit_index": r['pit_index'], "x": r['x'], "y": r['y'],
                "green_ratio": r['green_ratio'], "planted": r['planted']
            })
            
    csv_path = os.path.join(V2_MANIFEST_DIR, f"{args.field}.op2_planting.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved V2 planting results to {csv_path}")

def cmd_v2_confirm_establishment(args):
    # 1. Load V2 Planting
    planting_csv = os.path.join(V2_MANIFEST_DIR, f"{args.field}.op2_planting.csv")
    if not os.path.exists(planting_csv):
        print("Error: V2 Planting CSV not found.")
        return
    planting_df = pd.read_csv(planting_csv)
    
    # Filter for PLANTED pits only
    planted_df = planting_df[planting_df['planted'] == True]
    if planted_df.empty:
        print("No planted pits found to verify.")
        return
        
    # 2. Load OP3 Samples
    sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.op3.sample.json")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
        
    if not os.path.exists(sample_path):
        print("Error: OP3 sample manifest not found.")
        return
        
    with open(sample_path, 'r') as f:
        data = json.load(f)
    op3_items = [x for x in data if x.get('sampled')==True and x['stage']=='op3_post_sowing']
    print(f"V2: Confirming establishment for {len(op3_items)} images...")
    
    config = load_config()
    # Assume op3 config block or reuse/override
    op3_conf = config.get('op3_establishment', {}) # Might need to create this in config or use defaults
    if args.op3_threshold: op3_conf['establishment_threshold'] = args.op3_threshold
    
    csv_rows = []
    
    for i, item in enumerate(op3_items):
        if args.max_files and i >= args.max_files: break
        
        subfolder = f"{args.field}/{item['stage']}"
        local_path = download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)
        
        # Pits check (Match by image name assuming alignment)
        img_pits = planted_df[planted_df['op2_image_name'] == item['name']]
        if img_pits.empty: continue
        
        debug_path = None
        if args.debug_overlay:
            debug_path = os.path.join(V2_OP3_DIR, args.field, "overlays", f"{item['name']}.png")
            
        # Re-use planting dict but expect 'established' key in output
        results = check_establishment_v2(local_path, img_pits.to_dict('records'), config=op3_conf, debug_overlay_path=debug_path)
        
        out_dir = os.path.join(V2_OP3_DIR, args.field)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{item['name']}.op3.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        est_count = sum(1 for r in results if r['established'])
        print(f"[{i+1}] {item['name']}: {est_count}/{len(results)} established")
        
        for r in results:
            csv_rows.append({
                "field": args.field, "op3_image_name": item['name'],
                "pit_index": r['pit_index'], "x": r['x'], "y": r['y'],
                "green_ratio": r['green_ratio'], "established": r['established']
            })
            
    csv_path = os.path.join(V2_MANIFEST_DIR, f"{args.field}.op3_establishment.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved V2 establishment results to {csv_path}")


# --- Update Main ---
def main():
    parser = argparse.ArgumentParser(description="Antigravity V2")
    subparsers = parser.add_subparsers(dest='command')
    
    # Existing commands (preserved)
    subparsers.add_parser('index').add_argument('--field', required=True)
    p_sam = subparsers.add_parser('sample')
    p_sam.add_argument('--field', required=True)
    p_sam.add_argument('--seed', type=int, default=42)
    p_sam.add_argument('--sample-n', type=int)
    p_sam.add_argument('--sample-frac', type=float)
    p_sam.add_argument('--sample-max-mb', type=int)
    p_sam.add_argument('--stage', default='all')
    
    p_sync = subparsers.add_parser('sync-sample')
    p_sync.add_argument('--field', required=True)
    p_sync.add_argument('--stage', default='all')
    
    subparsers.add_parser('detect-pits').add_argument('--field', required=True) # Legacy
    subparsers.add_parser('confirm-planting').add_argument('--field', required=True) # Legacy
    
    # New V2 Commands
    v2_pits = subparsers.add_parser('v2-detect-pits')
    v2_pits.add_argument('--field', required=True)
    v2_pits.add_argument('--max-files', type=int)
    v2_pits.add_argument('--debug-overlay', action='store_true')
    v2_pits.add_argument('--min-area', type=int)
    v2_pits.add_argument('--max-area', type=int)
    
    v2_pl = subparsers.add_parser('v2-confirm-planting')
    v2_pl.add_argument('--field', required=True)
    v2_pl.add_argument('--max-files', type=int)
    v2_pl.add_argument('--debug-overlay', action='store_true')
    v2_pl.add_argument('--crop-size', type=int)
    v2_pl.add_argument('--op2-threshold', type=float)
    
    v2_est = subparsers.add_parser('v2-confirm-establishment')
    v2_est.add_argument('--field', required=True)
    v2_est.add_argument('--max-files', type=int)
    v2_est.add_argument('--debug-overlay', action='store_true')
    v2_est.add_argument('--op3-threshold', type=float)

    # Simplified dispatch for brevity (In real code, use functions)
    args = parser.parse_args()
    
    # ... Dispatch Logic ...
    if args.command == 'v2-detect-pits': cmd_v2_detect_pits(args)
    elif args.command == 'v2-confirm-planting': cmd_v2_confirm_planting(args)
    elif args.command == 'v2-confirm-establishment': cmd_v2_confirm_establishment(args)
    # ...
    # RE-IMPORT original functions for dispatch to work cleanly if I overwrote main.py entirely
    # But wait, I'm REPLACING main.py content. I need to keep the old functions.
    # I will paste the Full Merged Main below.

# ... (The Replace tool needs the full content or smart chunks. I will use Write to File to overwrite main.py with the FULL content merging old and new).
# I will include the imports and existing cmd functions from previous steps.

