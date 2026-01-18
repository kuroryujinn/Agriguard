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

# Pipeline Imports
from src.pipeline.op3_survival import analyze_survival
from src.pipeline.run_all import run_all_stages

# Configuration
CONFIG_PATH = os.path.join("config", "datasets.json")
MANIFEST_DIR = os.path.join("data", "manifests")
CACHE_DIR = os.path.join("data", "cache")
PITS_DIR = os.path.join("data", "manifests", "pits")
OP2_DIR = os.path.join("data", "manifests", "op2")

# V2 Output Paths (V3 uses "v3")
V2_MANIFEST_DIR = os.path.join("data", "manifests", "v2")
V2_PITS_DIR = os.path.join(V2_MANIFEST_DIR, "pits")
V2_OP2_DIR = os.path.join(V2_MANIFEST_DIR, "op2")
V2_OP3_DIR = os.path.join(V2_MANIFEST_DIR, "op3")

# V3 Output Paths
V3_MANIFEST_DIR = os.path.join("data", "manifests", "v3")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

# --- Helpers ---
def download_helper_fn(item):
    subfolder = f"{item['field']}/{item['stage']}" # Assuming item has field
    # If item doesn't have field, we need to pass it.
    # But analyze_survival passes item from op3_map which comes from manifest.manifest usually has field?
    # No, manifest items in 'files.json' usually don't have 'field' property explicitly unless added during index.
    # In index we do: f['stage'] = stage. We DON'T add field? 
    # Let's check cmd_index. It parses args.field but doesn't inject it into file dicts.
    # We must fix logic or pass field.
    # download_file accepts 'dest_subfolder'.
    return download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)

# --- Commands ---

def cmd_index(args):
    config = load_config()
    if args.field not in config:
        print(f"Error: Field '{args.field}' not found.")
        return

    field_conf = config[args.field]
    op1_id = field_conf.get('op1_post_pitting_folder_id')
    op2_id = field_conf.get('op2_post_plantating_folder_id')
    op3_id = field_conf.get('op3_post_sowing_folder_id')
    
    stages_to_scan = [("op1_post_pitting", op1_id), ("op3_post_sowing", op3_id)]
    
    # Optional OP2
    if op2_id and "PLACEHOLDER" not in op2_id:
         stages_to_scan.append(("op2_post_plantating", op2_id))
    
    # Validation Warning
    unique_ids = set([x[1] for x in stages_to_scan if x[1]])
    if len(unique_ids) < len([x for x in stages_to_scan if x[1]]):
         print(f"WARNING: ID collision detected in folder config.")

    all_files = []
    
    for stage, folder_id in stages_to_scan:
        if not folder_id: continue 
        print(f"Indexing {stage} ({folder_id})...")
        files = list_files(folder_id)
        print(f"Found {len(files)} files.")
        
        for f in files:
            f['stage'] = stage
            f['field'] = args.field # Inject field for easier downstream usage
            all_files.append(f)
            
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    manifest_path = os.path.join(MANIFEST_DIR, f"{args.field}.files.json")
    
    with open(manifest_path, 'w') as f:
        json.dump(all_files, f, indent=2)
        
    print(f"Saved manifest to {manifest_path} with {len(all_files)} total files.")

def cmd_sample(args):
    manifest_path = os.path.join(MANIFEST_DIR, f"{args.field}.files.json")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found. Run 'index' first.")
        return

    print(f"Sampling {args.field} (Stage: {args.stage}, Seed: {args.seed})...")
    
    # Load all files
    df = sample_dataset(manifest_path, seed=args.seed, sample_n=args.sample_n, 
                        sample_frac=args.sample_frac, sample_max_mb=args.sample_max_mb)
    
    # FILTER BY STAGE IF ARGUMENT PROVIDED
    stage_map = {'op1': 'op1_post_pitting', 'op2': 'op2_post_plantating', 'op3': 'op3_post_sowing'}
    target_stage = stage_map.get(args.stage, args.stage)
    
    if args.stage != 'all':
        with open(manifest_path, 'r') as f:
            all_files = json.load(f)
            
        stage_files = [x for x in all_files if x.get('stage') == target_stage]
        if not stage_files:
            print(f"No files found for stage {target_stage}.")
            return
            
        temp_path = os.path.join(MANIFEST_DIR, f"{args.field}.{args.stage}.temp_files.json")
        with open(temp_path, 'w') as f:
            json.dump(stage_files, f)
            
        # Sample derived
        df = sample_dataset(temp_path, seed=args.seed, sample_n=args.sample_n,
                            sample_frac=args.sample_frac, sample_max_mb=args.sample_max_mb)
        
        # Cleanup
        os.remove(temp_path)
        
        # Output filename
        json_out = os.path.join(MANIFEST_DIR, f"{args.field}.{args.stage}.sample.json")
        csv_out = os.path.join(MANIFEST_DIR, f"{args.field}.{args.stage}.sample.csv")
    else:
        # Default behavior (All stages)
        print("Warning: Sampling ALL stages. This replaces previous samples.")
        df = sample_dataset(manifest_path, seed=args.seed, sample_n=args.sample_n,
                            sample_frac=args.sample_frac, sample_max_mb=args.sample_max_mb)
        
        json_out = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
        csv_out = os.path.join(MANIFEST_DIR, f"{args.field}.sample.csv")

    df.to_json(json_out, orient='records', indent=2)
    df.to_csv(csv_out, index=False)
    
    print(f"Saved sample to {csv_out}")
    if not df.empty:
         print(df[df['sampled']==True].groupby('stage').size())

def cmd_sync_sample(args):
    # Determine which manifest to use
    if args.stage != 'all':
         # Check for specific manifest first
         spec_path = os.path.join(MANIFEST_DIR, f"{args.field}.{args.stage}.sample.json")
         gen_path = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
         
         if os.path.exists(spec_path):
             path_to_use = spec_path
         elif os.path.exists(gen_path):
             path_to_use = gen_path
         else:
             print("Error: No sample manifest found.")
             return
    else:
        path_to_use = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
        if not os.path.exists(path_to_use):
             print("Error: General sample manifest not found.")
             return

    print(f"Syncing from manifest: {os.path.basename(path_to_use)}")
    with open(path_to_use, 'r') as f:
        data = json.load(f)
        
    to_download = [x for x in data if x.get('sampled') == True]
    
    # Filter by stage logic
    stage_map = {'op1': 'op1_post_pitting', 'op2': 'op2_post_plantating', 'op3': 'op3_post_sowing'}
    target = stage_map.get(args.stage, args.stage)
    
    if args.stage != 'all':
        to_download = [x for x in to_download if x['stage'] == target]
        
    print(f"Syncing {len(to_download)} files for {args.field} (Stage: {args.stage})...")
    
    for item in to_download:
        subfolder = f"{args.field}/{item['stage']}"
        download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)

def cmd_summary(args):
    files_path = os.path.join(MANIFEST_DIR, f"{args.field}.files.json")
    sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
    
    print(f"=== Antigravity Summary: {args.field} ===")
    
    # Remote
    if os.path.exists(files_path):
        with open(files_path) as f:
            all_data = json.load(f)
        op1 = len([x for x in all_data if x['stage'] == 'op1_post_pitting'])
        op2 = len([x for x in all_data if x['stage'] == 'op2_post_plantating'])
        op3 = len([x for x in all_data if x['stage'] == 'op3_post_sowing'])
        print(f"Remote Total: {len(all_data)} (OP1: {op1}, OP2: {op2}, OP3: {op3})")
    else:
        print("Remote: Not indexed.")
        
    # Sampled
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            samp_data = json.load(f)
        sampled = [x for x in samp_data if x.get('sampled') == True]
        op1_s = len([x for x in sampled if x['stage'] == 'op1_post_pitting'])
        op2_s = len([x for x in sampled if x['stage'] == 'op2_post_plantating'])
        op3_s = len([x for x in sampled if x['stage'] == 'op3_post_sowing'])
        print(f"Sampled Total: {len(sampled)} (OP1: {op1_s}, OP2: {op2_s}, OP3: {op3_s})")
    else:
        print("Sampled: Not run.")

def cmd_detect_pits(args):
    # LEGACY COMMAND - see v2-detect-pits
    pass 

def cmd_confirm_planting(args):
    # LEGACY COMMAND - see v2-confirm-planting
    pass

# --- V2 Commands ---

def cmd_v2_detect_pits(args):
    # 1. Load OP1 Samples
    sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.op1.sample.json")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
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
            
        print(f"[{i+1}/{len(op1_items)}] {item['name']}: {result['pit_count']} pits")
        
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
        if args.field == 'test':
             # Bypass download for synthetic test
             local_path = os.path.join("data", "cache", subfolder, item['name'])
        else:
             local_path = download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)
        
        # Pits check
        img_pits = pits_df[pits_df['image_name'] == item['name']]
        
        # Debug
        print(f"DEBUG: Processing {item['name']}. Found {len(img_pits)} pits in CSV.")
        
        if img_pits.empty:
             print(f"Skipping {item['name']} (No pits found). Pits head: {pits_df.head()}")
             continue
        
        debug_path = None
        if args.debug_overlay:
            debug_path = os.path.join(V2_OP2_DIR, args.field, "overlays", f"{item['name']}.png")
            
        results = check_planting_v2(local_path, img_pits.to_dict('records'), config=op2_conf, debug_overlay_path=debug_path)
        
        out_dir = os.path.join(V2_OP2_DIR, args.field)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{item['name']}.op2.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        planted_count = sum(1 for r in results if r['planted'])
        print(f"[{i+1}/{len(op2_items)}] {item['name']}: {planted_count}/{len(results)} planted")
        
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
    op3_conf = config.get('op3_establishment', {}) 
    if args.op3_threshold: op3_conf['establishment_threshold'] = args.op3_threshold
    
    csv_rows = []
    
    for i, item in enumerate(op3_items):
        if args.max_files and i >= args.max_files: break
        
        subfolder = f"{args.field}/{item['stage']}"
        local_path = download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)
        
        # Pits check
        img_pits = planted_df[planted_df['op2_image_name'] == item['name']]
        if img_pits.empty: continue
        
        debug_path = None
        if args.debug_overlay:
            debug_path = os.path.join(V2_OP3_DIR, args.field, "overlays", f"{item['name']}.png")
            
        results = check_establishment_v2(local_path, img_pits.to_dict('records'), config=op3_conf, debug_overlay_path=debug_path)
        
        out_dir = os.path.join(V2_OP3_DIR, args.field)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{item['name']}.op3.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        est_count = sum(1 for r in results if r['established'])
        print(f"[{i+1}/{len(op3_items)}] {item['name']}: {est_count}/{len(results)} established")
        
        for r in results:
            csv_rows.append({
                "field": args.field, "op3_image_name": item['name'],
                "pit_index": r['pit_index'], "x": r['x'], "y": r['y'],
                "green_ratio": r['green_ratio'], "established": r['established']
            })
            
    csv_path = os.path.join(V2_MANIFEST_DIR, f"{args.field}.op3_establishment.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved V2 establishment results to {csv_path}")

# --- Pipeline Wrapper (OP3 Survival) ---

def cmd_op3_survival(args):
    # Requirements
    # v2 planting csv must exist
    # op3 samples must exist (and be downloaded or downloadable)
    
    op2_csv = os.path.join(V2_MANIFEST_DIR, f"{args.field}.op2_planting.csv")
    if not os.path.exists(op2_csv):
        print("Error: OP2 Planting CSV missing. Run v2-confirm-planting first.")
        return
        
    sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.op3.sample.json")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(MANIFEST_DIR, f"{args.field}.sample.json")
    
    config = {
        "crop_size_px": args.crop_size or 256,
        "offset_px": args.offset_px or 40,
        "alive_threshold": args.alive_threshold or 0.05,
        "dead_threshold": args.dead_threshold or 0.01
    }
    
    # Need to pass a customized downloader that injects 'field' if missing in item?
    # cmd_index now injects field into manifest items (in this updated file).
    # If using OLD manifest, field might be missing.
    # Safe to wrap:
    def safe_download(item):
        if 'field' not in item: item['field'] = args.field
        subfolder = f"{item['field']}/{item['stage']}"
        if args.field == 'test':
             return os.path.join("data", "cache", subfolder, item['name'])
        return download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)

    print(f"Running OP3 Survival Analysis (Mode: {args.mode})...")
    summary = analyze_survival(
        field=args.field,
        op2_planting_csv=op2_csv,
        op3_manifest_path=sample_path,
        output_dir=V3_MANIFEST_DIR,
        config=config,
        driver_download_fn=safe_download,
        mode=args.mode,
        max_gemini_calls=args.max_gemini_calls or 200,
        debug_overlay=args.debug_overlay
    )
    
    print("\n=== Survival Summary ===")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Antigravity")
    subparsers = parser.add_subparsers(dest='command')
    
    subparsers.add_parser('list')
    p_index = subparsers.add_parser('index')
    p_index.add_argument('--field', required=True)
    
    p_sample = subparsers.add_parser('sample')
    p_sample.add_argument('--field', required=True)
    p_sample.add_argument('--seed', type=int, default=42)
    p_sample.add_argument('--sample-n', type=int)
    p_sample.add_argument('--sample-frac', type=float)
    p_sample.add_argument('--sample-max-mb', type=int)
    p_sample.add_argument('--stage', default='all', choices=['all', 'op1', 'op2', 'op3'])
    
    p_sync = subparsers.add_parser('sync-sample')
    p_sync.add_argument('--field', required=True)
    p_sync.add_argument('--stage', default='all', choices=['all', 'op1', 'op2', 'op3'])
    
    p_sum = subparsers.add_parser('summary')
    p_sum.add_argument('--field', required=True)
    
    # Legacy
    p_detect = subparsers.add_parser('detect-pits')
    p_detect.add_argument('--field', required=True)
    p_conf = subparsers.add_parser('confirm-planting')
    p_conf.add_argument('--field', required=True)
    # ... args ...
    
    # V2
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

    # V3 (Survival)
    p_surv = subparsers.add_parser('op3-survival')
    p_surv.add_argument('--field', required=True)
    p_surv.add_argument('--mode', default='hybrid', choices=['cv', 'hybrid', 'gemini'])
    p_surv.add_argument('--crop-size', type=int)
    p_surv.add_argument('--offset-px', type=int)
    p_surv.add_argument('--alive-threshold', type=float)
    p_surv.add_argument('--dead-threshold', type=float)
    p_surv.add_argument('--max-gemini-calls', type=int)
    p_surv.add_argument('--debug-overlay', action='store_true')
    
    # Run All
    p_all = subparsers.add_parser('run-all')
    p_all.add_argument('--field', required=True)
    p_all.add_argument('--seed', type=int, default=42)
    p_all.add_argument('--sample-n', type=int)
    p_all.add_argument('--mode', default='hybrid')

    args = parser.parse_args()
    
    if args.command == 'index': cmd_index(args)
    elif args.command == 'sample': cmd_sample(args)
    elif args.command == 'sync-sample': cmd_sync_sample(args)
    elif args.command == 'summary': cmd_summary(args)
    
    elif args.command == 'v2-detect-pits': cmd_v2_detect_pits(args)
    elif args.command == 'v2-confirm-planting': cmd_v2_confirm_planting(args)
    elif args.command == 'v2-confirm-establishment': cmd_v2_confirm_establishment(args)
    
    elif args.command == 'op3-survival': cmd_op3_survival(args)
    elif args.command == 'run-all': run_all_stages(args)
    
    elif args.command == 'list': print("Use 'index'...")
    else: parser.print_help()

if __name__ == "__main__":
    main()
