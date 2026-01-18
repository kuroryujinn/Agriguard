import sys
import subprocess
import os

def run_all_stages(args):
    """
    Executes the pipeline stages sequentially via subprocess to ensure deep environment reset and modularity.
    """
    python_exe = sys.executable
    field = args.field
    
    def run_cmd(cmd_list):
        print(f"--- Running: {' '.join(cmd_list)} ---")
        res = subprocess.run(cmd_list, capture_output=False, text=True) # Pipeline output to stdout
        if res.returncode != 0:
            print(f"Command failed: {cmd_list}")
            sys.exit(res.returncode)

    # 1. Index
    run_cmd([python_exe, "-m", "src.main", "index", "--field", field])
    
    # 2. Sample (All stages)
    # Check if we need to set N. Default from args.
    sample_cmd = [python_exe, "-m", "src.main", "sample", "--field", field, "--stage", "all"]
    if args.sample_n:
        sample_cmd.extend(["--sample-n", str(args.sample_n)])
    if args.seed:
        sample_cmd.extend(["--seed", str(args.seed)])
    run_cmd(sample_cmd)
    
    # 3. Sync
    run_cmd([python_exe, "-m", "src.main", "sync-sample", "--field", field, "--stage", "all"])
    
    # 4. OP1 Pits (V2)
    run_cmd([python_exe, "-m", "src.main", "v2-detect-pits", "--field", field, "--debug-overlay"])
    
    # 5. OP2 Planting (V2)
    run_cmd([python_exe, "-m", "src.main", "v2-confirm-planting", "--field", field, "--debug-overlay"])
    
    # 6. OP3 Establishment (V2 - Optional but good for checks)
    run_cmd([python_exe, "-m", "src.main", "v2-confirm-establishment", "--field", field, "--debug-overlay"])
    
    # 7. Survival Analysis (Hybrid)
    survival_cmd = [python_exe, "-m", "src.main", "op3-survival", "--field", field, "--mode", args.mode]
    run_cmd(survival_cmd)
    
    print("\n=== RUN ALL COMPLETE ===")
    
