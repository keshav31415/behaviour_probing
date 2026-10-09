import os
import sys
import json
import glob

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROXY_ABBRS = [
    "PB", "PC", "HT", "NP", "DI", "ER", "CR",
    "RB", "PM", "TP", "DR", "TS", "RS"
]

def format_all_e4_phase_b_tables(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dataset = data["dataset"].upper()
    models  = data["models"]
    
    print("\n" + "=" * 76)
    print(f"TABLE 13: TRAINING SPARSITY DEGRADATION GRID (10%..100%) — {dataset}")
    print("=" * 76)
    print(f"{'Proxy':<5} | {'10% Data':<15} | {'25% Data':<15} | {'50% Data':<15} | {'100% Data':<15}")
    print("-" * 76)
    
    # We display SASRec by default
    m = "SASRec"
    grid = data.get("sparsity_grid", {}).get(m, {})
    r10  = grid.get("0.10", {})
    r25  = grid.get("0.25", {})
    r50  = grid.get("0.50", {})
    r100 = grid.get("1.00", {})
    
    for p in ["PB", "PC", "HT", "NP", "DI", "RB", "PM", "TP", "TS", "RS"]:
        v10  = r10.get(p, 0.0)
        v25  = r25.get(p, 0.0)
        v50  = r50.get(p, 0.0)
        v100 = r100.get(p, 0.0)
        print(f"{p:<5} | {v10:.3f}{'':<10} | {v25:.3f}{'':<10} | {v50:.3f}{'':<10} | {v100:.3f}{'':<10}")
    print("=" * 76)

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else "probe_results_e4/E4_PhaseB_results_ml-1m.json"
    format_all_e4_phase_b_tables(p)
