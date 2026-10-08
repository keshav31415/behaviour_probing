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

def format_all_e3_tables(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dataset = data["dataset"].upper()
    models  = data["models"]
    
    print("\n" + "=" * 78)
    print(f"TABLE 6: PROBE CAPACITY (LINEAR VS MLP) & SELECTIVITY — {dataset}")
    print("=" * 78)
    print(f"{'Proxy':<6} | {'SASRec (Lin / MLP / Δ)':<22} | {'BERT4Rec (Lin / MLP / Δ)':<22} | {'GRU4Rec (Lin / MLP / Δ)':<22}")
    print("-" * 78)
    for p in PROXY_ABBRS:
        cells = []
        for m in models:
            info = data["task1_probe_capacity"].get(m, {}).get(p, {})
            r_l = info.get("rho_lin", 0.0)
            r_m = info.get("rho_mlp", 0.0)
            d   = info.get("delta", 0.0)
            cells.append(f"{r_l:.2f}/{r_m:.2f} ({d:+.2f})")
        print(f"{p:<6} | {cells[0]:<22} | {cells[1]:<22} | {cells[2]:<22}")
    print("=" * 78)
    
    print("\n" + "=" * 78)
    print(f"TABLE 7: LENGTH CONVOLUTION & TERCILES (T1/T2/T3 & Partial ρ) — {dataset}")
    print("=" * 78)
    print(f"{'Proxy':<6} | {'SASRec (Part / T1/T2/T3)':<22} | {'BERT4Rec (Part / T1/T2/T3)':<22} | {'GRU4Rec (Part / T1/T2/T3)':<22}")
    print("-" * 78)
    for p in PROXY_ABBRS:
        cells = []
        for m in models:
            info = data["task2_length_terciles"].get(m, {}).get(p, {})
            pr = info.get("partial_rho", 0.0)
            t1 = info.get("t1_short", 0.0)
            t2 = info.get("t2_mid", 0.0)
            t3 = info.get("t3_long", 0.0)
            cells.append(f"{pr:.2f} [{t1:.2f}/{t2:.2f}/{t3:.2f}]")
        print(f"{p:<6} | {cells[0]:<22} | {cells[1]:<22} | {cells[2]:<22}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print(f"TABLE 8: LABEL LEAKAGE ABLATION (H_in vs H_full vs H_fut) — {dataset}")
    print("=" * 78)
    print(f"{'Proxy':<6} | {'SASRec (in / full / fut)':<22} | {'BERT4Rec (in / full / fut)':<22} | {'GRU4Rec (in / full / fut)':<22}")
    print("-" * 78)
    for p in PROXY_ABBRS:
        cells = []
        for m in models:
            info = data["task4_leakage"].get(m, {}).get(p, {})
            r_in = info.get("rho_in", 0.0)
            r_fl = info.get("rho_full", 0.0)
            r_ft = info.get("rho_fut", 0.0)
            cells.append(f"{r_in:.2f} / {r_fl:.2f} / {r_ft:.2f}")
        print(f"{p:<6} | {cells[0]:<22} | {cells[1]:<22} | {cells[2]:<22}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print(f"TABLE 9: PROBE DIRECTION STABILITY (Cosine Sim Across 5 Seeds) — {dataset}")
    print("=" * 78)
    print(f"{'Proxy':<6} | {'SASRec (cos θ)':<22} | {'BERT4Rec (cos θ)':<22} | {'GRU4Rec (cos θ)':<22}")
    print("-" * 78)
    for p in PROXY_ABBRS:
        cells = []
        for m in models:
            cos_val = data["task3_direction_cosine"].get(m, {}).get(p, 0.0)
            cells.append(f"{cos_val:.3f}")
        print(f"{p:<6} | {cells[0]:<22} | {cells[1]:<22} | {cells[2]:<22}")
    print("=" * 78)

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else "probe_results_e3/E3_results_ml-1m.json"
    format_all_e3_tables(p)
