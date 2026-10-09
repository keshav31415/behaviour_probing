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

def format_all_e4_tables(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dataset = data["dataset"].upper()
    seq_diag = data.get("sequentiality_diagnostics", {})
    
    print("\n" + "=" * 76)
    print(f"TABLE 10: PREFIX TRUNCATION GRID (FULL VS PREFIX LABELS) — {dataset}")
    print("=" * 76)
    print(f"{'Proxy':<5} | {'k=5 (Fl/Pr)':<13} | {'k=10 (Fl/Pr)':<13} | {'k=20 (Fl/Pr)':<13} | {'k=50 (Fl/Pr)':<13}")
    print("-" * 76)
    
    # We display SASRec by default or average
    m = "SASRec"
    t1 = data.get("task1_prefix_truncation", {}).get(m, {})
    for p in ["PB", "HT", "NP", "DI", "RB", "PM", "TP", "TS", "RS"]:
        cells = []
        for k in ["5", "10", "20", "50"]:
            info = t1.get(k, {}).get(p, {})
            r_fl = info.get("rho_full", 0.0)
            r_pr = info.get("rho_prefix", 0.0)
            cells.append(f"{r_fl:.2f}/{r_pr:.2f}")
        print(f"{p:<5} | {cells[0]:<13} | {cells[1]:<13} | {cells[2]:<13} | {cells[3]:<13}")
    print("=" * 76)

    print("\n" + "=" * 76)
    print(f"TABLE 11: COLD-START COHORT BREAKDOWN (ρ BY HISTORY LENGTH) — {dataset}")
    print("=" * 76)
    print(f"{'Proxy':<5} | {'Short [10,20)':<13} | {'Mid [20,50)':<13} | {'Warm [50,100)':<13} | {'Long [100+)':<13}")
    print("-" * 76)
    t2 = data.get("task2_cold_start_cohorts", {}).get(m, {})
    for p in ["PB", "PC", "HT", "NP", "DI", "RB", "PM", "TP", "TS", "RS"]:
        s_val = t2.get("Short_10_20", {}).get(p, 0.0)
        m_val = t2.get("Mid_20_50", {}).get(p, 0.0)
        w_val = t2.get("Warm_50_100", {}).get(p, 0.0)
        l_val = t2.get("Long_100plus", {}).get(p, 0.0)
        print(f"{p:<5} | {s_val:.2f}{'':<9} | {m_val:.2f}{'':<9} | {w_val:.2f}{'':<9} | {l_val:.2f}{'':<9}")
    print("=" * 76)

    print("\n" + "=" * 76)
    print(f"DATASET SEQUENTIALITY DIAGNOSTICS (KLENITSKIY ET AL. 2023) — {dataset}")
    print("=" * 76)
    r_rep = seq_diag.get("repetition_rate", 0.0)
    h_norm = seq_diag.get("norm_transition_entropy", 0.0)
    s_det = seq_diag.get("sequential_determinism", 0.0)
    n_trans = seq_diag.get("total_transitions", 0)
    print(f"  • Repeat Interaction Rate (r_rep):         {r_rep:.4f}")
    print(f"  • Normalized Transition Entropy (H_trans): {h_norm:.4f}")
    print(f"  • Sequential Determinism (1 - H_trans):    {s_det:.4f}")
    print(f"  • Total Transition Pairs Analyzed:         {n_trans:,}")
    print("=" * 76)

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else "probe_results_e4/E4_results_ml-1m.json"
    format_all_e4_tables(p)
