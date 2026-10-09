import os
import sys
import json
import glob

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def format_all_e5_tables(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dataset = data["dataset"].upper()
    models  = data["models"]
    m = "SASRec" # Main sequential architecture display
    
    print("\n" + "=" * 76)
    print(f"TABLE 14: QUARTILE-STRATIFIED NDCG@10 & MATCHED GAP — {dataset}")
    print("=" * 76)
    print(f"{'Proxy':<5} | {'Q1 (Low)':<10} | {'Q2':<8} | {'Q3':<8} | {'Q4 (High)':<10} | {'Gap (Raw/Match)':<17}")
    print("-" * 76)
    t1 = data.get("task1_quartile_ndcg", {}).get(m, {})
    for p in ["PB", "HT", "RB", "NP", "DI"]:
        info = t1.get(p, {})
        q1 = info.get("Q1", 0.0)
        q2 = info.get("Q2", 0.0)
        q3 = info.get("Q3", 0.0)
        q4 = info.get("Q4", 0.0)
        g_raw = info.get("gap_raw", 0.0)
        g_mat = info.get("gap_match", 0.0)
        print(f"{p:<5} | {q1:.4f}{'':<4} | {q2:.4f}{'':<2} | {q3:.4f}{'':<2} | {q4:.4f}{'':<4} | {g_raw:+.4f} / {g_mat:+.4f}")
    print("=" * 76)

    print("\n" + "=" * 76)
    print(f"TABLE 15: ENCODING-UTILIZATION GAP (EUG) & CALIBRATION — {dataset}")
    print("=" * 76)
    print(f"{'Proxy':<5} | {'Encoding ρ':<12} | {'Alignment γ':<13} | {'Cal Error':<11} | {'EUG (ρ - γ)':<13}")
    print("-" * 76)
    t2 = data.get("task2_calibration_alignment", {}).get(m, {})
    for p in ["PB", "HT"]:
        info = t2.get(p, {})
        rho = info.get("rho", 0.0)
        gamma = info.get("gamma", 0.0)
        cal = info.get("cal_err", 0.0)
        eug = info.get("eug", 0.0)
        print(f"{p:<5} | {rho:.3f}{'':<7} | {gamma:.3f}{'':<8} | {cal:.3f}{'':<6} | {eug:+.3f}{'':<8}")
    print("=" * 76)

    print("\n" + "=" * 76)
    print(f"TABLE 16: AMNESIC CONCEPT ERASURE & CAUSAL UTILIZATION — {dataset}")
    print("=" * 76)
    print(f"{'Proxy':<5} | {'Δ_raw (Drop)':<14} | {'Δ_rand (Control)':<18} | {'Δ_causal (Net Drop)':<19}")
    print("-" * 76)
    t3 = data.get("task3_amnesic_erasure", {}).get(m, {})
    for p in ["PB", "HT", "RB", "NP", "DI"]:
        info = t3.get(p, {})
        d_raw = info.get("delta_raw", 0.0)
        d_rnd = info.get("delta_rand", 0.0)
        d_cau = info.get("delta_causal", 0.0)
        print(f"{p:<5} | {d_raw:+.4f}{'':<7} | {d_rnd:+.4f}{'':<11} | {d_cau:+.4f}{'':<12}")
    print("=" * 76)

    print("\n" + "=" * 76)
    print(f"TABLE 17: REPRESENTATION STEERING (α ∈ [-2..+2]) — {dataset}")
    print("=" * 76)
    print(f"{'Proxy':<5} | {'α = -2':<12} | {'α = -1':<12} | {'α = 0 (Base)':<14} | {'α = +1':<12} | {'α = +2':<12}")
    print("-" * 76)
    t4 = data.get("task4_steering", {}).get(m, {})
    for p in ["PB", "HT"]:
        row = []
        for alpha in ["-2", "-1", "0", "1", "2"]:
            val = t4.get(p, {}).get(alpha, {}).get("rec_trait", 0.0)
            row.append(f"{val:.3f}")
        print(f"{p:<5} | {row[0]:<12} | {row[1]:<12} | {row[2]:<14} | {row[3]:<12} | {row[4]:<12}")
    print("=" * 76)

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else "probe_results_e5/E5_results_ml-1m.json"
    format_all_e5_tables(p)
