import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def format_all_e6_tables(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dataset = data["dataset"].upper()
    methods = data["methods"]
    
    print("\n" + "=" * 76)
    print(f"TABLE 18: UTILIZATION-AWARE CALIBRATION (UAC) VS BASELINES — {dataset}")
    print("=" * 76)
    print(f"{'Method / Regularizer':<24} | {'NDCG@10':<9} | {'Tail NDCG':<9} | {'Coverage':<8} | {'CalErr':<6}")
    print("-" * 76)
    
    order = [
        ("Base SASRec (λ=0.0)", "UAC_lambda=0.0"),
        ("LogQ-Correction", "LogQ_Correction"),
        ("IPW Reweighting", "IPW_Reweighting"),
        ("Steck Calib Re-rank", "Calibrated_ReRanking"),
        ("UAC (λ = 0.01)", "UAC_lambda=0.01"),
        ("UAC (λ = 0.10)", "UAC_lambda=0.1"),
        ("UAC (λ = 1.00)", "UAC_lambda=1.0"),
        ("Ablation: PB only", "Ablation_PB_only"),
        ("Ablation: Random Dir", "Ablation_Random_Control"),
    ]
    
    for label, key in order:
        if key in methods:
            m = methods[key]
            ondcg = m.get("overall_ndcg", 0.0)
            tndcg = m.get("tail_ndcg", 0.0)
            cov   = m.get("coverage", 0.0)
            cal   = m.get("cal_err", 0.0)
            print(f"{label:<24} | {ondcg:.4f}{'':<3} | {tndcg:.4f}{'':<3} | {cov*100:.1f}%{'':<4} | {cal:.4f}")
            
    print("=" * 76)

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else "probe_results_e6/E6_results_ml-1m.json"
    format_all_e6_tables(p)
