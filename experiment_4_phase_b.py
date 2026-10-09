import os
import sys
import json
import glob
import time
import subprocess
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from utils import data_partition
from probing import (
    PROXY_NAMES,
    compute_item_popularity,
    head_tail_split,
    compute_proxies,
    load_sasrec,
    extract_sasrec_embeddings,
)

PROXY_ABBRS = [
    "PB", "PC", "HT", "NP", "DI", "ER", "CR",
    "RB", "PM", "TP", "DR", "TS", "RS"
]

FRACTIONS = [0.10, 0.25, 0.50]

def run_experiment_4_phase_b(dataset_name="ml-1m", seeds=[42, 43, 44, 45, 46], models=["SASRec", "BERT4Rec", "GRU4Rec"], device="cuda" if torch.cuda.is_available() else "cpu"):
    print("=" * 80)
    print(f"EXPERIMENT 4 (PHASE B): TRAINING SPARSITY GRID (RQ4) — DATASET: {dataset_name.upper()}")
    print(f"Device: {device}")
    print("=" * 80)
    
    out_dir = "probe_results_e4"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load full base dataset interactions for standard ground-truth evaluation
    base_dataset = data_partition(dataset_name)
    [user_train, user_valid, user_test, usernum, itemnum] = base_dataset
    item_popularity, item_counts = compute_item_popularity(user_train)
    tail_items = head_tail_split(item_counts)
    
    # Ground truth proxies on the full evaluated cohort
    metrics_in, user_order = compute_proxies(user_train, item_popularity, tail_items, dataset_name, itemnum=itemnum, min_seq_len=10)
    n_users = len(user_order)
    Y_in = np.array([metrics_in[u] for u in user_order])
    print(f"Standard Evaluation Cohort: {n_users} users")
    
    # Check if 100% results exist from E3
    e3_file = f"probe_results_e3/E3_results_{dataset_name}.json"
    rhos_100 = {}
    if os.path.exists(e3_file):
        try:
            with open(e3_file, "r", encoding="utf-8") as f:
                e3_data = json.load(f)
            for m in models:
                rhos_100[m] = {p: e3_data["task1_probe_capacity"].get(m, {}).get(p, {}).get("rho_lin", 0.0) for p in PROXY_ABBRS}
            print(f"Loaded 100% baseline results from {e3_file}")
        except Exception as e:
            print(f"[WARN] Failed to read {e3_file}: {e}")
            
    e4_b_results = {
        "dataset": dataset_name,
        "n_users": n_users,
        "fractions": [0.10, 0.25, 0.50, 1.00],
        "models": models,
        "seeds": seeds,
        "sparsity_grid": {}, # model -> fraction -> proxy -> rho
    }
    
    maxlen = 200 if dataset_name == 'ml-1m' else 50
    
    for model_type in models:
        print(f"\n==================================================")
        print(f"Processing Model: {model_type}")
        print(f"==================================================")
        e4_b_results["sparsity_grid"][model_type] = {}
        
        # Add 1.00 baseline if available
        if model_type in rhos_100:
            e4_b_results["sparsity_grid"][model_type]["1.00"] = rhos_100[model_type]
            
        for f in FRACTIONS:
            f_str = f"{f:.2f}"
            sparse_ds = f"{dataset_name}_sparse_{f_str}"
            print(f"\n--- Training & Probing Sparsity Fraction: {f*100:.0f}% ({sparse_ds}) ---")
            
            seed_rhos = []
            for seed in seeds:
                train_dir = f"sparse_{f_str}_seed{seed}"
                
                # Check if checkpoint already exists
                ckpt_dir = f"{sparse_ds}_{train_dir}"
                existing = glob.glob(f"{ckpt_dir}/{model_type}.epoch=201*.pth")
                
                if not existing:
                    # Train model via main.py
                    cmd = [
                        sys.executable, "main.py",
                        f"--dataset={sparse_ds}",
                        f"--train_dir={train_dir}",
                        f"--model_type={model_type}",
                        f"--seed={seed}",
                        "--num_epochs=201",
                        f"--maxlen={maxlen}",
                        f"--device={device}"
                    ]
                    print(f"  Training {model_type} seed {seed} ({f*100:.0f}% data)...", flush=True)
                    t0 = time.time()
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    dur = time.time() - t0
                    if res.returncode != 0:
                        print(f"    [ERROR] Training failed for {model_type} seed {seed}: {res.stderr[-300:]}")
                        continue
                    else:
                        print(f"    [DONE] Trained in {dur:.1f}s", flush=True)
                        
                # Locate final checkpoint
                matches = glob.glob(f"{ckpt_dir}/{model_type}.epoch=201*.pth")
                if not matches:
                    matches = glob.glob(f"{ckpt_dir}/*.pth")
                if not matches:
                    print(f"    [WARN] No checkpoint found in {ckpt_dir}")
                    continue
                ckpt_path = matches[0]
                
                # Load trained sparse model and extract embeddings on standard full cohort
                model, model_args = load_sasrec(sparse_ds, ckpt_path, usernum, itemnum, device, model_type, maxlen=maxlen)
                X = extract_sasrec_embeddings(model, user_train, user_order, maxlen, device, model_type=model_type, itemnum=itemnum)
                
                # Fit linear Ridge probe
                idx_all = np.arange(n_users)
                tr_idx, te_idx = train_test_split(idx_all, test_size=0.2, random_state=seed)
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr_idx])
                X_te = scaler.transform(X[te_idx])
                
                rhos = []
                for p_idx in range(13):
                    clf = Ridge(alpha=1.0)
                    clf.fit(X_tr, Y_in[tr_idx, p_idx])
                    preds = clf.predict(X_te)
                    r, _ = spearmanr(Y_in[te_idx, p_idx], preds)
                    rhos.append(float(r) if not np.isnan(r) else 0.0)
                seed_rhos.append(rhos)
                
            if seed_rhos:
                mean_rhos = np.mean(np.array(seed_rhos), axis=0) # (13,)
                p_dict = {PROXY_ABBRS[p_idx]: float(mean_rhos[p_idx]) for p_idx in range(13)}
                e4_b_results["sparsity_grid"][model_type][f_str] = p_dict
                print(f"  Averaged {len(seed_rhos)} seeds for {model_type} at {f*100:.0f}% data.")
                
    # Save structured results
    out_file = os.path.join(out_dir, f"E4_PhaseB_results_{dataset_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(e4_b_results, f, indent=2)
    print(f"\n[COMPLETE] Experiment 4 Phase B saved -> {out_file}")
    return e4_b_results

if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    run_experiment_4_phase_b(dataset_name=ds)
