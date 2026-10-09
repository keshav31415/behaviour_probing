import os
import sys
import json
import glob
import math
import random
import numpy as np
import torch
from collections import Counter
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

KS = [5, 10, 20, 50, 100]

def compute_dataset_sequentiality(user_train, itemnum):
    """Klenitskiy et al. (2023) sequentiality diagnostics."""
    total_transitions = 0
    repeat_transitions = 0
    transition_counts = Counter()
    from_counts = Counter()
    
    for u, seq in user_train.items():
        if len(seq) < 2:
            continue
        for t in range(len(seq) - 1):
            i1, i2 = seq[t], seq[t+1]
            total_transitions += 1
            if i1 == i2:
                repeat_transitions += 1
            transition_counts[(i1, i2)] += 1
            from_counts[i1] += 1
            
    rep_rate = float(repeat_transitions / max(1, total_transitions))
    
    # Transition entropy
    entropy_sum = 0.0
    for (i1, i2), cnt in transition_counts.items():
        p_trans = cnt / from_counts[i1]
        entropy_sum -= (cnt / total_transitions) * math.log(max(p_trans, 1e-12))
        
    max_entropy = math.log(max(2, itemnum))
    norm_entropy = float(entropy_sum / max_entropy)
    sequential_determinism = float(max(0.0, 1.0 - norm_entropy))
    
    return {
        "repetition_rate": rep_rate,
        "norm_transition_entropy": norm_entropy,
        "sequential_determinism": sequential_determinism,
        "total_transitions": total_transitions
    }

def run_experiment_4_phase_a(dataset_name="ml-1m", seeds=[42, 43, 44, 45, 46], models=["SASRec", "BERT4Rec", "GRU4Rec"], device="cuda" if torch.cuda.is_available() else "cpu"):
    print("=" * 80)
    print(f"EXPERIMENT 4 (PHASE A): DATA REGIMES (RQ4) — DATASET: {dataset_name.upper()}")
    print(f"Device: {device}")
    print("=" * 80)
    
    out_dir = "probe_results_e4"
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = data_partition(dataset_name)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    item_popularity, item_counts = compute_item_popularity(user_train)
    tail_items = head_tail_split(item_counts)
    
    # Base proxies on full training sequence (H_in)
    metrics_full, user_order = compute_proxies(user_train, item_popularity, tail_items, dataset_name, itemnum=itemnum, min_seq_len=10)
    n_users = len(user_order)
    Y_full = np.array([metrics_full[u] for u in user_order])
    user_lens = np.array([len(user_train[u]) for u in user_order])
    
    print(f"Total evaluated users (len >= 10): {n_users}")
    
    # Compute sequentiality diagnostic
    seq_diag = compute_dataset_sequentiality(user_train, itemnum)
    print(f"Dataset Sequentiality Diagnostics: {seq_diag}")
    
    # Precompute prefix-consistent labels for each k
    print("\nPrecomputing prefix-consistent labels for k in [5, 10, 20, 50, 100]...")
    prefix_labels = {}
    for k in KS:
        user_prefix = {u: user_train[u][:k] for u in user_order}
        # Compute proxies on prefix with min_seq_len = min(k, 5)
        p_metrics, _ = compute_proxies(user_prefix, item_popularity, tail_items, dataset_name, itemnum=itemnum, min_seq_len=min(k, 5))
        prefix_labels[k] = np.array([p_metrics[u] for u in user_order])
        print(f"  Prefix k={k} labels ready.")
        
    e4_results = {
        "dataset": dataset_name,
        "n_users": n_users,
        "sequentiality_diagnostics": seq_diag,
        "task1_prefix_truncation": {},  # (model) -> {k: {proxy: {rho_full, rho_prefix}}}
        "task2_cold_start_cohorts": {}, # (model) -> {cohort: {proxy: rho}}
    }
    
    # Standard length cohorts for Task 2
    cohort_bins = [
        ("Short_10_20", (user_lens >= 10) & (user_lens < 20)),
        ("Mid_20_50",   (user_lens >= 20) & (user_lens < 50)),
        ("Warm_50_100", (user_lens >= 50) & (user_lens < 100)),
        ("Long_100plus",(user_lens >= 100)),
    ]
    
    for model_type in models:
        print(f"\n--- Processing Model Architecture: {model_type} ---")
        e4_results["task1_prefix_truncation"][model_type] = {}
        e4_results["task2_cold_start_cohorts"][model_type] = {}
        
        # We will accumulate across seeds
        k_seed_full = {k: [] for k in KS}   # k -> list of (n_seeds, 13)
        k_seed_prefix = {k: [] for k in KS}
        cohort_seed_rhos = {cname: [] for cname, _ in cohort_bins}
        
        for seed in seeds:
            ckpt_pattern = f"e1_checkpoints/{dataset_name}_default_seed{seed}/{model_type}*"
            ckpts = glob.glob(ckpt_pattern)
            if not ckpts:
                ckpt_pattern = f"checkpoints/{dataset_name}_default_seed{seed}/{model_type}*"
                ckpts = glob.glob(ckpt_pattern)
            if not ckpts:
                print(f"[WARN] No checkpoint for {model_type} seed {seed}")
                continue
            ckpt_path = ckpts[0]
            
            # Load model
            model = load_sasrec(model_type, dataset_name, usernum, itemnum, ckpt_path, device)
            maxlen = 200 if dataset_name == 'ml-1m' else 50
            
            # 1. Full embeddings (for Task 2 cold-start cohorts)
            X_full = extract_sasrec_embeddings(model, user_train, user_order, maxlen, device, model_type=model_type, itemnum=itemnum)
            
            # Task 2: Evaluate cohorts for this seed
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_full)
            train_idx, test_idx = train_test_split(np.arange(n_users), test_size=0.2, random_state=seed)
            
            for cname, cmask in cohort_bins:
                te_cmask = cmask[test_idx]
                if np.sum(te_cmask) < 5:
                    continue
                c_rhos = []
                for p_idx in range(13):
                    clf = Ridge(alpha=1.0)
                    clf.fit(X_sc[train_idx], Y_full[train_idx, p_idx])
                    preds = clf.predict(X_sc[test_idx][te_cmask])
                    r, _ = spearmanr(Y_full[test_idx][te_cmask, p_idx], preds)
                    c_rhos.append(float(r) if not np.isnan(r) else 0.0)
                cohort_seed_rhos[cname].append(c_rhos)
                
            # Task 1: Prefix Truncation Grid
            for k in KS:
                # Extract prefix embeddings z^(k)
                X_k = extract_sasrec_embeddings(model, user_train, user_order, maxlen, device, truncate_k=k, model_type=model_type, itemnum=itemnum)
                X_k_sc = StandardScaler().fit_transform(X_k)
                
                # Filter to users with length >= k for rigorous evaluation
                k_mask = (user_lens >= k)
                k_idx = np.where(k_mask)[0]
                if len(k_idx) < 20:
                    continue
                
                tr_k, te_k = train_test_split(k_idx, test_size=0.2, random_state=seed)
                
                rhos_full = []
                rhos_pref = []
                for p_idx in range(13):
                    # Predicting Full-History label
                    clf_fl = Ridge(alpha=1.0)
                    clf_fl.fit(X_k_sc[tr_k], Y_full[tr_k, p_idx])
                    r_fl, _ = spearmanr(Y_full[te_k, p_idx], clf_fl.predict(X_k_sc[te_k]))
                    rhos_full.append(float(r_fl) if not np.isnan(r_fl) else 0.0)
                    
                    # Predicting Prefix-Consistent label
                    clf_pr = Ridge(alpha=1.0)
                    clf_pr.fit(X_k_sc[tr_k], prefix_labels[k][tr_k, p_idx])
                    r_pr, _ = spearmanr(prefix_labels[k][te_k, p_idx], clf_pr.predict(X_k_sc[te_k]))
                    rhos_pref.append(float(r_pr) if not np.isnan(r_pr) else 0.0)
                    
                k_seed_full[k].append(rhos_full)
                k_seed_prefix[k].append(rhos_pref)
                
        # Average Task 1 results across seeds
        for k in KS:
            if k_seed_full[k]:
                arr_fl = np.mean(np.array(k_seed_full[k]), axis=0) # (13,)
                arr_pr = np.mean(np.array(k_seed_prefix[k]), axis=0)
                k_dict = {}
                for p_idx, p in enumerate(PROXY_ABBRS):
                    k_dict[p] = {
                        "rho_full": float(arr_fl[p_idx]),
                        "rho_prefix": float(arr_pr[p_idx])
                    }
                e4_results["task1_prefix_truncation"][model_type][str(k)] = k_dict
                
        # Average Task 2 results across seeds
        for cname, cmask in cohort_bins:
            if cohort_seed_rhos[cname]:
                arr_c = np.mean(np.array(cohort_seed_rhos[cname]), axis=0)
                c_dict = {}
                for p_idx, p in enumerate(PROXY_ABBRS):
                    c_dict[p] = float(arr_c[p_idx])
                e4_results["task2_cold_start_cohorts"][model_type][cname] = c_dict
                
    # Save structured results
    out_file = os.path.join(out_dir, f"E4_results_{dataset_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(e4_results, f, indent=2)
    print(f"\n[COMPLETE] Experiment 4 (Phase A) saved -> {out_file}")
    return e4_results

if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    run_experiment_4_phase_a(dataset_name=ds)
