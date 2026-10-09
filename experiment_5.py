import os
import sys
import copy
import json
import glob
import math
import random
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

def compute_ndcg_at_k(ranks, k=10):
    ndcgs = np.zeros(len(ranks))
    for idx, r in enumerate(ranks):
        if r < k:
            ndcgs[idx] = 1.0 / np.log2(r + 2)
    return ndcgs

def run_experiment_5(dataset_name="ml-1m", seeds=[42, 43, 44, 45, 46], models=["SASRec", "BERT4Rec", "GRU4Rec"], device="cuda" if torch.cuda.is_available() else "cpu"):
    print("=" * 80)
    print(f"EXPERIMENT 5: ENCODING-UTILIZATION GAP & AMNESIC ERASURE (RQ5) — {dataset_name.upper()}")
    print(f"Device: {device}")
    print("=" * 80)
    
    out_dir = "probe_results_e5"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load dataset interactions
    dataset = data_partition(dataset_name)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    item_popularity, item_counts = compute_item_popularity(user_train)
    tail_items = head_tail_split(item_counts)
    
    # Ground-truth proxies on evaluated cohort
    metrics_in, user_order = compute_proxies(user_train, item_popularity, tail_items, dataset_name, itemnum=itemnum, min_seq_len=10)
    n_users = len(user_order)
    Y_in = np.array([metrics_in[u] for u in user_order])
    print(f"Evaluated Users: {n_users}")
    
    # Target item popularity deciles for popularity matching
    target_items = [user_test[u][0] for u in user_order]
    target_pops = np.array([item_popularity.get(i, 0.0) for i in target_items])
    pop_deciles = np.digitize(target_pops, np.percentile(target_pops, np.linspace(10, 90, 9)))
    maxlen = 200
    
    # Pre-generate standard 100 negative items per user for test ranking
    np.random.seed(42)
    test_candidates = [] # list of 101 item ids per user: [pos_target, neg1, ..., neg100]
    for u in user_order:
        rated = set(user_train[u])
        if u in user_valid and user_valid[u]:
            rated.add(user_valid[u][0])
        if u in user_test and user_test[u]:
            rated.add(user_test[u][0])
        cand = [user_test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            cand.append(t)
        test_candidates.append(cand)
    test_candidates = np.array(test_candidates) # (n_users, 101)
    
    e5_results = {
        "dataset": dataset_name,
        "n_users": n_users,
        "models": models,
        "seeds": seeds,
        "task1_quartile_ndcg": {},      # model -> proxy -> {Q1..Q4, gap_raw, gap_match}
        "task2_calibration_alignment": {}, # model -> proxy -> {gamma, cal_err, eug}
        "task3_amnesic_erasure": {},    # model -> proxy -> {delta_raw, delta_rand, delta_causal}
        "task4_steering": {},           # model -> proxy -> alpha -> {avg_rec_trait, tail_ndcg}
    }
    
    for model_type in models:
        print(f"\n==================================================")
        print(f"Evaluating Model: {model_type}")
        print(f"==================================================")
        
        # Accumulators across seeds
        seed_quartiles = {p: [] for p in PROXY_ABBRS}
        seed_calibration = {p: [] for p in PROXY_ABBRS}
        seed_amnesic = {p: [] for p in PROXY_ABBRS}
        seed_steering = {p: {alpha: [] for alpha in [-2, -1, 0, 1, 2]} for p in ["PB", "HT", "RB", "NP"]}
        
        for seed in seeds:
            ckpt_pattern = f"e1_checkpoints/{dataset_name}_default_seed{seed}/{model_type}*"
            matches = glob.glob(ckpt_pattern)
            if not matches:
                matches = glob.glob(f"e1_checkpoints/**/{model_type}*seed={seed}.pth", recursive=True)
            if not matches:
                print(f"[WARN] No checkpoint for {model_type} seed {seed}")
                continue
            ckpt_path = matches[0]
            
            model, model_args = load_sasrec(dataset_name, ckpt_path, usernum, itemnum, device, model_type, maxlen=maxlen)
            model.eval()
            
            # Extract representations Z
            Z = extract_sasrec_embeddings(model, user_train, user_order, maxlen, device, model_type=model_type, itemnum=itemnum)
            
            # Candidate item embeddings
            cand_tensor = torch.LongTensor(test_candidates).to(device) # (n_users, 101)
            with torch.no_grad():
                cand_embs = model.item_emb(cand_tensor).cpu().numpy() # (n_users, 101, d)
                
            # Base rankings
            # score = z_u . e_i
            base_scores = np.einsum('ud,uid->ui', Z, cand_embs) # (n_users, 101)
            # True item is at index 0. Rank of index 0 in descending order:
            ranks = (base_scores >= base_scores[:, 0:1]).sum(axis=1) - 1 # 0-indexed rank
            base_ndcg = compute_ndcg_at_k(ranks, k=10)
            
            # Fit Ridge probes on Z to obtain probe directions w_phi and encoding rho_phi
            scaler = StandardScaler()
            Z_sc = scaler.fit_transform(Z)
            train_idx, test_idx = train_test_split(np.arange(n_users), test_size=0.2, random_state=seed)
            
            probe_weights = {} # proxy -> unit direction v_phi
            probe_rhos = {}
            for p_idx, p in enumerate(PROXY_ABBRS):
                clf = Ridge(alpha=1.0)
                clf.fit(Z_sc[train_idx], Y_in[train_idx, p_idx])
                preds = clf.predict(Z_sc[test_idx])
                r, _ = spearmanr(Y_in[test_idx, p_idx], preds)
                probe_rhos[p] = float(r) if not np.isnan(r) else 0.0
                w = clf.coef_
                norm_w = np.linalg.norm(w)
                v = w / (norm_w + 1e-9)
                probe_weights[p] = v
                
            # 1. Quartile Stratification & Popularity Matching
            for p_idx, p in enumerate(PROXY_ABBRS):
                vals = Y_in[:, p_idx]
                q_cuts = np.percentile(vals, [25, 50, 75])
                q_assign = np.digitize(vals, q_cuts) # 0 to 3 -> Q1 to Q4
                
                q_ndcgs = [float(np.mean(base_ndcg[q_assign == q])) for q in range(4)]
                gap_raw = q_ndcgs[3] - q_ndcgs[0]
                
                # Popularity matched gap
                # Re-weight Q1 and Q4 by target popularity decile distribution
                decile_weights = np.bincount(pop_deciles, minlength=10) / len(pop_deciles)
                match_q0 = 0.0
                match_q3 = 0.0
                for dec in range(10):
                    mask0 = (q_assign == 0) & (pop_deciles == dec)
                    mask3 = (q_assign == 3) & (pop_deciles == dec)
                    val0 = np.mean(base_ndcg[mask0]) if np.sum(mask0) > 0 else q_ndcgs[0]
                    val3 = np.mean(base_ndcg[mask3]) if np.sum(mask3) > 0 else q_ndcgs[3]
                    match_q0 += decile_weights[dec] * val0
                    match_q3 += decile_weights[dec] * val3
                gap_match = float(match_q3 - match_q0)
                
                seed_quartiles[p].append({
                    "Q1": q_ndcgs[0], "Q2": q_ndcgs[1], "Q3": q_ndcgs[2], "Q4": q_ndcgs[3],
                    "gap_raw": float(gap_raw), "gap_match": float(gap_match)
                })
                
            # 2. Calibration Alignment (Recommended List Trait vs User Trait)
            # Find top-10 items recommended for each user
            top10_indices = np.argsort(-base_scores, axis=1)[:, :10] # (n_users, 10)
            rec_items = np.take_along_axis(test_candidates, top10_indices, axis=1) # (n_users, 10)
            
            # Compute top-10 recommended popularity and head-tail
            rec_pops = np.mean(np.vectorize(lambda i: item_popularity.get(i, 0.0))(rec_items), axis=1)
            rec_ht   = np.mean(np.vectorize(lambda i: 1.0 if i in tail_items else 0.0)(rec_items), axis=1)
            
            for p_idx, p in enumerate(["PB", "HT"]):
                target_rec = rec_pops if p == "PB" else rec_ht
                user_trait = Y_in[:, p_idx]
                gamma, _ = spearmanr(target_rec, user_trait)
                gamma = float(gamma) if not np.isnan(gamma) else 0.0
                cal_err = float(np.mean(np.abs(target_rec - user_trait)))
                eug = float(probe_rhos[p] - gamma)
                seed_calibration[p].append({
                    "gamma": gamma, "cal_err": cal_err, "eug": eug, "rho": probe_rhos[p]
                })
                
            # 3. Amnesic Concept Erasure (Nullspace Projection)
            # Sample random direction control
            np.random.seed(seed)
            v_rand = np.random.randn(Z.shape[1])
            v_rand /= np.linalg.norm(v_rand)
            
            # Erase random direction
            Z_rand = Z - np.outer(Z.dot(v_rand), v_rand)
            scores_rand = np.einsum('ud,uid->ui', Z_rand, cand_embs)
            ranks_rand = (scores_rand >= scores_rand[:, 0:1]).sum(axis=1) - 1
            ndcg_rand = compute_ndcg_at_k(ranks_rand, k=10)
            delta_rand = float(np.mean(base_ndcg - ndcg_rand))
            
            for p in ["PB", "HT", "RB", "NP", "DI"]:
                v = probe_weights[p]
                # Nullspace projection
                Z_null = Z - np.outer(Z.dot(v), v)
                scores_null = np.einsum('ud,uid->ui', Z_null, cand_embs)
                ranks_null = (scores_null >= scores_null[:, 0:1]).sum(axis=1) - 1
                ndcg_null = compute_ndcg_at_k(ranks_null, k=10)
                delta_raw = float(np.mean(base_ndcg - ndcg_null))
                delta_causal = float(delta_raw - delta_rand)
                seed_amnesic[p].append({
                    "delta_raw": delta_raw, "delta_rand": delta_rand, "delta_causal": delta_causal
                })
                
            # 4. Steering along probe direction (alpha in [-2, -1, 0, 1, 2])
            for p in ["PB", "HT"]:
                v = probe_weights[p]
                std_proj = float(np.std(Z.dot(v)))
                vals_p = Y_in[:, PROXY_ABBRS.index(p)]
                tail_mask = (vals_p >= np.percentile(vals_p, 75)) if p == "HT" else (vals_p <= np.percentile(vals_p, 25))
                for alpha in [-2, -1, 0, 1, 2]:
                    Z_steer = Z + alpha * std_proj * v
                    scores_steer = np.einsum('ud,uid->ui', Z_steer, cand_embs)
                    ranks_steer = (scores_steer >= scores_steer[:, 0:1]).sum(axis=1) - 1
                    ndcg_steer = compute_ndcg_at_k(ranks_steer, k=10)
                    top10_steer = np.take_along_axis(test_candidates, np.argsort(-scores_steer, axis=1)[:, :10], axis=1)
                    if p == "HT":
                        rec_trait = float(np.mean(np.vectorize(lambda i: 1.0 if i in tail_items else 0.0)(top10_steer)))
                    else:
                        rec_trait = float(np.mean(np.vectorize(lambda i: item_popularity.get(i, 0.0))(top10_steer)))
                    tail_ndcg = float(np.mean(ndcg_steer[tail_mask]))
                    seed_steering[p][alpha].append({"rec_trait": rec_trait, "tail_ndcg": tail_ndcg})
                    
        # Average results across seeds for this model
        e5_results["task1_quartile_ndcg"][model_type] = {}
        for p in PROXY_ABBRS:
            if seed_quartiles[p]:
                keys = ["Q1", "Q2", "Q3", "Q4", "gap_raw", "gap_match"]
                e5_results["task1_quartile_ndcg"][model_type][p] = {
                    k: float(np.mean([item[k] for item in seed_quartiles[p]])) for k in keys
                }
                
        e5_results["task2_calibration_alignment"][model_type] = {}
        for p in ["PB", "HT"]:
            if seed_calibration[p]:
                keys = ["gamma", "cal_err", "eug", "rho"]
                e5_results["task2_calibration_alignment"][model_type][p] = {
                    k: float(np.mean([item[k] for item in seed_calibration[p]])) for k in keys
                }
                
        e5_results["task3_amnesic_erasure"][model_type] = {}
        for p in ["PB", "HT", "RB", "NP", "DI"]:
            if seed_amnesic[p]:
                keys = ["delta_raw", "delta_rand", "delta_causal"]
                e5_results["task3_amnesic_erasure"][model_type][p] = {
                    k: float(np.mean([item[k] for item in seed_amnesic[p]])) for k in keys
                }
                
        e5_results["task4_steering"][model_type] = {}
        for p in ["PB", "HT"]:
            e5_results["task4_steering"][model_type][p] = {}
            for alpha in [-2, -1, 0, 1, 2]:
                if seed_steering[p][alpha]:
                    e5_results["task4_steering"][model_type][p][str(alpha)] = {
                        "rec_trait": float(np.mean([item["rec_trait"] for item in seed_steering[p][alpha]])),
                        "tail_ndcg": float(np.mean([item["tail_ndcg"] for item in seed_steering[p][alpha]])),
                    }
                    
    # Save structured results
    out_file = os.path.join(out_dir, f"E5_results_{dataset_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(e5_results, f, indent=2)
    print(f"\n[COMPLETE] Experiment 5 saved -> {out_file}")
    return e5_results

if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    run_experiment_5(dataset_name=ds)
