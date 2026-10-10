import os
import sys
import copy
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
def fit_ridge_direction(Z, y, alpha=1.0):
    Z_mean = np.mean(Z, axis=0)
    Z_std = np.std(Z, axis=0) + 1e-8
    Z_sc = (Z - Z_mean) / Z_std
    d = Z_sc.shape[1]
    w = np.linalg.solve(Z_sc.T @ Z_sc + alpha * np.eye(d), Z_sc.T @ y)
    v = w / (np.linalg.norm(w) + 1e-8)
    return v

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

def compute_metrics_for_rankings(candidate_scores, test_candidates, item_counts, total_items, tail_items, k=10):
    """
    candidate_scores: (n_users, 101) array of scores for [true_item, neg_1, ..., neg_100]
    test_candidates: (n_users, 101) item ids
    item_counts: dict of item -> frequency
    """
    n_users = len(candidate_scores)
    ndcgs = np.zeros(n_users)
    topk_items = []
    
    ranks = (candidate_scores >= candidate_scores[:, 0:1]).sum(axis=1) - 1 # 0-indexed rank of true item
    for idx, r in enumerate(ranks):
        if r < k:
            ndcgs[idx] = 1.0 / np.log2(r + 2)
            
    # Top-k recommended item IDs per user
    topk_indices = np.argsort(-candidate_scores, axis=1)[:, :k]
    for u_idx in range(n_users):
        u_topk = test_candidates[u_idx, topk_indices[u_idx]]
        topk_items.extend(u_topk)
        
    topk_items = np.array(topk_items)
    unique_covered = len(np.unique(topk_items))
    catalog_coverage = unique_covered / float(total_items)
    
    # Average recommendation popularity
    pop_sum = sum(item_counts.get(i, 0) for i in topk_items)
    avg_rec_pop = pop_sum / float(len(topk_items)) if len(topk_items) > 0 else 0.0
    
    return ndcgs, catalog_coverage, avg_rec_pop, topk_indices

def run_experiment_6(dataset_name="ml-1m", seed=42, device="cuda" if torch.cuda.is_available() else "cpu"):
    print("=" * 80)
    print(f"EXPERIMENT 6: UTILIZATION-AWARE CALIBRATION REGULARIZER (RQ6) — {dataset_name.upper()}")
    print(f"Device: {device}")
    print("=" * 80)
    
    out_dirs = ["probe_results_e6"]
    if os.path.exists("/kaggle/working"):
        out_dirs.append("/kaggle/working/probe_results_e6")
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    # 1. Load dataset
    dataset = data_partition(dataset_name)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    item_popularity, item_counts = compute_item_popularity(user_train)
    tail_items = head_tail_split(item_counts)
    
    metrics_in, user_order = compute_proxies(user_train, item_popularity, tail_items, dataset_name, itemnum=itemnum, min_seq_len=10)
    n_users = len(user_order)
    user_to_idx = {u: idx for idx, u in enumerate(user_order)}
    
    # Ground truth traits
    user_pb = np.array([metrics_in[u][0] for u in user_order]) # Popularity Bias (0)
    user_ht = np.array([metrics_in[u][2] for u in user_order]) # Head-Tail ratio (2)
    user_tail_q1 = (user_ht <= np.percentile(user_ht, 25))     # Q1 tail / niche users
    
    # Pre-generate standard 100 negatives for testing
    np.random.seed(seed)
    test_candidates = []
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
    
    # Load base converged model from E1
    import glob
    ckpt_pattern = f"e1_checkpoints/{dataset_name}_default_seed{seed}/SASRec*"
    matches = glob.glob(ckpt_pattern)
    if not matches:
        matches = glob.glob(f"e1_checkpoints/**/SASRec*seed={seed}.pth", recursive=True)
    if not matches:
        raise FileNotFoundError(f"Base checkpoint not found for SASRec on {dataset_name}")
    base_ckpt = matches[0]
    print(f"Loaded Base Converged SASRec: {base_ckpt}")
    
    base_model, model_args = load_sasrec(dataset_name, base_ckpt, usernum, itemnum, device, 'SASRec')
    base_model.eval()
    
    maxlen = model_args.maxlen
    Z_base = extract_sasrec_embeddings(base_model, user_train, user_order, maxlen, device, model_type='SASRec', itemnum=itemnum)
    cand_tensor = torch.LongTensor(test_candidates).to(device)
    with torch.no_grad():
        cand_embs = base_model.item_emb(cand_tensor).cpu().numpy()
    base_scores = np.einsum('ud,uid->ui', Z_base, cand_embs)
    
    base_ndcgs, base_cov, base_pop, base_topk = compute_metrics_for_rankings(
        base_scores, test_candidates, item_counts, itemnum, tail_items, k=10
    )
    base_tail_ndcg = float(np.mean(base_ndcgs[user_tail_q1]))
    base_overall_ndcg = float(np.mean(base_ndcgs))
    
    # Calculate Base Calibration Error (Wasserstein / L1 between user trait and recommended trait)
    rec_tail_props = np.zeros(n_users)
    for u_idx in range(n_users):
        rec_items = test_candidates[u_idx, base_topk[u_idx]]
        rec_tail_props[u_idx] = sum(1 for i in rec_items if i in tail_items) / 10.0
    base_cal_err = float(np.mean(np.abs(rec_tail_props - user_ht)))
    
    print(f"Base SASRec -> Overall NDCG: {base_overall_ndcg:.4f} | Tail NDCG: {base_tail_ndcg:.4f} | CalErr: {base_cal_err:.4f}")
    
    e6_results = {
        "dataset": dataset_name,
        "n_users": n_users,
        "base_metrics": {
            "overall_ndcg": base_overall_ndcg,
            "tail_ndcg": base_tail_ndcg,
            "coverage": base_cov,
            "avg_pop": base_pop,
            "cal_err": base_cal_err,
        },
        "methods": {}
    }
    
    # 2. Benchmark 1: Steck Calibrated Re-ranking (2018)
    print("\n--- Evaluating Baseline: Steck Calibrated Re-ranking ---")
    calib_scores = np.copy(base_scores)
    # Penalize popularity of candidates for high-tail users
    for u_idx in range(n_users):
        target_tail_prob = user_ht[u_idx]
        cand_pops = np.array([item_popularity.get(c, 0.0) for c in test_candidates[u_idx]])
        # Calibrated score: trade off raw score with item popularity penalty
        calib_scores[u_idx] -= 0.15 * (1.0 - target_tail_prob) * cand_pops
        
    c_ndcgs, c_cov, c_pop, c_topk = compute_metrics_for_rankings(
        calib_scores, test_candidates, item_counts, itemnum, tail_items, k=10
    )
    rec_tail_c = np.array([sum(1 for i in test_candidates[u, c_topk[u]] if i in tail_items) / 10.0 for u in range(n_users)])
    c_cal = float(np.mean(np.abs(rec_tail_c - user_ht)))
    e6_results["methods"]["Calibrated_ReRanking"] = {
        "overall_ndcg": float(np.mean(c_ndcgs)),
        "tail_ndcg": float(np.mean(c_ndcgs[user_tail_q1])),
        "coverage": c_cov,
        "avg_pop": c_pop,
        "cal_err": c_cal
    }
    
    # 3. Benchmark 2: LogQ Correction (Popularity debiasing)
    print("--- Evaluating Baseline: LogQ Correction ---")
    logq_scores = np.copy(base_scores)
    item_probs = np.array([max(item_counts.get(c, 1), 1) / float(len(user_train)) for c in range(itemnum + 1)])
    for u_idx in range(n_users):
        q_vals = np.array([math.log(item_probs[min(c, itemnum)]) for c in test_candidates[u_idx]])
        logq_scores[u_idx] -= 0.05 * q_vals
    l_ndcgs, l_cov, l_pop, l_topk = compute_metrics_for_rankings(
        logq_scores, test_candidates, item_counts, itemnum, tail_items, k=10
    )
    rec_tail_l = np.array([sum(1 for i in test_candidates[u, l_topk[u]] if i in tail_items) / 10.0 for u in range(n_users)])
    l_cal = float(np.mean(np.abs(rec_tail_l - user_ht)))
    e6_results["methods"]["LogQ_Correction"] = {
        "overall_ndcg": float(np.mean(l_ndcgs)),
        "tail_ndcg": float(np.mean(l_ndcgs[user_tail_q1])),
        "coverage": l_cov,
        "avg_pop": l_pop,
        "cal_err": l_cal
    }
    
    # 4. Benchmark 3: Inverse Propensity Weighting (IPW)
    print("--- Evaluating Baseline: IPW Reweighted ---")
    ipw_scores = np.copy(base_scores)
    for u_idx in range(n_users):
        weights = np.array([1.0 / math.sqrt(max(item_counts.get(c, 1), 1)) for c in test_candidates[u_idx]])
        weights = weights / np.mean(weights)
        ipw_scores[u_idx] += 0.02 * weights
    ipw_ndcgs, ipw_cov, ipw_pop, ipw_topk = compute_metrics_for_rankings(
        ipw_scores, test_candidates, item_counts, itemnum, tail_items, k=10
    )
    rec_tail_ipw = np.array([sum(1 for i in test_candidates[u, ipw_topk[u]] if i in tail_items) / 10.0 for u in range(n_users)])
    ipw_cal = float(np.mean(np.abs(rec_tail_ipw - user_ht)))
    e6_results["methods"]["IPW_Reweighting"] = {
        "overall_ndcg": float(np.mean(ipw_ndcgs)),
        "tail_ndcg": float(np.mean(ipw_ndcgs[user_tail_q1])),
        "coverage": ipw_cov,
        "avg_pop": ipw_pop,
        "cal_err": ipw_cal
    }
    
    # 5. Utilization-Aware Calibration Regularizer (UAC) Lambda Sweep
    # UAC regularizer dynamically guides representations z_u along probe alignment direction v_HT
    print("\n--- Evaluating Proposed Method: Utilization-Aware Calibration (UAC) ---")
    # Obtain HT probe direction from Z_base
    v_ht = fit_ridge_direction(Z_base, user_ht, alpha=1.0)
    
    lambda_grid = [0.0, 0.01, 0.1, 1.0]
    for lam in lambda_grid:
        if lam == 0.0:
            e6_results["methods"]["UAC_lambda=0.0"] = {
                "overall_ndcg": base_overall_ndcg,
                "tail_ndcg": base_tail_ndcg,
                "coverage": base_cov,
                "avg_pop": base_pop,
                "cal_err": base_cal_err
            }
            continue
            
        print(f"  Evaluating UAC with lambda = {lam}...")
        # Calibrated representation: z'_u = z_u + lambda * (user_ht - rec_ht) * v_ht
        # Direct soft calibration alignment
        uac_Z = np.copy(Z_base)
        # Shift along probe direction proportionally to calibration gap
        for u_idx in range(n_users):
            misalignment = user_ht[u_idx] - rec_tail_props[u_idx]
            uac_Z[u_idx] += (lam * 0.15 * misalignment) * v_ht
            
        uac_scores = np.einsum('ud,uid->ui', uac_Z, cand_embs)
        u_ndcgs, u_cov, u_pop, u_topk = compute_metrics_for_rankings(
            uac_scores, test_candidates, item_counts, itemnum, tail_items, k=10
        )
        rec_tail_u = np.array([sum(1 for i in test_candidates[u, u_topk[u]] if i in tail_items) / 10.0 for u in range(n_users)])
        u_cal = float(np.mean(np.abs(rec_tail_u - user_ht)))
        
        e6_results["methods"][f"UAC_lambda={lam}"] = {
            "overall_ndcg": float(np.mean(u_ndcgs)),
            "tail_ndcg": float(np.mean(u_ndcgs[user_tail_q1])),
            "coverage": u_cov,
            "avg_pop": u_pop,
            "cal_err": u_cal
        }
        print(f"    -> Overall: {np.mean(u_ndcgs):.4f} | Tail: {np.mean(u_ndcgs[user_tail_q1]):.4f} | CalErr: {u_cal:.4f}")
        
    # 6. Proxy Ablations (lambda=0.1): HT only vs PB only vs Random Control
    print("\n--- Evaluating Proxy Ablations (at lambda=0.1) ---")
    # PB probe
    v_pb = fit_ridge_direction(Z_base, user_pb, alpha=1.0)
    
    # Random direction control
    np.random.seed(42)
    v_rand = np.random.randn(len(v_ht))
    v_rand /= np.linalg.norm(v_rand)
    
    # Ablation: PB only
    pb_Z = np.copy(Z_base)
    for u_idx in range(n_users):
        pb_Z[u_idx] -= (0.1 * 0.15 * (user_pb[u_idx] - 0.5)) * v_pb
    pb_scores = np.einsum('ud,uid->ui', pb_Z, cand_embs)
    pb_ndcgs, pb_cov, pb_pop, pb_topk = compute_metrics_for_rankings(pb_scores, test_candidates, item_counts, itemnum, tail_items, k=10)
    e6_results["methods"]["Ablation_PB_only"] = {
        "overall_ndcg": float(np.mean(pb_ndcgs)),
        "tail_ndcg": float(np.mean(pb_ndcgs[user_tail_q1])),
        "coverage": pb_cov,
        "avg_pop": pb_pop,
        "cal_err": float(np.mean(np.abs(np.array([sum(1 for i in test_candidates[u, pb_topk[u]] if i in tail_items)/10.0 for u in range(n_users)]) - user_ht)))
    }
    
    # Ablation: Random Proxy Control
    rnd_Z = np.copy(Z_base)
    for u_idx in range(n_users):
        rnd_Z[u_idx] += (0.1 * 0.15 * (user_ht[u_idx] - rec_tail_props[u_idx])) * v_rand
    rnd_scores = np.einsum('ud,uid->ui', rnd_Z, cand_embs)
    rnd_ndcgs, rnd_cov, rnd_pop, rnd_topk = compute_metrics_for_rankings(rnd_scores, test_candidates, item_counts, itemnum, tail_items, k=10)
    e6_results["methods"]["Ablation_Random_Control"] = {
        "overall_ndcg": float(np.mean(rnd_ndcgs)),
        "tail_ndcg": float(np.mean(rnd_ndcgs[user_tail_q1])),
        "coverage": rnd_cov,
        "avg_pop": rnd_pop,
        "cal_err": float(np.mean(np.abs(np.array([sum(1 for i in test_candidates[u, rnd_topk[u]] if i in tail_items)/10.0 for u in range(n_users)]) - user_ht)))
    }
    
    # Save results
    for d in out_dirs:
        out_file = os.path.join(d, f"E6_results_{dataset_name}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(e6_results, f, indent=2)
        print(f"Saved E6 results -> {out_file}")
        
    return e6_results

if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    run_experiment_6(dataset_name=ds)
