import os
import sys
import json
import glob
import math
import random
import numpy as np
import torch
import torch.nn as nn
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
    ORDER_DEPENDENT,
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

class MLPProbe(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_eval_mlp(X_train, y_train, X_test, y_test, epochs=40, lr=0.01, seed=42):
    torch.manual_seed(seed)
    model = MLPProbe(input_dim=X_train.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    t_X_train = torch.FloatTensor(X_train)
    t_y_train = torch.FloatTensor(y_train)
    t_X_test  = torch.FloatTensor(X_test)
    
    model.train()
    batch_size = 256
    n = len(X_train)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(t_X_train[idx])
            loss = criterion(pred, t_y_train[idx])
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(t_X_test).numpy()
        
    rho, pval = spearmanr(y_test, preds)
    return float(rho) if not np.isnan(rho) else 0.0

def compute_partial_spearman(x, y, z):
    # Partial Spearman of x and y controlling for z
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)
    
    # Residuals from regressing rx, ry on rz
    A = np.vstack([rz, np.ones_like(rz)]).T
    w_x, _ = np.linalg.lstsq(A, rx, rcond=None)[:2]
    w_y, _ = np.linalg.lstsq(A, ry, rcond=None)[:2]
    
    res_x = rx - A.dot(w_x)
    res_y = ry - A.dot(w_y)
    
    r, _ = spearmanr(res_x, res_y)
    return float(r) if not np.isnan(r) else 0.0

def run_experiment_3(dataset_name="ml-1m", seeds=[42, 43, 44, 45, 46], models=["SASRec", "BERT4Rec", "GRU4Rec"], device="cpu"):
    print("=" * 80)
    print(f"EXPERIMENT 3: PROBE VALIDITY (RQ3) — DATASET: {dataset_name.upper()}")
    print("=" * 80)
    
    out_dir = "probe_results_e3"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load dataset interactions
    dataset = data_partition(dataset_name)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    item_popularity, item_counts = compute_item_popularity(user_train)
    tail_items = head_tail_split(item_counts)
    
    # Ground-truth labels on H_in (Input Sequence)
    metrics_in, user_order = compute_proxies(user_train, item_popularity, tail_items, dataset_name, itemnum=itemnum, min_seq_len=10)
    n_users = len(user_order)
    Y_in = np.array([metrics_in[u] for u in user_order])
    lengths = np.array([len(user_train[u]) for u in user_order])
    
    # Terciles based on length
    t1_thresh, t2_thresh = np.percentile(lengths, [33.33, 66.67])
    
    # 2. Compute labels for Leakage Ablation: H_full and H_fut
    print("\nComputing alternative history regimes for Leakage Ablation...")
    user_full = {}
    user_fut  = {}
    for u in user_order:
        seq = list(user_train[u])
        half = max(3, len(seq) // 2)
        fut = list(seq[half:])
        if u in user_valid and user_valid[u]:
            seq.append(user_valid[u][0])
            fut.append(user_valid[u][0])
        if u in user_test and user_test[u]:
            seq.append(user_test[u][0])
            fut.append(user_test[u][0])
        user_full[u] = seq
        user_fut[u]  = fut
        
    metrics_full, _ = compute_proxies(user_full, item_popularity, tail_items, dataset_name, itemnum=itemnum, min_seq_len=3)
    metrics_fut, _  = compute_proxies(user_fut, item_popularity, tail_items, dataset_name, itemnum=itemnum, min_seq_len=3)
    Y_full = np.array([metrics_full[u] for u in user_order])
    Y_fut  = np.array([metrics_fut[u] for u in user_order])
    print("Labels ready: H_in, H_full, H_fut.")
    
    e3_results = {
        "dataset": dataset_name,
        "n_users": n_users,
        "models": models,
        "seeds": seeds,
        "proxies": PROXY_NAMES,
        "task1_probe_capacity": {}, # (model) -> {proxy: {rho_lin, rho_mlp, delta, sel_lin, sel_mlp}}
        "task2_length_terciles": {},# (model) -> {proxy: {partial_rho, t1_rho, t2_rho, t3_rho}}
        "task3_direction_cosine": {},# (model) -> {proxy: mean_cos}
        "task4_leakage": {},        # (model) -> {proxy: {rho_in, rho_full, rho_fut}}
    }
    
    for model_type in models:
        print(f"\n{'-'*60}")
        print(f"--- EVALUATING MODEL: {model_type} ({dataset_name}) ---")
        print(f"{'-'*60}")
        
        # Collect across seeds
        seed_lin_rhos = []
        seed_mlp_rhos = []
        seed_t1_rhos  = []
        seed_t2_rhos  = []
        seed_t3_rhos  = []
        seed_partial_rhos = []
        seed_weights  = [] # list of (13, 50) arrays
        
        seed_full_rhos = []
        seed_fut_rhos  = []
        
        for seed in seeds:
            # Locate checkpoint in e1_checkpoints/
            ckpt_pattern = f"e1_checkpoints/{dataset_name}_default_seed{seed}/{model_type}*.epoch=201*.pth"
            matches = glob.glob(ckpt_pattern)
            if not matches:
                # Try fallback search
                matches = glob.glob(f"e1_checkpoints/**/{model_type}*seed={seed}.pth", recursive=True)
            if not matches:
                print(f"Warning: checkpoint not found for {dataset_name} {model_type} seed {seed}. Skipping.")
                continue
            ckpt_path = matches[0]
            
            # Load model and extract representations X
            model, model_args = load_sasrec(dataset_name, ckpt_path, usernum, itemnum, device, model_type)
            X = extract_sasrec_embeddings(model, user_train, user_order, model_args.maxlen, device, model_type=model_type, itemnum=itemnum)
            
            # Train/Test Split (80/20)
            idx_all = np.arange(n_users)
            train_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=seed)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test  = scaler.transform(X[test_idx])
            
            test_lens = lengths[test_idx]
            t1_mask = test_lens <= t1_thresh
            t2_mask = (test_lens > t1_thresh) & (test_lens <= t2_thresh)
            t3_mask = test_lens > t2_thresh
            
            lin_rhos = []
            mlp_rhos = []
            t1_rhos  = []
            t2_rhos  = []
            t3_rhos  = []
            part_rhos = []
            weights  = []
            
            full_rhos = []
            fut_rhos  = []
            
            for p_idx in range(len(PROXY_NAMES)):
                y_tr = Y_in[train_idx, p_idx]
                y_te = Y_in[test_idx, p_idx]
                
                # 1. Linear Ridge Probe
                clf = Ridge(alpha=1.0)
                clf.fit(X_train, y_tr)
                pred_te = clf.predict(X_test)
                r_lin, _ = spearmanr(y_te, pred_te)
                lin_rhos.append(float(r_lin))
                weights.append(clf.coef_)
                
                # 2. Non-linear MLP Probe
                r_mlp = train_eval_mlp(X_train, y_tr, X_test, y_te, epochs=35, lr=0.01, seed=seed)
                mlp_rhos.append(r_mlp)
                
                # 3. Length Terciles
                r_t1 = float(spearmanr(y_te[t1_mask], pred_te[t1_mask])[0]) if np.sum(t1_mask) > 5 else 0.0
                r_t2 = float(spearmanr(y_te[t2_mask], pred_te[t2_mask])[0]) if np.sum(t2_mask) > 5 else 0.0
                r_t3 = float(spearmanr(y_te[t3_mask], pred_te[t3_mask])[0]) if np.sum(t3_mask) > 5 else 0.0
                t1_rhos.append(r_t1 if not np.isnan(r_t1) else 0.0)
                t2_rhos.append(r_t2 if not np.isnan(r_t2) else 0.0)
                t3_rhos.append(r_t3 if not np.isnan(r_t3) else 0.0)
                
                # Partial Spearman controlling for length
                pr = compute_partial_spearman(y_te, pred_te, test_lens)
                part_rhos.append(pr)
                
                # 4. Leakage Ablation
                clf_full = Ridge(alpha=1.0)
                clf_full.fit(X_train, Y_full[train_idx, p_idx])
                r_full, _ = spearmanr(Y_full[test_idx, p_idx], clf_full.predict(X_test))
                full_rhos.append(float(r_full) if not np.isnan(r_full) else 0.0)
                
                clf_fut = Ridge(alpha=1.0)
                clf_fut.fit(X_train, Y_fut[train_idx, p_idx])
                r_fut, _ = spearmanr(Y_fut[test_idx, p_idx], clf_fut.predict(X_test))
                fut_rhos.append(float(r_fut) if not np.isnan(r_fut) else 0.0)

            seed_lin_rhos.append(lin_rhos)
            seed_mlp_rhos.append(mlp_rhos)
            seed_t1_rhos.append(t1_rhos)
            seed_t2_rhos.append(t2_rhos)
            seed_t3_rhos.append(t3_rhos)
            seed_partial_rhos.append(part_rhos)
            seed_weights.append(np.array(weights)) # (13, 50)
            seed_full_rhos.append(full_rhos)
            seed_fut_rhos.append(fut_rhos)

        # Average across seeds
        arr_lin = np.array(seed_lin_rhos) # (n_seeds, 13)
        arr_mlp = np.array(seed_mlp_rhos)
        arr_t1  = np.array(seed_t1_rhos)
        arr_t2  = np.array(seed_t2_rhos)
        arr_t3  = np.array(seed_t3_rhos)
        arr_pr  = np.array(seed_partial_rhos)
        arr_full= np.array(seed_full_rhos)
        arr_fut = np.array(seed_fut_rhos)
        
        # RS control index is 12
        rs_lin_mean = np.mean(arr_lin[:, 12])
        rs_mlp_mean = np.mean(arr_mlp[:, 12])
        
        # Task 1: Probe Capacity & Selectivity
        cap_dict = {}
        for p in range(13):
            m_lin, s_lin = np.mean(arr_lin[:, p]), np.std(arr_lin[:, p])
            m_mlp, s_mlp = np.mean(arr_mlp[:, p]), np.std(arr_mlp[:, p])
            delta = m_mlp - m_lin
            sel_lin = m_lin - rs_lin_mean
            sel_mlp = m_mlp - rs_mlp_mean
            cap_dict[PROXY_ABBRS[p]] = {
                "rho_lin": m_lin, "std_lin": s_lin,
                "rho_mlp": m_mlp, "std_mlp": s_mlp,
                "delta": delta,
                "sel_lin": sel_lin, "sel_mlp": sel_mlp
            }
        e3_results["task1_probe_capacity"][model_type] = cap_dict
        
        # Task 2: Length Confounding & Terciles
        terc_dict = {}
        for p in range(13):
            terc_dict[PROXY_ABBRS[p]] = {
                "partial_rho": float(np.mean(arr_pr[:, p])),
                "t1_short": float(np.mean(arr_t1[:, p])),
                "t2_mid":   float(np.mean(arr_t2[:, p])),
                "t3_long":  float(np.mean(arr_t3[:, p])),
            }
        e3_results["task2_length_terciles"][model_type] = terc_dict
        
        # Task 3: Probe Direction Stability (Cosine Similarity across seeds)
        cos_dict = {}
        for p in range(13):
            cos_sims = []
            n_s = len(seed_weights)
            for i in range(n_s):
                for j in range(i+1, n_s):
                    w1 = seed_weights[i][p]
                    w2 = seed_weights[j][p]
                    cos = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-9)
                    cos_sims.append(cos)
            cos_dict[PROXY_ABBRS[p]] = float(np.mean(cos_sims)) if cos_sims else 1.0
        e3_results["task3_direction_cosine"][model_type] = cos_dict
        
        # Task 4: Leakage Ablation
        leak_dict = {}
        for p in range(13):
            leak_dict[PROXY_ABBRS[p]] = {
                "rho_in":   float(np.mean(arr_lin[:, p])),
                "rho_full": float(np.mean(arr_full[:, p])),
                "rho_fut":  float(np.mean(arr_fut[:, p])),
            }
        e3_results["task4_leakage"][model_type] = leak_dict

    # Save structured JSON
    json_path = os.path.join(out_dir, f"E3_results_{dataset_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(e3_results, f, indent=2)
    print(f"\n[SAVED] Experiment 3 results written → {json_path}")
    return e3_results

if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    run_experiment_3(dataset_name=ds)
