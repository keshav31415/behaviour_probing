import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import spearmanr, entropy
import json
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from model import SASRec
from utils import data_partition


# Constants and hyperparameters
PROXY_NAMES = [
    "Popularity Bias",           # distributional
    "Popularity Concentration",  # distributional
    "Head-Tail Ratio",           # distributional
    "Recency Bias",              # temporal
    "Popularity Momentum",       # temporal
    "Rank Stability",            # structural
    "Diversity Index",           # structural
    "Novelty Preference",        # distributional
    "Exploration Rate",          # distributional
    "Cross-Category Reach",      # structural
    "Temporal Stability",        # temporal
]
ORDER_DEPENDENT = [False, False, False, True, True, True, False, False, False, False, True]

MIN_SEQ_LEN    = 10
TAIL_THRESHOLD = 0.20
COLD_START_KS  = [5, 10, 20, 50, 100]
N_NEG_EVAL     = 100   
STRAT_BINS     = 4     



# Proxy Computation


def compute_item_popularity(user_train):
    counts = Counter()
    for u in user_train:
        counts.update(user_train[u])
    max_c = max(counts.values()) if counts else 1
    return {k: v / max_c for k, v in counts.items()}, counts


def head_tail_split(item_counts, tail_frac=TAIL_THRESHOLD):
    total = sum(item_counts.values())
    tail_items, cumulative = set(), 0
    for item, cnt in sorted(item_counts.items(), key=lambda x: x[1]):
        cumulative += cnt
        tail_items.add(item)
        if cumulative / total >= tail_frac:
            break
    return tail_items


def rank_stability(pop_seq):
    N    = len(pop_seq)
    half = N // 2
    if half < 4:
        return np.nan
    first  = pop_seq[:half]
    second = pop_seq[half:half * 2]   # symmetric length
    rho, _ = spearmanr(first, second)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_proxies(user_train, item_popularity, tail_items, dataset_name):
    try:
        with open(f'data/{dataset_name}_metadata.json', 'r') as f:
            meta = json.load(f)
    except Exception:
        meta = {'users': {}, 'items': {}}

    metrics, user_order = {}, []
    from collections import Counter
    import numpy as np
    from scipy.stats import spearmanr, entropy
    
    for u in user_train:
        seq = user_train[u]
        N   = len(seq)
        if N < MIN_SEQ_LEN:
            continue

        pop = np.array([item_popularity.get(i, 0.0) for i in seq])

        pop_bias = float(np.mean(pop))
        pop_conc = float(np.var(pop))
        ht_ratio = float(sum(1 for i in seq if i in tail_items) / N)

        recent_pop   = np.mean([item_popularity.get(i, 0.0) for i in seq[-5:]])
        old_pop      = np.mean([item_popularity.get(i, 0.0) for i in seq[:5]])
        recency_bias = float(recent_pop - old_pop)

        positions = np.arange(N, dtype=float)
        denom = ((positions - positions.mean()) ** 2).sum()
        slope = float(((positions - positions.mean()) * (pop - pop.mean())).sum() / denom) if denom > 1e-9 else 0.0

        rs = rank_stability(pop)
        if np.isnan(rs):
            continue   

        novelty = float(np.mean([-np.log(item_popularity.get(i, 1e-9)) for i in seq]))

        cats = []
        for item in seq:
            c_data = meta['items'].get(str(item), {})
            c = c_data.get('categories', c_data.get('genres', []))
            if isinstance(c, list):
                if len(c) > 0 and isinstance(c[0], list): 
                    c = [x for sub in c for x in sub]
            if isinstance(c, str):
                c = [c]
            cats.extend(c if c else ['unknown'])
            
        if not cats:
            diversity, exploration, cross_category, temp_stab = 0.0, 0.0, 0.0, 0.0
        else:
            cat_counts = Counter(cats)
            total_cats = sum(cat_counts.values())
            shares = np.array(list(cat_counts.values())) / total_cats
            
            diversity = float(1.0 - np.sum(shares ** 2))
            
            top3 = set(c for c, _ in cat_counts.most_common(3))
            exploration = float(sum(1 for c in cats if c not in top3) / len(cats))
            
            cross_category = float(entropy(shares))
            
            half = len(cats) // 2
            if half > 0:
                c1 = Counter(cats[:half])
                c2 = Counter(cats[half:])
                common = set(c1.keys()) | set(c2.keys())
                if len(common) > 1:
                    v1 = [c1.get(c, 0) for c in common]
                    v2 = [c2.get(c, 0) for c in common]
                    rho, _ = spearmanr(v1, v2)
                    temp_stab = float(rho) if not np.isnan(rho) else 0.0
                else:
                    temp_stab = 0.0
            else:
                temp_stab = 0.0

        metrics[u] = np.array([pop_bias, pop_conc, ht_ratio, recency_bias, slope, rs, 
                               diversity, novelty, exploration, cross_category, temp_stab], dtype=float)
        user_order.append(u)

    return metrics, user_order



# Embedding Extraction


def seq_to_arr(seq_list, maxlen):
    arr = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for item in reversed(seq_list):
        arr[idx] = item
        idx -= 1
        if idx == -1:
            break
    return arr


def extract_sasrec_embeddings(model, user_train, user_order, maxlen, device, truncate_k=None, batch_size=256):
    model.eval()
    reps = []
    with torch.no_grad():
        for i in range(0, len(user_order), batch_size):
            batch_u = user_order[i:i+batch_size]
            arrs = []
            for u in batch_u:
                seq = user_train[u]
                if truncate_k is not None:
                    seq = seq[:truncate_k]
                arrs.append(seq_to_arr(seq, maxlen))
            log_feats = model.log2feats(np.array(arrs))
            reps.extend(log_feats[:, -1, :].cpu().numpy())
    return np.array(reps)


def extract_mf_embeddings(user_train, user_order, item_popularity, hidden_dim=50):
    all_items   = sorted(item_popularity.keys())
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    n_users = len(user_order); n_items = len(all_items)
    rows, cols = [], []
    for ri, u in enumerate(user_order):
        for item in user_train[u]:
            if item in item_to_idx:
                rows.append(ri); cols.append(item_to_idx[item])
    mat = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items))
    k   = min(hidden_dim, min(n_users, n_items) - 1)
    U, S, _ = svds(mat, k=k)
    return (U * S[np.newaxis, :]).astype(np.float32)



# Probe Machinery


def scale_split(X, train_idx, test_idx):
    sc = StandardScaler()
    return sc.fit_transform(X[train_idx]), sc.transform(X[test_idx])


def probe_one(X_tr, X_te, y_tr, y_te, alpha=1.0):
    clf = Ridge(alpha=alpha)
    clf.fit(X_tr, y_tr)
    preds     = clf.predict(X_te)
    r2        = r2_score(y_te, preds)
    rho, pval = spearmanr(y_te, preds)
    return dict(r2=r2, rho=rho, pval=pval)


def run_probe_set(X, Y, train_idx, test_idx):
    X_tr, X_te = scale_split(X, train_idx, test_idx)
    Y_tr, Y_te = Y[train_idx], Y[test_idx]
    return [probe_one(X_tr, X_te, Y_tr[:, i], Y_te[:, i]) for i in range(Y.shape[1])]



# Proxy Correlation Matrix


def plot_proxy_correlation(Y, dataset_name, out_dir, fh):
    n = len(PROXY_NAMES)
    corr_mat = np.zeros((n, n))
    pval_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, pval = spearmanr(Y[:, i], Y[:, j])
            corr_mat[i, j] = rho
            pval_mat[i, j] = pval

    
    short = ["PopBias", "PopConc", "HT-Ratio", "RecBias", "PopMom", "RankStab", "DivIndex", "NovPref", "Explore", "CrossCat", "TempStab"]
    _pw(f"\n{'='*65}\n"
        f"  Proxy Correlation Matrix (Spearman ρ) — {dataset_name}\n"
        f"{'='*65}\n"
        f"  {'':12}" + "".join(f"{s:>10}" for s in short) + "\n"
        f"  {'-'*74}\n", fh)
    for i, name in enumerate(PROXY_NAMES):
        row = f"  {short[i]:12}" + "".join(f"{corr_mat[i,j]:10.3f}" for j in range(n))
        _pw(row + "\n", fh)

    
    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_mat[i, j]) > 0.5:
                high_pairs.append((PROXY_NAMES[i], PROXY_NAMES[j], corr_mat[i, j]))
    if high_pairs:
        _pw(f"\n  High correlations (|ρ| > 0.5) to note:\n", fh)
        for a, b, r in high_pairs:
            _pw(f"    {a}  ↔  {b}:  ρ = {r:.3f}\n", fh)
    else:
        _pw(f"\n  No proxy pair exceeds |ρ| = 0.5 — proxies are sufficiently independent.\n", fh)

    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Spearman ρ')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(short, fontsize=9)

    for i in range(n):
        for j in range(n):
            rho = corr_mat[i, j]
            sig = pval_mat[i, j] < 0.05
            txt = f"{rho:.2f}{'*' if sig and i != j else ''}"
            color = 'white' if abs(rho) > 0.6 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold' if i == j else 'normal')

    boundary = sum(1 for x in ORDER_DEPENDENT if not x) - 0.5
    ax.axhline(y=boundary, color='black', linewidth=2, alpha=0.7)
    ax.axvline(x=boundary, color='black', linewidth=2, alpha=0.7)
    ax.text(boundary / 2, -0.85, "static", ha='center', fontsize=8,
            color='#1f77b4', fontweight='bold')
    ax.text(boundary + (n - boundary) / 2, -0.85, "dynamic", ha='center',
            fontsize=8, color='#d62728', fontweight='bold')

    ax.set_title(f"Proxy Correlation Matrix (Spearman ρ)\n{dataset_name}  "
                 f"  * = p < 0.05", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f"proxy_correlation_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Proxy correlation figure saved → {path}")



# Per-User Evaluation


def compute_per_user_rank_percentile(model, user_train, user_valid, user_test, user_order, itemnum, maxlen, device, batch_size=256):
    model.eval()
    rank_pcts = {}
    eval_users = [u for u in user_order if user_train.get(u) and user_test.get(u)]
    with torch.no_grad():
        for i in range(0, len(eval_users), batch_size):
            batch_u = eval_users[i:i+batch_size]
            batch_arrs = []
            batch_item_idx = []
            for u in batch_u:
                seq_items = user_train[u].copy()
                if user_valid.get(u):
                    seq_items = seq_items + [user_valid[u][0]]
                batch_arrs.append(seq_to_arr(seq_items, maxlen))
                rated = set(user_train[u]) | {0}
                if user_valid.get(u):
                    rated.update(user_valid[u])

                true_item = user_test[u][0]
                item_idx  = [true_item]
                for _ in range(N_NEG_EVAL):
                    t = np.random.randint(1, itemnum + 1)
                    while t in rated:
                        t = np.random.randint(1, itemnum + 1)
                    item_idx.append(t)
                batch_item_idx.append(item_idx)

            logits = model.predict(np.array(batch_u), np.array(batch_arrs), np.array(batch_item_idx)).cpu().numpy()
            for b_idx, u in enumerate(batch_u):
                u_logits = logits[b_idx]
                rank  = int((u_logits > u_logits[0]).sum())   # items scoring higher
                rank_pcts[u] = 1.0 - rank / (N_NEG_EVAL + 1)
    return rank_pcts



# Experiment 3: Behavior-Error Correlation


def run_behavior_error_correlation(rank_pcts, metrics, user_order, dataset_name, out_dir, fh):
    valid = [u for u in user_order if u in rank_pcts]
    rp_vec    = np.array([rank_pcts[u] for u in valid])
    error_vec = 1.0 - rp_vec   # higher means model struggled more

    _pw(f"\n{'='*65}\n"
        f"  Experiment 3: Behavior-Error Correlation — {dataset_name}\n"
        f"  n={len(valid)}  |  error = 1 - rank_percentile\n"
        f"{'='*65}\n"
        f"  {'Proxy':<35}  {'Spearman ρ':>12}  {'p-val':>10}  note\n"
        f"  {'-'*68}\n", fh)

    for pi, name in enumerate(PROXY_NAMES):
        proxy_vec = np.array([metrics[u][pi] for u in valid])
        rho, pval = spearmanr(proxy_vec, error_vec)
        note = "harder users ↑" if rho > 0 else "easier users ↑"
        _pw(f"  {name:<35}  {rho:12.4f}  {pval:10.2e}  {note}\n", fh)

    _pw("  Positive ρ: high proxy value → higher model error\n"
        "  Negative ρ: high proxy value → lower model error\n", fh)



# Experiment 4: Behavioral Stratification (NDCG@10 by quartile)


def ndcg_at_k(rank, k=10):
    return 1 / np.log2(rank + 2) if rank < k else 0.0


def run_behavioral_stratification(model, user_train, user_valid, user_test, user_order, metrics, itemnum, maxlen, device, dataset_name, out_dir, fh, batch_size=256):
    print("  Computing per-user NDCG@10 ...")
    ndcg_map = {}
    model.eval()
    eval_users = [u for u in user_order if user_train.get(u) and user_test.get(u)]
    with torch.no_grad():
        for i in range(0, len(eval_users), batch_size):
            batch_u = eval_users[i:i+batch_size]
            batch_arrs = []
            batch_item_idx = []
            for u in batch_u:
                seq_items = user_train[u].copy()
                if user_valid.get(u):
                    seq_items = seq_items + [user_valid[u][0]]
                batch_arrs.append(seq_to_arr(seq_items, maxlen))
                rated = set(user_train[u]) | {0}
                if user_valid.get(u):
                    rated.update(user_valid[u])
                true_item = user_test[u][0]
                item_idx  = [true_item]
                for _ in range(N_NEG_EVAL):
                    t = np.random.randint(1, itemnum + 1)
                    while t in rated:
                        t = np.random.randint(1, itemnum + 1)
                    item_idx.append(t)
                batch_item_idx.append(item_idx)
            logits = model.predict(np.array(batch_u), np.array(batch_arrs), np.array(batch_item_idx)).cpu().numpy()
            for b_idx, u in enumerate(batch_u):
                u_logits = logits[b_idx]
                rank = int((u_logits > u_logits[0]).sum())
                ndcg_map[u] = ndcg_at_k(rank)

    valid_users = [u for u in user_order if u in ndcg_map]
    ndcg_arr    = np.array([ndcg_map[u] for u in valid_users])
    overall_ndcg = float(np.mean(ndcg_arr))

    _pw(f"\n{'='*75}\n"
        f"  Experiment 4: Behavioral Stratification — {dataset_name}\n"
        f"  n={len(valid_users)}  |  overall NDCG@10 = {overall_ndcg:.4f}\n"
        f"{'='*75}\n"
        f"  {'Proxy':<35}  {'Q1 (low)':>10}  {'Q2':>8}  {'Q3':>8}  "
        f"{'Q4 (high)':>10}  {'Q4−Q1':>8}\n"
        f"  {'-'*82}\n", fh)

    n_prox = len(PROXY_NAMES)
    rows = (n_prox + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.5 * rows))
    axes = axes.flatten()
    for extra_ax in axes[n_prox:]:
        extra_ax.set_visible(False)
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for pi, name in enumerate(PROXY_NAMES):
        proxy_arr = np.array([metrics[u][pi] for u in valid_users])
        bins      = np.percentile(proxy_arr, [0, 25, 50, 75, 100])
        q_ndcg    = [[] for _ in range(STRAT_BINS)]

        for i, u in enumerate(valid_users):
            pv = proxy_arr[i]
            for q in range(STRAT_BINS):
                if pv <= bins[q + 1] or q == STRAT_BINS - 1:
                    q_ndcg[q].append(ndcg_arr[i])
                    break

        means = [float(np.mean(q)) if q else 0.0 for q in q_ndcg]
        delta = means[-1] - means[0]
        tag   = "†" if ORDER_DEPENDENT[pi] else " "
        _pw(f"  {name+tag:<35}  {means[0]:10.4f}  {means[1]:8.4f}"
            f"  {means[2]:8.4f}  {means[3]:10.4f}  {delta:8.4f}\n", fh)

        ax = axes[pi]
        qlabels = ["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
        bars = ax.bar(qlabels, means, color=colors, alpha=0.85, edgecolor='white')
        ax.axhline(y=overall_ndcg, color='gray', linestyle='--', alpha=0.6,
                   label=f'overall={overall_ndcg:.3f}')
        ax.set_ylabel("NDCG@10", fontsize=9)
        ax.set_title(f"{'†' if ORDER_DEPENDENT[pi] else ''}  {name}", fontsize=9)
        ax.set_ylim(0, min(1.0, max(means) * 1.40) if max(means) > 0 else 0.1)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=7)
        ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)

    _pw("  Q4−Q1: NDCG difference between highest and lowest proxy quartile.\n"
        "  Negative Q4−Q1 = high proxy value → harder users.\n"
        "  † = order-dependent proxy\n", fh)

    fig.suptitle(f"NDCG@10 by Behavioral Quartile — {dataset_name}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, f"stratification_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Stratification figure saved → {path}")



# Experiment 2: Cold-Start


def run_coldstart(model, user_train, user_order, Y, train_idx, test_idx, maxlen, device, dataset_name, out_dir, fh,  strat_proxy_idx=3):   
    _pw(f"\n{'='*65}\n"
        f"  Experiment 2: Cold-Start Stability — {dataset_name}\n"
        f"  k values: {COLD_START_KS}\n"
        f"{'='*65}\n", fh)

    rho_matrix = np.zeros((len(COLD_START_KS), len(PROXY_NAMES)))
    for ki, k in enumerate(COLD_START_KS):
        X_k = extract_sasrec_embeddings(model, user_train, user_order,
                                         maxlen, device, truncate_k=k)
        X_tr, X_te = scale_split(X_k, train_idx, test_idx)
        for pi in range(len(PROXY_NAMES)):
            res = probe_one(X_tr, X_te, Y[train_idx, pi], Y[test_idx, pi])
            rho_matrix[ki, pi] = res['rho']
        print(f"  k={k:4d}  Popularity Bias ρ={rho_matrix[ki, 0]:.4f}")

    X_full = extract_sasrec_embeddings(model, user_train, user_order,
                                        maxlen, device, truncate_k=None)
    X_tr, X_te = scale_split(X_full, train_idx, test_idx)
    full_rho = [probe_one(X_tr, X_te, Y[train_idx, pi], Y[test_idx, pi])['rho']
                for pi in range(len(PROXY_NAMES))]

    rho_all  = np.vstack([rho_matrix, np.array(full_rho).reshape(1, -1)])
    ks_plot  = COLD_START_KS + ["full"]

    # ── Numeric table ─────────────────────────────────────────────────────────
    _pw(f"  {'k':<8}" + "".join(f"{n[:11]:>14}" for n in PROXY_NAMES) + "\n", fh)
    _pw("  " + "-" * (8 + 14 * len(PROXY_NAMES)) + "\n", fh)
    for ki, k in enumerate(ks_plot):
        row = f"  {str(k):<8}" + "".join(f"{rho_all[ki, pi]:14.4f}"
                                          for pi in range(len(PROXY_NAMES)))
        _pw(row + "\n", fh)

    # Figure 1: overall ρ vs k 
    n_prox = len(PROXY_NAMES)
    rows = (n_prox + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.5 * rows))
    axes = axes.flatten()
    for extra_ax in axes[n_prox:]:
        extra_ax.set_visible(False)
    x_ticks = list(range(len(ks_plot)))

    for pi, name in enumerate(PROXY_NAMES):
        ax    = axes[pi]
        vals  = rho_all[:, pi]
        color = '#d62728' if ORDER_DEPENDENT[pi] else '#1f77b4'
        ax.plot(x_ticks, vals, marker='o', color=color, linewidth=2, markersize=6)
        ax.axhline(y=full_rho[pi], color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(k) for k in ks_plot], fontsize=9)
        ax.set_xlabel("Sequence length k", fontsize=10)
        ax.set_ylabel("Spearman ρ", fontsize=10)
        ax.set_title(f"{'†' if ORDER_DEPENDENT[pi] else ''}  {name}", fontsize=9)
        ax.set_ylim(-0.15, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cold-Start: Signal Recovery vs k — {dataset_name}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, f"coldstart_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Cold-start figure saved → {path}")

    # Figure 2: stratified by behavioral group 
    # Split ALL users (not just test) by strat_proxy to get stable quartile boundaries
    strat_vals   = Y[:, strat_proxy_idx]
    q25          = np.percentile(strat_vals, 25)
    q75          = np.percentile(strat_vals, 75)
    strat_name   = PROXY_NAMES[strat_proxy_idx]

    high_mask = strat_vals >= q75    # top quartile users
    low_mask  = strat_vals <= q25    # bottom quartile users

    # Reindex: we need user-level masks aligned with user_order
    high_users = [u for i, u in enumerate(user_order) if high_mask[i]]
    low_users  = [u for i, u in enumerate(user_order) if low_mask[i]]

    _pw(f"\n  Stratified cold-start: split on '{strat_name}'\n"
        f"  High (top Q): n={len(high_users)}  |  Low (bottom Q): n={len(low_users)}\n", fh)

    def coldstart_rho_for_group(group_users, k):
        
        if len(group_users) < 20:
            return [float('nan')] * len(PROXY_NAMES)
        
        u_to_idx = {u: i for i, u in enumerate(user_order)}
        group_idx = np.array([u_to_idx[u] for u in group_users
                               if u in u_to_idx])
        Y_group = Y[group_idx]
        X_k = extract_sasrec_embeddings(model, user_train, group_users,
                                         maxlen, device, truncate_k=k)
        # For groups we do a simple 80/20 split (no shared split needed here)
        n_g = len(group_idx)
        tr_g, te_g = train_test_split(np.arange(n_g), test_size=0.2, random_state=42)
        results = []
        for pi in range(len(PROXY_NAMES)):
            res = probe_one(*scale_split(X_k, tr_g, te_g),
                            Y_group[tr_g, pi], Y_group[te_g, pi])
            results.append(res['rho'])
        return results

    high_rho_mat = np.zeros((len(COLD_START_KS) + 1, len(PROXY_NAMES)))
    low_rho_mat  = np.zeros((len(COLD_START_KS) + 1, len(PROXY_NAMES)))

    for ki, k in enumerate(COLD_START_KS):
        high_rho_mat[ki] = coldstart_rho_for_group(high_users, k)
        low_rho_mat[ki]  = coldstart_rho_for_group(low_users, k)
        print(f"  [stratified] k={k}")
    
    high_rho_mat[-1] = coldstart_rho_for_group(high_users, None)
    low_rho_mat[-1]  = coldstart_rho_for_group(low_users,  None)

    
    dyn_idx   = [i for i, x in enumerate(ORDER_DEPENDENT) if x]
    dyn_names = [PROXY_NAMES[i] for i in dyn_idx]
    n_dyn     = len(dyn_idx)

    fig2, axes2 = plt.subplots(2, n_dyn, figsize=(5 * n_dyn, 9), sharey=True)
    if n_dyn == 1:
        axes2 = axes2.reshape(2, 1)

    x_ticks  = list(range(len(ks_plot)))
    x_labels = [str(k) for k in ks_plot]

    for col, (pi, name) in enumerate(zip(dyn_idx, dyn_names)):
        for row, (label, rho_mat, color, marker) in enumerate([
            (f"High {strat_name} (Q4)", high_rho_mat, '#d62728', 'o'),
            (f"Low {strat_name}  (Q1)", low_rho_mat,  '#1f77b4', 's'),
        ]):
            ax = axes2[row, col]
            ax.plot(x_ticks, rho_mat[:, pi], marker=marker, color=color,
                    linewidth=2, markersize=6, label=label)
            ax.axhline(y=rho_mat[-1, pi], color='gray', linestyle='--',
                       alpha=0.5, label='full seq ref')
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_xlabel("Sequence length k", fontsize=10)
            ax.set_ylabel("Spearman ρ", fontsize=10)
            ax.set_title(f"{label}\n{name}", fontsize=9)
            ax.set_ylim(-0.20, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    fig2.suptitle(
        f"Cold-Start Signal Recovery by Behavioral Group\n"
        f"Split variable: {strat_name}  |  {dataset_name}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path2 = os.path.join(out_dir, f"coldstart_stratified_{dataset_name}.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Stratified cold-start figure saved → {path2}")

    _pw(f"\n  Signal emergence speed (avg ρ at k=5 → k=full, dynamic proxies):\n", fh)
    for pi_local, (pi, name) in enumerate(zip(dyn_idx, dyn_names)):
        h5   = high_rho_mat[0, pi];  hfull = high_rho_mat[-1, pi]
        l5   = low_rho_mat[0, pi];   lfull = low_rho_mat[-1, pi]
        h_gain = hfull - h5;  l_gain = lfull - l5
        faster = "High" if h5 > l5 else "Low"
        _pw(f"    {name:<30}  High: {h5:.3f}→{hfull:.3f} (gain={h_gain:+.3f})"
            f"  Low: {l5:.3f}→{lfull:.3f} (gain={l_gain:+.3f})"
            f"  Faster emergence: {faster}\n", fh)



# Output Helpers


def _pw(text, fh):
    print(text, end='')
    fh.write(text)


def format_probe_table(label, results, dataset_name, n_users, fh):
    _pw(f"\n{'='*65}\n"
        f"  {label}  |  {dataset_name}  |  n={n_users}\n"
        f"{'='*65}\n"
        f"  {'Proxy':<35}  {'R2':>7}  {'Spearman ρ':>12}  {'p-val':>10}\n"
        f"  {'-'*65}\n", fh)
    for i, name in enumerate(PROXY_NAMES):
        r   = results[i]
        tag = "†" if ORDER_DEPENDENT[i] else " "
        _pw(f"  {name+tag:<35}  {r['r2']:7.4f}  {r['rho']:12.4f}  {r['pval']:10.2e}\n", fh)
    _pw("  † = order-dependent proxy\n", fh)


def format_comparison_table(res_seq, res_shuf, res_mf, res_null, dataset_name, n_users, fh):
    _pw(f"\n{'='*90}\n"
        f"  Full Comparison — {dataset_name}  |  n={n_users}\n"
        f"{'='*90}\n"
        f"  {'Proxy':<35}  {'SASRec':>9}  {'Shuffled':>9}  {'MF-SVD':>9}"
        f"  {'Null':>7}  {'Δ seq-shuf':>11}\n"
        f"  {'-'*87}\n", fh)
    for i, name in enumerate(PROXY_NAMES):
        tag   = "†" if ORDER_DEPENDENT[i] else " "
        rs    = res_seq[i]['rho']
        rsh   = res_shuf[i]['rho'] if res_shuf else float('nan')
        rmf   = res_mf[i]['rho']   if res_mf   else float('nan')
        rn    = res_null[i]['rho']
        delta = rs - rsh if res_shuf else float('nan')
        _pw(f"  {name+tag:<35}  {rs:9.4f}  {rsh:9.4f}  {rmf:9.4f}"
            f"  {rn:7.4f}  {delta:11.4f}\n", fh)
    _pw("  Δ = SASRec(seq) − SASRec(shuffled). Positive Δ on order-dependent\n"
        "  proxies = sequence structure is being leveraged.\n"
        "  † = order-dependent proxy\n", fh)

    static_idx = [i for i, x in enumerate(ORDER_DEPENDENT) if not x]
    seq_idx    = [i for i, x in enumerate(ORDER_DEPENDENT) if x]

    def avg_rho(res, idxs):
        return float(np.mean([res[i]['rho'] for i in idxs]))

    mean_seq_static   = avg_rho(res_seq, static_idx)
    mean_seq_dynamic  = avg_rho(res_seq, seq_idx)

    _pw(f"\n  ── Static vs Sequential Signal Summary ──\n", fh)
    _pw(f"  {'':30}  {'SASRec(seq)':>12}", fh)

    if res_shuf:
        mean_shuf_static  = avg_rho(res_shuf, static_idx)
        mean_shuf_dynamic = avg_rho(res_shuf, seq_idx)
        delta_static  = mean_seq_static  - mean_shuf_static
        delta_dynamic = mean_seq_dynamic - mean_shuf_dynamic
        _pw(f"  {'SASRec(shuf)':>14}  {'Δ':>8}\n", fh)
        _pw(f"  {'Static proxies (avg ρ)':<30}  {mean_seq_static:12.4f}"
            f"  {mean_shuf_static:14.4f}  {delta_static:8.4f}\n", fh)
        _pw(f"  {'Dynamic proxies (avg ρ)':<30}  {mean_seq_dynamic:12.4f}"
            f"  {mean_shuf_dynamic:14.4f}  {delta_dynamic:8.4f}\n", fh)
        _pw(f"\n  Interpretation:\n", fh)
        _pw(f"    Sequential model gains on dynamic proxies after shuffle: {delta_dynamic:+.4f}\n", fh)
        _pw(f"    Sequential model gains on static  proxies after shuffle: {delta_static:+.4f}\n", fh)
        if abs(delta_dynamic) > abs(delta_static):
            _pw(f"    → Sequence structure contributes primarily to order-dependent signals.\n"
                f"      Static behavioral traits are encoded regardless of interaction order.\n", fh)
        else:
            _pw(f"    → Shuffle affects static and dynamic proxies similarly.\n", fh)
    else:
        _pw(f"\n", fh)
        _pw(f"  {'Static proxies (avg ρ)':<30}  {mean_seq_static:12.4f}\n", fh)
        _pw(f"  {'Dynamic proxies (avg ρ)':<30}  {mean_seq_dynamic:12.4f}\n", fh)




def load_sasrec(dataset_name, model_path, usernum, itemnum, device):
    ns = argparse.Namespace(
        dataset=dataset_name, maxlen=args.maxlen, hidden_units=args.hidden_units, num_blocks=2,
        num_epochs=201, num_heads=args.num_heads, dropout_rate=0.2, l2_emb=0.0,
        device=device, norm_first=args.norm_first
    )
    model = SASRec(usernum, itemnum, ns).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, ns



def run_niche_evaluation(model, user_train, user_valid, user_test, user_order, item_popularity, tail_items, itemnum, maxlen, device, dataset_name, out_dir, fh, batch_size=64):
    import time
    _pw(f"\n{'='*75}\n"
        f"  Experiment 5: Niche Preference & Coverage Metrics  [{dataset_name}]\n"
        f"{'='*75}\n", fh)
    
    print("  Computing full-catalog top-10 for Niche Evaluation...")
    eval_users = [u for u in user_order if user_train.get(u) and user_test.get(u)]
    
    # Pre-calculate popularities for stratification
    # Quartiles of item popularity
    pop_values = np.array(list(item_popularity.values()))
    pop_bins = np.percentile(pop_values, [0, 25, 50, 75, 100])
    
    tail_recall = []
    tail_ndcg = []
    recommended_items = set()
    total_tail_recs = 0
    total_recs = 0
    
    stratified_ndcg = [[] for _ in range(4)]
    
    calibration_errors = []
    
    model.eval()
    with torch.no_grad():
        all_items = np.arange(1, itemnum + 1)
        
        for i in range(0, len(eval_users), batch_size):
            batch_u = eval_users[i:i+batch_size]
            batch_arrs = []
            
            for u in batch_u:
                seq_items = user_train[u].copy()
                if user_valid.get(u):
                    seq_items = seq_items + [user_valid[u][0]]
                batch_arrs.append(seq_to_arr(seq_items, maxlen))
            
            # Predict scores for all items
            batch_item_idx = np.tile(all_items, (len(batch_u), 1))
            logits = model.predict(np.array(batch_u), np.array(batch_arrs), batch_item_idx).cpu().numpy()
            
            for b_idx, u in enumerate(batch_u):
                u_logits = logits[b_idx]
                target_i = user_test[u][0]
                
                # Mask out training/validation items
                rated = set(user_train.get(u, []))
                if user_valid.get(u):
                    rated.add(user_valid[u][0])
                for r_item in rated:
                    if 1 <= r_item <= itemnum:
                        u_logits[r_item - 1] = -1e9
                        
                # Get Top-10
                top10_idx = np.argsort(-u_logits)[:10] + 1
                top10_set = set(top10_idx)
                
                # Global metrics
                for rec in top10_idx:
                    recommended_items.add(rec)
                    if rec in tail_items:
                        total_tail_recs += 1
                total_recs += 10
                
                # Calibration
                train_seq = user_train.get(u, [])
                if len(train_seq) > 0:
                    hist_tail_ratio = sum(1 for tr_i in train_seq if tr_i in tail_items) / len(train_seq)
                    rec_tail_ratio = sum(1 for rec in top10_idx if rec in tail_items) / 10.0
                    calibration_errors.append(abs(hist_tail_ratio - rec_tail_ratio))
                
                # Target Evaluation
                # What rank did the true item get?
                target_score = u_logits[target_i - 1]
                rank = (u_logits > target_score).sum()
                ndcg10 = ndcg_at_k(rank, 10)
                
                if target_i in tail_items:
                    tail_recall.append(1.0 if rank < 10 else 0.0)
                    tail_ndcg.append(ndcg10)
                    
                # Stratification
                tgt_pop = item_popularity.get(target_i, 0)
                for q in range(4):
                    if tgt_pop <= pop_bins[q+1] or q == 3:
                        stratified_ndcg[q].append(ndcg10)
                        break

    avg_tail_rec = np.mean(tail_recall) if tail_recall else 0.0
    avg_tail_ndcg = np.mean(tail_ndcg) if tail_ndcg else 0.0
    coverage = len(recommended_items) / float(itemnum)
    exposure = total_tail_recs / float(total_recs) if total_recs > 0 else 0.0
    mae_calib = np.mean(calibration_errors) if calibration_errors else 0.0
    
    strat_means = [np.mean(q) if q else 0.0 for q in stratified_ndcg]
    
    _pw(f"  Tail Recall@10:       {avg_tail_rec:.4f}\n", fh)
    _pw(f"  Tail NDCG@10:         {avg_tail_ndcg:.4f}\n", fh)
    _pw(f"  Catalog Coverage:     {coverage:.4f}  ({len(recommended_items)} / {itemnum})\n", fh)
    _pw(f"  Tail Exposure:        {exposure:.4f}  ({total_tail_recs} / {total_recs})\n", fh)
    _pw(f"  Calibration MAE:      {mae_calib:.4f}\n", fh)
    
    _pw(f"\n  Target Item Popularity Stratification (NDCG@10):\n", fh)
    _pw(f"    Q1 (Rarest items):  {strat_means[0]:.4f}\n", fh)
    _pw(f"    Q2:                 {strat_means[1]:.4f}\n", fh)
    _pw(f"    Q3:                 {strat_means[2]:.4f}\n", fh)
    _pw(f"    Q4 (Most popular):  {strat_means[3]:.4f}\n", fh)


def run_probe(dataset_name, model_path,
              shuffled_model_path=None,
              run_mf=False,
              run_coldstart_flag=False,
              run_behavior_analysis=False,
              device='cuda',
              out_dir='probe_results'):

    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, f"results_{dataset_name}.txt")

    print(f"\n{'#'*65}\n#  Dataset: {dataset_name}\n{'#'*65}\n")

    dataset = data_partition(dataset_name)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    item_popularity, item_counts = compute_item_popularity(user_train)
    tail_items                   = head_tail_split(item_counts)
    metrics, user_order = compute_proxies(user_train, item_popularity, tail_items, dataset_name)
    n_users = len(user_order)
    print(f"[INFO] Qualified users: {n_users}  |  Items: {itemnum}  "
          f"|  Tail items: {len(tail_items)}")

    Y                       = np.array([metrics[u] for u in user_order])
    idx_all                 = np.arange(n_users)
    train_idx, test_idx     = train_test_split(idx_all, test_size=0.2,
                                               random_state=42)

    model, model_args = load_sasrec(dataset_name, model_path,
                                    usernum, itemnum, device)
    model_shuf = None
    if shuffled_model_path:
        model_shuf, _ = load_sasrec(dataset_name, shuffled_model_path,
                                     usernum, itemnum, device)

    with open(out_txt, 'w') as f:
        _pw(f"Behavioral Probing — {dataset_name}\n"
            f"Model : {model_path}\n"
            f"Users : {n_users}  Items: {itemnum}\n\n", f)

        # Probing
        print("\n[Exp 1] Probing representations ...")
        X_seq   = extract_sasrec_embeddings(model, user_train, user_order,
                                             model_args.maxlen, device)
        res_seq = run_probe_set(X_seq, Y, train_idx, test_idx)
        format_probe_table("SASRec (sequential)", res_seq,
                           dataset_name, n_users, f)

        res_shuf = None
        if model_shuf:
            X_shuf   = extract_sasrec_embeddings(model_shuf, user_train,
                                                  user_order,
                                                  model_args.maxlen, device)
            res_shuf = run_probe_set(X_shuf, Y, train_idx, test_idx)
            format_probe_table("SASRec (shuffled)", res_shuf,
                               dataset_name, n_users, f)

        res_mf = None
        if run_mf:
            print("[Exp 1] MF-SVD ...")
            X_mf   = extract_mf_embeddings(user_train, user_order,
                                            item_popularity, hidden_dim=50)
            res_mf = run_probe_set(X_mf, Y, train_idx, test_idx)
            format_probe_table("MF-SVD baseline", res_mf,
                               dataset_name, n_users, f)

        np.random.seed(42)
        X_null   = np.random.randn(n_users, 50).astype(np.float32)
        res_null = run_probe_set(X_null, Y, train_idx, test_idx)
        format_probe_table("Random Null", res_null, dataset_name, n_users, f)

        if res_shuf is not None or res_mf is not None:
            format_comparison_table(res_seq, res_shuf, res_mf, res_null,
                                    dataset_name, n_users, f)

        # Proxy Correlation 
        print("\n[Exp 1] Proxy correlation matrix ...")
        plot_proxy_correlation(Y, dataset_name, out_dir, f)

        # Cold-Start
        if run_coldstart_flag:
            print("\n[Exp 2] Cold-start stability ...")
            run_coldstart(model, user_train, user_order, Y,
                          train_idx, test_idx, model_args.maxlen, device,
                          dataset_name, out_dir, f)

        # Behavior Analysis
        if run_behavior_analysis:
            print("\n[Exp 3] Behavior-error correlation ...")
            rank_pcts = compute_per_user_rank_percentile(
                model, user_train, user_valid, user_test,
                user_order, itemnum, model_args.maxlen, device
            )
            run_behavior_error_correlation(rank_pcts, metrics, user_order,
                                           dataset_name, out_dir, f)

            print("\n[Exp 4] Behavioral stratification ...")
            run_behavioral_stratification(
                model, user_train, user_valid, user_test,
                user_order, metrics, itemnum, model_args.maxlen, device,
                dataset_name, out_dir, f
            )

    
        print("\n[Exp 5] Niche metrics ...")
        run_niche_evaluation(
            model, user_train, user_valid, user_test,
            user_order, item_popularity, tail_items, itemnum, model_args.maxlen, device,
            dataset_name, out_dir, f
        )

    print(f"\n[DONE] → {out_txt}\n")









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',               required=True)
    parser.add_argument('--model_path',            required=True)
    parser.add_argument('--shuffled_model_path',   default=None)
    parser.add_argument('--run_mf',                action='store_true')
    parser.add_argument('--run_coldstart',         action='store_true')
    parser.add_argument('--run_behavior_analysis', action='store_true')
    parser.add_argument('--device',                default='cuda')
    parser.add_argument('--out_dir',               default='probe_results')
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--norm_first', action='store_true', default=False)
    args = parser.parse_args()

    run_probe(
        dataset_name          = args.dataset,
        model_path            = args.model_path,
        shuffled_model_path   = args.shuffled_model_path,
        run_mf                = args.run_mf,
        run_coldstart_flag    = args.run_coldstart,
        run_behavior_analysis = args.run_behavior_analysis,
        device                = args.device,
        out_dir               = args.out_dir,
    )