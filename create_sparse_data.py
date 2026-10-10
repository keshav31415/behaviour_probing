import os
import random
from collections import defaultdict

fractions = [0.10, 0.25, 0.50]
datasets = ["ml-1m", "Beauty", "Steam"]

for ds in datasets:
    raw_path = f"data/{ds}.txt"
    if not os.path.exists(raw_path):
        print(f"Skipping {raw_path}, file not found")
        continue
        
    user_interactions = defaultdict(list)
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, i = int(parts[0]), int(parts[1])
                user_interactions[u].append(i)
                
    all_users = sorted(list(user_interactions.keys()))
    n_users = len(all_users)
    print(f"[{ds}] Total Users: {n_users}")
    
    for f in fractions:
        sparse_file = f"data/{ds}_sparse_{f:.2f}.txt"
        random.seed(42)
        n_keep = max(10, int(round(f * n_users)))
        kept_users = set(random.sample(all_users, n_keep))
        
        with open(sparse_file, "w", encoding="utf-8") as out:
            for new_u, u in enumerate(sorted(list(kept_users)), start=1):
                for i in user_interactions[u]:
                    out.write(f"{new_u} {i}\n")
                        
        print(f"  Created {sparse_file}: {len(kept_users)} users ({f*100:.0f}%)")
