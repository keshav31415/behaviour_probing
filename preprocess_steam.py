import os
import gzip
import ast
import json
import time
import shutil
import random
from datetime import datetime
from collections import defaultdict

def main():
    print("=" * 70)
    print("PREPROCESSING STEAM DATASET (94,220 Users with Full Metadata)")
    print("=" * 70)

    # 1. Backup existing files
    if os.path.exists('data/Steam.txt') and not os.path.exists('data/Steam_senior_backup.txt'):
        print("[1/6] Backing up existing senior Steam files...")
        shutil.copy('data/Steam.txt', 'data/Steam_senior_backup.txt')
        if os.path.exists('data/Steam_shuffled.txt'):
            shutil.copy('data/Steam_shuffled.txt', 'data/Steam_shuffled_senior_backup.txt')
        print("  Saved backup -> data/Steam_senior_backup.txt")
    else:
        print("[1/6] Senior backup already exists or Steam.txt not present. Proceeding...")

    # 2. Parse steam_games.json.gz
    print("\n[2/6] Parsing steam_games.json.gz for game metadata...")
    t0 = time.time()
    game_metadata = {} # raw_app_id -> {genres, title}
    with gzip.open('data/steam_games.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = ast.literal_eval(line)
                gid = str(d.get('id') or d.get('app_id') or '').strip()
                if gid:
                    genres = d.get('genres', [])
                    if isinstance(genres, list):
                        # clean genre names
                        genres = [str(g).strip() for g in genres if g]
                    else:
                        genres = [str(genres).strip()] if genres else []
                    title = str(d.get('app_name') or d.get('title') or '').strip()
                    game_metadata[gid] = {
                        'genres': genres,
                        'title': title
                    }
            except Exception:
                continue

    print(f"  Parsed {len(game_metadata):,} games with metadata in {time.time() - t0:.2f}s")

    # 3. Stream steam_reviews.json.gz to group by user
    print("\n[3/6] Streaming steam_reviews.json.gz and grouping user interactions...")
    t0 = time.time()
    user_interactions = defaultdict(list) # username -> [(ts, raw_app_id)]
    line_count = 0

    with gzip.open('data/steam_reviews.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"    Processed {line_count:,} reviews... ({len(user_interactions):,} unique users found)")
            line = line.strip()
            if not line:
                continue
            try:
                d = ast.literal_eval(line)
                uid = str(d.get('username') or d.get('user_id') or '').strip()
                iid = str(d.get('product_id') or d.get('app_id') or '').strip()
                date_str = str(d.get('date') or '').strip()
                if not uid or not iid or not date_str:
                    continue
                # Parse date to timestamp
                # Expected format: 'YYYY-MM-DD'
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                    ts = int(dt.timestamp())
                except Exception:
                    # fallback approximation or skip
                    continue
                
                # We retain interactions
                user_interactions[uid].append((ts, iid))
            except Exception:
                continue

    print(f"  Finished parsing {line_count:,} reviews in {time.time() - t0:.2f}s. Unique users: {len(user_interactions):,}")

    # 4. Filter users with >= 5 interactions and sort chronologically
    print("\n[4/6] Filtering users with >= 5 chronological interactions...")
    valid_users = {}
    for uid, acts in user_interactions.items():
        if len(acts) >= 5:
            # Sort chronologically by timestamp
            acts_sorted = sorted(acts, key=lambda x: x[0])
            valid_users[uid] = acts_sorted

    print(f"  Found {len(valid_users):,} users with >= 5 interactions.")

    # Target exactly 94,220 users (matching preliminary study)
    target_users = 94220
    if len(valid_users) >= target_users:
        # Sort users by activity / deterministic seed for reproducible high quality
        # Sort by number of interactions descending, take top or reproducible sample
        random.seed(42)
        sorted_uids = sorted(valid_users.keys())
        # To get realistic length distribution similar to preliminary study, take random sample with seed 42
        selected_uids = sorted(random.sample(sorted_uids, target_users))
    else:
        selected_uids = sorted(valid_users.keys())

    print(f"  Selected {len(selected_uids):,} users for the Steam dataset.")

    # 5. Build ID mappings (1-indexed integers)
    print("\n[5/6] Building 1-indexed ID mappings and metadata...")
    u2idx = {}
    i2idx = {}
    u_idx = 1
    i_idx = 1

    final_user_seqs = {} # u_int -> [(ts, i_int)]
    for uid in selected_uids:
        u2idx[uid] = u_idx
        seq_mapped = []
        for ts, raw_iid in valid_users[uid]:
            if raw_iid not in i2idx:
                i2idx[raw_iid] = i_idx
                i_idx += 1
            seq_mapped.append((ts, i2idx[raw_iid]))
        final_user_seqs[u_idx] = seq_mapped
        u_idx += 1

    num_users = len(u2idx)
    num_items = len(i2idx)
    total_interactions = sum(len(seq) for seq in final_user_seqs.values())
    print(f"  Final Dataset Stats:")
    print(f"    Users: {num_users:,}")
    print(f"    Items: {num_items:,}")
    print(f"    Interactions: {total_interactions:,}")
    print(f"    Avg History Length: {total_interactions / num_users:.2f}")

    # Build metadata dict
    # format matching probing.py: meta = {'users': {str(u): {'timestamps': [...]}}, 'items': {str(i): {'genres': [...], 'title': ...}}}
    meta_dict = {
        'users': {},
        'items': {}
    }

    for u_int, seq in final_user_seqs.items():
        meta_dict['users'][str(u_int)] = {
            'timestamps': [ts for ts, i_int in seq]
        }

    for raw_iid, i_int in i2idx.items():
        gdata = game_metadata.get(raw_iid, {'genres': ['Unknown'], 'title': 'Unknown'})
        meta_dict['items'][str(i_int)] = {
            'genres': gdata['genres'] if gdata.get('genres') else ['Unknown'],
            'title': gdata.get('title', 'Unknown')
        }

    # 6. Write output files
    print("\n[6/6] Writing Steam.txt, Steam_shuffled.txt, and Steam_metadata.json...")
    
    # 6a. Steam.txt
    with open('data/Steam.txt', 'w', encoding='utf-8') as f:
        for u_int in range(1, num_users + 1):
            for ts, i_int in final_user_seqs[u_int]:
                f.write(f"{u_int} {i_int}\n")
    print(f"  Written -> data/Steam.txt ({os.path.getsize('data/Steam.txt') / (1024*1024):.2f} MB)")

    # 6b. Steam_shuffled.txt
    random.seed(42)
    with open('data/Steam_shuffled.txt', 'w', encoding='utf-8') as f:
        for u_int in range(1, num_users + 1):
            items_shuffled = [i_int for ts, i_int in final_user_seqs[u_int]]
            random.shuffle(items_shuffled)
            for i_int in items_shuffled:
                f.write(f"{u_int} {i_int}\n")
    print(f"  Written -> data/Steam_shuffled.txt ({os.path.getsize('data/Steam_shuffled.txt') / (1024*1024):.2f} MB)")

    # 6c. Steam_metadata.json
    with open('data/Steam_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f)
    print(f"  Written -> data/Steam_metadata.json ({os.path.getsize('data/Steam_metadata.json') / (1024*1024):.2f} MB)")

    print("\n" + "=" * 70)
    print("SUCCESS: Steam dataset and complete metadata generated successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
