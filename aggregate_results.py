# aggregate_results.py
import os
import glob
import json
import argparse
import numpy as np

def aggregate_seed_results(results_base_dir='probe_results', output_dir='probe_results'):
    seed_dirs = sorted(glob.glob(os.path.join(results_base_dir, 'seed_*')))
    if not seed_dirs:
        # Check if json files are directly in results_base_dir
        json_files = glob.glob(os.path.join(results_base_dir, 'results_*.json'))
        if not json_files:
            print(f"No seed directories or result JSONs found in {results_base_dir}")
            return
        seed_dirs = [results_base_dir]

    print(f"Found {len(seed_dirs)} seed result directory/directories: {seed_dirs}")

    # Map: (dataset, model_type) -> list of seed results
    collected = {}

    for sdir in seed_dirs:
        for jpath in glob.glob(os.path.join(sdir, 'results_*.json')):
            try:
                with open(jpath, 'r') as f:
                    data = json.load(f)
                key = (data['dataset'], data['model_type'])
                if key not in collected:
                    collected[key] = []
                collected[key].append(data)
            except Exception as e:
                print(f"Error reading {jpath}: {e}")

    if not collected:
        print("No valid results collected.")
        return

    os.makedirs(output_dir, exist_ok=True)
    summary_txt = os.path.join(output_dir, 'statistical_summary.txt')

    with open(summary_txt, 'w') as out_f:
        header = f"{'='*80}\nBEHAVIORAL PROBING: MULTI-SEED STATISTICAL SUMMARY (Mean ± Std)\n{'='*80}\n"
        print(header)
        out_f.write(header + '\n')

        for (dataset, model_type), runs in sorted(collected.items()):
            n_runs = len(runs)
            proxies = runs[0]['proxies']
            
            section = f"\nDataset: {dataset.upper()} | Model: {model_type} | Seeds evaluated: {n_runs}\n"
            section += f"{'-'*80}\n"
            section += f"{'Proxy Name':<26} | {'Sequential (ρ)':<18} | {'Shuffled (ρ)':<18} | {'MF-SVD (ρ)':<18}\n"
            section += f"{'-'*80}\n"
            print(section)
            out_f.write(section)

            for pi, proxy in enumerate(proxies):
                def get_vals(key_name):
                    vals = []
                    for r in runs:
                        model_res = r.get(key_name)
                        if model_res and pi < len(model_res) and model_res[pi]:
                            rho = model_res[pi].get('rho')
                            if rho is not None and not np.isnan(rho):
                                vals.append(rho)
                    return vals

                seq_vals = get_vals('sequential')
                shuf_vals = get_vals('shuffled')
                mf_vals = get_vals('mf')

                def fmt(vals):
                    if not vals:
                        return "N/A"
                    if len(vals) == 1:
                        return f"{vals[0]:.4f}"
                    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

                line = f"{proxy:<26} | {fmt(seq_vals):<18} | {fmt(shuf_vals):<18} | {fmt(mf_vals):<18}\n"
                print(line, end="")
                out_f.write(line)

            out_f.write(f"{'-'*80}\n\n")

    print(f"\nStatistical summary saved to: {summary_txt}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate probing results across multiple seeds")
    parser.add_argument('--results_dir', default='probe_results', help='Base directory containing seed_* results')
    parser.add_argument('--out_dir', default='probe_results', help='Output directory for summary')
    args = parser.parse_args()

    aggregate_seed_results(args.results_dir, args.out_dir)
