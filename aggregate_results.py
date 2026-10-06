import os
import sys
import glob
import json
import argparse
import math

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def aggregate_table3_results(results_base_dir='probe_results', output_dir='probe_results'):
    json_files = glob.glob(os.path.join(results_base_dir, '**', 'Table3_E1_*.json'), recursive=True)
    if not json_files:
        print(f"No Table3_E1_*.json files found under {results_base_dir}")
        return

    print(f"Found {len(json_files)} Table 3 seed JSON files.")

    # Group by (dataset, model_type)
    collected = {}
    for jpath in json_files:
        try:
            with open(jpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            key = (data['dataset'], data['model_type'])
            if key not in collected:
                collected[key] = []
            collected[key].append(data)
        except Exception as e:
            print(f"Error reading {jpath}: {e}")

    os.makedirs(output_dir, exist_ok=True)
    out_txt = os.path.join(output_dir, 'Table3_multi_seed_summary.txt')

    with open(out_txt, 'w', encoding='utf-8') as out_f:
        header = f"{'='*105}\nTABLE 3: MULTI-SEED STATISTICAL SUMMARY (Mean ± Std across seeds)\n{'='*105}\n"
        print(header)
        out_f.write(header + '\n')

        for (dataset, model_type), runs in sorted(collected.items()):
            n_seeds = len(runs)
            proxies = runs[0]['proxies']
            osi_map = runs[0].get('osi', {})

            section = f"\nDataset: {dataset.upper()} | Model: {model_type} | Evaluated across {n_seeds} seeds\n"
            section += f"{'-'*105}\n"
            section += f"{'Proxy':<35} | {'OSI':>7} | {model_type + ' (mean ± std)':<22} | {'MF-SVD':<15} | {'BoI':<15} | {'Last-Item':<15}\n"
            section += f"{'-'*105}\n"
            print(section)
            out_f.write(section)

            for pi, proxy in enumerate(proxies):
                osi_val = osi_map.get(proxy, 0.0)

                def get_stat(model_key):
                    vals = []
                    part_vals = []
                    for r in runs:
                        m_res = r['results'].get(model_key, [])
                        if pi < len(m_res):
                            rho = m_res[pi].get('rho')
                            pr = m_res[pi].get('partial_rho')
                            if rho is not None and not math.isnan(float(rho)):
                                vals.append(float(rho))
                            if pr is not None and not math.isnan(float(pr)):
                                part_vals.append(float(pr))
                    if not vals:
                        return "N/A"
                    mean = sum(vals) / len(vals)
                    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
                    p_mean = sum(part_vals) / len(part_vals) if part_vals else None

                    if p_mean is not None:
                        if std > 0:
                            return f"{mean:.4f}±{std:.4f} ({p_mean:.4f})"
                        return f"{mean:.4f} ({p_mean:.4f})"
                    else:
                        if std > 0:
                            return f"{mean:.4f}±{std:.4f}"
                        return f"{mean:.4f}"

                c_model = get_stat(model_type)
                c_mf    = get_stat('MF-SVD')
                c_boi   = get_stat('BoI')
                c_last  = get_stat('Last-Item')

                row = f"{proxy:<35} | {osi_val:7.4f} | {c_model:<22} | {c_mf:<15} | {c_boi:<15} | {c_last:<15}\n"
                print(row, end="")
                out_f.write(row)

            out_f.write(f"{'-'*105}\n\n")

    print(f"\nMulti-seed Table 3 summary written to: {out_txt}\n")

def aggregate_seed_results(results_base_dir='probe_results', output_dir='probe_results'):
    aggregate_table3_results(results_base_dir, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate probing results across multiple seeds")
    parser.add_argument('--results_dir', default='probe_results', help='Base directory containing seed_* results')
    parser.add_argument('--out_dir', default='probe_results', help='Output directory for summary')
    args = parser.parse_args()

    aggregate_seed_results(args.results_dir, args.out_dir)
