# run_kaggle_pipeline.py
import os
import sys
import time
import json
import subprocess
import dotenv

dotenv.load_dotenv()

def get_account_config(account="main"):
    if account == "main":
        user = os.getenv("KAGGLE_USERNAME_MAIN") or os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_API_TOKEN_MAIN") or os.getenv("KAGGLE_KEY")
        slug = os.getenv("KAGGLE_NOTEBOOK_MAIN") or os.getenv("KAGGLE_NOTEBOOK_SLUG", "notebooke73a53a269")
    elif account == "acc2":
        user = os.getenv("KAGGLE_USERNAME_ACC2")
        key = os.getenv("KAGGLE_API_TOKEN_ACC2")
        slug = os.getenv("KAGGLE_NOTEBOOK_ACC2")
    elif account == "acc3":
        user = os.getenv("KAGGLE_USERNAME_ACC3")
        key = os.getenv("KAGGLE_API_TOKEN_ACC3")
        slug = os.getenv("KAGGLE_NOTEBOOK_ACC3")
    else:
        raise ValueError(f"Unknown account: {account}")
    return user, key, slug

def run_cmd(cmd, cwd=None, env=None):
    res = subprocess.run(cmd, cwd=cwd, env=env, shell=True, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr

def ensure_metadata(username, slug, title="Behaviour Probing Pipeline"):
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "kaggle_pipeline.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true"
    }
    with open("kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated kernel-metadata.json for {username}/{slug}")

def push_git():
    print("\n[1/4] Checking and pushing latest local changes to GitHub...")
    run_cmd("git add main.py probing.py utils.py shuffle_seqs.py aggregate_results.py kaggle_pipeline.ipynb .gitignore")
    code, out, _ = run_cmd("git status --porcelain")
    if any(k in out for k in ["main.py", "probing.py", "kaggle_pipeline.ipynb", "utils.py"]):
        run_cmd('git commit -m "Auto-update codebase and pipeline for Kaggle run"')
    
    code, out, err = run_cmd("git push keshav main")
    if code != 0 and "Everything up-to-date" not in err and "Everything up-to-date" not in out:
        print(f"Git push info/warning: {err.strip() or out.strip()}")
    else:
        print("Git push to GitHub successful!")

def push_kaggle(username, key, slug):
    print(f"\n[2/4] Pushing notebook to Kaggle GPU ({username}/{slug})...")
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key

    code, out, err = run_cmd("kaggle kernels push -p .", env=env)
    print(out.strip())
    if err and "Warning" not in err:
        print(f"Kaggle push info: {err.strip()}")
    return code == 0

def monitor_kaggle(username, key, slug, poll_interval=30):
    kernel_id = f"{username}/{slug}"
    print(f"\n[3/4] Monitoring Kaggle execution status for {kernel_id}...")
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key

    while True:
        code, out, err = run_cmd(f"kaggle kernels status {kernel_id}", env=env)
        status_text = (out + err).strip()
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status_text}")

        lower = status_text.lower()
        if "complete" in lower:
            print("\n>>> Pipeline run COMPLETED successfully on Kaggle!")
            out_dir = f"kaggle_output_{slug}"
            download_outputs(username, key, slug, out_dir=out_dir)
            return True, "complete", out_dir
        elif "error" in lower:
            print("\n>>> Pipeline run FAILED with error on Kaggle!")
            out_dir = f"kaggle_output_{slug}"
            download_outputs(username, key, slug, out_dir=out_dir)
            return False, "error", out_dir
        elif "cancel" in lower:
            print("\n>>> Pipeline was cancelled on Kaggle.")
            return False, "cancelled", None

        time.sleep(poll_interval)

def download_outputs(username, key, slug, out_dir="kaggle_output"):
    os.makedirs(out_dir, exist_ok=True)
    kernel_id = f"{username}/{slug}"
    print(f"\n[4/4] Downloading kernel outputs and logs to {out_dir}...")
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key

    code, out, err = run_cmd(f"kaggle kernels output {kernel_id} -p {out_dir}", env=env)
    print(out.strip())

    # Print log file content if available
    for f in os.listdir(out_dir):
        if f.endswith(".log") or f.endswith(".txt"):
            print(f"\n--- Output file: {f} ---")
            try:
                with open(os.path.join(out_dir, f), "r", encoding="utf-8", errors="ignore") as lf:
                    lines = lf.read().splitlines()
                    print("\n".join(lines[-40:]))
            except Exception as e:
                print(f"Could not read {f}: {e}")

if __name__ == "__main__":
    account = sys.argv[1] if len(sys.argv) > 1 else "main"
    user, key, slug = get_account_config(account)
    if not user or not key:
        print(f"Error: Credentials not found for account {account}")
        sys.exit(1)

    ensure_metadata(user, slug)
    push_git()
    success = push_kaggle(user, key, slug)
    if success:
        monitor_kaggle(user, key, slug)
