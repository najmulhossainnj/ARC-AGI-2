# Google Colab Worker Script for ARC Meta-Learning
# Run this notebook/script inside any Google Colab GPU/CPU instance!

import os
import sys
import time
import json
import requests
from pathlib import Path

# Master Coordinator Settings & Public Tunnel URL
COORDINATOR_URL = os.environ.get("ARC_COORDINATOR_URL", "https://smooth-bears-wave.loca.lt")
WORKER_ID = os.environ.get("ARC_WORKER_ID", f"colab_worker_{os.getpid()}_{int(time.time())}")

print(f"=== Starting Google Colab Worker: {WORKER_ID} ===")
print(f"Connecting to Coordinator at: {COORDINATOR_URL}")

# Walk up and down to find the directory containing 'arc_solver'
curr = Path(__file__).resolve().parent
search_paths = [curr, curr.parent, Path.cwd()]
for p in list(curr.parents) + list(Path.cwd().parents):
    search_paths.append(p)

solver_parent = None
for p in search_paths:
    if (p / "arc_solver").exists():
        solver_parent = p
        break
    elif (p / "arc-neurosymbolic-v1" / "arc_solver").exists():
        solver_parent = p / "arc-neurosymbolic-v1"
        break

if solver_parent:
    print(f"Adding arc_solver parent directory to sys.path: {solver_parent}")
    sys.path.insert(0, str(solver_parent))

# Fallback: search recursively in working directory
for root, dirs, files in os.walk(os.getcwd()):
    if "arc_solver" in dirs:
        print(f"Found arc_solver in: {root}")
        sys.path.insert(0, root)
        break

# Setup data directory paths
repo_root = solver_parent or Path.cwd()
DATA_DIR = repo_root / "data" / "arc-prize-2026-arc-agi-2"
os.makedirs(DATA_DIR, exist_ok=True)
CHALLENGES_PATH = str(DATA_DIR / "arc-agi_training_challenges.json")
SOLUTIONS_PATH = str(DATA_DIR / "arc-agi_training_solutions.json")

def download_arc_dataset(chal_path, sol_path):
    print("Downloading ARC training dataset JSONs directly from GitHub...")
    ch_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/main/data/arc-agi_training_challenges.json"
    sol_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/main/data/arc-agi_training_solutions.json"
    try:
        r1 = requests.get(ch_url, timeout=30)
        r2 = requests.get(sol_url, timeout=30)
        if r1.status_code == 200 and r2.status_code == 200:
            with open(chal_path, "wb") as f:
                f.write(r1.content)
            with open(sol_path, "wb") as f:
                f.write(r2.content)
            print("Dataset downloaded successfully!")
            return True
        else:
            # Fallback to alternate raw url layout
            ch_url2 = "https://raw.githubusercontent.com/arcprize/ARC-AGI/main/data/arc-agi_training_challenges.json"
            sol_url2 = "https://raw.githubusercontent.com/arcprize/ARC-AGI/main/data/arc-agi_training_solutions.json"
            r1 = requests.get(ch_url2, timeout=30)
            r2 = requests.get(sol_url2, timeout=30)
            if r1.status_code == 200 and r2.status_code == 200:
                with open(chal_path, "wb") as f:
                    f.write(r1.content)
                with open(sol_path, "wb") as f:
                    f.write(r2.content)
                print("Dataset downloaded from arcprize successfully!")
                return True
    except Exception as e:
        print(f"Dataset download notice: {e}")
    return False

# Download datasets if missing or empty
if not os.path.exists(CHALLENGES_PATH) or os.path.getsize(CHALLENGES_PATH) == 0:
    download_arc_dataset(CHALLENGES_PATH, SOLUTIONS_PATH)

from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver
from arc_solver.meta.diagnostic_engine import DiagnosticEngine
from arc_solver.meta.auto_primitive_injector import inject_llm_solve

# Load challenge set
try:
    train_ch = load_challenges(CHALLENGES_PATH)
    train_sol = load_solutions(SOLUTIONS_PATH)
except Exception as e:
    print(f"Retrying dataset download due to load error: {e}")
    download_arc_dataset(CHALLENGES_PATH, SOLUTIONS_PATH)
    train_ch = load_challenges(CHALLENGES_PATH)
    train_sol = load_solutions(SOLUTIONS_PATH)

solver = NeuroSymbolicARCSolver(beam_width=20, max_depth=2, ranker=None)
engine = DiagnosticEngine(use_llm=True)

headers = {"bypass-tunnel-reminder": "true"}

# Main worker execution loop
while True:
    try:
        # Request next task from central coordinator
        resp = requests.post(f"{COORDINATOR_URL}/api/get_task", json={"worker_id": WORKER_ID}, headers=headers, timeout=10)
        data = resp.json()
        
        if data.get("status") == "empty" or not data.get("task_id"):
            print("No more pending tasks in queue. Worker done!")
            break
            
        tid = data["task_id"]
        print(f"\n[{WORKER_ID}] Processing task: {tid}")
        
        task = train_ch[tid]
        truth = train_sol[tid]
        train_pairs = [(p.input, p.output) for p in task.train]
        
        # Step 1: Run baseline neurosymbolic solver
        t0 = time.time()
        preds, programs = solver.solve_task(train_pairs, task.test)
        dt = time.time() - t0
        
        solved = False
        prog_str = ""
        if preds and len(preds) > 0:
            for attempt in preds[0]:
                import numpy as np
                if np.array_equal(np.array(attempt), np.array(truth[0])):
                    solved = True
                    prog_str = programs[0].program if hasattr(programs[0], 'program') else str(programs[0])
                    break

        source = "original" if solved else "none"
        new_primitive_code = None

        # Step 2: If missed, run Diagnostic Engine (Rule-based + Gemini LLM fallback)
        if not solved:
            diagnosis = engine.diagnose(tid, train_pairs, task.test)
            if diagnosis.success:
                solved = True
                source = diagnosis.source
                if diagnosis.source == "llm" and diagnosis.solve_fn:
                    new_primitive_code = getattr(diagnosis.solve_fn, "_llm_code", None)
                    op_name = inject_llm_solve(tid, diagnosis.solve_fn)
                    prog_str = op_name
                elif diagnosis.candidate:
                    prog_str = str(diagnosis.candidate)

        # Step 3: Report results to coordinator
        report = {
            "worker_id": WORKER_ID,
            "task_id": tid,
            "solved": solved,
            "source": source,
            "program_str": prog_str,
            "new_primitive_code": new_primitive_code
        }
        requests.post(f"{COORDINATOR_URL}/api/submit_result", json=report, headers=headers, timeout=10)
        print(f"[{WORKER_ID}] Task {tid} finished. Solved: {solved} ({source})")
        
        # Brief pause between tasks
        time.sleep(1)
        
    except Exception as e:
        print(f"Worker loop error: {e}")
        time.sleep(5)
