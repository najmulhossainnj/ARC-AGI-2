# Google Colab Worker Script for ARC Meta-Learning
# Run this inside any Google Colab instance after cloning ARC-AGI-2 repo.
#
# Setup in Colab:
#   !git clone https://github.com/najmulhossainnj/ARC-AGI-2.git
#   %cd ARC-AGI-2
#   !pip install -q numpy scipy requests PyGithub
#   !python distributed/colab_worker.py

import os
import sys
import time
import json
import requests

# ─── Configuration ────────────────────────────────────────────────────────────
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "https://smooth-bears-wave.loca.lt")
WORKER_ID = f"local_worker_{os.getpid()}_{int(time.time())}"
EMPTY_QUEUE_PAUSE = 5   # seconds to wait when no tasks are pending
MAX_EMPTY_RETRIES = 0   # 0 = infinite (keep looping forever)
TASK_PAUSE = 0.5        # brief pause between tasks

print(f"=== Starting ARC Worker: {WORKER_ID} ===")
print(f"Connecting to Coordinator at: {COORDINATOR_URL}")

# ─── sys.path: support both flat (Colab) and nested (legacy) layouts ─────────
_cwd = os.getcwd()
_repo_root = _cwd

# Detect nested structures (e.g., /content/ARC-AGI-2/ARC-AGI-2/...)
while True:
    inner = os.path.join(_repo_root, os.path.basename(_repo_root))
    if os.path.isdir(inner) and os.path.abspath(inner) != os.path.abspath(_repo_root):
        _repo_root = inner
    else:
        break

# Add parent of arc_solver package to path
_arc_solver_parent = _repo_root
if not os.path.isdir(os.path.join(_arc_solver_parent, "arc_solver")):
    # try one level up
    candidate = os.path.dirname(_arc_solver_parent)
    if os.path.isdir(os.path.join(candidate, "arc_solver")):
        _arc_solver_parent = candidate

if _arc_solver_parent not in sys.path:
    sys.path.insert(0, _arc_solver_parent)

print(f"Adding arc_solver parent directory to sys.path: {_arc_solver_parent}")

# ─── Imports ──────────────────────────────────────────────────────────────────
try:
    from arc_solver.utils.arc_io import load_challenges, load_solutions
    from arc_solver.solver.pipeline import NeuroSymbolicARCSolver
    from arc_solver.meta.diagnostic_engine import DiagnosticEngine
    from arc_solver.meta.auto_primitive_injector import inject_llm_solve
    import numpy as np
except ImportError as e:
    print(f"[Worker] Import error: {e}")
    print("[Worker] Make sure you are running from the repo root: ARC-AGI-2/")
    sys.exit(1)

# ─── Load local challenge data ────────────────────────────────────────────────
DATA_SEARCH_PATHS = [
    os.path.join(_arc_solver_parent, "data", "arc-prize-2026-arc-agi-2", "arc-agi_training_challenges.json"),
    os.path.join(_repo_root, "data", "arc-prize-2026-arc-agi-2", "arc-agi_training_challenges.json"),
    "data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json",
]
SOL_SEARCH_PATHS = [p.replace("_challenges.json", "_solutions.json") for p in DATA_SEARCH_PATHS]

train_ch, train_sol = None, None
for ch_path, sol_path in zip(DATA_SEARCH_PATHS, SOL_SEARCH_PATHS):
    if os.path.exists(ch_path):
        print(f"[Worker] Loading challenges from: {ch_path}")
        train_ch = load_challenges(ch_path)
        if os.path.exists(sol_path):
            train_sol = load_solutions(sol_path)
        else:
            print(f"[Worker] Solutions file not found, accuracy check disabled.")
        break

if train_ch is None:
    print("[Worker] ERROR: Could not find arc-agi_training_challenges.json")
    print("[Worker] Searched:", DATA_SEARCH_PATHS)
    sys.exit(1)

# ─── Initialize solver & diagnostic engine ───────────────────────────────────
solver = NeuroSymbolicARCSolver(beam_width=20, max_depth=2, ranker=None)
engine = DiagnosticEngine(use_llm=True)

# ─── HTTP helpers with tunnel bypass header ───────────────────────────────────
TUNNEL_HEADERS = {"bypass-tunnel-reminder": "true", "Content-Type": "application/json"}

def coordinator_post(endpoint: str, payload: dict, timeout: int = 10) -> dict:
    url = f"{COORDINATOR_URL}{endpoint}"
    resp = requests.post(url, json=payload, headers=TUNNEL_HEADERS, timeout=timeout)
    return resp.json()

# ─── Main worker loop ─────────────────────────────────────────────────────────
empty_streak = 0

while True:
    try:
        # Request next task from coordinator
        data = coordinator_post("/api/get_task", {"worker_id": WORKER_ID})

        if data.get("status") == "empty" or not data.get("task_id"):
            empty_streak += 1
            if MAX_EMPTY_RETRIES > 0 and empty_streak >= MAX_EMPTY_RETRIES:
                print(f"No pending tasks in queue. Worker done after {empty_streak} empty checks.")
                break
            print(f"No pending tasks in queue right now. Pausing {EMPTY_QUEUE_PAUSE}s before checking again...")
            time.sleep(EMPTY_QUEUE_PAUSE)
            continue

        empty_streak = 0
        tid = data["task_id"]
        print(f"\n[{WORKER_ID}] Processing task: {tid}")

        if tid not in train_ch:
            print(f"[{WORKER_ID}] Task {tid} not found in local dataset, skipping.")
            coordinator_post("/api/submit_result", {
                "worker_id": WORKER_ID, "task_id": tid,
                "solved": False, "source": "not_found", "program_str": "",
            })
            continue

        task = train_ch[tid]
        train_pairs = [(p.input, p.output) for p in task.train]

        # Baseline truth (if solutions available)
        truth = None
        if train_sol and tid in train_sol:
            truth = train_sol[tid]

        # ── Step 1: Baseline neuro-symbolic solver ────────────────────────────
        t0 = time.time()
        preds, programs = solver.solve_task(train_pairs, task.test)
        dt = time.time() - t0

        solved = False
        prog_str = ""

        if preds and len(preds) > 0 and truth is not None:
            for attempt in preds[0]:
                if np.array_equal(np.array(attempt), np.array(truth[0])):
                    solved = True
                    prog_str = programs[0].program if hasattr(programs[0], "program") else str(programs[0])
                    break
        elif preds and len(preds) > 0 and truth is None:
            # Without ground truth, trust the solver if it produced output
            solved = False  # can't verify without truth

        source = "original" if solved else "none"
        new_primitive_code = None

        # ── Step 2: Diagnostic Engine (rule-based + LLM) ─────────────────────
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
                else:
                    prog_str = diagnosis.source

        # ── Step 3: Report to coordinator ────────────────────────────────────
        report = {
            "worker_id": WORKER_ID,
            "task_id": tid,
            "solved": solved,
            "source": source,
            "program_str": prog_str,
            "new_primitive_code": new_primitive_code,
        }
        coordinator_post("/api/submit_result", report, timeout=10)
        print(f"[{WORKER_ID}] Task {tid} finished. Solved: {solved} ({source})")

        time.sleep(TASK_PAUSE)

    except requests.exceptions.ReadTimeout:
        print(f"[{WORKER_ID}] Network/Tunnel temporary error: Read timed out. Pausing {EMPTY_QUEUE_PAUSE}s before reconnecting...")
        time.sleep(EMPTY_QUEUE_PAUSE)
    except requests.exceptions.ConnectionError as e:
        print(f"[{WORKER_ID}] Connection error: {e}. Pausing {EMPTY_QUEUE_PAUSE}s before reconnecting...")
        time.sleep(EMPTY_QUEUE_PAUSE)
    except ValueError as e:
        # JSON decode error (tunnel returning HTML)
        print(f"[{WORKER_ID}] Network/Tunnel temporary error: {e}. Pausing {EMPTY_QUEUE_PAUSE}s before reconnecting...")
        time.sleep(EMPTY_QUEUE_PAUSE)
    except Exception as e:
        print(f"[{WORKER_ID}] Worker loop error: {e}")
        time.sleep(EMPTY_QUEUE_PAUSE)
