"""
download_solved540_raw.py
-------------------------
Downloads solvers from GitMonsters/SOLVED-540-of-540 using raw.githubusercontent.com
(no auth needed, higher rate limits). Uses concurrent threads for speed.
"""
from pathlib import Path
import requests
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

REPO = 'GitMonsters/SOLVED-540-of-540'
BRANCH = 'main'
RAW_BASE = f'https://raw.githubusercontent.com/{REPO}/{BRANCH}'
SOLVES_ROOT = Path('external_solutions/solved_540/solves')
CATALOG_PATH = Path('external_solutions/solved_540/catalog.json')
PATHS_FILE   = Path('external_solutions/solved_540/solver_paths.json')
MAX_WORKERS = 12

# Load pre-fetched solver paths (avoids API rate limit on tree endpoint)
if PATHS_FILE.exists():
    solver_paths = json.loads(PATHS_FILE.read_text())
    print(f'Loaded {len(solver_paths)} solver paths from {PATHS_FILE}')
else:
    print('ERROR: solver_paths.json not found. Run the auth fetch first.')
    import sys; sys.exit(1)

# Download catalog.json
print('Downloading catalog.json...')
CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
cat_resp = requests.get(f'{RAW_BASE}/catalog.json', timeout=30)
if cat_resp.ok:
    CATALOG_PATH.write_bytes(cat_resp.content)
    catalog = json.loads(cat_resp.content)
    cat_by_path = {e.get('solver_file', ''): e for e in catalog}
    print(f'Catalog: {len(catalog)} entries')
else:
    cat_by_path = {}
    print(f'Catalog fetch failed: {cat_resp.status_code}')

# Step 2: Download each solver using raw URLs
already_downloaded = set(d.parent.name for d in SOLVES_ROOT.glob('*/solver.py'))
to_download = [p for p in solver_paths if p.split('/')[1] not in already_downloaded]
print(f'Already have: {len(already_downloaded)}  To download: {len(to_download)}')

lock = threading.Lock()
downloaded = [0]
errors = [0]

def download_one(path):
    task_id = path.split('/')[1]
    local_path = SOLVES_ROOT / task_id / 'solver.py'
    local_path.parent.mkdir(parents=True, exist_ok=True)

    url = f'{RAW_BASE}/{path}'
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            local_path.write_bytes(r.content)
            # Save metadata if in catalog
            meta = cat_by_path.get(path)
            if meta:
                (local_path.parent / 'meta.json').write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False)
                )
            with lock:
                downloaded[0] += 1
                if downloaded[0] % 50 == 0:
                    print(f'  Progress: {downloaded[0]}/{len(to_download)}')
            return True
        elif r.status_code == 429:
            time.sleep(5)
            return False
        else:
            with lock:
                errors[0] += 1
            return False
    except Exception as e:
        with lock:
            errors[0] += 1
        return False

print(f'\nDownloading with {MAX_WORKERS} threads...')
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_one, p): p for p in to_download}
    for future in as_completed(futures):
        future.result()

total = len(list(SOLVES_ROOT.glob('*/solver.py')))
print(f'\nDone! Total solvers on disk: {total}')
print(f'Downloaded: {downloaded[0]}  Errors: {errors[0]}')
