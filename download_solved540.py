"""
download_solved540.py
---------------------
Downloads all solver.py files from GitMonsters/SOLVED-540-of-540
and saves them locally for analysis.
Skips files already downloaded (incremental).
"""
from github import Github
from pathlib import Path
import base64, json, time

REPO = 'GitMonsters/SOLVED-540-of-540'
OUT_ROOT = Path('external_solutions/solved_540')

g = Github()
repo = g.get_repo(REPO)

print(f'Fetching file tree from {REPO}...')
tree = repo.get_git_tree('main', recursive=True)
all_files = [(f.path, f.sha) for f in tree.tree if f.type == 'blob']

# Get all solver paths from solves/ directory
solver_paths = [p for p, _ in all_files if p.startswith('solves/') and p.endswith('/solver.py')]
print(f'Found {len(solver_paths)} solvers in solves/')

# Also get catalog.json for metadata
try:
    cat_content = repo.get_contents('catalog.json')
    catalog = json.loads(base64.b64decode(cat_content.content))
    # Build lookup by solver_file
    cat_by_path = {e['solver_file']: e for e in catalog if 'solver_file' in e}
    print(f'Catalog entries: {len(catalog)}')
    (OUT_ROOT / 'catalog.json').parent.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / 'catalog.json').write_bytes(base64.b64decode(cat_content.content))
except Exception as e:
    print(f'Catalog error: {e}')
    cat_by_path = {}

downloaded = 0
skipped = 0
errors = 0

for path in solver_paths:
    task_id = path.split('/')[1]
    local_path = OUT_ROOT / 'solves' / task_id / 'solver.py'
    
    if local_path.exists():
        skipped += 1
        continue
    
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        fc = repo.get_contents(path)
        raw = base64.b64decode(fc.content)
        local_path.write_bytes(raw)
        
        # Save metadata alongside if available
        if path in cat_by_path:
            meta = cat_by_path[path]
            (local_path.parent / 'meta.json').write_text(
                json.dumps(meta, indent=2, ensure_ascii=False)
            )
        
        downloaded += 1
        if downloaded % 50 == 0:
            print(f'  Downloaded {downloaded}/{len(solver_paths)}...')
        
        # Small rate-limit pause every 100 files
        if downloaded % 100 == 0:
            time.sleep(2)
    
    except Exception as e:
        print(f'  ERROR {task_id}: {e}')
        errors += 1

print()
print(f'Done! Downloaded={downloaded}  Skipped={skipped}  Errors={errors}')
print(f'Solvers saved to: {OUT_ROOT}/solves/')
