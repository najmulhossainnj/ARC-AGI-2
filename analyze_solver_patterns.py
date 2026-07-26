"""
analyze_solver_patterns.py
--------------------------
Reads all solver.py files from external_solutions/solved_540/solves/
Extracts: docstring, function names, key algorithm patterns via regex/AST.
Clusters them into technique families.
Outputs a JSON report + a summary for building generalized analyzers.
"""
from pathlib import Path
import ast, re, json
from collections import Counter, defaultdict

SOLVES_ROOT = Path('external_solutions/solved_540/solves')
CATALOG_PATH = Path('external_solutions/solved_540/catalog.json')
OUT_PATH = Path('external_solutions/solved_540/technique_clusters.json')

# ── Technique keyword signatures ──────────────────────────────────────────────
# Maps technique name → list of keywords to search for in source code
TECHNIQUE_PATTERNS = {
    # Object / shape operations
    'connected_components':     ['connected_component', 'flood_fill', 'bfs', 'deque', 'visited'],
    'bounding_box':             ['min_r', 'max_r', 'min_c', 'max_c', 'bbox', 'bounding'],
    'object_count':             ['len(components)', 'len(comps)', 'num_objects'],
    # Spatial transforms
    'rotation':                 ['rot90', 'rotate', 'np.rot90', 'rotation'],
    'flip_mirror':              ['flipud', 'fliplr', 'flip', 'mirror', 'reflect'],
    'translation_shift':        ['translate', 'shift', 'offset', 'dr, dc'],
    'gravity':                  ['gravity', 'fall', 'settle', 'slide', 'snap_to'],
    # Color operations
    'color_mapping':            ['color_map', 'colormap', 'remap', 'substitute', 'mapping = {', 'palette'],
    'color_substitution':       ['replace_color', 'swap_color', 'recolor'],
    # Pattern / tiling
    'tiling':                   ['tile', 'tiling', 'repeat', 'modulo', '% period', 'periodic'],
    'pattern_match':            ['pattern', 'template', 'match', 'sliding'],
    # Fill operations
    'flood_fill':               ['flood_fill', 'fill_region', 'interior', 'exterior'],
    'concentric':               ['concentric', 'chebyshev', 'distance_transform', 'ring', 'shell'],
    'symmetry':                 ['symmetry', 'symmetric', 'flipud', 'fliplr', 'palindrome'],
    # Ray / physics
    'ray_casting':              ['ray', 'beam', 'cast', 'emit', 'propagate', 'bounce'],
    'gravity_physics':          ['gravity', 'fall', 'drop', 'projectile'],
    # Sorting / ordering
    'sort_objects':             ['sort', 'sorted', 'order', 'rank', 'argsort'],
    # Grid structure
    'grid_sections':            ['separator', 'divider', 'split', 'panel', 'section'],
    'border_frame':             ['border', 'frame', 'edge', 'boundary', 'perimeter'],
    # Counting / math
    'counting':                 ['count', 'sum', 'total', 'frequency', 'histogram'],
    'size_comparison':          ['area', 'size', 'largest', 'smallest', 'bigger'],
    # Stamping / copying
    'stamp':                    ['stamp', 'blit', 'paste', 'copy_shape', 'template'],
    'object_placement':         ['place', 'position', 'locate', 'anchor'],
    # Topology
    'topology':                 ['hole', 'enclosed', 'interior', 'topolog', 'genus'],
    # Arrows / indicators
    'arrow_indicator':          ['arrow', 'indicator', 'direction', 'pointer', 'head'],
    # Chaining / network
    'network_chain':            ['chain', 'network', 'path', 'reach', 'propagat'],
    # Crop / extract
    'crop':                     ['crop', 'extract', 'clip', 'trim', 'cutout'],
    # Output size change
    'resize':                   ['resize', 'scale', 'zoom', 'output_size', 'new_h', 'new_w'],
    # Diagonal
    'diagonal':                 ['diagonal', 'diag', 'antidiag', 'slope'],
    # Stacking / assembly
    'assembly':                 ['assemble', 'stitch', 'combine', 'merge', 'compose'],
}


def extract_docstring(source: str) -> str:
    """Extract top-level or first function docstring."""
    try:
        tree = ast.parse(source)
        ds = ast.get_docstring(tree)
        if ds:
            return ds[:500]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ds = ast.get_docstring(node)
                if ds:
                    return ds[:500]
    except Exception:
        pass
    return ''


def extract_function_names(source: str) -> list:
    try:
        tree = ast.parse(source)
        return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    except Exception:
        return []


def detect_techniques(source: str) -> list:
    src_lower = source.lower()
    found = []
    for tech, keywords in TECHNIQUE_PATTERNS.items():
        if any(kw.lower() in src_lower for kw in keywords):
            found.append(tech)
    return found


def analyze_all_solvers():
    if not SOLVES_ROOT.exists():
        print(f'ERROR: {SOLVES_ROOT} not found. Run download_solved540.py first.')
        return

    # Load catalog for metadata
    cat_by_id = {}
    if CATALOG_PATH.exists():
        catalog = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
        for e in catalog:
            tid = e.get('solver_file', '').split('/')[1] if e.get('solver_file') else ''
            if tid:
                cat_by_id[tid] = e

    task_dirs = [d for d in SOLVES_ROOT.iterdir() if d.is_dir() and (d / 'solver.py').exists()]
    print(f'Analyzing {len(task_dirs)} solvers...')

    results = []
    tech_counter = Counter()
    tech_to_tasks = defaultdict(list)

    for task_dir in sorted(task_dirs):
        task_id = task_dir.name
        solver_path = task_dir / 'solver.py'
        source = solver_path.read_text(encoding='utf-8', errors='replace')

        doc = extract_docstring(source)
        fns = extract_function_names(source)
        techs = detect_techniques(source)
        line_count = len(source.splitlines())

        meta = cat_by_id.get(task_id, {})
        rule_summary = meta.get('rule_summary', '')
        task_name = meta.get('name', '')

        entry = {
            'task_id': task_id,
            'task_name': task_name,
            'rule_summary': rule_summary,
            'docstring': doc[:300],
            'functions': fns,
            'techniques': techs,
            'line_count': line_count,
        }
        results.append(entry)

        for t in techs:
            tech_counter[t] += 1
            tech_to_tasks[t].append(task_id)

    # Build cluster report
    cluster_report = {
        'total_solvers': len(results),
        'technique_frequency': dict(tech_counter.most_common()),
        'technique_clusters': {
            t: {
                'count': len(tasks),
                'task_ids': tasks[:20],  # First 20 as sample
                'example_names': [
                    cat_by_id.get(tid, {}).get('name', '')[:80]
                    for tid in tasks[:5]
                ],
                'example_rules': [
                    cat_by_id.get(tid, {}).get('rule_summary', '')[:120]
                    for tid in tasks[:5]
                    if cat_by_id.get(tid, {}).get('rule_summary', '')
                ]
            }
            for t, tasks in sorted(tech_to_tasks.items(), key=lambda x: -len(x[1]))
        },
        'tasks': results,
    }

    OUT_PATH.write_text(json.dumps(cluster_report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nAnalysis complete! Saved to {OUT_PATH}')
    print(f'\n=== TECHNIQUE FREQUENCY (top 30) ===')
    for tech, count in tech_counter.most_common(30):
        bar = '█' * min(count // 5, 40)
        print(f'  {tech:30s} {count:4d}  {bar}')
    return cluster_report


if __name__ == '__main__':
    analyze_all_solvers()
