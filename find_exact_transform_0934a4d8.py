import sys
import json
import numpy as np

sys.path.insert(0, "./arc-neurosymbolic-v1")
from arc_solver.utils.arc_io import load_challenges, load_solutions

challenges = load_challenges("data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
solutions = load_solutions("data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

tid = "0934a4d8"
task = challenges[tid]
truth = solutions[tid][0]

def find_transform():
    print("--- Searching for exact 2D transformation for 0934a4d8 ---")
    
    # We test on train pair 0 first
    inp0 = task.train[0].input
    out0 = task.train[0].output
    h, w = inp0.shape
    oh, ow = out0.shape
    
    mask = (inp0 == 8)
    rows, cols = np.where(mask)
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()
    
    print(f"Train 0 noise bounding box: rows=[{r1}..{r2}], cols=[{c1}..{c2}] (shape: {r2-r1+1}x{c2-c1+1})")
    
    # Test all affine transformations maps: nr = a*r + b*c + dr, nc = d*r + e*c + dc
    coeffs = [
        (1, 0, 0, 1),   # identity
        (-1, 0, 0, 1),  # V flip
        (1, 0, 0, -1),  # H flip
        (-1, 0, 0, -1), # 180 rot
        (0, 1, 1, 0),   # transpose
        (0, -1, 1, 0),  # 90 rot CW
        (0, 1, -1, 0),  # 270 rot CW
        (0, -1, -1, 0), # anti-transpose
    ]
    
    found_rule = None
    
    for a, b, d, e in coeffs:
        for dr in range(-60, 60):
            for dc in range(-60, 60):
                # Try generating prediction for train pair 0
                pred = np.zeros_like(out0)
                valid = True
                for i in range(oh):
                    for j in range(ow):
                        r = r1 + i
                        c = c1 + j
                        nr = a * r + b * c + dr
                        nc = d * r + e * c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            pred[i, j] = inp0[nr, nc]
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and np.array_equal(pred, out0):
                    print(f"FOUND MATCH FOR PAIR 0! Coeffs=({a},{b},{d},{e}), Shift=({dr},{dc})")
                    # Check across ALL train pairs
                    all_match = True
                    for pair_idx, pair in enumerate(task.train):
                        in_p = pair.input
                        out_p = pair.output
                        m_p = (in_p == (8 if pair_idx == 0 else (in_p.max() if 8 not in in_p else 8)))
                        # Find noise color for pair
                        found_c_match = False
                        for noise_c in np.unique(in_p):
                            m_c = (in_p == noise_c)
                            r_c, c_c_idx = np.where(m_c)
                            if len(r_c) == 0: continue
                            pr1, pr2 = r_c.min(), r_c.max()
                            pc1, pc2 = c_c_idx.min(), c_c_idx.max()
                            if (pr2 - pr1 + 1, pc2 - pc1 + 1) != out_p.shape:
                                continue
                            p_pred = np.zeros_like(out_p)
                            p_val = True
                            for pi in range(out_p.shape[0]):
                                for pj in range(out_p.shape[1]):
                                    pr = pr1 + pi
                                    pc = pc1 + pj
                                    pnr = a * pr + b * pc + dr
                                    pnc = d * pr + e * pc + dc
                                    if 0 <= pnr < in_p.shape[0] and 0 <= pnc < in_p.shape[1]:
                                        p_pred[pi, pj] = in_p[pnr, pnc]
                                    else:
                                        p_val = False; break
                                if not p_val: break
                            if p_val and np.array_equal(p_pred, out_p):
                                found_c_match = True
                                break
                        if not found_c_match:
                            all_match = False
                            break
                    if all_match:
                        print(f"===> UNIVERSAL MATCH FOR ALL TRAIN PAIRS! Rule: Coeffs=({a},{b},{d},{e}), Shift=({dr},{dc})")
                        found_rule = (a, b, d, e, dr, dc)
                        return found_rule
    return found_rule

find_transform()
