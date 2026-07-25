import argparse,json
from arc_solver.utils.io import load_single
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("task")
    args=ap.parse_args()
    task=load_single(args.task)
    preds,valid=NeuroSymbolicARCSolver().solve(task)
    print("Exact programs:")
    for c in valid[:10]: print(c.program,c.error)
    print(json.dumps(preds,indent=2))

if __name__=="__main__":
    main()
