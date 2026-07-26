import argparse,json
from arc_solver.utils.io import load_tasks
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--output",default="submission.json")
    args=ap.parse_args()
    tasks=load_tasks(args.data)
    solver=NeuroSymbolicARCSolver()
    submission={}
    for tid,task in tasks.items():
        preds,_=solver.solve(task)
        rows=[]
        for choices in preds:
            if not choices:
                fallback=task.test[len(rows)].tolist()
                choices=[fallback,fallback]
            elif len(choices)==1:
                choices=[choices[0],choices[0]]
            rows.append({"attempt_1":choices[0],"attempt_2":choices[1]})
        submission[tid]=rows
    with open(args.output,"w") as f: json.dump(submission,f)
    print("Wrote",args.output)

if __name__=="__main__":
    main()
