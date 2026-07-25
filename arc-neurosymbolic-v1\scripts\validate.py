import argparse
from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    tasks = load_challenges(args.challenges)
    solutions = load_solutions(args.solutions)

    solver = NeuroSymbolicARCSolver(
        beam_width=args.beam_width,
        max_depth=args.max_depth,
    )

    total = 0
    correct = 0

    for task_id, task in tasks.items():
        predictions, _ = solver.solve_task(
            [(p.input, p.output) for p in task.train],
            task.test,
        )

        truth = solutions[task_id]

        if len(predictions) != len(truth):
            print(
                f"WARNING {task_id}: predicted {len(predictions)} "
                f"test outputs, expected {len(truth)}"
            )

        for i, choices in enumerate(predictions):
            if i >= len(truth):
                break

            total += 1
            expected = truth[i]

            if expected in choices:
                correct += 1

    print(f"Correct test inputs: {correct}/{total}")
    print(f"Accuracy: {correct / max(total, 1):.4f}")

if __name__ == "__main__":
    main()
