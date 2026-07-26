import argparse
from arc_solver.utils.arc_io import (
    load_challenges,
    make_submission,
    write_submission,
)
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--output", default="submission.json")
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    challenges = load_challenges(args.challenges)
    solver = NeuroSymbolicARCSolver(
        beam_width=args.beam_width,
        max_depth=args.max_depth,
    )

    submission = make_submission(challenges, solver)
    write_submission(submission, args.output)

    print(f"Solved tasks: {len(submission)}")
    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()
