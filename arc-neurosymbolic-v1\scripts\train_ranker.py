import argparse
import os

from arc_solver.utils.arc_io import load_challenges
from arc_solver.neural.trainer import train_ranker, save_ranker
from arc_solver.neural.ranker import DEFAULT_WEIGHTS_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True,
                         help="Path to arc-agi_training_challenges.json")
    parser.add_argument("--output", default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-negatives-per-task", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = load_challenges(args.challenges)

    model, stats = train_ranker(
        tasks,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        max_negatives_per_task=args.max_negatives_per_task,
        max_tasks=args.max_tasks,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        verbose=True,
    )

    print("Training stats:", stats)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_ranker(model, args.output)
    print(f"Saved ranker weights to {args.output}"
          f"{'.npz' if not args.output.endswith('.npz') else ''}")


if __name__ == "__main__":
    main()
