"""
Builds a training set for the program ranker directly from the beam
searcher's own search traces, then trains a small classifier on it.

For every training task we run the (ranker-free) beam search and record
every candidate program it evaluated along with its training error. A
program with error 0 is a positive example ("this program solves this
task"); everything else is a negative example. Because a single task can
produce thousands of negatives and only a handful of positives, negatives
are subsampled per task, biased toward the lowest-error ("near miss")
candidates -- those are the ones a ranker most needs to learn to distinguish
from true solutions.
"""
from __future__ import annotations
import random
import numpy as np

from ..synthesis.beam_search import BeamSearcher
from .encoders import encode_pair
from .numpy_mlp import NumpyMLP


def collect_examples(tasks, beam_width=40, max_depth=2, max_negatives_per_task=8,
                      max_tasks=None, seed=0, verbose=False):
    """tasks: dict[task_id -> ARCTask]. Returns list of (program, train_pairs, label)."""
    rng = random.Random(seed)
    task_ids = list(tasks.keys())
    rng.shuffle(task_ids)
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]

    searcher = BeamSearcher(beam_width=beam_width, max_depth=max_depth)
    examples = []

    for n, task_id in enumerate(task_ids):
        task = tasks[task_id]
        train_pairs = [(p.input, p.output) for p in task.train]
        if not train_pairs:
            continue

        collector = []
        try:
            found, _beam = searcher.search(train_pairs, collector=collector)
        except Exception:
            continue

        positives = [c for c in collector if c.error == 0]
        negatives = [c for c in collector if c.error != 0]
        negatives.sort(key=lambda c: c.error)  # near-misses first

        for c in positives:
            examples.append((c.program, train_pairs, 1))
        for c in negatives[:max_negatives_per_task]:
            examples.append((c.program, train_pairs, 0))

        if verbose and (n % 50 == 0):
            print(f"[{n}/{len(task_ids)}] {task_id}: "
                  f"{len(positives)} pos, {len(collector)} evaluated")

    return examples


def examples_to_arrays(examples):
    X = np.stack([encode_pair(prog, pairs) for prog, pairs, _ in examples])
    y = np.asarray([label for _, _, label in examples], dtype=np.float32)
    return X, y


def train_ranker(tasks, beam_width=40, max_depth=2, max_negatives_per_task=8,
                  max_tasks=200, hidden=32, epochs=300, lr=0.05, seed=0,
                  verbose=False):
    """Train the NumPy ranker MLP on a sample of ARC training tasks.

    Returns (model, stats) where stats has basic dataset/accuracy info.
    """
    examples = collect_examples(
        tasks, beam_width=beam_width, max_depth=max_depth,
        max_negatives_per_task=max_negatives_per_task,
        max_tasks=max_tasks, seed=seed, verbose=verbose,
    )
    if not examples:
        raise ValueError("No training examples collected -- check `tasks`.")

    X, y = examples_to_arrays(examples)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    model = NumpyMLP(input_dim=X.shape[1], hidden=hidden, seed=seed)
    model.fit(X, y, epochs=epochs, lr=lr, verbose=verbose)

    p = model.predict_proba(X)
    train_acc = float(np.mean((p > 0.5) == (y > 0.5)))

    stats = {
        "num_examples": len(y),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "train_accuracy": train_acc,
        "input_dim": X.shape[1],
    }
    return model, stats


def save_ranker(model, path):
    model.save(path)


def load_ranker_model(path):
    return NumpyMLP.load(path)
