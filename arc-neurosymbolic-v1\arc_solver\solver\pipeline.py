from ..neural.ranker import load_ranker
from .specialized_solvers import CategorySpecializedARCSolver

class NeuroSymbolicARCSolver:
    """
    ARC solver interface with category-specialized solver routing.
    """

    def __init__(self, beam_width=100, max_depth=3, ranker=None, ranker_weights=None):
        self.ranker = ranker if ranker is not None else load_ranker(ranker_weights)
        self.specialized_router = CategorySpecializedARCSolver(beam_width, max_depth, self.ranker)

    def solve_task(self, train_pairs, test_inputs):
        return self.specialized_router.solve_task(train_pairs, test_inputs)
