class NullRanker:
    def score(self,program,train_pairs):
        return 0.0


class TrainedRanker:
    """Wraps a trained NumpyMLP; higher score = more likely to be a correct
    program for this task. Used to bias beam search toward promising
    candidates instead of relying on raw MDL complexity alone."""
    def __init__(self, model):
        self.model = model

    def score(self, program, train_pairs):
        from .encoders import encode_pair
        x = encode_pair(program, train_pairs)
        try:
            return float(self.model.predict_proba(x)[0])
        except Exception:
            return 0.0


import os

DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "ranker.npz"
)


def load_ranker(weights_path=None):
    """Load the trained ranker if weights exist on disk, else NullRanker.

    Never raises: a missing/corrupt checkpoint just means beam search falls
    back to plain MDL scoring, which is the existing (pre-ranker) behavior.
    """
    weights_path = weights_path or DEFAULT_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        return NullRanker()
    try:
        from .numpy_mlp import NumpyMLP
        model = NumpyMLP.load(weights_path)
        return TrainedRanker(model)
    except Exception:
        return NullRanker()


try:
    import torch
    import torch.nn as nn

    class ProgramRanker(nn.Module):
        def __init__(self,input_dim,hidden=128):
            super().__init__()
            self.net=nn.Sequential(
                nn.Linear(input_dim,hidden),nn.ReLU(),
                nn.Linear(hidden,hidden),nn.ReLU(),
                nn.Linear(hidden,1)
            )
        def forward(self,x):
            return self.net(x)

except Exception:
    ProgramRanker=None
