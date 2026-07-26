from ..core.scene import Scene
from .signatures import make_signature

class TaskAnalyzer:
    def analyze(self, train_pairs):
        sig=make_signature(train_pairs)
        scenes=[(Scene.from_grid(a),Scene.from_grid(b)) for a,b in train_pairs]
        return sig, scenes
