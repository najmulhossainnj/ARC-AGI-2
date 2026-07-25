from dataclasses import dataclass
import numpy as np

@dataclass
class TaskSignature:
    same_grid_size: bool
    color_count_changed: bool
    object_count_delta: int
    nonzero_count_delta: int
    likely_geometry: bool
    likely_recolor: bool
    likely_creation_deletion: bool

def make_signature(train_pairs):
    if not train_pairs:
        return TaskSignature(True,False,0,0,False,False,False)
    same_size=all(a.shape==b.shape for a,b in train_pairs)
    color_changed=any(set(np.unique(a))!=set(np.unique(b)) for a,b in train_pairs)
    nz_delta=sum(int(np.count_nonzero(b)-np.count_nonzero(a)) for a,b in train_pairs)
    return TaskSignature(
        same_size,
        color_changed,
        0,
        nz_delta,
        not same_size,
        color_changed,
        nz_delta!=0
    )
