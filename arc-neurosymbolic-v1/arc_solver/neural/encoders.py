"""
Fixed-size feature encoders for the program ranker.

Both encoders MUST return a constant-length vector regardless of the input
(regardless of how many training pairs a task has, or which/how many
instructions a program contains), because they get concatenated and fed to a
small feed-forward net with a fixed input width.
"""
import numpy as np

# Every op the DSL can currently produce. Order matters (defines the
# per-op count layout in encode_program) but is otherwise arbitrary.
ALL_OPS = (
    "IDENTITY", "ROTATE", "FLIP", "CROP", "GRAVITY", "SCALE", "RECOLOR",
    "TRANSPOSE", "COLORMAP", "TILE", "MOSAIC", "DOWNSCALE", "BORDER",
    "STRIP_BORDER", "FILL_HOLES", "SYMMETRY_REPAIR", "PANEL_LOGIC",
    "PATTERN_COMPLETE", "SELECT_RECOLOR", "SELECT_CROP",
    "RELOCATE_OBJECT", "COPY_OBJECT",
)

PROGRAM_FEATURE_DIM = 2 + len(ALL_OPS)
TASK_FEATURE_DIM = 14


def encode_program(program):
    """[num_instructions, num_unique_ops, count(op) for op in ALL_OPS]."""
    ops_used = [i.op for i in program.instructions]
    x = [len(ops_used), len(set(ops_used))]
    x += [ops_used.count(op) for op in ALL_OPS]
    return np.asarray(x, dtype=np.float32)


def encode_task(train_pairs):
    """Aggregate, fixed-size statistics describing the training pairs.

    Deliberately shape-agnostic (mean/max over pairs) so the vector length
    doesn't depend on how many training pairs the task has.
    """
    if not train_pairs:
        return np.zeros(TASK_FEATURE_DIM, dtype=np.float32)

    same_shape = []
    h_ratio, w_ratio = [], []
    in_h, in_w, out_h, out_w = [], [], [], []
    in_colors, out_colors = [], []
    in_density, out_density = [], []
    color_growth = []

    for a, b in train_pairs:
        a = np.asarray(a)
        b = np.asarray(b)
        ah, aw = a.shape
        bh, bw = b.shape
        same_shape.append(1.0 if (ah, aw) == (bh, bw) else 0.0)
        h_ratio.append(bh / ah if ah else 0.0)
        w_ratio.append(bw / aw if aw else 0.0)
        in_h.append(ah); in_w.append(aw)
        out_h.append(bh); out_w.append(bw)
        ac = len(np.unique(a)); bc = len(np.unique(b))
        in_colors.append(ac); out_colors.append(bc)
        in_density.append(np.count_nonzero(a) / (ah * aw) if ah * aw else 0.0)
        out_density.append(np.count_nonzero(b) / (bh * bw) if bh * bw else 0.0)
        color_growth.append(bc - ac)

    x = [
        len(train_pairs),
        float(np.mean(same_shape)),
        float(np.mean(h_ratio)), float(np.mean(w_ratio)),
        float(np.mean(in_h)), float(np.mean(in_w)),
        float(np.mean(out_h)), float(np.mean(out_w)),
        float(np.mean(in_colors)), float(np.mean(out_colors)),
        float(np.mean(in_density)), float(np.mean(out_density)),
        float(np.mean(color_growth)),
        float(np.std(same_shape)),
    ]
    return np.asarray(x, dtype=np.float32)


def encode_pair(program, train_pairs):
    """Concatenated (task_features, program_features) -> fixed-size vector."""
    return np.concatenate([encode_task(train_pairs), encode_program(program)])
